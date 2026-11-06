package bm25

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"unsafe"
)

const (
	cuckooCacheMagic   = "B25CUK01"
	cuckooCacheVersion = uint64(1)
	cuckooHeaderBytes  = 80
)

var errCuckooCycle = errors.New("cuckoo insertion cycle")

// CuckooBuildConfig controls the physical Appendix-E dictionary. Each item is
// stored in one of exactly two public hash locations.
type CuckooBuildConfig struct {
	LoadFactor    float64
	MaxKicks      int
	BuildAttempts int
	CacheDir      string
}

func DefaultCuckooBuildConfig() CuckooBuildConfig {
	return CuckooBuildConfig{
		LoadFactor:    0.4,
		MaxKicks:      1000,
		BuildAttempts: 8,
	}
}

// CuckooTable is the actual fixed-width PIR database. Word zero of every slot
// is the packed (term,node) key; the remaining words hold the score vector.
// An empty slot is all zero and therefore has key zero.
type CuckooTable struct {
	Name       string
	SlotCount  uint64
	ItemCount  uint64
	EntryWords uint64
	ValueBytes int
	Seed1      uint64
	Seed2      uint64
	Data       []uint64

	cacheFile *os.File
	cacheMmap []byte
}

type cuckooSource func(yield func(key uint64, value []byte) error) error

func normalizeCuckooConfig(config CuckooBuildConfig) (CuckooBuildConfig, error) {
	if config.LoadFactor == 0 {
		config.LoadFactor = 0.4
	}
	if config.MaxKicks == 0 {
		config.MaxKicks = 1000
	}
	if config.BuildAttempts == 0 {
		config.BuildAttempts = 8
	}
	if config.LoadFactor <= 0 || config.LoadFactor >= 0.5 {
		return config, fmt.Errorf("Cuckoo load factor must be in (0, 0.5), got %g", config.LoadFactor)
	}
	if config.MaxKicks < 1 || config.BuildAttempts < 1 {
		return config, fmt.Errorf("Cuckoo max kicks and build attempts must be positive")
	}
	return config, nil
}

func cuckooSlotCount(itemCount uint64, loadFactor float64) uint64 {
	if itemCount == 0 {
		return 1
	}
	return uint64(math.Ceil(float64(itemCount) / loadFactor))
}

func cuckooEntryWords(valueBytes int) uint64 {
	return uint64((8 + valueBytes + 7) / 8)
}

func splitMix64(value uint64) uint64 {
	value += 0x9e3779b97f4a7c15
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9
	value = (value ^ (value >> 27)) * 0x94d049bb133111eb
	return value ^ (value >> 31)
}

func cuckooBuildSeed(name string, attempt int, which uint64) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(name))
	return splitMix64(h.Sum64() + uint64(attempt)*0x9e3779b97f4a7c15 + which*0xd1b54a32d192ed03)
}

func (table *CuckooTable) positions(key uint64) (uint64, uint64) {
	first := splitMix64(key^table.Seed1) % table.SlotCount
	second := splitMix64(key^table.Seed2) % table.SlotCount
	if second == first && table.SlotCount > 1 {
		second = (second + 1) % table.SlotCount
	}
	return first, second
}

func (table *CuckooTable) verifyResponse(slot uint64, response []uint64) error {
	if slot >= table.SlotCount {
		return fmt.Errorf("%s Cuckoo slot %d is out of range", table.Name, slot)
	}
	if uint64(len(response)) != table.EntryWords {
		return fmt.Errorf("%s Cuckoo slot %d returned %d words; want %d", table.Name, slot, len(response), table.EntryWords)
	}
	expected := table.Data[slot*table.EntryWords : (slot+1)*table.EntryWords]
	for word := range expected {
		if response[word] != expected[word] {
			return fmt.Errorf("%s Cuckoo slot %d word %d mismatch: got %#x, want %#x", table.Name, slot, word, response[word], expected[word])
		}
	}
	return nil
}

func cuckooResponseValue(response []uint64, key uint64, valueBytes int) ([]byte, bool) {
	if len(response) == 0 || response[0] != key {
		return nil, false
	}
	value := make([]byte, valueBytes)
	for offset := range value {
		value[offset] = byte(response[1+offset/8] >> (8 * (offset % 8)))
	}
	return value, true
}

func encodeCuckooEntry(scratch []uint64, key uint64, value []byte, valueBytes int) error {
	if key == 0 {
		return fmt.Errorf("Cuckoo key zero is reserved for empty slots")
	}
	if len(value) != valueBytes {
		return fmt.Errorf("Cuckoo value has %d bytes; want %d", len(value), valueBytes)
	}
	clear(scratch)
	scratch[0] = key
	for offset := 0; offset < len(value); offset++ {
		scratch[1+offset/8] |= uint64(value[offset]) << (8 * (offset % 8))
	}
	return nil
}

func (table *CuckooTable) insert(key uint64, value []byte, scratch []uint64, maxKicks int) error {
	if err := encodeCuckooEntry(scratch, key, value, table.ValueBytes); err != nil {
		return err
	}
	first, second := table.positions(key)
	if table.Data[first*table.EntryWords] == key || table.Data[second*table.EntryWords] == key {
		return fmt.Errorf("duplicate Cuckoo key %#x", key)
	}

	position := first
	if splitMix64(key^table.Seed1^table.Seed2)&1 != 0 {
		position = second
	}
	for kick := 0; kick < maxKicks; kick++ {
		start := position * table.EntryWords
		if table.Data[start] == 0 {
			copy(table.Data[start:start+table.EntryWords], scratch)
			return nil
		}
		for word := uint64(0); word < table.EntryWords; word++ {
			scratch[word], table.Data[start+word] = table.Data[start+word], scratch[word]
		}
		displacedKey := scratch[0]
		displacedFirst, displacedSecond := table.positions(displacedKey)
		if position == displacedFirst {
			position = displacedSecond
		} else {
			position = displacedFirst
		}
	}
	return errCuckooCycle
}

func buildOrLoadCuckooTable(
	name string,
	itemCount uint64,
	valueBytes int,
	sourceTag uint64,
	config CuckooBuildConfig,
	cacheName string,
	source cuckooSource,
) (*CuckooTable, error) {
	var err error
	config, err = normalizeCuckooConfig(config)
	if err != nil {
		return nil, err
	}
	slotCount := cuckooSlotCount(itemCount, config.LoadFactor)
	entryWords := cuckooEntryWords(valueBytes)

	var cachePath string
	if config.CacheDir != "" {
		cachePath = filepath.Join(config.CacheDir, cacheName)
		if table, loadErr := loadCuckooTable(cachePath, name, itemCount, slotCount, entryWords, valueBytes, sourceTag, config.LoadFactor); loadErr == nil {
			fmt.Printf("%s Cuckoo: loaded %d items in %d slots from %s\n", name, itemCount, slotCount, cachePath)
			return table, nil
		} else if !errors.Is(loadErr, os.ErrNotExist) {
			fmt.Printf("%s Cuckoo: ignoring cache: %v\n", name, loadErr)
		}
	}

	for attempt := 0; attempt < config.BuildAttempts; attempt++ {
		table := &CuckooTable{
			Name:       name,
			SlotCount:  slotCount,
			ItemCount:  itemCount,
			EntryWords: entryWords,
			ValueBytes: valueBytes,
			Seed1:      cuckooBuildSeed(name, attempt, 1),
			Seed2:      cuckooBuildSeed(name, attempt, 2),
			Data:       make([]uint64, slotCount*entryWords),
		}
		scratch := make([]uint64, entryWords)
		inserted := uint64(0)
		buildErr := source(func(key uint64, value []byte) error {
			if err := table.insert(key, value, scratch, config.MaxKicks); err != nil {
				return err
			}
			inserted++
			if inserted%10_000_000 == 0 {
				fmt.Printf("%s Cuckoo: inserted %d/%d rows\n", name, inserted, itemCount)
			}
			return nil
		})
		if buildErr == nil && inserted != itemCount {
			buildErr = fmt.Errorf("source yielded %d rows; want %d", inserted, itemCount)
		}
		if buildErr == nil {
			fmt.Printf("%s Cuckoo: built %d items in %d slots (load %.4f, entry %d bytes)\n",
				name, itemCount, slotCount, float64(itemCount)/float64(slotCount), entryWords*8)
			if cachePath != "" {
				if err := saveCuckooTable(cachePath, table, sourceTag, config.LoadFactor); err != nil {
					return nil, fmt.Errorf("saving %s Cuckoo cache: %w", name, err)
				}
				fmt.Printf("%s Cuckoo: cached at %s\n", name, cachePath)
			}
			return table, nil
		}
		if !errors.Is(buildErr, errCuckooCycle) {
			return nil, fmt.Errorf("building %s Cuckoo table: %w", name, buildErr)
		}
		fmt.Printf("%s Cuckoo: insertion cycle on attempt %d/%d; rehashing\n", name, attempt+1, config.BuildAttempts)
		table.Data = nil
		runtime.GC()
	}
	return nil, fmt.Errorf("building %s Cuckoo table failed after %d hash seeds", name, config.BuildAttempts)
}

func cacheHeader(table *CuckooTable, sourceTag uint64, loadFactor float64) []byte {
	header := make([]byte, cuckooHeaderBytes)
	copy(header[:8], cuckooCacheMagic)
	values := []uint64{
		cuckooCacheVersion, table.SlotCount, table.ItemCount, table.EntryWords,
		uint64(table.ValueBytes), table.Seed1, table.Seed2, sourceTag,
		math.Float64bits(loadFactor),
	}
	for i, value := range values {
		binary.LittleEndian.PutUint64(header[8+i*8:16+i*8], value)
	}
	return header
}

func uint64Bytes(values []uint64) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*8)
}

func saveCuckooTable(path string, table *CuckooTable, sourceTag uint64, loadFactor float64) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".cuckoo-*")
	if err != nil {
		return err
	}
	tmpPath := file.Name()
	succeeded := false
	defer func() {
		_ = file.Close()
		if !succeeded {
			_ = os.Remove(tmpPath)
		}
	}()
	if _, err := file.Write(cacheHeader(table, sourceTag, loadFactor)); err != nil {
		return err
	}
	data := uint64Bytes(table.Data)
	const writeChunk = 64 << 20
	for len(data) > 0 {
		end := writeChunk
		if end > len(data) {
			end = len(data)
		}
		if _, err := file.Write(data[:end]); err != nil {
			return err
		}
		data = data[end:]
	}
	if err := file.Sync(); err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	succeeded = true
	return nil
}

func loadCuckooTable(
	path, name string,
	itemCount, slotCount, entryWords uint64,
	valueBytes int,
	sourceTag uint64,
	loadFactor float64,
) (*CuckooTable, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	closeOnError := true
	defer func() {
		if closeOnError {
			_ = file.Close()
		}
	}()
	stat, err := file.Stat()
	if err != nil {
		return nil, err
	}
	expectedSize := int64(cuckooHeaderBytes) + int64(slotCount*entryWords*8)
	if stat.Size() != expectedSize {
		return nil, fmt.Errorf("cache size is %d; want %d", stat.Size(), expectedSize)
	}
	mapped, err := mmap(file, stat.Size())
	if err != nil {
		return nil, err
	}
	unmapOnError := true
	defer func() {
		if unmapOnError {
			_ = unmmap(mapped)
		}
	}()
	if string(mapped[:8]) != cuckooCacheMagic {
		return nil, fmt.Errorf("invalid Cuckoo cache magic")
	}
	read := func(index int) uint64 {
		return binary.LittleEndian.Uint64(mapped[8+index*8 : 16+index*8])
	}
	if read(0) != cuckooCacheVersion || read(1) != slotCount || read(2) != itemCount ||
		read(3) != entryWords || read(4) != uint64(valueBytes) || read(7) != sourceTag ||
		read(8) != math.Float64bits(loadFactor) {
		return nil, fmt.Errorf("Cuckoo cache metadata does not match source/configuration")
	}
	wordCount := int(slotCount * entryWords)
	data := unsafe.Slice((*uint64)(unsafe.Pointer(&mapped[cuckooHeaderBytes])), wordCount)
	table := &CuckooTable{
		Name:       name,
		SlotCount:  slotCount,
		ItemCount:  itemCount,
		EntryWords: entryWords,
		ValueBytes: valueBytes,
		Seed1:      read(5),
		Seed2:      read(6),
		Data:       data,
		cacheFile:  file,
		cacheMmap:  mapped,
	}
	closeOnError = false
	unmapOnError = false
	return table, nil
}

func (table *CuckooTable) Close() error {
	var firstErr error
	if table.cacheMmap != nil {
		firstErr = unmmap(table.cacheMmap)
		table.cacheMmap = nil
		table.Data = nil
	}
	if table.cacheFile != nil {
		if err := table.cacheFile.Close(); firstErr == nil {
			firstErr = err
		}
		table.cacheFile = nil
	}
	return firstErr
}

func packedCuckooKey(termKey uint32, nodeID uint32) uint64 {
	return uint64(termKey)<<32 | uint64(nodeID)
}

func storedTermKey(termID int) (uint32, error) {
	if termID < 0 || uint64(termID) >= uint64(math.MaxInt32) {
		return 0, fmt.Errorf("term ID %d does not fit the Cuckoo key space", termID)
	}
	return uint32(termID) + 1, nil
}

func queryTermKey(term string, vocab map[string]int) uint32 {
	if termID, ok := vocab[term]; ok {
		key, err := storedTermKey(termID)
		if err == nil {
			return key
		}
	}
	hash := uint32(2166136261)
	for i := 0; i < len(term); i++ {
		hash ^= uint32(term[i])
		hash *= 16777619
	}
	return 0x80000000 | (hash & 0x7fffffff)
}

func idMapSourceTag(paths ...string) (uint64, error) {
	tag := uint64(0x6a09e667f3bcc909)
	for _, path := range paths {
		stat, err := os.Stat(path)
		if err != nil {
			return 0, err
		}
		tag = splitMix64(tag ^ uint64(stat.Size()))
		tag = splitMix64(tag ^ uint64(stat.ModTime().UnixNano()))
	}
	return tag, nil
}

func forEachIDMapRecord(path string, visit func(termID, nodeID uint32, rowIndex uint64) error) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	record := make([]byte, 16)
	for {
		_, err := io.ReadFull(file, record)
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if err := visit(
			binary.LittleEndian.Uint32(record[0:4]),
			binary.LittleEndian.Uint32(record[4:8]),
			binary.LittleEndian.Uint64(record[8:16]),
		); err != nil {
			return err
		}
	}
}
