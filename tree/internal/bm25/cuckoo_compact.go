package bm25

import (
	"fmt"
	"math"
)

// Client-storage options for the Cuckoo tables. Every Piano hint is a whole
// number of table rows, so the hint shrinks with fewer rows (a fuller table)
// and with shorter rows. None of this touches the documents or embeddings: it
// only changes how the tree's own (term,node) -> upper-bound tables are laid
// out. The defaults (2 hashes, 8-byte keys, 8-bit bounds) take the original
// code path exactly, so existing tables, caches and results are unchanged.
//
//   Hashes     2 or 3 public locations per key. Two-choice Cuckoo only fills
//              about 50% of the slots (LoadFactor < 0.5); three-choice fills
//              about 91%, so the same items need far fewer rows. Every key is
//              then probed 3 times instead of 2 (a fixed, public number).
//   KeyBytes   bytes of the key kept in a row: 8 is the full (term,node) key;
//              fewer keep a non-zero fingerprint of it. A lookup then accepts a
//              different key with the same fingerprint, probability about
//              Hashes/2^(8*KeyBytes) per lookup (2 bytes: 1 in 22,000). The
//              client keeps the element-wise max over every matching probe, and
//              the true row is always one of them, so a false match only loosens
//              that node's bound, never lowers it. It never affects privacy.
//   ValueBits  bits per upper bound (8 = the bytes as built). Fewer bits store
//              ceil(v/step) with step = ceil(255/(2^bits-1)) and read it back as
//              min(255, code*step) >= v, so every bound stays an upper bound,
//              only looser.
//
// A row is a little-endian bit stream: KeyBytes*8 key bits, then ValueBits per
// value, padded to whole uint64 words (PIR entries are words).

func (config CuckooBuildConfig) hashes() int {
	if config.Hashes == 0 {
		return 2
	}
	return config.Hashes
}

func (config CuckooBuildConfig) keyBytes() int {
	if config.KeyBytes == 0 {
		return 8
	}
	return config.KeyBytes
}

func (config CuckooBuildConfig) valueBits() int {
	if config.ValueBits == 0 {
		return 8
	}
	return config.ValueBits
}

// IsDefaultLayout reports whether the original (2-hash, full-key, 8-bit) code
// path is used.
func (config CuckooBuildConfig) IsDefaultLayout() bool {
	return config.hashes() == 2 && config.keyBytes() == 8 && config.valueBits() == 8
}

// CacheSuffix keeps caches of different layouts apart; empty for the default
// so existing cache files keep their names.
func (config CuckooBuildConfig) CacheSuffix() string {
	if config.IsDefaultLayout() {
		return ""
	}
	return fmt.Sprintf("_h%d_k%d_v%d", config.hashes(), config.keyBytes(), config.valueBits())
}

func validateCompactConfig(config CuckooBuildConfig) error {
	switch h := config.hashes(); h {
	case 2:
		if config.LoadFactor >= 0.5 {
			return fmt.Errorf("two-hash Cuckoo load factor must be below 0.5, got %g", config.LoadFactor)
		}
	case 3:
		if config.LoadFactor >= 0.91 {
			return fmt.Errorf("three-hash Cuckoo load factor must be below 0.91, got %g", config.LoadFactor)
		}
	default:
		return fmt.Errorf("Cuckoo hashes must be 2 or 3, got %d", h)
	}
	if k := config.keyBytes(); k < 1 || k > 8 {
		return fmt.Errorf("Cuckoo key bytes must be 1..8, got %d", k)
	}
	if v := config.valueBits(); v < 1 || v > 8 {
		return fmt.Errorf("upper-bound bits must be 1..8, got %d", v)
	}
	return nil
}

func compactEntryWords(keyBytes, valueBits, valueCount int) uint64 {
	bits := 8*keyBytes + valueBits*valueCount
	return uint64((bits + 63) / 64)
}

// valueStep is the quantization step for `bits`-bit bounds.
func valueStep(bits int) int {
	if bits >= 8 {
		return 1
	}
	levels := (1 << bits) - 1
	return (255 + levels - 1) / levels
}

// storedKey is what a row keeps of key: the key itself, or a non-zero
// fingerprint of KeyBytes bytes.
func (table *CuckooTable) storedKey(key uint64) uint64 {
	if table.KeyBytes >= 8 {
		return key
	}
	space := uint64(1)<<(8*table.KeyBytes) - 1
	return 1 + splitMix64(key^table.Seed1^0x5bd1e9955bd1e995)%space
}

func putBits(row []uint64, offset, width int, value uint64) {
	for width > 0 {
		word, shift := offset/64, offset%64
		take := min(width, 64-shift)
		mask := uint64(1)<<take - 1
		if take == 64 {
			mask = math.MaxUint64
		}
		row[word] |= (value & mask) << shift
		value >>= take
		offset += take
		width -= take
	}
}

func getBits(row []uint64, offset, width int) uint64 {
	value, done := uint64(0), 0
	for done < width {
		word, shift := offset/64, offset%64
		take := min(width-done, 64-shift)
		mask := uint64(1)<<take - 1
		if take == 64 {
			mask = math.MaxUint64
		}
		value |= ((row[word] >> shift) & mask) << done
		offset += take
		done += take
	}
	return value
}

func (table *CuckooTable) encodeCompact(row []uint64, key uint64, value []byte) error {
	if key == 0 {
		return fmt.Errorf("Cuckoo key zero is reserved for empty slots")
	}
	if len(value) != table.ValueBytes {
		return fmt.Errorf("Cuckoo value has %d bytes; want %d", len(value), table.ValueBytes)
	}
	clear(row)
	putBits(row, 0, 8*table.KeyBytes, table.storedKey(key))
	step := valueStep(table.ValueBits)
	for i, v := range value {
		code := (int(v) + step - 1) / step // rounds up: decoded value >= v
		putBits(row, 8*table.KeyBytes+i*table.ValueBits, table.ValueBits, uint64(code))
	}
	return nil
}

// decodeValue returns the row's upper bounds when the row holds key (or a key
// with the same fingerprint). It replaces cuckooResponseValue for any layout.
func (table *CuckooTable) decodeValue(response []uint64, key uint64) ([]byte, bool) {
	if table.Hashes <= 2 && (table.KeyBytes == 0 || table.KeyBytes >= 8) && (table.ValueBits == 0 || table.ValueBits >= 8) {
		return cuckooResponseValue(response, key, table.ValueBytes)
	}
	if uint64(len(response)) < table.EntryWords || getBits(response, 0, 8*table.KeyBytes) != table.storedKey(key) {
		return nil, false
	}
	step := valueStep(table.ValueBits)
	value := make([]byte, table.ValueBytes)
	for i := range value {
		code := int(getBits(response, 8*table.KeyBytes+i*table.ValueBits, table.ValueBits))
		value[i] = byte(min(255, code*step))
	}
	return value, true
}

// probes returns every public location of key: the two of positions(), plus a
// third for a three-hash table. Lookups probe all of them.
func (table *CuckooTable) probes(key uint64) []uint64 {
	first, second := table.positions(key)
	if table.Hashes <= 2 {
		return []uint64{first, second}
	}
	third := splitMix64(key^table.Seed3) % table.SlotCount
	for table.SlotCount > 2 && (third == first || third == second) {
		third = (third + 1) % table.SlotCount
	}
	return []uint64{first, second, third}
}

func (table *CuckooTable) rowEmpty(slot uint64) bool {
	for _, w := range table.Data[slot*table.EntryWords : (slot+1)*table.EntryWords] {
		if w != 0 {
			return false
		}
	}
	return true
}

// insertCompact is the random-walk Cuckoo insertion for the non-default
// layouts. A fingerprint cannot be rehashed, so the full key of every
// occupied slot is kept in keys while the table is built.
func (table *CuckooTable) insertCompact(key uint64, value []byte, keys []uint64, scratch []uint64, maxKicks int) error {
	if err := table.encodeCompact(scratch, key, value); err != nil {
		return err
	}
	locations := table.probes(key)
	for _, slot := range locations {
		if keys[slot] == key {
			return fmt.Errorf("duplicate Cuckoo key %#x", key)
		}
	}
	for _, slot := range locations { // a free location: no kicking needed
		if keys[slot] == 0 {
			copy(table.Data[slot*table.EntryWords:(slot+1)*table.EntryWords], scratch)
			keys[slot] = key
			return nil
		}
	}
	rng := splitMix64(key ^ table.Seed2)
	position := locations[rng%uint64(len(locations))]
	for kick := 0; kick < maxKicks; kick++ {
		start := position * table.EntryWords
		if keys[position] == 0 {
			copy(table.Data[start:start+table.EntryWords], scratch)
			keys[position] = key
			return nil
		}
		for word := uint64(0); word < table.EntryWords; word++ {
			scratch[word], table.Data[start+word] = table.Data[start+word], scratch[word]
		}
		key, keys[position] = keys[position], key
		// move the evicted key to one of its other locations
		alternatives := table.probes(key)
		rng = splitMix64(rng + uint64(kick) + 1)
		next := alternatives[rng%uint64(len(alternatives))]
		if next == position {
			next = alternatives[(rng+1)%uint64(len(alternatives))]
		}
		position = next
	}
	return errCuckooCycle
}

// maxValue merges a decoded row into acc by element-wise max. With fingerprint
// keys several probes can match; the true row is one of them, so the max is
// still an upper bound. With full keys at most one probe matches.
func maxValue(acc, value []byte) []byte {
	if acc == nil {
		return value
	}
	for i, v := range value {
		acc[i] = max(acc[i], v)
	}
	return acc
}
