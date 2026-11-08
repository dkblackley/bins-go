package bm25

import (
	"fmt"
	"math"
	"os"
	"strings"
	"sync"

	pianopir "github.com/dkblackley/bins-go/pir-old"
)

// Stage2Precomputed stores (term,leaf block) -> B/s sub-block score vectors.
// In PIR mode Table is a physical two-hash Cuckoo table whose rows include the
// key; LookupTable exists only for the explicit plaintext diagnostic mode.
type Stage2Precomputed struct {
	S           int
	R           int
	Scale       float64
	DataMmap    []byte
	DataUint64  []uint64 // Deprecated compact representation; nil in Cuckoo mode.
	dataFile    *os.File
	LookupTable map[int]map[int]int
	Vocab       map[string]int
	Table       *CuckooTable
	pir         *pianopir.SimpleBatchPianoPIR
	mu          sync.RWMutex
}

func NewStage2Precomputed(dataBinPath, idmapBinPath, vocabPath string, B, s int, scale float64) (*Stage2Precomputed, error) {
	return NewStage2PrecomputedWithPIR(dataBinPath, idmapBinPath, vocabPath, B, s, scale, false, 128)
}

func NewStage2PrecomputedWithPIR(dataBinPath, idmapBinPath, vocabPath string, B, s int, scale float64, enablePIR bool, batchSize uint64) (*Stage2Precomputed, error) {
	return NewStage2PrecomputedWithCuckoo(
		dataBinPath, idmapBinPath, vocabPath, B, s, scale,
		enablePIR, batchSize, DefaultCuckooBuildConfig(),
	)
}

func NewStage2PrecomputedWithCuckoo(
	dataBinPath, idmapBinPath, vocabPath string,
	B, s int,
	scale float64,
	enablePIR bool,
	batchSize uint64,
	cuckooConfig CuckooBuildConfig,
) (*Stage2Precomputed, error) {
	if s <= 0 || B%s != 0 {
		return nil, fmt.Errorf("block size B=%d must be divisible by positive s=%d", B, s)
	}
	pre := &Stage2Precomputed{
		S:           s,
		R:           B / s,
		Scale:       scale,
		LookupTable: make(map[int]map[int]int),
		Vocab:       make(map[string]int),
	}

	var err error
	pre.Vocab, err = loadVocabulary(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("loading Stage2 vocabulary: %w", err)
	}

	dataFile, err := os.Open(dataBinPath)
	if err != nil {
		return nil, fmt.Errorf("opening Stage2 data: %w", err)
	}
	pre.dataFile = dataFile
	dataStat, err := dataFile.Stat()
	if err != nil {
		pre.Close()
		return nil, fmt.Errorf("stat Stage2 data: %w", err)
	}
	if dataStat.Size()%int64(pre.R) != 0 {
		pre.Close()
		return nil, fmt.Errorf("Stage2 data size %d is not divisible by score width %d", dataStat.Size(), pre.R)
	}
	pre.DataMmap, err = mmap(dataFile, dataStat.Size())
	if err != nil {
		pre.Close()
		return nil, fmt.Errorf("mapping Stage2 data: %w", err)
	}

	if !enablePIR {
		if err := pre.loadPlaintextLookup(idmapBinPath); err != nil {
			pre.Close()
			return nil, err
		}
		return pre, nil
	}

	config, err := normalizeCuckooConfig(cuckooConfig)
	if err != nil {
		pre.Close()
		return nil, err
	}
	idmapStat, err := os.Stat(idmapBinPath)
	if err != nil {
		pre.Close()
		return nil, fmt.Errorf("stat Stage2 id map: %w", err)
	}
	if idmapStat.Size()%16 != 0 {
		pre.Close()
		return nil, fmt.Errorf("Stage2 id map size %d is not a multiple of 16", idmapStat.Size())
	}
	itemCount := uint64(idmapStat.Size() / 16)
	rowCount := uint64(len(pre.DataMmap) / pre.R)
	if itemCount != rowCount {
		pre.Close()
		return nil, fmt.Errorf("Stage2 id map has %d rows but score data has %d", itemCount, rowCount)
	}
	sourceTag, err := idMapSourceTag(idmapBinPath, dataBinPath)
	if err != nil {
		pre.Close()
		return nil, fmt.Errorf("fingerprinting Stage2 artifacts: %w", err)
	}
	cacheName := fmt.Sprintf("stage2_B%d_s%d_alpha_%0.4f%s.cuckoo", B, s, config.LoadFactor, config.CacheSuffix())
	pre.Table, err = buildOrLoadCuckooTable(
		"Stage2",
		itemCount,
		pre.R,
		sourceTag,
		config,
		cacheName,
		func(yield func(key uint64, value []byte) error) error {
			return forEachIDMapRecord(idmapBinPath, func(termID, nodeID uint32, rowIndex uint64) error {
				if nodeID == 0 {
					return fmt.Errorf("Stage2 row %d has invalid node zero", rowIndex)
				}
				if rowIndex >= rowCount {
					return fmt.Errorf("Stage2 row %d is outside %d rows", rowIndex, rowCount)
				}
				termKey, err := storedTermKey(int(termID))
				if err != nil {
					return err
				}
				start := rowIndex * uint64(pre.R)
				return yield(packedCuckooKey(termKey, nodeID), pre.DataMmap[start:start+uint64(pre.R)])
			})
		},
	)
	if err != nil {
		pre.Close()
		return nil, err
	}

	pre.pir = pianopir.NewSimpleBatchPianoPIR(
		pre.Table.SlotCount,
		pre.Table.EntryWords*8,
		batchSize,
		pre.Table.Data,
		20,
	)
	pre.pir.SetThreadNum(PIRThreads)
	pre.pir.SetConstantWork(PIRConstantWork)
	fmt.Printf("Stage2 Cuckoo PIR: DBSize=%d, EntrySize=%d bytes, BatchSize=%d\n",
		pre.Table.SlotCount, pre.Table.EntryWords*8, batchSize)
	pre.pir.Preprocessing()
	if StrictPIR {
		if err := verifyPackedPIR("Stage2", pre.pir, pre.Table.Data, pre.Table.EntryWords); err != nil {
			pre.Close()
			return nil, err
		}
		fmt.Printf("Stage2 PIR: verified 32 sampled Cuckoo slots\n")
	}
	return pre, nil
}

func (pre *Stage2Precomputed) loadPlaintextLookup(idmapBinPath string) error {
	return forEachIDMapRecord(idmapBinPath, func(termID, nodeID uint32, rowIndex uint64) error {
		if rowIndex >= uint64(len(pre.DataMmap)/pre.R) {
			return fmt.Errorf("Stage2 row %d is out of range", rowIndex)
		}
		term := int(termID)
		if pre.LookupTable[term] == nil {
			pre.LookupTable[term] = make(map[int]int)
		}
		pre.LookupTable[term][int(nodeID)] = int(rowIndex)
		return nil
	})
}

func (pre *Stage2Precomputed) Close() error {
	pre.mu.Lock()
	defer pre.mu.Unlock()
	var firstErr error
	if pre.Table != nil {
		firstErr = pre.Table.Close()
		pre.Table = nil
	}
	if pre.DataMmap != nil {
		if err := unmmap(pre.DataMmap); firstErr == nil {
			firstErr = err
		}
		pre.DataMmap = nil
	}
	if pre.dataFile != nil {
		if err := pre.dataFile.Close(); firstErr == nil {
			firstErr = err
		}
		pre.dataFile = nil
	}
	return firstErr
}

func (pre *Stage2Precomputed) sumBoundsDirectUnlocked(terms []string, leafNodes []int) map[int][]int {
	acc := make(map[int][]int, len(leafNodes))
	for _, leaf := range leafNodes {
		acc[leaf] = make([]int, pre.R)
	}
	for _, term := range terms {
		termID, ok := pre.Vocab[strings.ToLower(term)]
		if !ok {
			continue
		}
		for _, leaf := range leafNodes {
			rowIndex, ok := pre.LookupTable[termID][leaf]
			if !ok {
				continue
			}
			start := rowIndex * pre.R
			for subblock, score := range pre.DataMmap[start : start+pre.R] {
				acc[leaf][subblock] += int(score)
			}
		}
	}
	return acc
}

func (pre *Stage2Precomputed) SumBoundsForTerms(terms []string, leafNodes []int) map[int][]int {
	pre.mu.RLock()
	defer pre.mu.RUnlock()
	return pre.sumBoundsDirectUnlocked(terms, leafNodes)
}

// SumBoundsForTermsWithPIR issues exactly Hashes (2 or 3) physical PIR queries for every
// padded (term,leaf) key, validates returned keys, and sums only matching rows.
func (pre *Stage2Precomputed) SumBoundsForTermsWithPIR(terms []string, leafNodes []int) map[int][]int {
	pre.mu.RLock()
	defer pre.mu.RUnlock()
	if len(leafNodes) == 0 {
		return nil
	}
	if pre.pir == nil || pre.Table == nil {
		return pre.sumBoundsDirectUnlocked(terms, leafNodes)
	}

	acc := make(map[int][]int, len(leafNodes))
	for _, leaf := range leafNodes {
		acc[leaf] = make([]int, pre.R)
	}
	termKeys := termKeysForQuery(terms, pre.Vocab)
	logicalCount := len(termKeys) * len(leafNodes)
	keys := make([]uint64, 0, logicalCount)
	rowIndices := make([]uint64, 0, logicalCount*max(pre.Table.Hashes, 2))
	for _, termKey := range termKeys {
		for _, leaf := range leafNodes {
			if leaf <= 0 || uint64(leaf) > math.MaxUint32 {
				panic(fmt.Sprintf("Stage2 leaf ID %d does not fit the Cuckoo key", leaf))
			}
			key := packedCuckooKey(termKey, uint32(leaf))
			keys = append(keys, key)
			rowIndices = append(rowIndices, pre.Table.probes(key)...)
		}
	}
	Debugf("Stage2: %d logical keys, %d Cuckoo PIR probes, %d physical slots, %d-byte entries\n",
		logicalCount, len(rowIndices), pre.Table.SlotCount, pre.Table.EntryWords*8)
	responses, err := pre.pir.Query(rowIndices)
	if err != nil {
		panic(fmt.Sprintf("Stage2 Cuckoo PIR query failed: %v", err))
	}
	if len(responses) != len(rowIndices) {
		panic(fmt.Sprintf("Stage2 returned %d PIR rows; want %d", len(responses), len(rowIndices)))
	}
	probeCount := len(rowIndices) / max(len(keys), 1)
	for logical, key := range keys {
		var value []byte
		for j := logical * probeCount; j < (logical+1)*probeCount; j++ {
			if StrictPIR {
				if err := pre.Table.verifyResponse(rowIndices[j], responses[j]); err != nil {
					panic(err)
				}
			}
			if decoded, ok := pre.Table.decodeValue(responses[j], key); ok {
				value = maxValue(value, decoded)
			}
		}
		if value == nil {
			continue
		}
		leaf := leafNodes[logical%len(leafNodes)]
		for subblock, score := range value {
			acc[leaf][subblock] += int(score)
		}
	}
	return acc
}

// QueryPIR exposes physical Cuckoo slot reads for diagnostics.
func (pre *Stage2Precomputed) QueryPIR(rowIndices []uint64) ([][]uint64, error) {
	if pre.pir == nil {
		return nil, fmt.Errorf("PIR not initialized")
	}
	return pre.pir.Query(rowIndices)
}

func (pre *Stage2Precomputed) GetMetrics() map[string]float64 {
	if pre.pir == nil {
		return nil
	}
	return pre.pir.GetMetrics()
}
