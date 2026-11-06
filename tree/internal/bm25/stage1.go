package bm25

import (
	"fmt"
	"math"
	"os"
	"strings"
	"sync"

	pianopir "github.com/dkblackley/bins-go/pir-old"
)

// Stage1LevelDB is one physical Cuckoo/PIR database for one tree expansion
// level. Keys are (term,parent node); values are the r child scores.
type Stage1LevelDB struct {
	Level int
	Table *CuckooTable
	pir   *pianopir.SimpleBatchPianoPIR
}

// Stage1PIRDB represents the first protocol round. In PIR mode Levels contains
// a separate Appendix-E Cuckoo table for every tree level. LookupTable is kept
// only for the explicit --disable-pir plaintext diagnostic path.
type Stage1PIRDB struct {
	R           int
	DataMmap    []byte
	DataUint64  []uint64 // Deprecated compact representation; nil in Cuckoo mode.
	dataFile    *os.File
	LookupTable map[int]map[int]int
	Vocab       map[string]int
	Levels      []*Stage1LevelDB
	mu          sync.RWMutex
}

func NewStage1PIRDB(dataBinPath, idmapBinPath, vocabPath string, r int) (*Stage1PIRDB, error) {
	return NewStage1PIRDBWithPIR(dataBinPath, idmapBinPath, vocabPath, r, false, 128)
}

func NewStage1PIRDBWithPIR(dataBinPath, idmapBinPath, vocabPath string, r int, enablePIR bool, batchSize uint64) (*Stage1PIRDB, error) {
	return NewStage1PIRDBWithCuckoo(dataBinPath, idmapBinPath, vocabPath, r, enablePIR, batchSize, DefaultCuckooBuildConfig())
}

// NewStage1PIRDBWithCuckoo builds or loads the faithful physical Cuckoo tables
// and initializes one BatchPIR instance per tree level.
func NewStage1PIRDBWithCuckoo(
	dataBinPath, idmapBinPath, vocabPath string,
	r int,
	enablePIR bool,
	batchSize uint64,
	cuckooConfig CuckooBuildConfig,
) (*Stage1PIRDB, error) {
	if r < 2 {
		return nil, fmt.Errorf("Stage1 fan-out r must be at least 2")
	}
	db := &Stage1PIRDB{
		R:           r,
		LookupTable: make(map[int]map[int]int),
		Vocab:       make(map[string]int),
	}

	var err error
	db.Vocab, err = loadVocabulary(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("loading Stage1 vocabulary: %w", err)
	}

	dataFile, err := os.Open(dataBinPath)
	if err != nil {
		return nil, fmt.Errorf("opening Stage1 data: %w", err)
	}
	db.dataFile = dataFile
	dataStat, err := dataFile.Stat()
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("stat Stage1 data: %w", err)
	}
	if dataStat.Size()%int64(r) != 0 {
		db.Close()
		return nil, fmt.Errorf("Stage1 data size %d is not divisible by r=%d", dataStat.Size(), r)
	}
	db.DataMmap, err = mmap(dataFile, dataStat.Size())
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("mapping Stage1 data: %w", err)
	}

	if !enablePIR {
		if err := db.loadPlaintextLookup(idmapBinPath); err != nil {
			db.Close()
			return nil, err
		}
		return db, nil
	}

	config, err := normalizeCuckooConfig(cuckooConfig)
	if err != nil {
		db.Close()
		return nil, err
	}
	sourceTag, err := idMapSourceTag(idmapBinPath, dataBinPath)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("fingerprinting Stage1 artifacts: %w", err)
	}
	levelCounts, err := countStage1Levels(idmapBinPath, r)
	if err != nil {
		db.Close()
		return nil, err
	}
	db.Levels = make([]*Stage1LevelDB, len(levelCounts))
	rowCount := uint64(len(db.DataMmap) / r)
	totalItems := uint64(0)
	for _, count := range levelCounts {
		totalItems += count
	}
	if totalItems != rowCount {
		db.Close()
		return nil, fmt.Errorf("Stage1 id map has %d rows but score data has %d", totalItems, rowCount)
	}

	for level, itemCount := range levelCounts {
		level := level
		name := fmt.Sprintf("Stage1 level %d", level)
		cacheName := fmt.Sprintf("stage1_level_%02d_r%d_alpha_%0.4f.cuckoo", level, r, config.LoadFactor)
		table, err := buildOrLoadCuckooTable(
			name,
			itemCount,
			r,
			splitMix64(sourceTag^uint64(level)),
			config,
			cacheName,
			func(yield func(key uint64, value []byte) error) error {
				return forEachIDMapRecord(idmapBinPath, func(termID, nodeID uint32, rowIndex uint64) error {
					recordLevel, ok := stage1ParentLevel(nodeID, r)
					if !ok || recordLevel != level {
						return nil
					}
					if rowIndex >= rowCount {
						return fmt.Errorf("Stage1 row %d is outside %d rows", rowIndex, rowCount)
					}
					termKey, err := storedTermKey(int(termID))
					if err != nil {
						return err
					}
					start := rowIndex * uint64(r)
					return yield(packedCuckooKey(termKey, nodeID), db.DataMmap[start:start+uint64(r)])
				})
			},
		)
		if err != nil {
			db.Close()
			return nil, err
		}

		levelDB := &Stage1LevelDB{Level: level, Table: table}
		// Register the table before preprocessing so Close releases it if PianoPIR
		// initialization or strict verification fails.
		db.Levels[level] = levelDB
		levelDB.pir = pianopir.NewSimpleBatchPianoPIR(
			table.SlotCount,
			table.EntryWords*8,
			batchSize,
			table.Data,
			20,
		)
		levelDB.pir.SetThreadNum(PIRThreads)
		levelDB.pir.SetConstantWork(PIRConstantWork)
		fmt.Printf("%s PIR: DBSize=%d, EntrySize=%d bytes, BatchSize=%d\n",
			name, table.SlotCount, table.EntryWords*8, batchSize)
		levelDB.pir.Preprocessing()
		if StrictPIR {
			if err := verifyPackedPIR(name, levelDB.pir, table.Data, table.EntryWords); err != nil {
				db.Close()
				return nil, err
			}
			fmt.Printf("%s PIR: verified 32 sampled Cuckoo slots\n", name)
		}
	}
	return db, nil
}

func (db *Stage1PIRDB) loadPlaintextLookup(idmapBinPath string) error {
	return forEachIDMapRecord(idmapBinPath, func(termID, nodeID uint32, rowIndex uint64) error {
		if rowIndex >= uint64(len(db.DataMmap)/db.R) {
			return fmt.Errorf("Stage1 row %d is out of range", rowIndex)
		}
		term := int(termID)
		if db.LookupTable[term] == nil {
			db.LookupTable[term] = make(map[int]int)
		}
		db.LookupTable[term][int(nodeID)] = int(rowIndex)
		return nil
	})
}

func countStage1Levels(idmapBinPath string, r int) ([]uint64, error) {
	counts := make([]uint64, 0, 8)
	err := forEachIDMapRecord(idmapBinPath, func(_ uint32, nodeID uint32, _ uint64) error {
		level, ok := stage1ParentLevel(nodeID, r)
		if !ok {
			return fmt.Errorf("Stage1 parent node %d is outside the r=%d tree level ranges", nodeID, r)
		}
		for len(counts) <= level {
			counts = append(counts, 0)
		}
		counts[level]++
		return nil
	})
	if err != nil {
		return nil, err
	}
	for level, count := range counts {
		fmt.Printf("Stage1 level %d: %d nonzero packed (term,parent) rows\n", level, count)
	}
	return counts, nil
}

// stage1ParentLevel follows the node numbering used by the artifact builder:
// root [1,2), then [r,2r), [r^2,2r^2), and so on.
func stage1ParentLevel(nodeID uint32, r int) (int, bool) {
	node := uint64(nodeID)
	start := uint64(1)
	for level := 0; level < 32; level++ {
		end := start * 2
		if node >= start && node < end {
			return level, true
		}
		if start > uint64(math.MaxUint32)/uint64(r) {
			break
		}
		start *= uint64(r)
	}
	return 0, false
}

func (db *Stage1PIRDB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()
	var firstErr error
	for _, level := range db.Levels {
		if level != nil && level.Table != nil {
			if err := level.Table.Close(); firstErr == nil {
				firstErr = err
			}
		}
	}
	if db.DataMmap != nil {
		if err := unmmap(db.DataMmap); firstErr == nil {
			firstErr = err
		}
		db.DataMmap = nil
	}
	if db.dataFile != nil {
		if err := db.dataFile.Close(); firstErr == nil {
			firstErr = err
		}
		db.dataFile = nil
	}
	return firstErr
}

func (db *Stage1PIRDB) level(level int) *Stage1LevelDB {
	if level < 0 || level >= len(db.Levels) || db.Levels[level] == nil {
		panic(fmt.Sprintf("Stage1 has no Cuckoo PIR database for tree level %d", level))
	}
	return db.Levels[level]
}

func termKeysForQuery(terms []string, vocab map[string]int) []uint32 {
	keys := make([]uint32, len(terms))
	for i, term := range terms {
		keys[i] = queryTermKey(strings.ToLower(term), vocab)
	}
	return keys
}

func (db *Stage1PIRDB) queryCuckooRows(level int, termKeys []uint32, nodeIDs []int) [][]byte {
	levelDB := db.level(level)
	table := levelDB.Table
	logicalCount := len(termKeys) * len(nodeIDs)
	rowIndices := make([]uint64, 0, logicalCount*2)
	keys := make([]uint64, 0, logicalCount)
	for _, termKey := range termKeys {
		for _, nodeID := range nodeIDs {
			if nodeID <= 0 || uint64(nodeID) > math.MaxUint32 {
				panic(fmt.Sprintf("Stage1 node ID %d does not fit the Cuckoo key", nodeID))
			}
			key := packedCuckooKey(termKey, uint32(nodeID))
			first, second := table.positions(key)
			keys = append(keys, key)
			rowIndices = append(rowIndices, first, second)
		}
	}
	Debugf("Stage1 level %d: %d logical keys, %d Cuckoo PIR probes, %d physical slots, %d-byte entries\n",
		level, logicalCount, len(rowIndices), table.SlotCount, table.EntryWords*8)
	responses, err := levelDB.pir.Query(rowIndices)
	if err != nil {
		panic(fmt.Sprintf("Stage1 level %d Cuckoo PIR query failed: %v", level, err))
	}
	if len(responses) != len(rowIndices) {
		panic(fmt.Sprintf("Stage1 level %d returned %d PIR rows; want %d", level, len(responses), len(rowIndices)))
	}
	rows := make([][]byte, logicalCount)
	for i, key := range keys {
		firstResponse := responses[2*i]
		secondResponse := responses[2*i+1]
		if StrictPIR {
			if err := table.verifyResponse(rowIndices[2*i], firstResponse); err != nil {
				panic(err)
			}
			if err := table.verifyResponse(rowIndices[2*i+1], secondResponse); err != nil {
				panic(err)
			}
		}
		if value, ok := cuckooResponseValue(firstResponse, key, db.R); ok {
			rows[i] = value
		} else if value, ok := cuckooResponseValue(secondResponse, key, db.R); ok {
			rows[i] = value
		} else {
			rows[i] = make([]byte, db.R)
		}
	}
	return rows
}

// SumChildScoresAtLevel performs exactly two PIR probes for every
// (query term,parent) pair and checks returned keys before summing scores.
func (db *Stage1PIRDB) SumChildScoresAtLevel(level int, terms []string, nodeIDs []int) map[int][]int {
	db.mu.RLock()
	defer db.mu.RUnlock()
	if len(nodeIDs) == 0 {
		return nil
	}
	acc := make(map[int][]int, len(nodeIDs))
	for _, nodeID := range nodeIDs {
		acc[nodeID] = make([]int, db.R)
	}
	if len(db.Levels) == 0 {
		for _, term := range terms {
			termID, ok := db.Vocab[strings.ToLower(term)]
			if !ok {
				continue
			}
			for _, nodeID := range nodeIDs {
				if rowIndex, ok := db.LookupTable[termID][nodeID]; ok {
					row := db.readRowDirectByte(rowIndex)
					for child, score := range row {
						acc[nodeID][child] += int(score)
					}
				}
			}
		}
		return acc
	}

	rows := db.queryCuckooRows(level, termKeysForQuery(terms, db.Vocab), nodeIDs)
	rowIndex := 0
	for range terms {
		for _, nodeID := range nodeIDs {
			for child, score := range rows[rowIndex] {
				acc[nodeID][child] += int(score)
			}
			rowIndex++
		}
	}
	return acc
}

// GetScoreBatch is retained for diagnostic callers. Production tree traversal
// uses SumChildScoresAtLevel so padded/OOV terms also issue their two probes.
func (db *Stage1PIRDB) GetScoreBatch(termIDs []int, nodeIDs []int) map[int]map[int][]byte {
	return db.GetScoreBatchCovered(termIDs, nodeIDs, len(termIDs))
}

func (db *Stage1PIRDB) GetScoreBatchCovered(termIDs []int, nodeIDs []int, coverTermCount int) map[int]map[int][]byte {
	db.mu.RLock()
	defer db.mu.RUnlock()
	if len(nodeIDs) == 0 {
		return map[int]map[int][]byte{}
	}
	if len(db.Levels) == 0 {
		return db.getScoreDirect(termIDs, nodeIDs)
	}
	level, ok := stage1ParentLevel(uint32(nodeIDs[0]), db.R)
	if !ok {
		panic("cannot infer Stage1 level")
	}
	termKeys := make([]uint32, 0, coverTermCount)
	for _, termID := range termIDs {
		key, err := storedTermKey(termID)
		if err != nil {
			panic(err)
		}
		termKeys = append(termKeys, key)
	}
	for len(termKeys) < coverTermCount {
		termKeys = append(termKeys, 0x80000000|uint32(len(termKeys)+1))
	}
	rows := db.queryCuckooRows(level, termKeys, nodeIDs)
	result := make(map[int]map[int][]byte)
	for termIndex, termID := range termIDs {
		result[termID] = make(map[int][]byte)
		for nodeIndex, nodeID := range nodeIDs {
			result[termID][nodeID] = rows[termIndex*len(nodeIDs)+nodeIndex]
		}
	}
	return result
}

func (db *Stage1PIRDB) getScoreDirect(termIDs []int, nodeIDs []int) map[int]map[int][]byte {
	result := make(map[int]map[int][]byte)
	for _, termID := range termIDs {
		for _, nodeID := range nodeIDs {
			if rowIndex, ok := db.LookupTable[termID][nodeID]; ok {
				if result[termID] == nil {
					result[termID] = make(map[int][]byte)
				}
				result[termID][nodeID] = db.readRowDirectByte(rowIndex)
			}
		}
	}
	return result
}

func (db *Stage1PIRDB) readRowDirectByte(rowIndex int) []byte {
	start := rowIndex * db.R
	end := start + db.R
	if start >= 0 && end <= len(db.DataMmap) {
		row := make([]byte, db.R)
		copy(row, db.DataMmap[start:end])
		return row
	}
	return make([]byte, db.R)
}

// GetMetrics sums every level database into one set of Stage1 metrics.
func (db *Stage1PIRDB) GetMetrics() map[string]float64 {
	total := make(map[string]float64)
	for _, levelDB := range db.Levels {
		if levelDB == nil || levelDB.pir == nil {
			continue
		}
		for key, value := range levelDB.pir.GetMetrics() {
			total[key] += value
		}
	}
	return total
}
