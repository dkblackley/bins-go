package bm25

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/dkblackley/bins-go/pianopir"
)

// Stage2Precomputed represents the precomputed database for stage 2
type Stage2Precomputed struct {
	S           int     // Sub-block size
	R           int     // R = B/S
	Scale       float64 // Scale factor for scores
	DataMmap    []byte
	DataUint64  [][]uint64
	dataFile    *os.File
	LookupTable map[int]map[int]int // term_id -> node_id -> row_index
	Vocab       map[string]int      // term_str -> term_id
	Pir         *pianopir.SimpleBatchPianoPIR
	mu          sync.RWMutex
}

// NewStage2Precomputed creates a new Stage2Precomputed instance
func NewStage2Precomputed(dataBinPath, idmapBinPath, vocabPath string, B, s int, scale float64) (*Stage2Precomputed, error) {
	return NewStage2PrecomputedWithPIR(dataBinPath, idmapBinPath, vocabPath, B, s, scale, false, 128)
}

// NewStage2PrecomputedWithPIR creates a Stage2Precomputed instance with optional PIR support
func NewStage2PrecomputedWithPIR(dataBinPath, idmapBinPath, vocabPath string, B, s int, scale float64, enablePIR bool, batchSize uint64) (*Stage2Precomputed, error) {
	if B%s != 0 {
		return nil, fmt.Errorf("block size B (%d) must be divisible by s (%d)", B, s)
	}

	pre := &Stage2Precomputed{
		S:           s,
		R:           B / s,
		Scale:       scale,
		LookupTable: make(map[int]map[int]int),
		Vocab:       make(map[string]int),
		Pir:         nil,
	}

	// Load vocab
	vocabData, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("error reading vocab file: %v", err)
	}

	var vocabRaw map[string]string
	if err := json.Unmarshal(vocabData, &vocabRaw); err != nil {
		return nil, fmt.Errorf("error unmarshaling vocab: %v", err)
	}

	for term, tidStr := range vocabRaw {
		var tid int
		fmt.Sscanf(tidStr, "%d", &tid)
		// Convert term to lowercase for case-insensitive matching
		pre.Vocab[term] = tid
	}

	// Open and mmap data file
	dataFile, err := os.OpenFile(dataBinPath, os.O_RDONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("error opening data file: %v", err)
	}
	pre.dataFile = dataFile

	dataStat, err := dataFile.Stat()
	if err != nil {
		pre.Close()
		return nil, fmt.Errorf("error getting data file stats: %v", err)
	}

	pre.DataMmap, err = mmap(dataFile, dataStat.Size())
	if err != nil {
		pre.Close()
		return nil, fmt.Errorf("error mmapping data file: %v", err)
	}

	// Load idmap
	idmapFile, err := os.OpenFile(idmapBinPath, os.O_RDONLY, 0)
	if err != nil {
		pre.Close()
		return nil, fmt.Errorf("error opening idmap file: %v", err)
	}
	defer idmapFile.Close()

	// Read records from idmap
	recordSize := 16 // 4 + 4 + 8 bytes
	buf := make([]byte, recordSize)

	for {
		n, err := idmapFile.Read(buf)
		if n == 0 || err != nil {
			break
		}

		termID := int(binary.LittleEndian.Uint32(buf[0:4]))
		nodeID := int(binary.LittleEndian.Uint32(buf[4:8]))
		rowIndex := int(binary.LittleEndian.Uint64(buf[8:16]))

		if _, exists := pre.LookupTable[termID]; !exists {
			pre.LookupTable[termID] = make(map[int]int)
		}
		pre.LookupTable[termID][nodeID] = rowIndex
	}

	// Initialize PIR if enabled
	// Note: stage2 rows are R uint8 values (one byte per sub-block).
	// Pack each row into one uint64 word (little-endian), then pass to PIR.
	if enablePIR {
		// Pack each Stage2 row into one uint64 word
		pre.DataUint64 = convertRowsToUint64(pre.DataMmap, pre.R)
		// DBSize is number of rows
		dbSize := uint64(len(pre.DataUint64))
		// DBEntryByteNum for PIR: since each row is packed into 1 uint64 word,
		// and PIR will compute DBEntrySize = ceil(DBEntryByteNum / 8),
		// we pass DBEntryByteNum = 8 (1 word) to get DBEntrySize = 1 word per row.
		dbEntryByteNum := uint64(8) // 1 uint64 word
		//fmt.Printf("Stage2 PIR debug: pre.R=%d, len(DataMmap)=%d (rows), len(DataUint64)=%d (words), DBSize=%d\n", pre.R, len(pre.DataMmap)/pre.R, len(pre.DataUint64), dbSize)
		pre.Pir = pianopir.NewSimpleBatchPianoPIR(
			dbSize,
			1,
			dbEntryByteNum, // 8 bytes = 1 uint64 word
			batchSize,
			pre.DataUint64,
			20, // FailureProbLog2
			batchSize,
		)
		fmt.Printf("Stage2 PIR: initialized with DBSize=%d, EntryByteNum=8 (1 uint64 word), BatchSize=%d\n", dbSize, batchSize)
		//pre.Pir.Preprocessing()
		//fmt.Printf("Stage2 PIR: preprocessing complete\n")
	}

	return pre, nil
}

// Close releases resources
func (pre *Stage2Precomputed) Close() error {
	pre.mu.Lock()
	defer pre.mu.Unlock()

	if pre.DataMmap != nil {
		if err := unmmap(pre.DataMmap); err != nil {
			return fmt.Errorf("error unmapping data: %v", err)
		}
		pre.DataMmap = nil
	}

	if pre.dataFile != nil {
		if err := pre.dataFile.Close(); err != nil {
			return fmt.Errorf("error closing data file: %v", err)
		}
		pre.dataFile = nil
	}

	return nil
}

// SumBoundsForTerms sums the bounds for given terms and leaf nodes
func (pre *Stage2Precomputed) SumBoundsForTerms(terms []string, leafNodes []int) map[int][]int {
	pre.mu.RLock()
	defer pre.mu.RUnlock()

	if len(leafNodes) == 0 {
		return nil
	}

	// Convert leaf nodes to set for O(1) lookup
	wantedNodes := make(map[int]bool)
	for _, ln := range leafNodes {
		wantedNodes[ln] = true
	}

	// Convert terms to term IDs, ignore OOV terms
	var termIDs []int
	for _, term := range terms {
		// Try exact match first
		tid, ok := pre.Vocab[term]
		if !ok {
			// Try lowercase
			tid, ok = pre.Vocab[strings.ToLower(term)]
		}
		if ok {
			termIDs = append(termIDs, tid)
		}
	}

	if len(termIDs) == 0 {
		// No terms exist in precomputation
		acc := make(map[int][]int)
		for ln := range wantedNodes {
			acc[ln] = make([]int, pre.R)
		}
		return acc
	}

	// Initialize accumulators per leaf
	acc := make(map[int][]int)
	for ln := range wantedNodes {
		acc[ln] = make([]int, pre.R)
	}

	// Per term, find rows for the wanted leaf nodes, then add
	for _, tid := range termIDs {
		if termMap, ok := pre.LookupTable[tid]; ok {
			for ln := range wantedNodes {
				if rowIdx, ok := termMap[ln]; ok {
					// Read the row bytes (each row is pre.R bytes)
					start := rowIdx * pre.R
					end := start + pre.R
					if end <= len(pre.DataMmap) {
						row := pre.DataMmap[start:end]
						// Add to accumulator
						for j := 0; j < pre.R; j++ {
							acc[ln][j] += int(row[j])
						}
					}
				}
			}
		}
	}

	return acc
}

// SumBoundsForTermsWithPIR sums the bounds for given terms and leaf nodes using batch PIR
func (pre *Stage2Precomputed) SumBoundsForTermsWithPIR(terms []string, leafNodes []int) map[int][]int {
	pre.mu.RLock()
	defer pre.mu.RUnlock()

	if len(leafNodes) == 0 {
		return nil
	}

	// If PIR not available, use direct method
	if pre.Pir == nil {
		return pre.SumBoundsForTerms(terms, leafNodes)
	}

	// Convert leaf nodes to set for O(1) lookup
	wantedNodes := make(map[int]bool)
	for _, ln := range leafNodes {
		wantedNodes[ln] = true
	}

	// Convert terms to term IDs, ignore OOV terms
	var termIDs []int
	for _, term := range terms {
		// Try exact match first
		tid, ok := pre.Vocab[term]
		if !ok {
			// Try lowercase
			tid, ok = pre.Vocab[strings.ToLower(term)]
		}
		if ok {
			termIDs = append(termIDs, tid)
		}
	}

	if len(termIDs) == 0 {
		// No terms exist in precomputation
		acc := make(map[int][]int)
		for ln := range wantedNodes {
			acc[ln] = make([]int, pre.R)
		}
		return acc
	}

	// Initialize accumulators per leaf
	acc := make(map[int][]int)
	for ln := range wantedNodes {
		acc[ln] = make([]int, pre.R)
	}

	// Collect all row indices needed
	type rowRequest struct {
		termID int
		nodeID int
		rowIdx uint64
	}

	var requests []rowRequest
	for _, termID := range termIDs {
		if termMap, ok := pre.LookupTable[termID]; ok {
			for nodeID := range wantedNodes {
				if rowIdx, ok := termMap[nodeID]; ok {
					requests = append(requests, rowRequest{
						termID: termID,
						nodeID: nodeID,
						rowIdx: uint64(rowIdx),
					})
				}
			}
		}
	}

	if len(requests) == 0 {
		return acc
	}

	// Extract row indices for batch query
	rowIndices := make([]uint64, len(requests))
	for i, req := range requests {
		rowIndices[i] = req.rowIdx
	}

	// BATCH QUERY - single call for all rows
	responses, err := pre.Pir.Query(rowIndices)
	if err != nil {
		fmt.Printf("Stage2 batch query error: %v, falling back to direct read\n", err)
		// Fallback: use direct method
		return pre.SumBoundsForTerms(terms, leafNodes)
	}

	// Process responses
	for i, req := range requests {
		rowWords := responses[i]

		// Handle miss (all zeros) by falling back to direct read
		if allZeroResponseUint64(rowWords) {
			rowWords = pre.readRowDirectUint64(req.rowIdx)
			if allZeroResponseUint64(rowWords) {
				// Still zero, treat as zero scores (all zeros)
				// create empty words corresponding to ceil(pre.R/8) words
				wordsPerRow := (pre.R + 7) / 8
				rowWords = make([]uint64, wordsPerRow)
			}
		}

		// Convert uint64 word(s) back to bytes
		rowBytes := convertUint64ToBytes(rowWords)

		// Debug: Compare PIR response with direct read for first few requests
		if Debug && i < 3 {
			start := int(req.rowIdx) * pre.R
			end := start + pre.R
			var directBytes []byte
			if end <= len(pre.DataMmap) {
				directBytes = make([]byte, pre.R)
				copy(directBytes, pre.DataMmap[start:end])
			} else {
				directBytes = make([]byte, pre.R)
			}

			pirMatch := true
			for j := 0; j < pre.R; j++ {
				if rowBytes[j] != directBytes[j] {
					pirMatch = false
					break
				}
			}
			if !pirMatch {
				Debugf("Stage2 PIR MISMATCH: termID=%d, nodeID=%d, rowIdx=%d\n  PIR:    %v\n  Direct: %v",
					req.termID, req.nodeID, req.rowIdx, rowBytes[:pre.R], directBytes[:pre.R])
			} else {
				Debugf("Stage2 PIR OK: termID=%d, nodeID=%d, rowIdx=%d, bytes match", req.termID, req.nodeID, req.rowIdx)
			}
		}

		// Use first pre.R bytes (each is one uint8 sub-block score)
		for j := 0; j < pre.R && j < len(rowBytes); j++ {
			acc[req.nodeID][j] += int(rowBytes[j])
		}
	}

	return acc
}

// readRowDirectUint64 reads a row directly from uint64-format mmap
// Each row is packed into 1 uint64 word
func (pre *Stage2Precomputed) readRowDirectUint64(rowIdx uint64) []uint64 {
	if pre.DataUint64 == nil {
		return make([]uint64, 1) // 1 word per row
	}
	// Each row occupies exactly 1 uint64 word (since pre.R <= 8)
	if rowIdx < uint64(len(pre.DataUint64)) {
		return pre.DataUint64[rowIdx] // SHould be fine to call directly now(?)
	}
	return make([]uint64, 1)
}

// QueryPIR performs a PIR query for the given row indices (for diagnostic purposes)
func (pre *Stage2Precomputed) QueryPIR(rowIndices []uint64) ([][]uint64, error) {
	if pre.Pir == nil {
		return nil, fmt.Errorf("PIR not initialized")
	}
	return pre.Pir.Query(rowIndices)
}
