package bm25

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"sync"

	"github.com/dkblackley/bins-go/pianopir"
	"github.com/sirupsen/logrus"
)

// Stage1PIRDB represents the database for stage 1 of the PIR protocol
type Stage1PIRDB struct {
	R           int
	DataMmap    []byte
	DataUint64  [][]uint64
	dataFile    *os.File
	LookupTable map[int]map[int]int // term_id -> node_id -> row_index
	Vocab       map[string]int      // term_str -> term_id
	Pir         *pianopir.SimpleBatchPianoPIR
	mu          sync.RWMutex
}

// NewStage1PIRDB creates a new Stage1PIRDB instance
func NewStage1PIRDB(dataBinPath, idmapBinPath, vocabPath string, r int) (*Stage1PIRDB, error) {
	return NewStage1PIRDBWithPIR(dataBinPath, idmapBinPath, vocabPath, r, false, 128)
}

// NewStage1PIRDBWithPIR creates a Stage1PIRDB instance with optional PIR support
func NewStage1PIRDBWithPIR(dataBinPath, idmapBinPath, vocabPath string, r int, enablePIR bool, batchSize uint64) (*Stage1PIRDB, error) {
	db := &Stage1PIRDB{
		R:           r,
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
		db.Vocab[term] = tid
		// Add debug logging for term IDs
		// if term == "androgen" || term == "receptor" || term == "define" {
		// 	fmt.Printf("Stage1 Vocab: term '%s' -> id %d\n", term, tid)
		// }
	}

	// Open and mmap data file
	dataFile, err := os.OpenFile(dataBinPath, os.O_RDONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("error opening data file: %v", err)
	}
	db.dataFile = dataFile

	dataStat, err := dataFile.Stat()
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("error getting data file stats: %v", err)
	}

	db.DataMmap, err = mmap(dataFile, dataStat.Size())
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("error mmapping data file: %v", err)
	}

	// Load idmap
	idmapFile, err := os.OpenFile(idmapBinPath, os.O_RDONLY, 0)
	if err != nil {
		db.Close()
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

		if _, exists := db.LookupTable[termID]; !exists {
			db.LookupTable[termID] = make(map[int]int)
		}
		db.LookupTable[termID][nodeID] = rowIndex
	}

	// Initialize PIR if enabled
	if enablePIR {
		// DBSize is number of rows. Each row is r bytes = r/8 uint64 elements
		entrySizeUint64 := uint64(r) / 8
		db.DataUint64 = convertBytesToUint64(db.DataMmap, entrySizeUint64)
		dbSize := uint64(len(db.DataUint64)) / entrySizeUint64
		db.Pir = pianopir.NewSimpleBatchPianoPIR(
			uint64(len(db.DataUint64)),
			uint64(len(db.DataUint64[0])),
			//uint64(r),
			dbSize,
			batchSize,
			db.DataUint64,
			20, // FailureProbLog2
			batchSize,
		)
		fmt.Printf("Stage1 PIR: initialized with DBSize=%d, EntrySize=%d bytes, BatchSize=%d\n", dbSize, r, batchSize)
		// db.Pir.Preprocessing()
		// fmt.Printf("Stage1 PIR: preprocessing complete\n")
	}

	return db, nil
}

// Close releases resources
func (db *Stage1PIRDB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.DataMmap != nil {
		if err := unmmap(db.DataMmap); err != nil {
			return fmt.Errorf("error unmapping data: %v", err)
		}
		db.DataMmap = nil
	}

	if db.dataFile != nil {
		if err := db.dataFile.Close(); err != nil {
			return fmt.Errorf("error closing data file: %v", err)
		}
		db.dataFile = nil
	}

	return nil
}

// GetScoreBatch retrieves scores for multiple term-node pairs using batch PIR
func (db *Stage1PIRDB) GetScoreBatch(termIDs []int, nodeIDs []int) map[int]map[int][]byte {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.Pir == nil {
		// Fallback to direct read if PIR not available
		return db.getScoreDirect(termIDs, nodeIDs)
	}

	// Collect row indices and metadata
	type rowRequest struct {
		termID int
		nodeID int
		rowIdx uint64
	}

	var requests []rowRequest
	for _, termID := range termIDs {
		if termMap, ok := db.LookupTable[termID]; ok {
			for _, nodeID := range nodeIDs {
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
		return make(map[int]map[int][]byte)
	}

	result := make(map[int]map[int][]byte)
	batchSize := int(db.Pir.Config().BatchSize) - 1

	for i := 0; i < len(requests); i += batchSize {
		end := i + batchSize
		if end > len(requests) {
			end = len(requests)
		}

		chunkRequests := requests[i:end]
		rowIndices := make([]uint64, len(chunkRequests))
		for j, req := range chunkRequests {
			rowIndices[j] = req.rowIdx
		}
		logrus.Errorf("Stage1 PIR: Querying %d rows", len(rowIndices))
		logrus.Errorf("Total queries to be made %d, start %d, end ,%d", len(requests), i, end)
		logrus.Errorf("Chunk size %d", len(chunkRequests))
		logrus.Errorf("batch size %d", batchSize)
		responses, err := db.Pir.Query(rowIndices)
		if err != nil {
			// Fallback logic for just this chunk
			for _, req := range chunkRequests {
				if _, ok := result[req.termID]; !ok {
					result[req.termID] = make(map[int][]byte)
				}
				result[req.termID][req.nodeID] = db.readRowDirectByte(int(req.rowIdx))
			}
			continue
		}

		for j, response := range responses {
			req := chunkRequests[j]
			if _, ok := result[req.termID]; !ok {
				result[req.termID] = make(map[int][]byte)
			}
			result[req.termID][req.nodeID] = convertUint64ToBytes(response)
		}
	}

	//// Extract row indices for batch query
	//rowIndices := make([]uint64, len(requests))
	//for i, req := range requests {
	//	rowIndices[i] = req.rowIdx
	//}
	//
	//// Single batch PIR query
	//responses, err := db.Pir.Query(rowIndices)
	//if err != nil {
	//	fmt.Printf("Stage1 batch query error: %v, falling back to direct read\n", err)
	//	return db.getScoreDirect(termIDs, nodeIDs)
	//}
	//
	//// Aggregate responses into result map
	//result := make(map[int]map[int][]byte)
	//for i, req := range requests {
	//	if _, ok := result[req.termID]; !ok {
	//		result[req.termID] = make(map[int][]byte)
	//	}
	//
	//	response := responses[i]
	//
	//	// Handle miss (all zeros) by falling back to direct read
	//	if allZeroResponseUint64(response) {
	//		response = db.readRowDirectUint64(req.rowIdx)
	//		if allZeroResponseUint64(response) {
	//			// Still zero, use zero response
	//			response = make([]uint64, db.R)
	//		}
	//	}
	//
	//	// Convert uint64 to byte array
	//	row := convertUint64ToBytes(response)
	//
	//	// Debug: Compare PIR response with direct read for first few requests
	//	if Debug && i < 3 {
	//		directRow := db.readRowDirectByte(int(req.rowIdx))
	//		pirMatch := true
	//		for j := 0; j < min(len(row), len(directRow)); j++ {
	//			if row[j] != directRow[j] {
	//				pirMatch = false
	//				break
	//			}
	//		}
	//		if !pirMatch {
	//			Debugf("Stage1 PIR MISMATCH: termID=%d, nodeID=%d, rowIdx=%d\n  PIR:    %v\n  Direct: %v",
	//				req.termID, req.nodeID, req.rowIdx, row[:min(10, len(row))], directRow[:min(10, len(directRow))])
	//		} else {
	//			Debugf("Stage1 PIR OK: termID=%d, nodeID=%d, rowIdx=%d, first 10 bytes match", req.termID, req.nodeID, req.rowIdx)
	//		}
	//	}
	//
	//	result[req.termID][req.nodeID] = row
	//}

	return result
}

// getScoreDirect retrieves scores directly from mmap (no PIR)
func (db *Stage1PIRDB) getScoreDirect(termIDs []int, nodeIDs []int) map[int]map[int][]byte {
	result := make(map[int]map[int][]byte)

	for _, termID := range termIDs {
		if termMap, ok := db.LookupTable[termID]; ok {
			for _, nodeID := range nodeIDs {
				if rowIdx, ok := termMap[nodeID]; ok {
					row := db.readRowDirectByte(rowIdx)
					if _, ok := result[termID]; !ok {
						result[termID] = make(map[int][]byte)
					}
					result[termID][nodeID] = row
				}
			}
		}
	}

	return result
}

// readRowDirectByte reads a row directly from byte-format mmap
func (db *Stage1PIRDB) readRowDirectByte(rowIdx int) []byte {
	// Leave this as-is just now
	start := rowIdx * db.R
	end := start + db.R
	if end <= len(db.DataMmap) {
		row := make([]byte, db.R)
		copy(row, db.DataMmap[start:end])
		return row
	}
	return make([]byte, db.R)
}

// readRowDirectUint64 reads a row directly from uint64-format mmap
func (db *Stage1PIRDB) readRowDirectUint64(rowIdx uint64) []uint64 {
	if db.DataUint64 == nil {
		return make([]uint64, db.R/8)
	}
	// Should just be direct now.
	return db.DataUint64[rowIdx]

	// Each row is R bytes = R/8 uint64 elements
	//entrySizeUint64 := uint64(db.R) / 8
	//start := rowIdx * entrySizeUint64
	//end := start + entrySizeUint64
	//if end <= uint64(len(db.DataUint64)) {
	//	row := make([]uint64, entrySizeUint64)
	//	copy(row, db.DataUint64[start:end])
	//	return row
	//}
	//return make([]uint64, entrySizeUint64)
}
