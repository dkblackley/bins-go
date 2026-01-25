package bm25

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"sync"

	"github.com/dkblackley/bins-go/pianopir"
	"github.com/sirupsen/logrus"
)

// Stage3Reranker performs dense reranking using query/document embeddings
// Similar to step4.py: loads embeddings, computes cosine similarity, reranks
type Stage3Reranker struct {
	// Query embeddings: one embedding per query
	QueryEmbeddings [][]float32    // indexed by query index in order
	QueryIDMap      map[string]int // query_id -> index in QueryEmbeddings

	// Document embeddings: mmap of dimension D (typically 192 for PCA'd 768)
	DocEmbeddings [][]float32 // mmap'd to avoid memory overhead
	docFile       *os.File
	docEmbedMmap  []byte
	EmbedDim      int // e.g., 192
	BytesPerElem  int // 4 for float32, 8 for float64

	// Document ID map: maps internal doc index to external doc ID
	DocIDMap []string
	// Reverse map: external doc ID -> embedding array index
	DocIDReverseMap map[string]int

	// PIR support: optional PIR-backed embedding access
	Pir       *pianopir.SimpleBatchPianoPIR
	pirUsable bool // true if PIR is initialized and can be used

	// Blocking & Permutation Support
	BlockPermutation []int // Maps LogicalBlockID -> PhysicalBlockID

	mu sync.RWMutex
}

// NewStage3Reranker initializes the reranker with embeddings
// docEmbedPath: path to document embeddings (mmap npy format)
// docIDMapPath: path to document ID mapping file (one doc ID per line)
// queryEmbedPath: path to query embeddings (npy format, loaded fully in memory)
// permutationPath: path to block permutation JSON map (optional)
// enablePIR: if true, initialize PIR for private embedding access
// batchSize: PIR batch size
func NewStage3Reranker(
	docEmbedPath, docIDMapPath, queryEmbedPath string,
	permutationPath string, // <--- New parameter
	embedDim int,
	enablePIR bool,
	batchSize uint64,
	queryList []string,
) (*Stage3Reranker, error) {

	logrus.Debugf("Loading permutation from path: %v", permutationPath)

	QueryIDMap := make(map[string]int)

	for i, qid := range queryList {
		QueryIDMap[qid] = i

		if i < 3 {
			logrus.Debugf("QueryIDMap[%d] = %s\n", i, qid)
		}
	}

	sr := &Stage3Reranker{
		EmbedDim:     embedDim,
		BytesPerElem: 0, // will be set from NPY header
		QueryIDMap:   make(map[string]int),
		Pir:          nil,
		pirUsable:    false,
	}

	// Load document embeddings (mmap)
	docFile, err := os.OpenFile(docEmbedPath, os.O_RDONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("error opening doc embeddings file: %v", err)
	}
	sr.docFile = docFile

	docStat, err := docFile.Stat()
	if err != nil {
		sr.Close()
		return nil, fmt.Errorf("error getting doc embeddings file stats: %v", err)
	}

	// Memory-map the entire file (including header)
	fullMmap, err := mmap(docFile, docStat.Size())
	if err != nil {
		sr.Close()
		return nil, fmt.Errorf("error mmapping doc embeddings: %v", err)
	}

	// Parse NPY header to get actual shape and dtype
	numDocs, npyEmbedDim, bytesPerElem, dataStart, err := ParseNPYHeader(fullMmap)
	if err != nil {
		sr.Close()
		return nil, fmt.Errorf("error parsing NPY header: %v", err)
	}

	// Verify embedding dimension matches what we expect
	if npyEmbedDim != embedDim {
		fmt.Printf("WARNING: NPY file has embedding dim=%d, but embedDim parameter=%d. Using NPY value.\n", npyEmbedDim, embedDim)
	}

	// Point docEmbedMmap to the data portion only (skip header)
	sr.docEmbedMmap = fullMmap[dataStart:]
	sr.EmbedDim = npyEmbedDim
	sr.BytesPerElem = bytesPerElem

	// Verify the data size matches expectations
	expectedDataSize := numDocs * npyEmbedDim * bytesPerElem
	if len(sr.docEmbedMmap) < expectedDataSize {
		sr.Close()
		return nil, fmt.Errorf("data size mismatch: file has %d bytes, expected at least %d bytes for %d docs x %d dim x %d bytes/elem",
			len(sr.docEmbedMmap), expectedDataSize, numDocs, npyEmbedDim, bytesPerElem)
	}

	fmt.Printf("Stage3 Reranker: loaded %d document embeddings (dim=%d, bytes/elem=%d)\n", numDocs, npyEmbedDim, bytesPerElem)

	// Load document ID map
	docIDFile, err := os.OpenFile(docIDMapPath, os.O_RDONLY, 0)
	if err != nil {
		sr.Close()
		return nil, fmt.Errorf("error opening doc ID map file: %v", err)
	}
	defer docIDFile.Close()

	idmapStat, err := docIDFile.Stat()
	if err != nil {
		sr.Close()
		return nil, fmt.Errorf("error getting doc ID map file stats: %v", err)
	}

	idmapData := make([]byte, idmapStat.Size())
	_, err = docIDFile.Read(idmapData)
	if err != nil {
		sr.Close()
		return nil, fmt.Errorf("error reading doc ID map: %v", err)
	}

	// Parse line-by-line (one doc ID per line)
	sr.DocIDMap = parseDocIDs(string(idmapData))
	fmt.Printf("Stage3 Reranker: loaded %d document IDs\n", len(sr.DocIDMap))

	// Build reverse map: external doc ID -> embedding array index
	sr.DocIDReverseMap = make(map[string]int, len(sr.DocIDMap))
	for idx, extID := range sr.DocIDMap {
		sr.DocIDReverseMap[extID] = idx
	}
	fmt.Printf("Stage3 Reranker: built reverse map with %d entries\n", len(sr.DocIDReverseMap))

	// Load query embeddings (fully in memory)
	queryEmbedData, err := os.ReadFile(queryEmbedPath)
	if err != nil {
		sr.Close()
		return nil, fmt.Errorf("error reading query embeddings: %v", err)
	}

	// Parse NPY header for query embeddings
	numQueries, queryEmbedDim, queryBytesPerElem, queryDataStart, err := ParseNPYHeader(queryEmbedData)
	if err != nil {
		sr.Close()
		return nil, fmt.Errorf("error parsing query embeddings NPY header: %v", err)
	}

	logrus.Debugf("[Stage3 DEBUG] Managed to load %d number of queries from file %s", numQueries, queryEmbedPath)

	// Verify query embedding dimension matches document embedding dimension
	if queryEmbedDim != sr.EmbedDim {
		fmt.Printf("WARNING: Query embeddings have dim=%d, but document embeddings have dim=%d. Using query dim.\n", queryEmbedDim, sr.EmbedDim)
		sr.EmbedDim = queryEmbedDim
	}

	// Extract query embeddings data (skip header)
	queryData := queryEmbedData[queryDataStart:]
	sr.QueryEmbeddings = parseQueryEmbeddings(queryData, numQueries, queryEmbedDim, queryBytesPerElem)
	fmt.Printf("Stage3 Reranker: loaded %d query embeddings (dim=%d, bytes/elem=%d)\n", len(sr.QueryEmbeddings), queryEmbedDim, queryBytesPerElem)

	// --- NEW: Load Block Permutation Map ---
	if permutationPath != "" {
		permData, err := os.ReadFile(permutationPath)
		if err != nil {
			return nil, fmt.Errorf("error reading permutation map: %v", err)
		}
		if err := json.Unmarshal(permData, &sr.BlockPermutation); err != nil {
			return nil, fmt.Errorf("error parsing permutation map: %v", err)
		}
		fmt.Printf("Stage3 Reranker: loaded block permutation map with %d entries\n", len(sr.BlockPermutation))
	}

	// Initialize PIR for document embeddings if enabled
	if enablePIR {
		// --- BLOCKING & PADDING LOGIC ---
		blockSize := 8 // Treat 8 documents as 1 block

		// Calculate DB dimensions in terms of BLOCKS
		dbSize := uint64((numDocs + blockSize - 1) / blockSize)
		entrySizeBytes := uint64(npyEmbedDim * bytesPerElem * blockSize)

		dbEntrySizeUint64 := entrySizeBytes / 8

		// Convert raw bytes to uint64 array for PIR
		embedUint64 := convertBytesToUint64(sr.docEmbedMmap, dbEntrySizeUint64)
		if Debug {
			wordsPerBlock := 0
			if len(embedUint64) > 0 {
				wordsPerBlock = len(embedUint64[0])
			}
			logrus.Debugf("[Stage3 DEBUG] PIR init: numDocs=%d blockSize=8 dbSizeBlocks=%d embedUint64Blocks=%d wordsPerBlock=%d expectedWordsPerBlock=%d entrySizeBytes=%d\n",
				numDocs, dbSize, len(embedUint64), wordsPerBlock, dbEntrySizeUint64, entrySizeBytes)
		}

		// PADDING: Ensure the array is exactly the size PIR expects.
		// PIR expects (dbSize * entrySizeInUint64s)
		// expectedLen := int(dbSize * dbEntrySizeUint64)

		//if len(embedUint64) < int(dbSize) { // Should just be dbSIze now... TODO: What is this doing? - I might've broke it
		//	padAmount := int(dbSize) - len(embedUint64)
		//	fmt.Printf("Stage3 PIR: Padding embedding data with %d zeros to match block alignment\n", padAmount)
		//	padding := make([]uint64, padAmount)
		//	embedUint64 = append(embedUint64, padding...)
		//}

		sr.Pir = pianopir.NewSimpleBatchPianoPIR(
			// dbSize,
			uint64(len(embedUint64)),
			dbEntrySizeUint64,
			entrySizeBytes,
			batchSize,
			embedUint64,
			20, // FailureProbLog2
			batchSize,
		)
		fmt.Printf("Stage3 PIR (Blocking Mode): initialized with DBSize=%d blocks, EntrySize=%d bytes (8 docs/block), BatchSize=%d\n", dbSize, entrySizeBytes, batchSize)
		//sr.Pir.Preprocessing()
		//fmt.Printf("Stage3 PIR: preprocessing complete\n")
		sr.pirUsable = true
	}

	if sr.pirUsable {
		testIdxs := []int{0, 1, 7, 8, 123} // any in-range
		for _, ei := range testIdxs {
			b := uint64(ei / 8)
			resp, err := sr.Pir.Query([]uint64{b})
			if err != nil {
				continue
			}
			raw := convertUint64ToBytes(resp[0])

			embSize := sr.EmbedDim * sr.BytesPerElem
			start := (ei % 8) * embSize
			end := start + embSize

			directStart := ei * embSize
			directEnd := directStart + embSize

			if !bytes.Equal(raw[start:end], sr.docEmbedMmap[directStart:directEnd]) {
				logrus.Errorf("[Stage3 SELFTEST] mismatch at embeddingIdx=%d block=%d", ei, b)
			}
		}
	}

	logrus.Debugf("[Stage3 DEBUG] permLen=%d first10=%v",
		len(sr.BlockPermutation),
		sr.BlockPermutation[:min(10, len(sr.BlockPermutation))],
	)

	seen := make([]bool, len(sr.BlockPermutation))
	dups := 0
	for _, v := range sr.BlockPermutation {
		if v < 0 || v >= len(sr.BlockPermutation) {
			logrus.Errorf("[Stage3 DEBUG] perm out of range: %d (len=%d)", v, len(sr.BlockPermutation))
			break
		}
		if seen[v] {
			dups++
		}
		seen[v] = true
	}
	logrus.Debugf("[Stage3 DEBUG] perm duplicates=%d", dups)

	return sr, nil
}

// parseDocIDs parses line-separated doc IDs
func parseDocIDs(data string) []string {
	var ids []string
	var current []byte
	for i := 0; i < len(data); i++ {
		if data[i] == '\n' {
			if len(current) > 0 {
				ids = append(ids, string(current))
				current = nil
			}
		} else {
			current = append(current, data[i])
		}
	}
	if len(current) > 0 {
		ids = append(ids, string(current))
	}
	return ids
}

// parseQueryEmbeddings parses embeddings from raw bytes (supports both float32 and float64)
func parseQueryEmbeddings(data []byte, numQueries, embedDim, bytesPerElem int) [][]float32 {
	embeddings := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		emb := make([]float32, embedDim)
		for j := 0; j < embedDim; j++ {
			offset := (i*embedDim + j) * bytesPerElem
			if offset+bytesPerElem <= len(data) {
				switch bytesPerElem {
				case 4:
					emb[j] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset : offset+4]))
				case 8:
					emb[j] = float32(math.Float64frombits(binary.LittleEndian.Uint64(data[offset : offset+8])))
				}
			}
		}
		embeddings[i] = emb
	}
	log.Printf("[Stage3] queryEmbeddings: rows=%d dim=%v", len(embeddings), len(embeddings[0]))
	log.Printf("[Stage3] docEmbeddings: rows=%d dim=%v", len(embeddings), len(embeddings[0]))
	return embeddings
}

// GetDocEmbedding retrieves a document embedding by internal doc index
// Returns float32 slice of length EmbedDim
func (sr *Stage3Reranker) GetDocEmbedding(docIdx int) []float32 {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	emb := make([]float32, sr.EmbedDim)
	for j := 0; j < sr.EmbedDim; j++ {
		offset := (docIdx*sr.EmbedDim + j) * sr.BytesPerElem
		if offset+sr.BytesPerElem <= len(sr.docEmbedMmap) {
			switch sr.BytesPerElem {
			case 4:
				emb[j] = math.Float32frombits(binary.LittleEndian.Uint32(sr.docEmbedMmap[offset : offset+4]))
			case 8:
				emb[j] = float32(math.Float64frombits(binary.LittleEndian.Uint64(sr.docEmbedMmap[offset : offset+8])))
			}
		}
	}
	return emb
}

// GetDocEmbeddingBatch retrieves a batch of document embeddings using PIR if available
func (sr *Stage3Reranker) GetDocEmbeddingBatch(docIndices []int) map[int][]float32 {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	result := make(map[int][]float32)

	// If PIR is not available or disabled, use direct reads
	if !sr.pirUsable || sr.Pir == nil {
		for _, idx := range docIndices {
			result[idx] = sr.getDocEmbeddingDirect(idx)
		}
		return result
	}

	// --- BLOCKING & SHUFFLING LOGIC START ---
	blockSize := 8
	uniqueBlocks := make(map[uint64]bool)

	// 1. Identify which PHYSICAL blocks we need
	for _, idx := range docIndices {
		logicalBlockID := idx / blockSize
		physicalBlockID := logicalBlockID
		if len(sr.BlockPermutation) > 0 && logicalBlockID < len(sr.BlockPermutation) {
			physicalBlockID = sr.BlockPermutation[logicalBlockID]
		}
		uniqueBlocks[uint64(physicalBlockID)] = true
	}

	//for _, idx := range docIndices {
	//	blockID := idx / blockSize // idx is already physical (embeddingIdx)
	//	uniqueBlocks[uint64(blockID)] = true
	//}

	// 2. Prepare block query list
	queryBlocks := make([]uint64, 0, len(uniqueBlocks))
	for bid := range uniqueBlocks {
		queryBlocks = append(queryBlocks, bid)
	}

	// 3. EXECUTE BATCHED QUERIES
	// Retrieve the configured batch size
	batchSize := int(sr.Pir.Config().BatchSize)
	if batchSize <= 0 {
		batchSize = 16
	}

	blockData := make(map[uint64][]byte)

	for i := 0; i < len(queryBlocks); i += batchSize {
		end := i + batchSize
		if end > len(queryBlocks) {
			end = len(queryBlocks)
		}

		chunkBlocks := queryBlocks[i:end]

		responses, err := sr.Pir.Query(chunkBlocks)
		if err != nil {
			fmt.Printf("Stage3 PIR batch query error at index %d: %v, falling back to direct read for this chunk\n", i, err)
			// Fallback: If PIR fails for this chunk, just skip adding to blockData
			// The final loop (step 5) will miss these blocks, so we must fill them via direct read later if needed
			// or simpler: just switch to direct read for EVERYTHING if PIR fails?
			// For robustness, let's let the loop continue and handle missing data below.
			continue
		}

		// 4. Unpack responses into map: PhysicalBlockID -> Data
		for j, bid := range chunkBlocks {
			blockData[bid] = convertUint64ToBytes(responses[j])
		}
	}

	debugOnce := true

	//// 5. Extract specific embeddings from retrieved blocks
	//for _, docIdx := range docIndices {
	//	blockID := docIdx / blockSize // physical block
	//	offsetInBlock := docIdx % blockSize
	//
	//	rawBlock, ok := blockData[uint64(blockID)]
	//
	//	if debugOnce {
	//		debugOnce = false
	//		logrus.Debugf("[Stage3 DEBUG] docID=%d offsetInBlock=%d BlockID=%d\n", docIdx, offsetInBlock, blockID)
	//		logrus.Debugf("res ok?: %v\n", ok)
	//		logrus.Debugf("Doc embedding: %v\n", sr.GetDocEmbedding(docIdx))
	//	}
	//
	//	if !ok {
	//		result[docIdx] = sr.getDocEmbeddingDirect(docIdx)
	//		continue
	//	}
	//
	//	embSize := sr.EmbedDim * sr.BytesPerElem
	//	start := offsetInBlock * embSize
	//	end := start + embSize
	//
	//	if end <= len(rawBlock) {
	//		embBytes := rawBlock[start:end]
	//
	//		if Debug {
	//			embSize := sr.EmbedDim * sr.BytesPerElem
	//			directStart := docIdx * embSize
	//			directEnd := directStart + embSize
	//			if directEnd <= len(sr.docEmbedMmap) {
	//				directBytes := sr.docEmbedMmap[directStart:directEnd]
	//				if !bytes.Equal(directBytes, embBytes) {
	//					// print a small prefix so logs are readable
	//					n := 16
	//					if len(embBytes) < n {
	//						n = len(embBytes)
	//					}
	//					Debugf("[Stage3 DEBUG] PIR!=direct docIdx=%d blockID=%d offsetInBlock=%d\n", docIdx, blockID, offsetInBlock)
	//					Debugf("  direct[:%d]=% x\n", n, directBytes[:n])
	//					Debugf("  pir   [:%d]=% x\n", n, embBytes[:n])
	//				}
	//			}
	//		}
	//
	//		emb := make([]float32, sr.EmbedDim)
	//		for j := 0; j < sr.EmbedDim; j++ {
	//			bOffset := j * sr.BytesPerElem
	//			if sr.BytesPerElem == 4 {
	//				emb[j] = math.Float32frombits(binary.LittleEndian.Uint32(embBytes[bOffset : bOffset+4]))
	//			} else {
	//				emb[j] = float32(math.Float64frombits(binary.LittleEndian.Uint64(embBytes[bOffset : bOffset+8])))
	//			}
	//		}
	//		result[docIdx] = emb
	//	}
	//
	//}
	//// --- BLOCKING & SHUFFLING LOGIC END ---
	//
	//return result

	// 5. Extract specific embeddings from retrieved blocks
	for _, docIdx := range docIndices {
		logicalBlockID := docIdx / blockSize
		offsetInBlock := (docIdx % blockSize)

		physicalBlockID := logicalBlockID

		if len(sr.BlockPermutation) > 0 && logicalBlockID < len(sr.BlockPermutation) {
			physicalBlockID = sr.BlockPermutation[logicalBlockID]
		}

		if debugOnce {
			debugOnce = false
			logrus.Tracef("[Stage3 DEBUG] docID=%d offsetInBlock=%d physicalBlockID=%d, logicalBlockID=%d\n", docIdx, offsetInBlock, physicalBlockID, logicalBlockID)
		}

		// Check if we successfully retrieved this block via PIR
		if rawBlock, ok := blockData[uint64(physicalBlockID)]; ok {
			embSize := sr.EmbedDim * sr.BytesPerElem
			start := offsetInBlock * embSize
			end := start + embSize

			if end <= len(rawBlock) {
				embBytes := rawBlock[start:end]

				if Debug {
					embSize := sr.EmbedDim * sr.BytesPerElem
					directStart := docIdx * embSize
					directEnd := directStart + embSize
					if directEnd <= len(sr.docEmbedMmap) {
						directBytes := sr.docEmbedMmap[directStart:directEnd]
						if !bytes.Equal(directBytes, embBytes) {
							// print a small prefix so logs are readable
							n := 16
							if len(embBytes) < n {
								n = len(embBytes)
							}
							logrus.Tracef("[Stage3 DEBUG] PIR!=direct docIdx=%d blockID=%d offsetInBlock=%d\n", docIdx, physicalBlockID, offsetInBlock)
							logrus.Tracef("  direct[:%d]=% x\n", n, directBytes[:n])
							logrus.Tracef("  pir   [:%d]=% x\n", n, embBytes[:n])
						}
					}
				}

				emb := make([]float32, sr.EmbedDim)
				for j := 0; j < sr.EmbedDim; j++ {
					bOffset := j * sr.BytesPerElem
					if sr.BytesPerElem == 4 {
						emb[j] = math.Float32frombits(binary.LittleEndian.Uint32(embBytes[bOffset : bOffset+4]))
					} else {
						emb[j] = float32(math.Float64frombits(binary.LittleEndian.Uint64(embBytes[bOffset : bOffset+8])))
					}
				}
				result[docIdx] = emb
			}
		} else {
			// Fallback: Block missing (likely PIR error on that chunk), use direct read
			result[docIdx] = sr.getDocEmbeddingDirect(docIdx)
		}
	}
	// --- BLOCKING & SHUFFLING LOGIC END ---

	return result
}

//// GetDocEmbeddingBatch retrieves a batch of document embeddings using PIR if available
//// docIndices: list of internal document indices
//// Returns map of index -> embedding
//func (sr *Stage3Reranker) GetDocEmbeddingBatch(docIndices []int) map[int][]float32 {
//	sr.mu.RLock()
//	defer sr.mu.RUnlock()
//
//	result := make(map[int][]float32)
//
//	// If PIR is not available or disabled, use direct reads
//	if !sr.pirUsable || sr.Pir == nil {
//		if len(docIndices) > 0 && docIndices[0] == 7542593 {
//			Debugf("DEBUG: GetDocEmbeddingBatch using DIRECT READ path (pirUsable=%v, Pir==nil=%v)\n", sr.pirUsable, sr.Pir == nil)
//		}
//		for _, idx := range docIndices {
//			result[idx] = sr.getDocEmbeddingDirect(idx)
//		}
//		return result
//	}
//
//	// --- BLOCKING & SHUFFLING LOGIC START ---
//	blockSize := 8
//	uniqueBlocks := make(map[uint64]bool)
//
//	// 1. Identify which PHYSICAL blocks we need
//	for _, idx := range docIndices {
//		logicalBlockID := idx / blockSize
//
//		// Map Logical -> Physical (Shuffled)
//		physicalBlockID := logicalBlockID
//		if len(sr.BlockPermutation) > 0 {
//			if logicalBlockID < len(sr.BlockPermutation) {
//				physicalBlockID = sr.BlockPermutation[logicalBlockID]
//			}
//		}
//
//		uniqueBlocks[uint64(physicalBlockID)] = true
//	}
//
//	// 2. Prepare block query
//	queryBlocks := make([]uint64, 0, len(uniqueBlocks))
//	for bid := range uniqueBlocks {
//		queryBlocks = append(queryBlocks, bid)
//	}
//
//	// 3. Execute Batch Query
//	responses, err := sr.Pir.Query(queryBlocks)
//	if err != nil {
//		fmt.Printf("Stage3 PIR batch query error: %v, falling back to direct read\n", err)
//		for _, idx := range docIndices {
//			result[idx] = sr.getDocEmbeddingDirect(idx)
//		}
//		return result
//	}
//
//	// 4. Unpack responses into map: PhysicalBlockID -> Data
//	blockData := make(map[uint64][]byte)
//	for i, bid := range queryBlocks {
//		blockData[bid] = convertUint64ToBytes(responses[i])
//	}
//
//	// 5. Extract specific embeddings from retrieved blocks
//	for _, docIdx := range docIndices {
//		logicalBlockID := docIdx / blockSize
//		offsetInBlock := (docIdx % blockSize)
//
//		// Re-calculate Physical ID to look up in blockData
//		physicalBlockID := logicalBlockID
//		if len(sr.BlockPermutation) > 0 {
//			physicalBlockID = sr.BlockPermutation[logicalBlockID]
//		}
//
//		if rawBlock, ok := blockData[uint64(physicalBlockID)]; ok {
//			// Calculate byte offsets for this specific embedding within the block
//			embSize := sr.EmbedDim * sr.BytesPerElem
//			start := offsetInBlock * embSize
//			end := start + embSize
//
//			if end <= len(rawBlock) {
//				embBytes := rawBlock[start:end]
//				// Convert bytes to float32
//				emb := make([]float32, sr.EmbedDim)
//				for j := 0; j < sr.EmbedDim; j++ {
//					bOffset := j * sr.BytesPerElem
//					if sr.BytesPerElem == 4 {
//						emb[j] = math.Float32frombits(binary.LittleEndian.Uint32(embBytes[bOffset : bOffset+4]))
//					} else {
//						emb[j] = float32(math.Float64frombits(binary.LittleEndian.Uint64(embBytes[bOffset : bOffset+8])))
//					}
//				}
//				result[docIdx] = emb
//			}
//		}
//	}
//	// --- BLOCKING & SHUFFLING LOGIC END ---
//
//	return result
//}

// getDocEmbeddingDirect reads embedding directly without PIR
func (sr *Stage3Reranker) getDocEmbeddingDirect(docIdx int) []float32 {
	emb := make([]float32, sr.EmbedDim)
	for j := 0; j < sr.EmbedDim; j++ {
		offset := (docIdx*sr.EmbedDim + j) * sr.BytesPerElem
		if offset+sr.BytesPerElem <= len(sr.docEmbedMmap) {
			switch sr.BytesPerElem {
			case 4:
				emb[j] = math.Float32frombits(binary.LittleEndian.Uint32(sr.docEmbedMmap[offset : offset+4]))
			case 8:
				emb[j] = float32(math.Float64frombits(binary.LittleEndian.Uint64(sr.docEmbedMmap[offset : offset+8])))
			}
		}
	}
	// Debug: Check for specific NaN-producing index
	if docIdx == 7542593 {
		Debugf("DEBUG: getDocEmbeddingDirect(idx=%d): first 5 values = [%.6f %.6f %.6f %.6f %.6f]\n",
			docIdx, emb[0], emb[1], emb[2], emb[3], emb[4])
		hasNaN := false
		for _, v := range emb {
			if math.IsNaN(float64(v)) {
				hasNaN = true
				break
			}
		}
		if hasNaN {
			Debugf("  WARNING: Embedding contains NaN!\n")
		}
	}
	return emb
}

// CosineSimilarity computes cosine similarity between two embeddings
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProd, normA, normB float32
	for i := 0; i < len(a); i++ {
		dotProd += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProd / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// RerankerResults represents reranking output
type RerankerResults struct {
	QueryID string
	DocIDs  []string  // reranked doc IDs
	Scores  []float32 // cosine similarity scores
	MRRAt10 float32   // MRR@10 (if gold set provided)
}

// RerankedResult represents one query's reranking result
// Rerank sorts hitSubBlocks by dense similarity scores
// hitSubBlocks: from Round2 (doc ranges with BM25 internal doc IDs)
// queryIdx: index into QueryEmbeddings
// goldDocIDs: set of relevant doc IDs for evaluation (optional)
// luceneIdx: for converting BM25 internal IDs to external IDs
// Returns: reranked doc IDs with scores
func (sr *Stage3Reranker) Rerank(
	hitSubBlocks []HitSubBlock,
	queryNum string,
	goldDocIDs map[string]bool,
	luceneIdx *LuceneIndex,
) ([]string, []float32, float32) {

	queryIdx, err := sr.QueryIDMap[queryNum]
	if err {
		logrus.Errorf("ERROR: Stage3Reranker.Rerank: queryNum=%q not found in QueryIDMap", queryNum)
		os.Exit(1)
	}
	// Should just be in order?
	logrus.Debugf("[Stage3 DEBUG] queryNum=%q queryIdx=%d\n", queryNum, queryIdx)

	if queryIdx < 0 || queryIdx >= len(sr.QueryEmbeddings) {
		logrus.Errorf("ERROR: Stage3Reranker.Rerank: queryIdx=%d out of range (len(QueryEmbeddings)=%d)", queryIdx, len(sr.QueryEmbeddings))
		return nil, nil, 0
	}

	queryEmb := sr.QueryEmbeddings[queryIdx]

	// Collect all candidate doc indices from hitSubBlocks
	// Note: hitSubBlocks contain BM25 internal doc IDs, NOT embedding indices!
	// We need to convert: BM25 internal ID -> external ID -> embedding index
	type docCandidate struct {
		bm25InternalID int
		externalID     string
		embeddingIdx   int
	}
	var candidates []docCandidate

	for _, hit := range hitSubBlocks {
		for bm25InternalID := hit.Start; bm25InternalID < hit.End; bm25InternalID++ {
			// Convert BM25 internal ID to external ID
			var externalID string
			if luceneIdx != nil {
				if extID, err := luceneIdx.ConvertInternalToExternalID(bm25InternalID); err == nil && extID != 0 {
					externalID = fmt.Sprintf("%d", extID)
				} else {
					// Fallback: use internal ID as string
					externalID = fmt.Sprintf("%d", bm25InternalID)
				}
			} else {
				// No luceneIdx, use internal ID as-is
				externalID = fmt.Sprintf("%d", bm25InternalID)
			}

			//resolved := false
			//
			//if bm25InternalID < len(sr.DocIDMap) {
			//	externalID = sr.DocIDMap[bm25InternalID]
			//	resolved = true
			//}
			//
			//if !resolved && luceneIdx != nil {
			//	if extID, err := luceneIdx.ConvertInternalToExternalID(bm25InternalID); err == nil && extID != 0 {
			//		externalID = fmt.Sprintf("%d", extID)
			//		resolved = true
			//	}
			//}
			//
			//// 3. Last resort
			//if !resolved {
			//	externalID = fmt.Sprintf("%d", bm25InternalID)
			//}

			// DEBUG: compare DocIDMap vs Lucene docmap for the same internal ID
			if Debug { // assuming your Debug/Debugf gating exists in-package
				luceneExternal := "<nil-lucene>"
				if luceneIdx != nil {
					if extID, err := luceneIdx.ConvertInternalToExternalID(bm25InternalID); err == nil && extID != 0 {
						luceneExternal = fmt.Sprintf("%d", extID)
					} else {
						luceneExternal = fmt.Sprintf("<err=%v extID=%d>", err, extID)
					}
				}
				logrus.Tracef("[Stage3 DEBUG] internal=%d external(DocIDMap/choice)=%q external(lucene)=%q\n",
					bm25InternalID, externalID, luceneExternal)
			}

			// Look up embedding index for this external ID
			embIdx, ok := sr.DocIDReverseMap[externalID]
			if Debug {
				if ok {
					logrus.Tracef("[Stage3 DEBUG] externalID=%q -> embeddingIdx=%d (internal=%d)\n",
						externalID, embIdx, bm25InternalID)
				} else {
					logrus.Tracef("[Stage3 DEBUG] externalID=%q missing from DocIDReverseMap (internal=%d)\n",
						externalID, bm25InternalID)
				}
			}

			if ok {
				candidates = append(candidates, docCandidate{
					bm25InternalID: bm25InternalID,
					externalID:     externalID,
					embeddingIdx:   embIdx,
				})
			} else {
				// No embedding found for this doc - skip it
				fmt.Printf("WARNING: No embedding found for external ID %s (BM25 internal %d)\n", externalID, bm25InternalID)
			}
		}
	}

	if len(candidates) == 0 {
		return nil, nil, 0
	}

	// Debug: check if gold doc is in candidate pool
	if Debug && len(goldDocIDs) > 0 {
		goldFoundInPool := false
		for _, cand := range candidates {
			if goldDocIDs[cand.externalID] {
				Debugf("DEBUG Stage3: Gold doc %s (BM25 internal=%d, emb idx=%d) FOUND in candidate pool\n",
					cand.externalID, cand.bm25InternalID, cand.embeddingIdx)
				goldFoundInPool = true
				break
			}
		}
		if !goldFoundInPool {
			Debugf("DEBUG Stage3: Gold doc NOT FOUND in candidate pool of %d docs\n", len(candidates))

			var gold string
			for k := range goldDocIDs {
				gold = k
				break
			}

			// however you store them; external docID
			goldEmbIdx, ok := sr.DocIDReverseMap[gold]

			if ok {
				if goldEmbIdx < len(sr.DocIDMap) {
					back := sr.DocIDMap[goldEmbIdx]
					logrus.Debugf("[Stage3 DEBUG] roundtrip: gold=%d -> embIdx=%d -> back=%d", gold, goldEmbIdx, back)
				}
			}

			logrus.Debugf("[Stage3 DEBUG] gold external=%d -> embIdx=%d (ok=%v)", gold, goldEmbIdx, ok)

			// If your candidate pool is in embeddingIdx space:
			foundEmb := false
			for _, embIdx := range candidates { // docIndices are embeddingIdx
				if embIdx.embeddingIdx == goldEmbIdx {
					foundEmb = true
					break
				}
			}
			logrus.Debugf("[Stage3 DEBUG] gold embIdx=%d in docIndices? %v", goldEmbIdx, foundEmb)

		}

	}

	// Fetch embeddings using embedding indices
	embeddingIndices := make([]int, len(candidates))
	for i, cand := range candidates {
		embeddingIndices[i] = cand.embeddingIdx
	}
	embeddings := sr.GetDocEmbeddingBatch(embeddingIndices)

	// Debug: check embeddings for gold doc
	if Debug && len(goldDocIDs) > 0 {
		for _, cand := range candidates {
			if goldDocIDs[cand.externalID] {
				if emb, ok := embeddings[cand.embeddingIdx]; ok && len(emb) >= 5 {
					Debugf("DEBUG Stage3: Gold doc embedding (first 5): %.4f %.4f %.4f %.4f %.4f\n",
						emb[0], emb[1], emb[2], emb[3], emb[4])
				}
				break
			}
		}
		// Also print query embedding sample
		if len(queryEmb) >= 5 {
			Debugf("DEBUG Stage3: Query embedding (first 5): %.4f %.4f %.4f %.4f %.4f\n",
				queryEmb[0], queryEmb[1], queryEmb[2], queryEmb[3], queryEmb[4])
		}
	}

	// Score and rerank
	type scoredDoc struct {
		externalID string
		score      float32
	}

	scored := make([]scoredDoc, 0, len(candidates))
	for _, cand := range candidates {
		embIdx := cand.embeddingIdx
		if emb, ok := embeddings[embIdx]; ok {
			score := CosineSimilarity(queryEmb, emb)
			// Check for NaN
			if math.IsNaN(float64(score)) {
				fmt.Printf("WARNING: NaN score for external ID %s (emb idx %d)\n", cand.externalID, embIdx)
				// Check if embedding contains NaN
				hasNaN := false
				for _, v := range emb {
					if math.IsNaN(float64(v)) {
						hasNaN = true
						break
					}
				}
				if hasNaN {
					fmt.Printf("  Embedding contains NaN values!\n")
				}
			}
			scored = append(scored, scoredDoc{cand.externalID, score})
		}
	}

	// Sort by score descending
	for i := 0; i < len(scored); i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[j].score > scored[i].score {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}

	// Extract results
	docIDs := make([]string, len(scored))
	scores := make([]float32, len(scored))
	for i, s := range scored {
		docIDs[i] = s.externalID
		scores[i] = s.score
	}

	// Compute MRR@10 if gold set provided
	// Debug: print gold-set sample keys and top returned docIDs to diagnose mismatches
	if Debug && len(goldDocIDs) > 0 { // only debug when gold docs exist
		Debugf("DEBUG Stage3: gold size=%d\n", len(goldDocIDs))
		// print up to 5 sample gold keys
		i := 0
		for k := range goldDocIDs {
			if i < 5 {
				Debugf("  gold[%d]=%s\n", i+1, k)
				i++
			} else {
				break
			}
		}
		// Find where gold docs rank
		for goldID := range goldDocIDs {
			for rank, docID := range docIDs {
				if docID == goldID {
					Debugf("DEBUG Stage3: Gold doc %s found at rank %d (score=%.4f)\n", goldID, rank+1, scores[rank])
					break
				}
			}
		}
		// print top returned docIDs
		Debugf("DEBUG Stage3: top returned docIDs (top 10):\n")
		for t := 0; t < 10 && t < len(docIDs); t++ {
			Debugf("  %d -> %s (score=%.4f)\n", t+1, docIDs[t], scores[t])
		}
	}

	var mrr float32 = 0
	if len(goldDocIDs) > 0 {
		for rank := 0; rank < 10 && rank < len(docIDs); rank++ {
			if _, ok := goldDocIDs[docIDs[rank]]; ok {
				mrr = 1.0 / float32(rank+1)
				break
			}
		}
	}

	return docIDs, scores, mrr
}

// Close releases resources
func (sr *Stage3Reranker) Close() error {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	if sr.docEmbedMmap != nil {
		if err := unmmap(sr.docEmbedMmap); err != nil {
			return fmt.Errorf("error unmapping doc embeddings: %v", err)
		}
		sr.docEmbedMmap = nil
	}

	if sr.docFile != nil {
		if err := sr.docFile.Close(); err != nil {
			return fmt.Errorf("error closing doc embeddings file: %v", err)
		}
		sr.docFile = nil
	}

	return nil
}
