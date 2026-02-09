package bm25

import (
	"encoding/binary"
	"fmt"
	"math"
	"regexp"
	"strconv"
)

func flattenBlock(batchVectors [][]float32, targetBlockSize, dim int) []uint64 {
	// Calculate total bytes needed for a full block
	totalBytes := targetBlockSize * dim * 4

	// Create a byte buffer to hold the raw bits
	vectorBytes := make([]byte, totalBytes)

	// 1. Convert Floats -> Bytes (uint32 bits)
	cursor := 0
	for _, vector := range batchVectors {
		for j := 0; j < dim; j++ {
			binary.LittleEndian.PutUint32(vectorBytes[cursor:], math.Float32bits(vector[j]))
			cursor += 4
		}
	}
	// Note: If batchVectors < targetBlockSize, the remaining bytes in vectorBytes stay 0 (padding)

	// 2. Convert Bytes -> Uint64s for the PIR DB
	// We assume totalBytes is divisible by 8. If dim * 4 is not divisible by 8,
	// ensure blockSize is even, otherwise we might need explicit padding logic here.
	u64Len := totalBytes / 8
	entry := make([]uint64, u64Len)

	for j := 0; j < u64Len; j++ {
		entry[j] = binary.LittleEndian.Uint64(vectorBytes[j*8:])
	}

	return entry
}

// recoverBlock maps a single uint64 slice (PIR DB row) back into a slice of float vectors.
func recoverBlock(entry []uint64, blockSize, dim int) [][]float32 {
	totalBytes := blockSize * dim * 4
	vectorBytes := make([]byte, totalBytes)

	// 1. Convert Uint64s -> Bytes
	for j := 0; j < len(entry); j++ {
		binary.LittleEndian.PutUint64(vectorBytes[j*8:], entry[j])
	}

	// 2. Convert Bytes -> Floats
	result := make([][]float32, blockSize)
	cursor := 0

	for i := 0; i < blockSize; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			bits := binary.LittleEndian.Uint32(vectorBytes[cursor:])
			vec[j] = math.Float32frombits(bits)
			cursor += 4
		}
		result[i] = vec
	}

	return result
}

// getUniqueBlockIndices takes a list of requested Doc IDs and returns:
// 1. A slice of unique Block IDs (uint64) to send to the PIR Query.
// 2. A map lookup that tells us where a BlockID sits in that query slice.
func (sr *Stage3Reranker) getUniqueBlockIndices(docIndices []int) ([]uint64, map[int]int) {
	uniqueBlocks := make([]uint64, 0)

	// blockIndexMap: maps RealBlockID -> IndexInQuerySlice
	// e.g., if Block 50 is the 0th item in our query, blockIndexMap[50] = 0
	blockIndexMap := make(map[int]int)
	seen := make(map[int]bool)

	for _, docID := range docIndices {
		// Look up which block this doc belongs to
		blockID, ok := sr.blockMap[docID]
		if !ok {
			continue // Handle missing doc error if needed
		}

		if !seen[blockID] {
			seen[blockID] = true
			blockIndexMap[blockID] = len(uniqueBlocks)
			uniqueBlocks = append(uniqueBlocks, uint64(blockID))
		}
	}

	return uniqueBlocks, blockIndexMap
}

// mapPIRResultsToDocs extracts the specific document vectors from the retrieved blocks.
func (sr *Stage3Reranker) mapPIRResultsToDocs(
	resultsRaw [][]uint64,
	docIndices []int,
	blockIndexMap map[int]int, // The map returned from the first function
	blockSize int, // e.g. 8
	embedDim int, // e.g. 192
) map[int][]float32 {

	finalResults := make(map[int][]float32)

	// validBlockCache prevents us from decoding the same block bytes multiple times
	// if multiple docs from the same block were requested.
	decodedBlocks := make(map[int][][]float32)

	for _, docID := range docIndices {
		blockID, ok := sr.blockMap[docID]
		if !ok {
			continue
		}

		// 1. Find where this block's data is in the resultsRaw
		resultIdx, found := blockIndexMap[blockID]
		if !found {
			continue // Should not happen if logic is correct
		}

		// 2. Decode the block (if we haven't already)
		if _, cached := decodedBlocks[blockID]; !cached {
			// Uses the recoverBlock helper from previous step
			decodedBlocks[blockID] = recoverBlock(resultsRaw[resultIdx], blockSize, embedDim)
		}

		// 3. Calculate the offset of this doc within the block.
		// Assuming docs are packed sequentially: Doc 0 is at index 0, Doc 8 is at index 0 of Block 1.
		localIndex := docID % blockSize

		// 4. Extract the specific vector
		blockVectors := decodedBlocks[blockID]
		if localIndex < len(blockVectors) {
			finalResults[docID] = blockVectors[localIndex]
		}
	}

	return finalResults
}

// convertBytesToUint64 converts a byte slice to a uint64 slice (little-endian)
// Does NOT pad - assumes data length is compatible with the expected entry size
func convertBytesToUint64Old(data []byte) []uint64 {
	// Calculate how many uint64 words we need (ceil), pad last word with zeros if necessary
	numUint64s := (len(data) + 7) / 8
	result := make([]uint64, numUint64s)
	for i := 0; i < numUint64s; i++ {
		start := i * 8
		end := start + 8
		if end <= len(data) {
			result[i] = binary.LittleEndian.Uint64(data[start:end])
		} else {
			// pad remaining bytes with zeros
			tmp := make([]byte, 8)
			copy(tmp, data[start:len(data)])
			result[i] = binary.LittleEndian.Uint64(tmp)
		}
	}
	return result
}

func convertBytesToUint64(data []byte, entrySize uint64) [][]uint64 {

	// Calculate how many uint64 words we need (ceil), pad last word with zeros if necessary
	numUint64s := (len(data) + 7) / 8

	// First build the flat slice
	flat := make([]uint64, numUint64s)
	for i := 0; i < numUint64s; i++ {
		start := i * 8
		end := start + 8
		if end <= len(data) {
			flat[i] = binary.LittleEndian.Uint64(data[start:end])
		} else {
			tmp := make([]byte, 8)
			copy(tmp, data[start:])
			flat[i] = binary.LittleEndian.Uint64(tmp)
		}
	}

	// Reshape into [][]uint64 with `entrySize` items per row
	// Just assume the DB fits perfectly.
	converted := int(entrySize)
	numRows := (numUint64s + converted - 1) / converted
	result := make([][]uint64, numRows)
	for r := 0; r < numRows; r++ {
		start := r * converted
		end := start + converted
		if end > numUint64s {
			end = numUint64s
		}
		result[r] = flat[start:end]
	}

	return result
}

// convertUint64ToBytes converts a uint64 slice to a byte slice (little-endian)
func convertUint64ToBytes(data []uint64) []byte {
	result := make([]byte, len(data)*8)
	for i, v := range data {
		binary.LittleEndian.PutUint64(result[i*8:(i+1)*8], v)
	}
	return result
}

func flatten(matrix [][]uint64) []uint64 {
	// 1. Calculate the total number of elements
	totalLen := 0
	for _, row := range matrix {
		totalLen += len(row)
	}

	// 2. Pre-allocate the slice with the exact capacity needed
	// make(type, length, capacity)
	result := make([]uint64, 0, totalLen)

	// 3. Append the rows using the variadic "..." operator
	for _, row := range matrix {
		result = append(result, row...)
	}

	return result
}

// convertRowsToUint64 packs each row (rowSize bytes) into one uint64 word (little-endian).
// Assumes rowSize <= 8. If data length is not a multiple of rowSize, the last partial row
// is padded with zeros.
//func convertRowsToUint64(data []byte, rowSize int) []uint64 {
//	if rowSize <= 0 { // Shouldn't it always be 1?
//		return nil
//	}
//	numRows := (len(data) + rowSize - 1) / rowSize
//	result := make([]uint64, numRows)
//	for i := 0; i < numRows; i++ {
//		start := i * rowSize
//		end := start + rowSize
//		if end > len(data) {
//			end = len(data)
//		}
//		tmp := make([]byte, 8)
//		copy(tmp, data[start:end])
//		result[i] = binary.LittleEndian.Uint64(tmp)
//	}
//	return result
//}

func convertRowsToUint64(data []byte, rowSize int) [][]uint64 {
	if rowSize <= 0 {
		return nil
	}

	numRows := (len(data) + rowSize - 1) / rowSize
	out := make([][]uint64, numRows)

	for i := 0; i < numRows; i++ {
		start := i * rowSize
		end := start + rowSize
		if end > len(data) {
			end = len(data)
		}

		tmp := make([]byte, 8)
		copy(tmp, data[start:end])
		out[i] = []uint64{binary.LittleEndian.Uint64(tmp)}
	}

	return out
}

// allZeroResponseUint64 checks if a uint64 response is all zeros
func allZeroResponseUint64(response []uint64) bool {
	for _, v := range response {
		if v != 0 {
			return false
		}
	}
	return true
}

// allZeroResponseByte checks if a byte response is all zeros
func allZeroResponseByte(response []byte) bool {
	for _, v := range response {
		if v != 0 {
			return false
		}
	}
	return true
}

// ParseNPYHeader parses a NumPy .npy file header and extracts shape and dtype information
// Returns: (numDocs, embeddingDim, bytesPerElement, headerSize, error)
// For a 2D array of shape (N, D), returns (N, D, bytesPerElement, headerSize)
func ParseNPYHeader(data []byte) (int, int, int, int, error) {
	if len(data) < 16 {
		return 0, 0, 0, 0, fmt.Errorf("NPY file too small")
	}

	// Check magic number: \x93NUMPY
	if data[0] != 0x93 || string(data[1:6]) != "NUMPY" {
		return 0, 0, 0, 0, fmt.Errorf("invalid NPY magic number")
	}

	majorVersion := data[6]
	minorVersion := data[7]
	if majorVersion != 1 {
		return 0, 0, 0, 0, fmt.Errorf("unsupported NPY version %d.%d", majorVersion, minorVersion)
	}

	// Header length is stored as uint16 (little-endian) for version 1.0
	headerLen := int(binary.LittleEndian.Uint16(data[8:10]))
	headerStart := 10
	headerEnd := headerStart + headerLen

	if headerEnd > len(data) {
		return 0, 0, 0, 0, fmt.Errorf("header length extends beyond file")
	}

	headerStr := string(data[headerStart:headerEnd])

	// Parse dtype to get bytes per element
	bytesPerElem := 8 // default float64
	dtypeMatch := regexp.MustCompile(`'descr':\s*'([^']+)'`).FindStringSubmatch(headerStr)
	if len(dtypeMatch) > 1 {
		dtype := dtypeMatch[1]
		switch dtype {
		case "<f4": // float32 (little-endian)
			bytesPerElem = 4
		case "<f8": // float64 (little-endian)
			bytesPerElem = 8
		case "<i4": // int32 (little-endian)
			bytesPerElem = 4
		case "<i8": // int64 (little-endian)
			bytesPerElem = 8
		default:
			// Try to parse format: <u1, <i2, etc.
			if len(dtype) >= 2 {
				sizeStr := dtype[2:]
				if size, err := strconv.Atoi(sizeStr); err == nil {
					bytesPerElem = size
				}
			}
		}
	}

	// Parse shape: 'shape': (N, D)
	shapeMatch := regexp.MustCompile(`'shape':\s*\((\d+),\s*(\d+)\)`).FindStringSubmatch(headerStr)
	if len(shapeMatch) != 3 {
		return 0, 0, 0, 0, fmt.Errorf("could not parse 2D shape from NPY header")
	}

	numDocs, err := strconv.Atoi(shapeMatch[1])
	if err != nil {
		return 0, 0, 0, 0, fmt.Errorf("failed to parse num docs: %v", err)
	}

	embeddingDim, err := strconv.Atoi(shapeMatch[2])
	if err != nil {
		return 0, 0, 0, 0, fmt.Errorf("failed to parse embedding dim: %v", err)
	}

	dataStart := headerEnd
	return numDocs, embeddingDim, bytesPerElem, dataStart, nil
}
