package bm25

import (
	"encoding/binary"
	"fmt"
	"regexp"
	"strconv"
)

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
