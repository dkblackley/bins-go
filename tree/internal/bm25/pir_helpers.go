package bm25

import (
	"encoding/binary"
	"fmt"
	"regexp"
	"strconv"

	pianopir "github.com/dkblackley/bins-go/pir-old"
)

// verifyPackedPIR checks a small, evenly distributed sample after offline
// preprocessing. It is intentionally used only in strict benchmark runs: a
// successful call with an incorrect response must never be mistaken for a
// legitimate quantized zero row.
//
// data may stop short of the last row's end (less than one row: a zero-padded partial last row, as the Stage3 DB
// is when it is the mmap'd embedding file); the sample is still spread over every row, so the same rows (and the
// same number of PIR queries, which count towards the byte totals) are sampled as with a padded copy.
func verifyPackedPIR(name string, pir *pianopir.SimpleBatchPianoPIR, data []uint64, wordsPerRow uint64) error {
	if pir == nil || wordsPerRow == 0 {
		return fmt.Errorf("%s PIR verification has invalid dimensions", name)
	}
	rowCount := (uint64(len(data)) + wordsPerRow - 1) / wordsPerRow
	if rowCount == 0 {
		return fmt.Errorf("%s PIR verification has an empty database", name)
	}

	const maxSamples = 32
	sampleCount := uint64(maxSamples)
	if rowCount < sampleCount {
		sampleCount = rowCount
	}
	indices := make([]uint64, sampleCount)
	for i := uint64(0); i < sampleCount; i++ {
		indices[i] = i * rowCount / sampleCount
	}
	responses, err := pir.Query(indices)
	if err != nil {
		return fmt.Errorf("%s PIR verification query failed: %w", name, err)
	}
	if len(responses) != len(indices) {
		return fmt.Errorf("%s PIR verification returned %d rows for %d requests", name, len(responses), len(indices))
	}
	for i, rowIndex := range indices {
		expected := make([]uint64, wordsPerRow) // zeros past the end of data
		end := (rowIndex + 1) * wordsPerRow
		if end > uint64(len(data)) {
			end = uint64(len(data))
		}
		copy(expected, data[rowIndex*wordsPerRow:end])
		actual := responses[i]
		if len(actual) != len(expected) {
			return fmt.Errorf("%s PIR row %d returned %d words; want %d", name, rowIndex, len(actual), len(expected))
		}
		for word := range expected {
			if actual[word] != expected[word] {
				return fmt.Errorf("%s PIR row %d word %d mismatch: got %#x, want %#x", name, rowIndex, word, actual[word], expected[word])
			}
		}
	}
	return nil
}

// convertBytesToUint64 converts a byte slice to a uint64 slice (little-endian)
// Does NOT pad - assumes data length is compatible with the expected entry size
func convertBytesToUint64(data []byte) []uint64 {
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

// convertUint64ToBytes converts a uint64 slice to a byte slice (little-endian)
func convertUint64ToBytes(data []uint64) []byte {
	result := make([]byte, len(data)*8)
	for i, v := range data {
		binary.LittleEndian.PutUint64(result[i*8:(i+1)*8], v)
	}
	return result
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
