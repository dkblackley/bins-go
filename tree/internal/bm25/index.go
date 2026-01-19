package bm25

import (
	"encoding/binary"
	"fmt"
	"os"
)

// LuceneIndex handles document ID mapping and index statistics
type LuceneIndex struct {
	docMappingFile *os.File
	docMappingMmap []byte
	stats          IndexStats
}

type IndexStats struct {
	NumDocs    int
	TotalTerms int64
	AverageDL  float64
}

// NewLuceneIndex creates a new index interface from mapping files
func NewLuceneIndex(docMappingPath string) (*LuceneIndex, error) {
	// Open document mapping file
	docMapFile, err := os.OpenFile(docMappingPath, os.O_RDONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("error opening doc mapping file: %v", err)
	}

	// Get file stats
	stat, err := docMapFile.Stat()
	if err != nil {
		docMapFile.Close()
		return nil, fmt.Errorf("error getting file stats: %v", err)
	}

	// Memory map the file
	docMap, err := mmap(docMapFile, stat.Size())
	if err != nil {
		docMapFile.Close()
		return nil, fmt.Errorf("error mmapping doc mapping file: %v", err)
	}

	idx := &LuceneIndex{
		docMappingFile: docMapFile,
		docMappingMmap: docMap,
	}

	// Read index stats from metadata file if available
	// This is a placeholder - actual stats should come from index metadata
	idx.stats = IndexStats{
		NumDocs:    int(stat.Size() / 8), // Assuming 8 bytes per mapping
		TotalTerms: 0,                    // Should come from index metadata
		AverageDL:  0,                    // Should come from index metadata
	}

	return idx, nil
}

// Close releases resources
func (idx *LuceneIndex) Close() error {
	if idx.docMappingMmap != nil {
		if err := unmmap(idx.docMappingMmap); err != nil {
			return fmt.Errorf("error unmapping doc mapping: %v", err)
		}
		idx.docMappingMmap = nil
	}

	if idx.docMappingFile != nil {
		if err := idx.docMappingFile.Close(); err != nil {
			return fmt.Errorf("error closing doc mapping file: %v", err)
		}
		idx.docMappingFile = nil
	}

	return nil
}

// ConvertInternalToExternalID converts an internal document ID to its external ID
func (idx *LuceneIndex) ConvertInternalToExternalID(internalID int) (int, error) {
	if internalID < 0 || internalID*8 >= len(idx.docMappingMmap) {
		return 0, fmt.Errorf("internal ID out of range: %d", internalID)
	}

	// Read 8 bytes starting at internalID * 8
	externalID := binary.LittleEndian.Uint64(idx.docMappingMmap[internalID*8 : internalID*8+8])
	return int(externalID), nil
}

// GetStats returns index statistics
func (idx *LuceneIndex) GetStats() IndexStats {
	return idx.stats
}
