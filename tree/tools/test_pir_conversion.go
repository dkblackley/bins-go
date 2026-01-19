package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"math/rand"

	"bm25-msmarco/internal/bm25"
)

func main() {
	fmt.Println("=== Testing PIR Data Conversion ===")
	fmt.Println()

	// Test Stage1: Load data files and test conversion
	fmt.Println("--- Stage1 Test ---")
	stage1DB, err := bm25.NewStage1PIRDBWithPIR(
		"data/stage1_data.bin",
		"data/stage1/stage1_idmap.bin",
		"data/stage1_vocab.json",
		128,   // r
		false, // Don't enable PIR to avoid preprocessing
		1024,
	)
	if err != nil {
		log.Fatalf("Failed to load Stage1: %v", err)
	}
	defer stage1DB.Close()

	fmt.Printf("Stage1 loaded: %d rows, r=%d\n", len(stage1DB.DataMmap)/stage1DB.R, stage1DB.R)

	// Test conversion roundtrip for Stage1
	testStage1Conversion(stage1DB)

	fmt.Println()
	fmt.Println("--- Stage2 Test ---")

	// Test Stage2
	stage2DB, err := bm25.NewStage2PrecomputedWithPIR(
		"data/stage2/data.bin",
		"data/stage2/idmap.bin",
		"data/stage1_vocab.json",
		32,    // B
		8,     // s
		10.0,  // scale
		false, // Don't enable PIR
		1024,
	)
	if err != nil {
		log.Fatalf("Failed to load Stage2: %v", err)
	}
	defer stage2DB.Close()

	fmt.Printf("Stage2 loaded: %d rows, R=%d\n", len(stage2DB.DataMmap)/stage2DB.R, stage2DB.R)

	// Test conversion roundtrip for Stage2
	testStage2Conversion(stage2DB)
}

func testStage1Conversion(db *bm25.Stage1PIRDB) {
	// Test random rows
	numTests := 10
	numRows := len(db.DataMmap) / db.R

	fmt.Printf("Testing %d random rows...\n", numTests)

	allMatch := true
	for i := 0; i < numTests; i++ {
		rowIdx := rand.Intn(numRows)

		// Read original bytes
		start := rowIdx * db.R
		end := start + db.R
		originalBytes := make([]byte, db.R)
		copy(originalBytes, db.DataMmap[start:end])

		// Convert to uint64 (simulating PIR storage)
		uint64Words := make([]uint64, db.R)
		for j := 0; j < db.R; j++ {
			uint64Words[j] = uint64(originalBytes[j])
		}

		// Convert back to bytes (simulating PIR retrieval)
		convertedBytes := make([]byte, db.R*8)
		for j, word := range uint64Words {
			binary.LittleEndian.PutUint64(convertedBytes[j*8:(j+1)*8], word)
		}

		// Compare first R bytes
		match := true
		for j := 0; j < db.R; j++ {
			if originalBytes[j] != convertedBytes[j*8] {
				match = false
				break
			}
		}

		if !match {
			fmt.Printf("  ❌ Row %d: MISMATCH\n", rowIdx)
			fmt.Printf("     Original: %v...\n", originalBytes[:min(10, len(originalBytes))])
			fmt.Printf("     Converted: %v...\n", convertedBytes[:min(10, len(convertedBytes))])
			allMatch = false
		} else if i < 3 {
			fmt.Printf("  ✅ Row %d: OK\n", rowIdx)
		}
	}

	if allMatch {
		fmt.Println("✅ All Stage1 conversions correct!")
	} else {
		fmt.Println("❌ Stage1 conversion has issues!")
	}
}

func testStage2Conversion(db *bm25.Stage2Precomputed) {
	// Test random rows
	numTests := 10
	numRows := len(db.DataMmap) / db.R

	fmt.Printf("Testing %d random rows...\n", numTests)

	allMatch := true
	for i := 0; i < numTests; i++ {
		rowIdx := rand.Intn(numRows)

		// Read original bytes
		start := rowIdx * db.R
		end := start + db.R
		originalBytes := make([]byte, db.R)
		copy(originalBytes, db.DataMmap[start:end])

		// Convert to uint64 (pack into one word, simulating PIR storage)
		tmp := make([]byte, 8)
		copy(tmp, originalBytes)
		word := binary.LittleEndian.Uint64(tmp)

		// Convert back to bytes (simulating PIR retrieval)
		convertedBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(convertedBytes, word)

		// Compare first R bytes
		match := true
		for j := 0; j < db.R; j++ {
			if originalBytes[j] != convertedBytes[j] {
				match = false
				break
			}
		}

		if !match {
			fmt.Printf("  ❌ Row %d: MISMATCH\n", rowIdx)
			fmt.Printf("     Original: %v\n", originalBytes)
			fmt.Printf("     Word: 0x%016x\n", word)
			fmt.Printf("     Converted: %v\n", convertedBytes[:db.R])
			allMatch = false
		} else if i < 3 {
			fmt.Printf("  ✅ Row %d: OK (bytes: %v)\n", rowIdx, originalBytes)
		}
	}

	if allMatch {
		fmt.Println("✅ All Stage2 conversions correct!")
	} else {
		fmt.Println("❌ Stage2 conversion has issues!")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
