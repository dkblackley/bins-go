package main

import (
	"fmt"
	"log"

	"bm25-msmarco/internal/bm25"
)

func main() {
	fmt.Println("=== Minimal PIR Query Test ===")
	fmt.Println("Loading Stage1 data (this will take ~1 minute for PIR preprocessing)...")

	stage1DB, err := bm25.NewStage1PIRDBWithPIR(
		"data/stage1_data.bin",
		"data/stage1/stage1_idmap.bin",
		"data/stage1_vocab.json",
		128,  // r
		true, // Enable PIR
		128,  // Small batch size for faster preprocessing
	)
	if err != nil {
		log.Fatalf("Failed to load Stage1: %v", err)
	}
	defer stage1DB.Close()

	fmt.Println("Stage1 PIR loaded successfully!")
	fmt.Println()

	// Test a few specific rows
	testRows := []uint64{0, 1, 100, 1000, 1000000}

	for _, rowIdx := range testRows {
		fmt.Printf("Testing row %d...\n", rowIdx)

		// Direct read from DataMmap
		start := int(rowIdx) * stage1DB.R
		end := start + stage1DB.R
		var directBytes []byte
		if end <= len(stage1DB.DataMmap) {
			directBytes = make([]byte, stage1DB.R)
			copy(directBytes, stage1DB.DataMmap[start:end])
		} else {
			fmt.Printf("  ❌ Row out of bounds\n\n")
			continue
		}

		// PIR query for this single row
		// Note: We need to access the internal PIR, but it's not exported
		// So we'll use the GetScoreBatch method which uses PIR internally
		// But that requires term/node IDs. Let's find a valid one from the lookup table.

		fmt.Printf("  (Skipping PIR test - need to use GetScoreBatch with valid term/node IDs)\n")
		fmt.Printf("  Direct read first 10 bytes: %v\n", directBytes[:10])
		fmt.Println()
	}

	fmt.Println("===")
	fmt.Println("To properly test PIR, we need to query using actual term/node pairs.")
	fmt.Println("Let's test with a real query term...")
	fmt.Println()

	// Find a term that exists
	testTerm := "what"
	termID, ok := stage1DB.Vocab[testTerm]
	if !ok {
		fmt.Printf("Term '%s' not found in vocab\n", testTerm)
		return
	}

	fmt.Printf("Testing with term '%s' (ID=%d)\n", testTerm, termID)

	// Find nodes for this term
	termMap, ok := stage1DB.LookupTable[termID]
	if !ok {
		fmt.Printf("Term ID %d not in lookup table\n", termID)
		return
	}

	// Get first few nodes
	testNodes := []int{}
	for nodeID := range termMap {
		testNodes = append(testNodes, nodeID)
		if len(testNodes) >= 5 {
			break
		}
	}

	fmt.Printf("Testing %d nodes for this term\n", len(testNodes))
	fmt.Println()

	// Query with PIR (via GetScoreBatch)
	pirResults := stage1DB.GetScoreBatch([]int{termID}, testNodes)

	// Query with direct (disable PIR temporarily by using getScoreDirect - but it's not exported)
	// So let's just verify the PIR results look reasonable
	for _, nodeID := range testNodes {
		rowIdx, ok := termMap[nodeID]
		if !ok {
			continue
		}

		pirRow := pirResults[termID][nodeID]

		// Direct read
		start := rowIdx * stage1DB.R
		end := start + stage1DB.R
		directRow := make([]byte, stage1DB.R)
		copy(directRow, stage1DB.DataMmap[start:end])

		// Compare
		match := true
		for i := 0; i < stage1DB.R; i++ {
			if pirRow[i] != directRow[i] {
				match = false
				break
			}
		}

		if match {
			fmt.Printf("  ✅ Node %d (row %d): MATCH\n", nodeID, rowIdx)
		} else {
			fmt.Printf("  ❌ Node %d (row %d): MISMATCH!\n", nodeID, rowIdx)
			fmt.Printf("     PIR:    %v...\n", pirRow[:min(20, len(pirRow))])
			fmt.Printf("     Direct: %v...\n", directRow[:min(20, len(directRow))])
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
