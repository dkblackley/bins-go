package main

import (
	"bm25-msmarco/internal/bm25"
	"fmt"
	"log"
)

func main() {
	fmt.Println("Testing more PIR queries...")

	stage1DB, err := bm25.NewStage1PIRDBWithPIR(
		"data/stage1_data.bin",
		"data/stage1/stage1_idmap.bin",
		"data/stage1_vocab.json",
		128,
		true,
		128,
	)
	if err != nil {
		log.Fatalf("Failed: %v", err)
	}
	defer stage1DB.Close()

	// Test the problematic row and nearby rows
	testTermID := 2560442 // "what"
	testNodeIDs := []int{17346, 17345, 17347, 17344, 17348}

	for _, nodeID := range testNodeIDs {
		rowIdx, ok := stage1DB.LookupTable[testTermID][nodeID]
		if !ok {
			fmt.Printf("Node %d: not in lookup\n", nodeID)
			continue
		}

		// PIR query
		pirResults := stage1DB.GetScoreBatch([]int{testTermID}, []int{nodeID})
		pirRow := pirResults[testTermID][nodeID]

		// Direct read
		start := rowIdx * stage1DB.R
		end := start + stage1DB.R
		directRow := make([]byte, stage1DB.R)
		copy(directRow, stage1DB.DataMmap[start:end])

		// Check if all zeros
		allZeros := true
		for _, b := range directRow {
			if b != 0 {
				allZeros = false
				break
			}
		}

		// Compare
		match := len(pirRow) == len(directRow)
		if match {
			for i := 0; i < len(pirRow); i++ {
				if pirRow[i] != directRow[i] {
					match = false
					break
				}
			}
		}

		status := "✅"
		if !match {
			status = "❌"
		}

		fmt.Printf("%s Node %d (row %d): zeros=%v, match=%v, direct[:8]=%v, pir[:8]=%v\n",
			status, nodeID, rowIdx, allZeros, match, directRow[:8], pirRow[:8])
	}
}
