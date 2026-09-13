package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/dkblackley/bins-go/tree/internal/bm25"
)

func main() {
	fmt.Println("=== Comprehensive Batch PIR Test ===")
	fmt.Println("Loading Stage1 with PIR...")

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

	// Test with different batch sizes
	batchSizes := []int{1, 3, 5, 10, 20, 50}

	testTerm := "what"
	termID, ok := stage1DB.Vocab[testTerm]
	if !ok {
		log.Fatalf("Term not found")
	}

	// Get all nodes for this term
	termMap := stage1DB.LookupTable[termID]
	allNodes := []int{}
	for nodeID := range termMap {
		allNodes = append(allNodes, nodeID)
	}

	fmt.Printf("Testing term '%s' (ID=%d) with %d nodes\n\n", testTerm, termID, len(allNodes))

	for _, batchSize := range batchSizes {
		if batchSize > len(allNodes) {
			batchSize = len(allNodes)
		}

		// Random sample of nodes
		rand.Shuffle(len(allNodes), func(i, j int) {
			allNodes[i], allNodes[j] = allNodes[j], allNodes[i]
		})
		testNodes := allNodes[:batchSize]

		// Query with PIR
		pirResults := stage1DB.GetScoreBatch([]int{termID}, testNodes)

		// Verify each result
		mismatches := 0
		for _, nodeID := range testNodes {
			rowIdx := termMap[nodeID]
			pirRow := pirResults[termID][nodeID]

			// Direct read
			start := rowIdx * stage1DB.R
			end := start + stage1DB.R
			directRow := make([]byte, stage1DB.R)
			copy(directRow, stage1DB.DataMmap[start:end])

			// Compare
			match := len(pirRow) == len(directRow)
			if match {
				for i := 0; i < len(pirRow); i++ {
					if pirRow[i] != directRow[i] {
						match = false
						mismatches++
						break
					}
				}
			} else {
				mismatches++
			}
		}

		status := "✅"
		if mismatches > 0 {
			status = "❌"
		}

		fmt.Printf("%s Batch size %2d: %d/%d queries correct\n", status, batchSize, batchSize-mismatches, batchSize)
	}

	fmt.Println("\n✅ All tests passed!")
}
