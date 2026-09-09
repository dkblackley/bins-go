package main

import (
	"encoding/binary"
	"fmt"
	"log"

	"bm25-msmarco/internal/bm25"
)

func main() {
	// Test configuration
	dataBinPath := "data/stage2/data.bin"
	idmapBinPath := "data/stage2/idmap.bin"
	vocabPath := "data/stage1_vocab.json"
	B := 32
	s := 8
	scale := 10.0

	fmt.Println("=== Stage2 PIR vs Plaintext Diagnostic ===")
	fmt.Println()

	// Load Stage2 without PIR (plaintext mode)
	fmt.Println("Loading Stage2 in plaintext mode...")
	plaintext, err := bm25.NewStage2PrecomputedWithPIR(dataBinPath, idmapBinPath, vocabPath, B, s, scale, false, 128)
	if err != nil {
		log.Fatalf("Failed to load plaintext Stage2: %v", err)
	}
	defer plaintext.Close()

	// Load Stage2 with PIR
	fmt.Println("Loading Stage2 with PIR mode...")
	pirMode, err := bm25.NewStage2PrecomputedWithPIR(dataBinPath, idmapBinPath, vocabPath, B, s, scale, true, 128)
	if err != nil {
		log.Fatalf("Failed to load PIR Stage2: %v", err)
	}
	defer pirMode.Close()

	fmt.Printf("Stage2 config: B=%d, s=%d, R=%d, scale=%.1f\n", B, s, plaintext.R, scale)
	fmt.Printf("DataMmap size: %d bytes (%d rows of %d bytes each)\n", len(plaintext.DataMmap), len(plaintext.DataMmap)/plaintext.R, plaintext.R)
	fmt.Printf("DataUint64 size: %d uint64 words\n", len(pirMode.DataUint64))
	fmt.Println()

	// Test query from the problematic example: query 1048585
	// Query terms: ["what", "paula", "deen", "brother"]
	testTerms := []string{"what", "paula", "deen", "brother"}

	// Gold doc 2465873 is in leaf node that contains this doc range
	// With B=32, doc 2465872-2465904 would be in leaf node 77058
	// Test with some nodes around that range
	testLeafNodes := []int{77057, 77058, 77059, 0, 1}

	fmt.Println("=== Testing query terms:", testTerms, "===")
	fmt.Println()

	// Get results from plaintext
	fmt.Println("--- PLAINTEXT MODE ---")
	plaintextResults := plaintext.SumBoundsForTerms(testTerms, testLeafNodes)
	for _, node := range testLeafNodes {
		if vec, ok := plaintextResults[node]; ok {
			fmt.Printf("Node %d: %v\n", node, vec)
		}
	}
	fmt.Println()

	// Get results from PIR
	fmt.Println("--- PIR MODE ---")
	pirResults := pirMode.SumBoundsForTermsWithPIR(testTerms, testLeafNodes)
	for _, node := range testLeafNodes {
		if vec, ok := pirResults[node]; ok {
			fmt.Printf("Node %d: %v\n", node, vec)
		}
	}
	fmt.Println()

	// Compare results
	fmt.Println("=== COMPARISON ===")
	allMatch := true
	for _, node := range testLeafNodes {
		plaintextVec, ok1 := plaintextResults[node]
		pirVec, ok2 := pirResults[node]

		if !ok1 && !ok2 {
			continue
		}

		if !ok1 || !ok2 {
			fmt.Printf("❌ Node %d: Missing in %s\n", node, map[bool]string{true: "PIR", false: "plaintext"}[!ok2])
			allMatch = false
			continue
		}

		match := true
		for j := 0; j < len(plaintextVec) && j < len(pirVec); j++ {
			if plaintextVec[j] != pirVec[j] {
				match = false
				break
			}
		}

		if match {
			fmt.Printf("✅ Node %d: MATCH\n", node)
		} else {
			fmt.Printf("❌ Node %d: MISMATCH\n", node)
			fmt.Printf("   Plaintext: %v\n", plaintextVec)
			fmt.Printf("   PIR:       %v\n", pirVec)
			allMatch = false
		}
	}
	fmt.Println()

	if allMatch {
		fmt.Println("✅ All results match! PIR is working correctly.")
	} else {
		fmt.Println("❌ Results differ! PIR is producing incorrect results.")
		fmt.Println()

		// Additional diagnostic: check raw row data
		fmt.Println("=== RAW DATA DIAGNOSTIC ===")

		// Find a specific term-node pair to inspect
		term := "contain"
		node := 0

		if termID, ok := plaintext.Vocab[term]; ok {
			if termMap, ok := plaintext.LookupTable[termID]; ok {
				if rowIdx, ok := termMap[node]; ok {
					fmt.Printf("Term '%s' (ID=%d), Node=%d → Row index=%d\n", term, termID, node, rowIdx)
					fmt.Println()

					// Read raw bytes from plaintext DataMmap
					start := rowIdx * plaintext.R
					end := start + plaintext.R
					if end <= len(plaintext.DataMmap) {
						rawBytes := plaintext.DataMmap[start:end]
						fmt.Printf("Raw bytes from DataMmap[%d:%d]: %v\n", start, end, rawBytes)

						// Read from PIR DataUint64
						if rowIdx < len(pirMode.DataUint64) {
							word := pirMode.DataUint64[rowIdx]
							fmt.Printf("PIR DataUint64[%d]: 0x%016x\n", rowIdx, word)

							// Unpack word back to bytes
							unpackedBytes := make([]byte, 8)
							binary.LittleEndian.PutUint64(unpackedBytes, word)
							fmt.Printf("Unpacked bytes from PIR: %v (first %d bytes)\n", unpackedBytes[:plaintext.R], plaintext.R)

							// Compare
							bytesMatch := true
							for j := 0; j < plaintext.R; j++ {
								if rawBytes[j] != unpackedBytes[j] {
									bytesMatch = false
									break
								}
							}

							if bytesMatch {
								fmt.Println("✅ Raw bytes MATCH between DataMmap and DataUint64")
							} else {
								fmt.Println("❌ Raw bytes MISMATCH! Conversion error detected!")
							}
						}
					}
					fmt.Println()

					// Test PIR Query for this specific row
					fmt.Println("Testing PIR Query() for this row...")
					pirResponse, err := pirMode.QueryPIR([]uint64{uint64(rowIdx)})
					if err != nil {
						fmt.Printf("❌ PIR Query error: %v\n", err)
					} else {
						fmt.Printf("PIR Query response: %d words returned\n", len(pirResponse))
						if len(pirResponse) > 0 && len(pirResponse[0]) > 0 {
							respWord := pirResponse[0][0]
							fmt.Printf("PIR response word: 0x%016x\n", respWord)

							respBytes := make([]byte, 8)
							binary.LittleEndian.PutUint64(respBytes, respWord)
							fmt.Printf("PIR response bytes: %v (first %d bytes)\n", respBytes[:plaintext.R], plaintext.R)

							// Compare with original
							if respWord == pirMode.DataUint64[rowIdx] {
								fmt.Println("✅ PIR Query returned CORRECT data (matches DataUint64)")
							} else {
								fmt.Println("❌ PIR Query returned INCORRECT data!")
								fmt.Printf("   Expected: 0x%016x\n", pirMode.DataUint64[rowIdx])
								fmt.Printf("   Got:      0x%016x\n", respWord)
							}
						}
					}
				}
			}
		}
	}
}
