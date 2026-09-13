package bm25

import (
	"fmt"
	"sort"
)

// Round1BeamPrecomputed performs beam search using precomputed stage-1 database with batch PIR queries
func Round1BeamPrecomputed(layout *LayoutWithStats, queryTerms []string, L int, db *Stage1PIRDB) []int {
	//fmt.Println("Enter Round1BeamPrecomputed")
	frontier := []int{1} // Always start at root
	H := layout.Height()
	r := db.R

	// Convert query term strings to term_ids
	var queryTermIDs []int
	for _, term := range queryTerms {
		if termID, ok := db.Vocab[term]; ok {
			queryTermIDs = append(queryTermIDs, termID)
		}
	}

	if len(queryTermIDs) == 0 {
		return []int{}
	}

	// Loop through each level of the tree
	for level := 0; level < H; level++ {
		// Debug: show term and frontier info
		//fmt.Printf("Round1 start level %d: termIDs=%d frontier=%v\n", level, len(queryTermIDs), frontier)
		// Step 1: Gather all (termID, parentNodeID) pairs to fetch in batch
		type queryRequest struct {
			termID       int
			parentNodeID int
			rowIndex     int
		}
		var requests []queryRequest

		for _, parentNodeID := range frontier {
			for _, termID := range queryTermIDs {
				// Look up the row_index for this (term, parent_node)
				if termMap, ok := db.LookupTable[termID]; ok {
					if rowIndex, ok := termMap[parentNodeID]; ok {
						requests = append(requests, queryRequest{
							termID:       termID,
							parentNodeID: parentNodeID,
							rowIndex:     rowIndex,
						})
					}
				}
			}
		}

		// Debug: report if no requests
		if len(requests) == 0 {
			fmt.Printf("Round1 level %d: no requests found (maybe lookup table mismatch)\n", level)
		}

		if len(requests) == 0 {
			break
		}

		// Step 2: Perform batch PIR query for all requested term/node pairs
		// GetScoreBatch returns map[termID]map[nodeID][]byte
		scoresByTermNode := db.GetScoreBatch(queryTermIDs, frontier)

		// Step 3: Accumulate child scores using batch results
		childScores := make(map[int]int)
		retrieved := 0
		for _, req := range requests {
			parentNodeID := req.parentNodeID

			// Retrieve the score row from batch results (termID -> nodeID)
			termMap, ok := scoresByTermNode[req.termID]
			if !ok {
				continue
			}
			scoreRow := termMap[req.parentNodeID]
			if len(scoreRow) == 0 {
				continue // Skip if no data
			}
			retrieved++

			// Add scores to all 'r' children
			for j := 0; j < r && j < len(scoreRow); j++ {
				childNodeID := parentNodeID*r + j
				childScores[childNodeID] += int(scoreRow[j])
			}
		}

		// Debug: print stats for this level
		//fmt.Printf("Round1 level %d: requests=%d retrieved=%d childScores=%d\n", level, len(requests), retrieved, len(childScores))

		if len(childScores) == 0 {
			break
		}

		// Sort children by score
		type scoreNode struct {
			nodeID int
			score  int
		}
		sortedChildren := make([]scoreNode, 0, len(childScores))
		for nodeID, score := range childScores {
			sortedChildren = append(sortedChildren, scoreNode{nodeID, score})
		}
		sort.Slice(sortedChildren, func(i, j int) bool {
			return sortedChildren[i].score > sortedChildren[j].score
		})

		// Set new frontier (top L node_ids)
		frontier = make([]int, 0, L)
		for i := 0; i < L && i < len(sortedChildren); i++ {
			frontier = append(frontier, sortedChildren[i].nodeID)
		}
	}

	return frontier
}

// Round2BMWPrecomputed performs BMW algorithm using precomputed stage-2 database with batch PIR queries
func Round2BMWPrecomputed(
	layout *LayoutWithStats,
	leafNodes []int,
	queryTerms []string,
	s int,
	kCandidates int,
	pre *Stage2Precomputed,
) []HitSubBlock {
	if pre.S != s {
		panic(fmt.Sprintf("Precomputed s=%d does not match requested s=%d", pre.S, s))
	}
	r := pre.R

	// Step 1: Perform batch PIR query for all leaf nodes and all terms at once
	// This collects all required IDs upfront: map[leafNodeID][]int (per-leaf bounds)
	sumsByLeaf := pre.SumBoundsForTermsWithPIR(queryTerms, leafNodes)

	var hits []HitSubBlock
	var totalDocs int

	// Step 2: Expand each leaf into sub-blocks and attach summed scores
	for _, node := range leafNodes {
		a, b := layout.NodeToDocRange(node)
		if a >= b {
			continue // Skip invalid ranges
		}

		vec, ok := sumsByLeaf[node]
		if !ok {
			continue // Skip if no data for this leaf
		}

		// Create sub-blocks
		for j := 0; j < r; j++ {
			start := a + j*s
			if start >= b {
				break
			}
			end := min(b, start+s)

			// Score: divide by scale to get back to approx float UB
			scoreUB := float64(vec[j]) / pre.Scale

			if scoreUB <= 0.0 {
				continue
			}

			hits = append(hits, HitSubBlock{
				Start:   start,
				End:     end,
				ScoreUB: scoreUB,
			})
			totalDocs += end - start
		}
	}

	// Global top-k by score
	sort.Slice(hits, func(i, j int) bool {
		return hits[i].ScoreUB > hits[j].ScoreUB
	})

	// Keep top k_candidates sub-blocks for Stage 3 reranking
	// With default B=32, s=8: we get 200 leaf nodes × 4 sub-blocks = 800 candidates
	// Keep top 200 for Stage 3: 200 × s(=8) = 1,600 documents
	if kCandidates > 0 && len(hits) > kCandidates {
		hits = hits[:kCandidates]
	}

	return hits
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
