package bm25

import (
	"fmt"
	"sort"
)

// Round1BeamPrecomputed performs beam search using precomputed stage-1 database with batch PIR queries
func Round1BeamPrecomputed(layout *LayoutWithStats, queryTerms []string, L int, db *Stage1PIRDB) []int {
	frontier := []int{1}
	H := layout.Height()
	r := db.R
	for level := 0; level < H; level++ {
		// Each row contains all r child scores for one (term,parent) key. The
		// level-specific database performs both Cuckoo probes for every padded
		// query term and verifies the returned key before accumulation.
		scoresByParent := db.SumChildScoresAtLevel(level, queryTerms, frontier)
		childScores := make(map[int]int)
		for _, parentNodeID := range frontier {
			for child, score := range scoresByParent[parentNodeID] {
				childScores[parentNodeID*r+child] = score
			}
		}
		type scoreNode struct {
			nodeID int
			score  int
		}
		sortedChildren := make([]scoreNode, 0, len(childScores))
		for nodeID, score := range childScores {
			sortedChildren = append(sortedChildren, scoreNode{nodeID, score})
		}
		sort.Slice(sortedChildren, func(i, j int) bool {
			if sortedChildren[i].score == sortedChildren[j].score {
				return sortedChildren[i].nodeID < sortedChildren[j].nodeID
			}
			return sortedChildren[i].score > sortedChildren[j].score
		})
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
