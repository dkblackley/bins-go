package bm25

import (
	"math"
)

// IDF calculates the Inverse Document Frequency
func IDF(df int, N int) float64 {
	return math.Log(1 + (float64(N-df)+0.5)/(float64(df)+0.5))
}

// BM25Term calculates the BM25 score for a single term
func BM25Term(tf int, dl int, avgDL float64, p *BM25Params) float64 {
	num := (p.K1 + 1.0) * float64(tf)
	denom := float64(tf) + p.K1*(1.0-p.B+p.B*(float64(dl)/math.Max(1.0, avgDL)))
	return num / math.Max(1e-9, denom)
}

// BM25TermUBWithMinDL calculates an upper bound for BM25 score using minimum document length
func BM25TermUBWithMinDL(tf int, minDL int, avgDL float64, p *BM25Params) float64 {
	return BM25Term(tf, minDL, avgDL, p)
}

// NodeToDocRange calculates document range for a given node
func (l *LayoutWithStats) NodeToDocRange(node int) (int, int) {
	var level float64
	if l.R > 1 {
		level = math.Floor(math.Log(math.Max(1, float64(node-1))) / math.Log(float64(l.R)))
	} else {
		level = 0
	}
	h := l.Height()
	leavesPerNode := math.Pow(float64(l.R), float64(h)-level)
	var firstNodeAtLevel int
	if l.R > 1 {
		firstNodeAtLevel = int(math.Pow(float64(l.R), level))
	} else {
		firstNodeAtLevel = 1
	}
	idxInLevel := node - firstNodeAtLevel
	leafStart := float64(idxInLevel) * leavesPerNode
	docStart := int(leafStart) * l.B
	docEnd := int(math.Min(float64(l.N), float64(docStart)+leavesPerNode*float64(l.B)))
	return docStart, docEnd
}

// Height calculates the height of the tree
func (l *LayoutWithStats) Height() int {
	if l.R <= 1 {
		return 0
	}
	numLeaves := float64(l.NumLeaves())
	return int(math.Ceil(math.Log(numLeaves) / math.Log(float64(l.R))))
}

// NumLeaves returns the number of leaf nodes
func (l *LayoutWithStats) NumLeaves() int {
	return int(math.Ceil(float64(l.N) / float64(l.B)))
}

// NodeChildren returns the children of given nodes
func (l *LayoutWithStats) NodeChildren(nodes []int) []int {
	children := make([]int, 0, len(nodes)*l.R)
	for _, node := range nodes {
		for j := 0; j < l.R; j++ {
			children = append(children, node*l.R+j)
		}
	}
	return children
}
