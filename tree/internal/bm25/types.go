package bm25

// BM25Params holds the parameters for BM25 scoring
type BM25Params struct {
	K1 float64
	B  float64
}

// HitSubBlock represents a block of documents with their upper bound score
type HitSubBlock struct {
	Start   int
	End     int
	ScoreUB float64
}

// QueryData holds preprocessed query information
type QueryData struct {
	Terms  []string
	DocIDs map[string][]int
	TFs    map[string][]int
	IDFs   map[string]float64
}

// LayoutWithStats represents the document layout with precomputed statistics
type LayoutWithStats struct {
	N        int
	B        int
	R        int
	MinDLMap map[int]int
}
