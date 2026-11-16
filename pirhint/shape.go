package pirhint

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// Corpus is a synthetic stand-in for a dataset: D documents of L term draws
// each from a Zipf(V, a) vocabulary. Only the sizes of the PIR DBs depend on
// it, so it does not need real text.
type Corpus struct {
	Docs     uint64
	Tokens   float64 // term draws per document
	Vocab    uint64
	ZipfExp  float64
	Clusters float64 // in (0,1]: fraction of the random-assignment distinct count kept, 1 = no clustering
}

var (
	MSMarco = Corpus{Docs: 8_841_823, Tokens: 35, Vocab: 2_600_000, ZipfExp: 1.0, Clusters: 1}
	SciFact = Corpus{Docs: 5_183, Tokens: 150, Vocab: 30_000, ZipfExp: 1.0, Clusters: 1}
)

// distinctTerms is the expected number of distinct terms among the m*Tokens
// draws of m documents: sum over terms of 1-(1-p_v)^(m*Tokens).
func (c Corpus) distinctTerms(m float64) float64 {
	h := 0.0
	for v := uint64(1); v <= c.Vocab; v++ {
		h += math.Pow(float64(v), -c.ZipfExp)
	}
	draws := m * c.Tokens
	sum := 0.0
	for v := uint64(1); v <= c.Vocab; v++ {
		p := math.Pow(float64(v), -c.ZipfExp) / h
		sum += -math.Expm1(draws * math.Log1p(-p))
	}
	return sum * c.Clusters
}

// TreeShape holds the item (non-zero row) counts of the tree method's DBs.
type TreeShape struct {
	Source      string
	Docs        uint64
	Leaves      uint64
	Stage1Items []uint64 // per tree level, root first
	Stage2Items uint64
	// average distinct terms of one leaf block, for the one-DB-per-leaf layout
	ItemsPerLeaf float64
}

// TreeHeight follows build_small_test_artifacts.py: ceil(log_r(leaves)).
func TreeHeight(leaves uint64, r int) int {
	if leaves <= 1 {
		return 0
	}
	h := 0
	for span := uint64(1); span < leaves; span *= uint64(r) {
		h++
	}
	return h
}

// SyntheticTreeShape estimates the DB sizes for a random (unclustered) block
// assignment, an upper bound on what clustering produces.
func SyntheticTreeShape(c Corpus, B, r int) TreeShape {
	leaves := (c.Docs + uint64(B) - 1) / uint64(B)
	height := TreeHeight(leaves, r)
	shape := TreeShape{Source: "synthetic", Docs: c.Docs, Leaves: leaves}
	for level := 0; level < height; level++ {
		span := uint64(math.Pow(float64(r), float64(height-level))) // leaves under one parent
		parents := (leaves + span - 1) / span
		docs := math.Min(float64(span*uint64(B)), float64(c.Docs))
		shape.Stage1Items = append(shape.Stage1Items, uint64(float64(parents)*c.distinctTerms(docs)))
	}
	shape.ItemsPerLeaf = c.distinctTerms(float64(B))
	shape.Stage2Items = uint64(float64(leaves) * shape.ItemsPerLeaf)
	return shape
}

// stage1ParentLevel mirrors bm25.stage1ParentLevel.
func stage1ParentLevel(node uint64, r int) (int, bool) {
	start := uint64(1)
	for level := 0; level < 32; level++ {
		if node >= start && node < start*2 {
			return level, true
		}
		if start > uint64(math.MaxUint32)/uint64(r) {
			break
		}
		start *= uint64(r)
	}
	return 0, false
}

func readIDMap(path string, visit func(term, node uint32)) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	rec := make([]byte, 16)
	for {
		if _, err := io.ReadFull(f, rec); errors.Is(err, io.EOF) {
			return nil
		} else if err != nil {
			return err
		}
		visit(binary.LittleEndian.Uint32(rec[0:4]), binary.LittleEndian.Uint32(rec[4:8]))
	}
}

// RealTreeShape counts the rows of a built artifact directory's id maps
// (stage1_idmap.bin, stage2/idmap.bin), the exact item counts the run uses.
func RealTreeShape(stage1IDMap, stage2IDMap string, docs uint64, B, r int) (TreeShape, error) {
	shape := TreeShape{Source: "artifacts", Docs: docs, Leaves: (docs + uint64(B) - 1) / uint64(B)}
	var bad error
	err := readIDMap(stage1IDMap, func(_, node uint32) {
		level, ok := stage1ParentLevel(uint64(node), r)
		if !ok {
			bad = fmt.Errorf("node %d is outside the r=%d level ranges", node, r)
			return
		}
		for len(shape.Stage1Items) <= level {
			shape.Stage1Items = append(shape.Stage1Items, 0)
		}
		shape.Stage1Items[level]++
	})
	if err != nil {
		return shape, err
	}
	if bad != nil {
		return shape, bad
	}
	if stage2IDMap != "" {
		if err := readIDMap(stage2IDMap, func(_, _ uint32) { shape.Stage2Items++ }); err != nil {
			return shape, err
		}
	}
	shape.ItemsPerLeaf = float64(shape.Stage2Items) / float64(shape.Leaves)
	return shape, nil
}
