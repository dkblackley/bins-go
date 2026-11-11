package bins

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/dkblackley/bins-go/globals"
)

// dummyBins makes a corpus where only part of the docs land in any bin: bins
// draw from the first `pool` docs with a skew, like BM25 top-k lists do.
func dummyBins(rng *rand.Rand, numDocs, pool, numBins, docsPerBin, dim int) ([][]string, map[int]string, [][]float32) {
	docMap := make(map[int]string, numDocs)
	vectors := make([][]float32, numDocs)
	for i := range numDocs {
		docMap[i] = fmt.Sprintf("doc%d", i)
		vectors[i] = make([]float32, dim)
		for k := range vectors[i] {
			vectors[i][k] = rng.Float32()*2 - 1
		}
	}
	bins := make([][]string, numBins)
	for b := range bins {
		if rng.Intn(10) == 0 {
			continue // some empty bins
		}
		seen := map[int]bool{}
		for len(bins[b]) < 1+rng.Intn(docsPerBin) {
			d := int(float64(pool) * math.Pow(rng.Float64(), 2)) // skewed towards low ids
			if !seen[d] {
				seen[d] = true
				bins[b] = append(bins[b], docMap[d])
			}
		}
	}
	return bins, docMap, vectors
}

func copyBins(db [][]string) [][]string {
	out := make([][]string, len(db))
	for i := range db {
		out[i] = append([]string(nil), db[i]...)
	}
	return out
}

// Pruning the embedding DB must not change a single search: same candidate
// docs, in the same order, and the pruned DB must return each doc's own
// embedding at its new row.
func TestVecPruneSameResults(t *testing.T) {
	const numDocs, pool, numBins, docsPerBin, dim = 3000, 1200, 500, 12, 8
	rng := rand.New(rand.NewSource(1))
	db, docMap, vectors := dummyBins(rng, numDocs, pool, numBins, docsPerBin, dim)

	build := func(prune bool) (VecBins, *globals.Args) {
		config := &globals.Args{Dimensions: dim, VecPrune: prune, Metadata: map[string]string{}}
		v := buildVecBins(config, copyBins(db), docMap, vectors)
		v.Preprocess()
		return v, config
	}
	full, fullConfig := build(false)
	pruned, prunedConfig := build(true)

	distinct := map[string]bool{}
	for _, bin := range db {
		for _, id := range bin {
			distinct[id] = true
		}
	}
	if got := pruned.vecPIR.Config().DBSize; got != uint64(len(distinct)) {
		t.Fatalf("pruned embedding DB has %d rows; want the %d binned docs", got, len(distinct))
	}
	if got := full.vecPIR.Config().DBSize; got != numDocs {
		t.Fatalf("unpruned embedding DB has %d rows; want %d", got, numDocs)
	}
	t.Logf("embedding DB rows %d -> %d; client storage %s MB -> %s MB", numDocs, len(distinct),
		full.vecPIR.PrintInfo()["ClientStorageMB"], pruned.vecPIR.PrintInfo()["ClientStorageMB"])
	if prunedConfig.Metadata["VecDocsStored"] != fmt.Sprint(len(distinct)) || fullConfig.Metadata["VecDocsStored"] != fmt.Sprint(numDocs) {
		t.Fatalf("metadata: %v / %v", prunedConfig.Metadata, fullConfig.Metadata)
	}

	total := 0
	for q := range 200 {
		indices := make([]uint64, 1+rng.Intn(6))
		for i := range indices {
			indices[i] = uint64(rng.Intn(numBins))
		}
		a, errA := full.searchBins("q", indices)
		b, errB := pruned.searchBins("q", indices)
		if errA != nil || errB != nil {
			t.Fatalf("query %d: %v / %v", q, errA, errB)
		}
		gotFull, gotPruned := a.(DBentry).entry, b.(DBentry).entry
		if !reflect.DeepEqual(gotFull, gotPruned) {
			t.Fatalf("query %d bins %v: unpruned %v, pruned %v", q, indices, gotFull, gotPruned)
		}
		total += len(gotPruned)
		// the pruned DB really serves each retrieved doc's embedding
		rows := make([]uint64, len(gotPruned))
		for i, idx := range gotPruned {
			rows[i] = pruned.rowOfCorpus(idx)
		}
		if len(rows) == 0 {
			continue
		}
		resp, err := pruned.vecPIR.Query(rows)
		if err != nil {
			t.Fatal(err)
		}
		for i, idx := range gotPruned {
			if len(resp[i]) <= 1 {
				continue // dropped lookup, same as the search treats it
			}
			v := vectors[idx]
			for k := 0; k+1 < dim; k += 2 {
				want := uint64(math.Float32bits(v[k])) | uint64(math.Float32bits(v[k+1]))<<32
				if resp[i][k/2] != want {
					t.Fatalf("doc %d row %d word %d: got %#x want %#x", idx, rows[i], k/2, resp[i][k/2], want)
				}
			}
		}
	}
	if total == 0 {
		t.Fatal("no query retrieved anything")
	}
	t.Logf("%d docs retrieved over 200 queries, identical", total)
}

func TestBinnedDocs(t *testing.T) {
	flipped := map[string]int{"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
	vecDocs, rowOf := binnedDocs([][]string{{"d", "b"}, nil, {"b", "e"}}, flipped, 5)
	if !reflect.DeepEqual(vecDocs, []uint32{1, 3, 4}) || !reflect.DeepEqual(rowOf, []int32{-1, 0, -1, 1, 2}) {
		t.Fatalf("vecDocs %v rowOf %v", vecDocs, rowOf)
	}
}
