package pianopir

import (
	"math/rand"
	"reflect"
	"testing"
)

func randDB(n int, maxWords int, rng *rand.Rand) [][]uint64 {
	db := make([][]uint64, n)
	for i := range db {
		w := rng.Intn(maxWords + 1) // variable length, some empty, like the bins
		if w == 0 {
			continue
		}
		db[i] = make([]uint64, w)
		for k := range db[i] {
			db[i][k] = rng.Uint64()
		}
	}
	return db
}

// preprocessBlocked must build exactly the parities UpdatePreprocessing builds.
func TestPreprocessBlockedMatches(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	db := randDB(3000, 12, rng)
	cfg := partitionConfig(3000, 12, 96, 20)
	c := NewPianoPIRClient(&cfg)
	padded := c.Preprocessing(db)

	wantP := DeepCopy2DUint64(c.primaryParity)
	wantB := make([][][]uint64, len(c.backupParity))
	for i := range c.backupParity {
		wantB[i] = DeepCopy2DUint64(c.backupParity[i])
		for _, r := range c.backupParity[i] {
			for k := range r {
				r[k] = 0xffff // garbage: preprocessBlocked must clear rows itself
			}
		}
	}
	for _, r := range c.primaryParity {
		for k := range r {
			r[k] = 0xffff
		}
	}
	defer func(b uint64) { HintBlockBytes = b }(HintBlockBytes)
	HintBlockBytes = 7 * 12 * 8 // 7-row blocks, so many blocks and a short last one
	c.preprocessBlocked(padded)
	if !reflect.DeepEqual(wantP, c.primaryParity) {
		t.Fatal("primary parities differ")
	}
	if !reflect.DeepEqual(wantB, c.backupParity) {
		t.Fatal("backup parities differ")
	}
}

// End to end through the batch PIR with the hint tables in a HintStore: every answer is the DB row.
func TestBatchPIRWithHintStore(t *testing.T) {
	defer func(h func(string, uint64, uint64) [][]uint64, m, b uint64) {
		HintStore, HintDiskMinBytes, HintBlockBytes = h, m, b
	}(HintStore, HintDiskMinBytes, HintBlockBytes)
	allocs := 0
	HintStore = func(_ string, rows, width uint64) [][]uint64 {
		allocs++
		flat := make([]uint64, rows*width)
		t := make([][]uint64, rows)
		for i := range t {
			t[i] = flat[uint64(i)*width : uint64(i+1)*width]
		}
		return t
	}
	HintDiskMinBytes, HintBlockBytes = 0, 4096

	rng := rand.New(rand.NewSource(2))
	const n, w = 20000, 10
	db := randDB(n, w, rng)
	p := NewSimpleBatchPianoPIR(n, w, w*8, 8, db, 20, 1)
	p.Preprocessing()
	if allocs == 0 {
		t.Fatal("HintStore was never used")
	}
	for round := 0; round < 1500; round++ { // enough to run out of hints and re-preprocess
		idx := make([]uint64, 4)
		for k := range idx {
			idx[k] = uint64(rng.Intn(n))
		}
		got, err := p.Query(idx)
		if err != nil {
			t.Fatal(err)
		}
		for k, i := range idx {
			want := make([]uint64, w)
			copy(want, db[i])
			if len(got[k]) == 1 && len(want) != 1 { // dropped lookup (two in one partition), not an error
				continue
			}
			if !reflect.DeepEqual(got[k], want) {
				t.Fatalf("round %d: row %d: got %v want %v", round, i, got[k], want)
			}
		}
	}
}
