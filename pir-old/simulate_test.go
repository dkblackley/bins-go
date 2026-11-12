package pianopir

import (
	"math/rand"
	"testing"
)

// Equal-size rows: the simulated PIR returns the same answers and exactly the same metrics as the real one.
func TestSimulateMatchesReal(t *testing.T) {
	const n, words, batch = 20000, 6, 32
	rng := rand.New(rand.NewSource(1))
	db := make([]uint64, n*words)
	for i := range db {
		db[i] = rng.Uint64()
	}
	batches := make([][]uint64, 300)
	for i := range batches {
		batches[i] = make([]uint64, 20)
		for j := range batches[i] {
			batches[i][j] = uint64(rng.Intn(n))
		}
		batches[i][3] = batches[i][2] // repeat: local cache
	}
	run := func(simulate bool) (*SimpleBatchPianoPIR, [][][]uint64) {
		Simulate = simulate
		defer func() { Simulate = false }()
		p := NewSimpleBatchPianoPIR(n, words*8, batch, db, 20)
		p.SetThreadNum(4)
		p.Preprocessing()
		var out [][][]uint64
		for _, b := range batches {
			res, err := p.Query(b)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, res)
		}
		return p, out
	}
	real, realOut := run(false)
	sim, simOut := run(true)
	for i := range realOut {
		for k := range realOut[i] {
			for j := range realOut[i][k] {
				if realOut[i][k][j] != simOut[i][k][j] {
					t.Fatalf("batch %d answer %d word %d differs", i, k, j)
				}
			}
		}
	}
	rm, sm := real.GetMetrics(), sim.GetMetrics()
	for _, k := range []string{"TotalByteSent", "TotalWANTime", "TotalLANTime", "Rounds", "TotalPreProc", "ClientStorageMB", "DBSizeInBytesMB", "CommCostPerBatchKB"} {
		if rm[k] != sm[k] {
			t.Errorf("%s: real %v sim %v", k, rm[k], sm[k])
		}
	}
	if rm["TotalPreProc"] < 2 {
		t.Fatalf("test should cover re-preprocessing, got %v", rm["TotalPreProc"])
	}
	t.Logf("bytes %v, WAN %v, preprocessings %v", sm["TotalByteSent"], sm["TotalWANTime"], sm["TotalPreProc"])
}
