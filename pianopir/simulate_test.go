package pianopir

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
)

// skewedDB looks like a bins DB: most rows short, a few long, some empty.
func skewedDB(n, maxWords int, seed int64) [][]uint64 {
	rng := rand.New(rand.NewSource(seed))
	db := make([][]uint64, n)
	for i := range db {
		var l int
		switch r := rng.Intn(100); {
		case r < 10:
			l = 0
		case r < 80:
			l = 1 + rng.Intn(maxWords/8)
		case r < 98:
			l = 1 + rng.Intn(maxWords/2)
		default:
			l = maxWords/2 + rng.Intn(maxWords/2+1)
		}
		db[i] = make([]uint64, l)
		for j := range db[i] {
			db[i][j] = rng.Uint64() | 1
		}
	}
	return db
}

func runBatches(t *testing.T, simulate bool, db [][]uint64, maxWords, batch uint64, batches [][]uint64) (*SimpleBatchPianoPIR, [][][]uint64) {
	t.Helper()
	Simulate = simulate
	defer func() { Simulate = false }()
	pir := NewSimpleBatchPianoPIR(uint64(len(db)), maxWords, maxWords*8, batch, db, 20, 1)
	pir.Preprocessing()
	var out [][][]uint64
	for _, b := range batches {
		if pir.FinishedBatchNum+pir.Config().BatchNumNeeded >= pir.SupportBatchNum {
			pir.Preprocessing()
		}
		res, err := pir.Query(b)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, res)
	}
	return pir, out
}

// The simulated PIR gives the same answers, rounds and refreshes as the real one, the same
// padded byte counts, and trimmed bytes / client storage within a fraction of a percent.
func TestSimulateMatchesReal(t *testing.T) {
	const n, maxWords, batch = 30000, 96, 16
	db := skewedDB(n, maxWords, 3)
	rng := rand.New(rand.NewSource(4))
	batches := make([][]uint64, 400)
	for i := range batches {
		b := make([]uint64, batch)
		for j := range b {
			b[j] = uint64(rng.Intn(n))
		}
		if i%7 == 0 {
			b[1] = b[0] // a repeat inside a batch hits the local cache
		}
		batches[i] = b
	}

	real, realOut := runBatches(t, false, db, maxWords, batch, batches)
	sim, simOut := runBatches(t, true, db, maxWords, batch, batches)

	for i := range realOut {
		for k := range realOut[i] {
			r, s := realOut[i][k], simOut[i][k]
			if len(r) != len(s) {
				t.Fatalf("batch %d answer %d: length %d vs %d", i, k, len(r), len(s))
			}
			for j := range r {
				if r[j] != s[j] {
					t.Fatalf("batch %d answer %d word %d: real %x sim %x", i, k, j, r[j], s[j])
				}
			}
		}
	}
	if real.roundsMade != sim.roundsMade || real.TotalPreProcs != sim.TotalPreProcs {
		t.Fatalf("rounds %d/%d, preprocessings %d/%d", real.roundsMade, sim.roundsMade, real.TotalPreProcs, sim.TotalPreProcs)
	}
	if real.TotalPreProcs < 2 {
		t.Fatalf("test should cover re-preprocessing, got %d", real.TotalPreProcs)
	}

	rm, sm := real.PrintInfo(), sim.PrintInfo()
	for _, k := range []string{"TotalByteSentPadded", "ClientStoragePaddedMB", "CommCostPerBatchKB", "DBSizeInBytesMB", "TotalLANTimePadded", "TotalWANTimePadded"} {
		if rm[k] != sm[k] {
			t.Errorf("%s: real %s sim %s", k, rm[k], sm[k])
		}
	}
	for _, k := range []string{"TotalByteSent", "TotalWANTime", "ClientStorageMB"} {
		r, _ := strconv.ParseFloat(rm[k], 64)
		s, _ := strconv.ParseFloat(sm[k], 64)
		rel := math.Abs(r-s) / r
		t.Logf("%s: real %v sim %v (%.3f%%)", k, r, s, 100*rel)
		if rel > 0.01 {
			t.Errorf("%s differs by %.2f%%", k, 100*rel)
		}
	}
	// storage at full precision, ClientStorageMB is rounded down to whole MB
	r, s := real.LocalStorageSizeTrimmed(), sim.LocalStorageSizeTrimmed()
	t.Logf("trimmed storage bytes: real %.0f sim %.0f (%.3f%%)", r, s, 100*math.Abs(r-s)/r)
	if math.Abs(r-s)/r > 0.01 {
		t.Errorf("trimmed storage differs by %.2f%%", 100*math.Abs(r-s)/r)
	}
}

// For equal-size rows the bytes are exact. The storage is exact up to the random replacement picks in a
// partly padded last chunk (a pick that lands on padding stores nothing).
func TestSimulateEqualRowsExact(t *testing.T) {
	const n, words, batch = 20000, 24, 16
	rng := rand.New(rand.NewSource(5))
	db := make([][]uint64, n)
	for i := range db {
		db[i] = make([]uint64, words)
		for j := range db[i] {
			db[i][j] = rng.Uint64() | 1
		}
	}
	batches := make([][]uint64, 50)
	for i := range batches {
		batches[i] = make([]uint64, batch)
		for j := range batches[i] {
			batches[i][j] = uint64(rng.Intn(n))
		}
	}
	real, _ := runBatches(t, false, db, words, batch, batches)
	sim, _ := runBatches(t, true, db, words, batch, batches)
	rm, sm := real.PrintInfo(), sim.PrintInfo()
	for _, k := range []string{"TotalByteSent", "TotalWANTime", "TotalLANTime", "TotalByteSentPadded"} {
		if rm[k] != sm[k] {
			t.Errorf("%s: real %s sim %s", k, rm[k], sm[k])
		}
	}
	if r, s := real.LocalStorageSizeTrimmed(), sim.LocalStorageSizeTrimmed(); math.Abs(r-s)/r > 0.001 {
		t.Errorf("trimmed storage: real %v sim %v", r, s)
	}
}
