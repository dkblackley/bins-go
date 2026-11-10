package pianopir

import (
	"math/rand"
	"testing"
)

func randomDB(rows, words uint64, seed int64) []uint64 {
	rng := rand.New(rand.NewSource(seed))
	db := make([]uint64, rows*words)
	for i := range db {
		db[i] = rng.Uint64()
	}
	return db
}

// The size formula used before building (BatchHintStorageBytes, and the
// pirhint calculator that mirrors it) must equal what the clients allocate.
func TestBatchHintStorageMatchesClients(t *testing.T) {
	defer func(c, q float64) { ChunkSizeFactor, QueryBudgetScale = c, q }(ChunkSizeFactor, QueryBudgetScale)
	for _, knobs := range [][2]float64{{2, 1}, {1, 1}, {0.5, 1}, {2, 0.25}, {1, 0.5}} {
		ChunkSizeFactor, QueryBudgetScale = knobs[0], knobs[1]
		for _, tc := range []struct{ rows, entry, batch uint64 }{
			{1000, 16, 8}, {5000, 24, 32}, {70000, 8, 128}, {123457, 64, 16},
		} {
			db := make([]uint64, tc.rows*tc.entry/8)
			pir := NewSimpleBatchPianoPIR(tc.rows, tc.entry, tc.batch, db, 20)
			got := pir.LocalStorageSize()
			want := BatchHintStorageBytes(tc.rows, tc.entry, tc.batch, 20)
			if got != want {
				t.Fatalf("knobs %v rows %d entry %d batch %d: clients hold %v bytes, formula says %v",
					knobs, tc.rows, tc.entry, tc.batch, got, want)
			}
		}
	}
}

// Smaller chunks and a smaller query budget must still return correct rows,
// including across a forced re-preprocessing.
func TestBatchPIRStorageKnobsCorrect(t *testing.T) {
	defer func(c, q float64) { ChunkSizeFactor, QueryBudgetScale = c, q }(ChunkSizeFactor, QueryBudgetScale)
	const rows, words, batch = 20000, 3, 16
	db := randomDB(rows, words, 7)
	for _, knobs := range [][2]float64{{1, 1}, {0.5, 1}, {2, 0.25}, {1, 0.3}} {
		ChunkSizeFactor, QueryBudgetScale = knobs[0], knobs[1]
		pir := NewSimpleBatchPianoPIR(rows, words*8, batch, db, 20)
		pir.Preprocessing()
		rng := rand.New(rand.NewSource(1))
		for round := 0; round < 40; round++ {
			idx := make([]uint64, batch)
			for i := range idx {
				idx[i] = uint64(rng.Intn(rows))
			}
			resp, err := pir.Query(idx)
			if err != nil {
				t.Fatalf("knobs %v round %d: %v", knobs, round, err)
			}
			for i, row := range idx {
				for w := uint64(0); w < words; w++ {
					if resp[i][w] != db[row*words+w] {
						t.Fatalf("knobs %v round %d: row %d word %d wrong", knobs, round, row, w)
					}
				}
			}
		}
	}
}

func TestBatchPIRDownloadIfCheaper(t *testing.T) {
	defer func(d bool) { DownloadIfCheaper = d }(DownloadIfCheaper)
	DownloadIfCheaper = true

	// 2000 rows over 64 partitions: ~31 rows each, far below the hint size.
	small := randomDB(2000, 2, 3)
	pir := NewSimpleBatchPianoPIR(2000, 16, 128, small, 20)
	if !pir.Downloaded() {
		t.Fatalf("a 2000-row DB in 64 partitions should be downloaded (hints %v B, DB %v B)",
			BatchHintStorageBytes(2000, 16, 128, 20), 2000*16)
	}
	pir.Preprocessing()
	if got := pir.LocalStorageSize(); got != 2000*16 {
		t.Fatalf("downloaded DB should cost exactly its size, got %v", got)
	}
	resp, err := pir.Query([]uint64{0, 1999, 7, 7})
	if err != nil {
		t.Fatal(err)
	}
	for i, row := range []uint64{0, 1999, 7, 7} {
		if resp[i][0] != small[row*2] || resp[i][1] != small[row*2+1] {
			t.Fatalf("row %d wrong", row)
		}
	}
	if m := pir.GetMetrics(); m["TotalByteSent"] != 0 || m["Rounds"] != 0 || m["DownloadedDBs"] != 1 {
		t.Fatalf("a downloaded DB sends nothing online, got %v", m)
	}

	// A large DB in few partitions keeps PIR.
	big := make([]uint64, 400000)
	if NewSimpleBatchPianoPIR(400000, 8, 4, big, 20).Downloaded() {
		t.Fatalf("a 400k-row DB in 2 partitions should keep PIR")
	}
}
