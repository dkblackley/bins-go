package pianopir

import (
	"math/rand"
	"testing"
)

// Variable-length entries: queries still decode correctly, and the trimmed byte / storage counts are below the
// padded ones (and never above them).
func TestTrimmedAccounting(t *testing.T) {
	const n, maxWords = 4096, 64
	rng := rand.New(rand.NewSource(1))
	db := make([][]uint64, n)
	for i := range db {
		l := 1 + rng.Intn(maxWords/2) // most entries are much shorter than the largest
		if i == 7 {
			l = maxWords
		}
		db[i] = make([]uint64, l)
		for j := range db[i] {
			db[i][j] = rng.Uint64() | 1
		}
	}
	pir := NewSimpleBatchPianoPIR(n, maxWords, maxWords*8, 16, db, 20, 1)
	pir.Preprocessing()

	trim, pad := pir.LocalStorageSizeTrimmed(), pir.LocalStorageSize()
	if trim >= pad {
		t.Fatalf("trimmed storage %v should be below padded %v", trim, pad)
	}
	for q := 0; q < 5; q++ {
		idx := []uint64{uint64(rng.Intn(n)), 7}
		res, err := pir.Query(idx)
		if err != nil {
			t.Fatal(err)
		}
		for k, i := range idx {
			for j, w := range db[i] {
				if res[k][j] != w {
					t.Fatalf("entry %d word %d: got %x want %x", i, j, res[k][j], w)
				}
			}
		}
	}
	var sent, padded uint64
	for _, s := range pir.subPIR {
		sent += s.server.RetrievalCount
		padded += s.server.RetrievalCountPadded
	}
	if sent == 0 || sent >= padded {
		t.Fatalf("trimmed bytes %d should be in (0, padded %d)", sent, padded)
	}
	t.Logf("storage trimmed/padded = %.0f/%.0f KB, sent trimmed/padded = %d/%d B", trim/1024, pad/1024, sent, padded)
}
