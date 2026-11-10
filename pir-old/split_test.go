package pianopir

import (
	"math/rand"
	"testing"
)

// The scifact Stage3 case: 648 blocks, batch 32 (16 partitions of 41 rows), a request for 256 distinct blocks.
// Query refuses it in one go; split, every sub-batch is accepted and the rows come back right, in order.
func TestSplitByPartitionLimit(t *testing.T) {
	defer func(d bool) { DownloadIfCheaper = d }(DownloadIfCheaper)
	DownloadIfCheaper = false
	const rows, words, batch = 648, 4, 32
	db := randomDB(rows, words, 3)
	p := NewSimpleBatchPianoPIR(rows, words*8, batch, db, 20)
	p.Preprocessing()
	idx := make([]uint64, 0, 256)
	for _, r := range rand.New(rand.NewSource(5)).Perm(rows)[:256] {
		idx = append(idx, uint64(r))
	}
	if _, err := p.Query(idx); err == nil {
		t.Fatal("expected the unsplit request to be refused")
	}
	subs := p.SplitByPartitionLimit(idx)
	if len(subs) < 2 {
		t.Fatalf("got %d sub-batches", len(subs))
	}
	var got [][]uint64
	n := 0
	for _, sb := range subs {
		n += len(sb)
		r, err := p.Query(sb)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, r...)
	}
	if n != len(idx) || len(got) != len(idx) {
		t.Fatalf("lost lookups: %d / %d of %d", n, len(got), len(idx))
	}
	for i, row := range idx {
		for w := uint64(0); w < words; w++ {
			if got[i][w] != db[row*words+w] {
				t.Fatalf("lookup %d (row %d) word %d wrong", i, row, w)
			}
		}
	}
	small := idx[:8]
	if s := p.SplitByPartitionLimit(small); len(s) != 1 || len(s[0]) != 8 {
		t.Fatalf("a request that fits was split: %v", s)
	}
}

// A DB view up to one entry short (the tree's mmap'd Stage3 embeddings: a partial last block) reads as zero-padded.
func TestShortRawDBIsZeroPadded(t *testing.T) {
	defer func(d bool) { DownloadIfCheaper = d }(DownloadIfCheaper)
	for _, download := range []bool{false, true} {
		DownloadIfCheaper = download
		const rows, words, batch = 1000, 6, 32
		full := randomDB(rows, words, 9)
		for k := words*(rows-1) + 1; k < words*rows; k++ {
			full[k] = 0 // last entry: only its first word is real
		}
		short := full[:words*(rows-1)+1]
		p := NewSimpleBatchPianoPIR(rows, words*8, batch, short, 20)
		p.Preprocessing()
		idx := []uint64{0, 1, rows / 2, rows - 2, rows - 1}
		got, err := p.Query(idx)
		if err != nil {
			t.Fatal(err)
		}
		for i, row := range idx {
			for w := 0; w < words; w++ {
				if got[i][w] != full[int(row)*words+w] {
					t.Fatalf("download=%v row %d word %d: got %d want %d", download, row, w, got[i][w], full[int(row)*words+w])
				}
			}
		}
	}
}
