package bins

import "testing"

func TestVecBatchAndRounds(t *testing.T) {
	for _, tc := range []struct{ autoB, autoR, b, r, wantB, wantR int }{
		{300, 1, 0, 0, 300, 1},   // nothing set: automatic
		{300, 1, 50, 0, 50, 6},   // batch only: same 300-doc coverage in 6 rounds
		{3000, 2, 0, 4, 1500, 4}, // rounds only: batch spread over them
		{300, 1, 64, 2, 64, 2},   // both: as given
		{100, 6, 1, 0, 2, 300},   // batch floored at 2 (one partition)
	} {
		b, r := vecBatchAndRounds(tc.autoB, tc.autoR, tc.b, tc.r)
		if b != tc.wantB || r != tc.wantR {
			t.Errorf("%+v: got batch %d rounds %d", tc, b, r)
		}
	}
}

// packBin stores doc rows the way BuildIndexDB does: row+1, two per word, 0 = padding
func packBin(rows ...uint32) []uint64 {
	e := make([]uint64, (len(rows)+1)/2)
	for j, r := range rows {
		e[j/2] |= uint64(r+1) << (32 * (j % 2))
	}
	return e
}

func TestCollectDocIdxOrder(t *testing.T) {
	// term A's bin ranks 10, 11, 12; term B's ranks 20, 11 (shared with A), 21, 22
	results := [][]uint64{packBin(10, 11, 12), packBin(20, 11, 21, 22)}
	for _, tc := range []struct {
		rr   bool
		want []uint64
	}{
		{false, []uint64{10, 11, 12, 20, 21, 22}}, // term by term
		{true, []uint64{10, 20, 11, 12, 21, 22}},  // rank 1 of each, then rank 2 (11 already taken), ...
	} {
		got := collectDocIdx(results, 4, tc.rr)
		if len(got) != len(tc.want) {
			t.Fatalf("rr=%v: got %v, want %v", tc.rr, got, tc.want)
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Fatalf("rr=%v: got %v, want %v", tc.rr, got, tc.want)
			}
		}
	}
}
