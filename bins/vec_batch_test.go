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
