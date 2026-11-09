package pianopir

import "testing"

func TestShufflePermutationMatches(t *testing.T) {
	const n = 5000
	raw := make([][]uint64, n)
	for i := range raw {
		raw[i] = []uint64{uint64(i) + 1}
	}
	p := NewSimpleBatchPianoPIR(n, 1, 8, 4, raw, 20, 1)
	phys := ShufflePermutation(n)
	for physIdx, logical := range phys {
		if p.permutation[logical] != uint64(physIdx) {
			t.Fatalf("logical row %d: batch PIR stores it at %d, ShufflePermutation says %d",
				logical, p.permutation[logical], physIdx)
		}
	}
}
