package pianopir

import "math/rand"

// ShuffleSeed is the seed NewSimpleBatchPianoPIR shuffles the DB rows with.
const ShuffleSeed = 12345

// ShufflePermutation returns, for every physical position p, the logical row NewSimpleBatchPianoPIR stores there
// (its permInts). The -noram mode (diskstore) writes the rows to disk in this order, so the preprocessing, which
// walks the shuffled rows in order, reads the file sequentially. It only affects speed: any order gives the same
// answers. TestShufflePermutationMatches keeps it in step with batch-pir.go.
func ShufflePermutation(DBSize uint64) []int {
	return rand.New(rand.NewSource(ShuffleSeed)).Perm(int(DBSize))
}
