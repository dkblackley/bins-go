package pir

import "math"

// Returns the i-th elem in the representation of m in base p.
func Base_p(p, m, i uint64) uint64 {
	for j := uint64(0); j < i; j++ {
		m = m / p
	}
	return (m % p)
}

// Returns the element whose base-p decomposition is given by the values in vals
func Reconstruct_from_base_p(p uint64, vals []uint64) uint64 {
	res := uint64(0)
	coeff := uint64(1)
	for _, v := range vals {
		res += coeff * v
		coeff *= p
	}
	return res
}

// Returns how many entries in Z_p are needed to represent an element in Z_q
func Compute_num_entries_base_p(p, log_q uint64) uint64 {
	log_p := math.Log2(float64(p))
	return uint64(math.Ceil(float64(log_q) / log_p))
}

func PrevPowerOfTwo(v uint64) uint64 {
	if v == 0 {
		return 0
	}

	digits := 0
	for ; v > 0; v = (v >> 1) {
		digits += 1
	}

	return uint64(1 << (digits - 1))
}
