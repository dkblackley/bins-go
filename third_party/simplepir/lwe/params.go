package lwe

import (
  "math"
  "math/big"
  "fmt"
)

// NOTE: These parameters were chosen to support ternary secrets

// For 32-bit ciphertext modulus
const secretDimension32 = uint64(1408)
const lweErrorStdDev32 = float64(6.4)

// For 64-bit ciphertext modulus
const secretDimension64 = uint64(2048)
const lweErrorStdDev64 = float64(81920.0)

/* Maps #samples ==> plaintext modulus */
var plaintextModulus32 = map[uint64]uint64{
	1 << 13: 991,
	1 << 14: 833,
	1 << 15: 701,
	1 << 16: 589,
	1 << 17: 495,
	1 << 18: 416,
	1 << 19: 350,
	1 << 20: 294,
}

/* Maps #samples ==> plaintext modulus */
var plaintextModulus64 = map[uint64]uint64{
	1 << 13: 574457,
	1 << 14: 483058,
	1 << 15: 406202,
	1 << 16: 341574,
	1 << 17: 287228,
	1 << 18: 241529,
	1 << 19: 203101,
	1 << 20: 170787,
	1 << 21: 143614,
	//1 << 22: 120764,
	//1 << 23: 101550,
	//1 << 24: 85393,
	//1 << 25: 71807,
	//1 << 26: 60382,
	//1 << 27: 50775,
}

type Params struct {
	N     uint64  // LWE secret dimension
	Sigma float64 // LWE error distribution stddev
	M     uint64  // LWE samples supported

	Logq uint64 // (logarithm of) ciphertext modulus
	P    uint64 // plaintext modulus

	Delta uint64 // Plaintext multiplier
}


func (p *Params) Round(x uint64) uint64 {
	v := (x + p.Delta/2) / p.Delta
	return v % p.P
}

func (p *Params) PrintParams() {
	fmt.Printf("Working with: n=%d; m=%d; logq=%d; p=%d; sigma=%f\n",
		p.N, p.M, p.Logq, p.P, p.Sigma)
}

// Output LWE parameters for Regev encryption where
// each ciphertext can support up to 'nSamples'
// homomorphic additions.
func NewParams(logq uint64, nSamples uint64) *Params {
	max := uint64(math.MaxUint64)
	m := max
	pmod := uint64(0)

	options := plaintextModulus32
	if logq == 64 {
		options = plaintextModulus64
	}
	for mNew, pNew := range options {
		if mNew < m && nSamples <= mNew {
			m = mNew
			pmod = pNew
		}
	}

	// No good parameters found
	if m == max {
		return nil
	}

	return newParamsFixedP(logq, m, pmod)
}

func NewParamsFixedP(logq uint64, nSamples uint64, pMod uint64) *Params {
	if CheckParams(logq, nSamples, pMod) {
		return newParamsFixedP(logq, nSamples, pMod)
	}

	return nil
}

func CheckParams(logq uint64, nSamples uint64, pMod uint64) bool {
	options := plaintextModulus32
	if logq == 64 {
		options = plaintextModulus64
	}

	for mNew, pNew := range options {
		if nSamples <= mNew && pMod <= pNew {
			return true
		}
	}

	return false
}

func newParamsFixedP(logq uint64, nSamples uint64, pMod uint64) *Params {
	p := &Params{
		Logq: logq,
		M:    nSamples,
		P:    pMod,
	}

	b := big.NewInt(int64(1))
	pInt := big.NewInt(int64(pMod))
	b.Lsh(b, uint(logq))
	b.Div(b, pInt)
	p.Delta = uint64(b.Int64())

	if logq == 32 {
		p.N = secretDimension32
		p.Sigma = lweErrorStdDev32
	} else if logq == 64 {
		p.N = secretDimension64
		p.Sigma = lweErrorStdDev64
	} else {
		panic("Not yet implemented")
	}

	return p
}
