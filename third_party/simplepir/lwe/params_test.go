package lwe

//import "fmt"
//import "math/rand"
import "testing"

func TestTooManySamples(t *testing.T) {
	p := NewParams(32, 1000000000)
	if p != nil {
		t.Fail()
	}
}

func TestGood(t *testing.T) {
	p := NewParams(32, 10)
	if p == nil {
		t.Fail()
	}

	if p.P != uint64(991) || p.Logq != 32 || p.M !=
		1<<13 || p.Sigma != float64(6.4) || p.N !=
		uint64(1024) {
		t.Fail()
	}
}

func TestGoodPicked(t *testing.T) {
	p := NewParamsFixedP(32, 10, 900)
	if p == nil {
		t.Fail()
	}

	if p.P != uint64(900) || p.Logq != 32 || p.M !=
		10 || p.Sigma != float64(6.4) || p.N !=
		uint64(1024) {
		t.Fail()
	}
}

func TestDelta(t *testing.T) {
	p := NewParams(32, 10)
	if p.Delta != 4333973 {
		t.Fail()
	}
}

func TestGood64(t *testing.T) {
	p := NewParams(64, 100)
	if p == nil {
		t.Fail()
	}

	if p.P != uint64(574457) || p.Logq != 64 || p.M !=
		1<<13 || p.Sigma != float64(81920.0) || p.N !=
		uint64(2048) {
		t.Fail()
	}
}

/*
func TestGauss64(t *testing.T) {
  r := rand.New(rand.NewSource(99))
  bins := make([]int, 1000)

  for i := 0; i<1024*1024*4; i++ {
    v := GaussSample64(r)
    v = (v / 8000) + 500
    bins[v] += 1
  }

  for i := 0; i<len(bins); i++ {
    fmt.Printf("bin[%v] = %v\n", i-500, bins[i])
  }
}
*/
