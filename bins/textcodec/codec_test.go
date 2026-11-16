package textcodec

import (
	"math/rand"
	"sort"
	"testing"
)

func randTerm(rng *rand.Rand) string {
	n := 2 + rng.Intn(9)
	b := make([]byte, n)
	for i := range b {
		b[i] = byte('a' + rng.Intn(26))
	}
	if rng.Intn(50) == 0 {
		b[rng.Intn(n)] = byte(0xc3 + rng.Intn(2)) // non a-z byte -> ESC path
	}
	return string(b)
}

func randDoc(rng *rand.Rand, vocab []string) []string {
	n := rng.Intn(60)
	if rng.Intn(20) == 0 {
		n = 0
	}
	d := make([]string, n)
	for i := range d {
		d[i] = vocab[int(float64(len(vocab))*rng.Float64()*rng.Float64())] // skewed, gives repeats
	}
	return d
}

func tfOf(terms []string) map[string]int {
	m := map[string]int{}
	for _, t := range terms {
		m[t]++
	}
	return m
}

func TestBinRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	vocab := make([]string, 3000)
	for i := range vocab {
		vocab[i] = randTerm(rng)
	}
	docs := make([][]string, 2000)
	for i := range docs {
		docs[i] = randDoc(rng, vocab)
	}
	for _, format := range []string{FormatBOW, FormatSeq} {
		for order := 0; order <= maxOrder; order++ {
			tr := NewTrainer(Params{Format: format, Order: order})
			for _, d := range docs[:500] {
				tr.Add(d)
			}
			m := tr.Build()
			bp := NewBinParams(len(docs), 40)
			for bin := 0; bin < 30; bin++ {
				var codes []DocCode
				want := map[uint32][]string{}
				for _, i := range rng.Perm(len(docs))[:rng.Intn(80)] {
					b, n := m.EncodeDoc(docs[i])
					codes = append(codes, DocCode{uint32(i), b, n})
					want[uint32(i)] = docs[i]
				}
				got := m.DecodeBin(bp, EncodeBin(bp, codes))
				if len(got) != len(want) {
					t.Fatalf("%s/%d: got %d docs want %d", format, order, len(got), len(want))
				}
				for _, g := range got {
					w, ok := want[g.Idx]
					if !ok {
						t.Fatalf("%s/%d: unexpected doc %d", format, order, g.Idx)
					}
					wtf := tfOf(w)
					if g.Len != len(w) || len(g.Terms) != len(wtf) {
						t.Fatalf("%s/%d doc %d: len %d/%d terms %d/%d", format, order, g.Idx, g.Len, len(w), len(g.Terms), len(wtf))
					}
					for _, tt := range g.Terms {
						if wtf[tt.Term] != tt.TF {
							t.Fatalf("%s/%d doc %d: term %q tf %d want %d", format, order, g.Idx, tt.Term, tt.TF, wtf[tt.Term])
						}
					}
				}
			}
		}
	}
}

func TestBitIO(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	var w BitWriter
	var vals []uint64
	var ks []uint
	for i := 0; i < 1000; i++ {
		v, k := uint64(rng.Intn(100000)), uint(rng.Intn(12))
		vals, ks = append(vals, v), append(ks, k)
		w.WriteExpGolomb(v, k)
		var w2 BitWriter
		w2.WriteBits(uint64(rng.Intn(1<<13)), 13)
		w.AppendBitString(w2.Bytes(), 5) // odd-length, unaligned appends
	}
	r := NewBitReader(w.Words())
	for i := range vals {
		if g := r.ReadExpGolomb(ks[i]); g != vals[i] {
			t.Fatalf("%d: got %d want %d", i, g, vals[i])
		}
		r.ReadBits(5)
	}
}

func TestIDFSketch(t *testing.T) {
	c := NewIDFCounter(12)
	for i := 0; i < 1000; i++ {
		d := []string{"common", "common"}
		if i%100 == 0 {
			d = append(d, "rare")
		}
		c.AddDoc(d)
	}
	s := c.Build()
	if df := s.DF("common"); df < 900 || df > 1100 {
		t.Fatalf("df(common) = %v", df)
	}
	if df := s.DF("rare"); df < 9 || df > 11 {
		t.Fatalf("df(rare) = %v", df)
	}
	if s.IDF("rare") <= s.IDF("common") {
		t.Fatal("rare term should have the larger idf")
	}
	docs := [][]TermTF{{{"common", 2}}, {{"common", 1}, {"rare", 1}}}
	sc := []float64{s.BM25([]string{"rare", "common"}, docs[0], 2, 0.9, 0.4), s.BM25([]string{"rare", "common"}, docs[1], 2, 0.9, 0.4)}
	if !sort.Float64sAreSorted(sc) || sc[0] == sc[1] {
		t.Fatalf("doc with the rare term should score higher: %v", sc)
	}
}
