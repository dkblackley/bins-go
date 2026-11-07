package textcodec

import (
	"fmt"
	"sort"
)

// Symbols of the character model. Terms come out of the bins analyzer (lowercase, stopwords removed, Porter-stemmed),
// so they are almost always a-z; anything else is sent as ESC + the raw byte.
const (
	nLetters  = 26
	symESC    = 26
	symEnd1   = 27 // end of term, tf=1 (seq: end of token)
	symEnd2   = 28 // end of term, tf=2
	symEnd3   = 29 // end of term, tf=3
	symEndN   = 30 // end of term, tf>=4 (tf-4 follows, Exp-Golomb)
	nCharSyms = 31

	nCtxChars = 28 // context alphabet: 0 = start of term, 1..26 = a-z, 27 = anything else
	maxPrefix = 15 // bow: shared prefix with the previous (sorted) term, capped
	nPrefix   = maxPrefix + 1
	maxOrder  = 3
)

const (
	FormatBOW = "bow" // sorted bag of stems with tf (all BM25 needs): front-coded, order-k arithmetic coded
	FormatSeq = "seq" // the stem sequence in document order, order-k arithmetic coded
)

// Params are public: they, the model's frequency counts and the IDF sketch are all the client needs to decode.
type Params struct {
	Format string
	Order  int  // characters of context for the character model (0-3)
	CountK uint // Exp-Golomb order for the per-document term count
}

func (p Params) Validate() error {
	if p.Format != FormatBOW && p.Format != FormatSeq {
		return fmt.Errorf("text format %q: want %q or %q", p.Format, FormatBOW, FormatSeq)
	}
	if p.Order < 0 || p.Order > maxOrder {
		return fmt.Errorf("text order %d: want 0..%d", p.Order, maxOrder)
	}
	return nil
}

func numContexts(order int) int {
	n := 1
	for i := 0; i < order; i++ {
		n *= nCtxChars
	}
	return n
}

func ctxChar(b byte) int {
	if b >= 'a' && b <= 'z' {
		return int(b-'a') + 1
	}
	return nCtxChars - 1
}

// context of the next character of term, given its first pos bytes.
func contextOf(term string, pos, order int) int {
	c := 0
	for i := 1; i <= order; i++ {
		c *= nCtxChars
		if pos-i >= 0 {
			c += ctxChar(term[pos-i])
		}
	}
	return c
}

// Model is a static (semi-)adaptive-free model: frequency tables trained once on a sample of the corpus, published
// with the DB, and used for every document. Decoding is arithmetic over these counts.
type Model struct {
	Params
	charCum   [][]uint32 // [context][symbol+1] cumulative frequencies
	prefixCum []uint32
}

// Trainer counts symbols over a sample of documents.
type Trainer struct {
	p          Params
	charCount  [][]uint64
	prefCount  []uint64
	termCounts []uint64
}

func NewTrainer(p Params) *Trainer {
	t := &Trainer{p: p, charCount: make([][]uint64, numContexts(p.Order)), prefCount: make([]uint64, nPrefix)}
	for i := range t.charCount {
		t.charCount[i] = make([]uint64, nCharSyms)
	}
	return t
}

// Add counts one document's terms (analyzer output, in document order).
func (t *Trainer) Add(terms []string) {
	items := prepare(t.p.Format, terms)
	t.termCounts = append(t.termCounts, uint64(len(items)))
	prev := ""
	for i, it := range items {
		pre := 0
		if t.p.Format == FormatBOW && i > 0 {
			pre = sharedPrefix(prev, it.term)
			t.prefCount[pre]++
		}
		for pos := pre; pos < len(it.term); pos++ {
			ctx := contextOf(it.term, pos, t.p.Order)
			b := it.term[pos]
			if b >= 'a' && b <= 'z' {
				t.charCount[ctx][b-'a']++
			} else {
				t.charCount[ctx][symESC]++
			}
		}
		t.charCount[contextOf(it.term, len(it.term), t.p.Order)][endSym(it.tf)]++
		prev = it.term
	}
}

// Build turns the counts into a Model. CountK is set from the median term count.
func (t *Trainer) Build() *Model {
	m := &Model{Params: t.p, charCum: make([][]uint32, len(t.charCount)), prefixCum: toCum(t.prefCount)}
	for i, c := range t.charCount {
		m.charCum[i] = toCum(c)
	}
	if len(t.termCounts) > 0 {
		s := append([]uint64(nil), t.termCounts...)
		sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
		med := s[len(s)/2]
		k := uint(0)
		for (uint64(1) << (k + 1)) <= med {
			k++
		}
		m.CountK = k
	}
	return m
}

// toCum scales counts to a total <= MaxTotal, every symbol keeping a frequency of at least 1.
func toCum(counts []uint64) []uint32 {
	var sum uint64
	for _, c := range counts {
		sum += c
	}
	budget := uint64(MaxTotal - len(counts))
	cum := make([]uint32, len(counts)+1)
	for i, c := range counts {
		f := uint64(1)
		if sum > 0 {
			f += c * budget / sum
		}
		cum[i+1] = cum[i] + uint32(f)
	}
	return cum
}

// SizeBytes is what the client stores for the model: 2 bytes per frequency.
func (m *Model) SizeBytes() int {
	return 2 * (len(m.charCum)*nCharSyms + nPrefix)
}

func endSym(tf int) int {
	switch {
	case tf <= 1:
		return symEnd1
	case tf == 2:
		return symEnd2
	case tf == 3:
		return symEnd3
	}
	return symEndN
}

func sharedPrefix(a, b string) int {
	n := 0
	for n < len(a) && n < len(b) && n < maxPrefix && a[n] == b[n] {
		n++
	}
	return n
}

type item struct {
	term string
	tf   int
}

// prepare turns analyzer output into what gets coded: sorted unique terms with tf (bow) or the sequence (seq).
func prepare(format string, terms []string) []item {
	if format == FormatSeq {
		out := make([]item, 0, len(terms))
		for _, t := range terms {
			if t != "" {
				out = append(out, item{t, 1})
			}
		}
		return out
	}
	tf := make(map[string]int, len(terms))
	for _, t := range terms {
		if t != "" {
			tf[t]++
		}
	}
	out := make([]item, 0, len(tf))
	for t, c := range tf {
		out = append(out, item{t, c})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].term < out[j].term })
	return out
}
