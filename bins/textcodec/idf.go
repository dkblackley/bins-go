package textcodec

import (
	"hash/fnv"
	"math"
)

// IDFSketch gives the client document frequencies for BM25 without a vocabulary: a count-min sketch (2 hashed rows
// of 2^Bits one-byte cells), each cell a log-quantised df (df ~ 1.08^(q-1)). A term's df is the min over its cells.
// It is public corpus statistics, built once with the DB; the client never needs to know which words exist.
type IDFSketch struct {
	Bits  uint
	rows  [2][]uint8
	N     uint64  // number of documents
	AvgDL float64 // average analysed document length (tokens), for BM25 length normalisation
}

const idfBase = 1.08

func sketchHash(term string, row int, bits uint) uint32 {
	h := fnv.New64a()
	h.Write([]byte{byte(row)})
	h.Write([]byte(term))
	x := h.Sum64()
	x ^= x >> 29
	return uint32(x & (1<<bits - 1))
}

// IDFCounter accumulates exact df counts per cell; merge one per worker, then Build.
type IDFCounter struct {
	bits   uint
	counts [2][]uint32
	Docs   uint64
	Tokens uint64
}

func NewIDFCounter(bits uint) *IDFCounter {
	c := &IDFCounter{bits: bits}
	for r := range c.counts {
		c.counts[r] = make([]uint32, 1<<bits)
	}
	return c
}

// AddDoc counts one document (terms in document order, duplicates allowed).
func (c *IDFCounter) AddDoc(terms []string) {
	c.Docs++
	c.Tokens += uint64(len(terms))
	seen := make(map[string]struct{}, len(terms))
	for _, t := range terms {
		if _, ok := seen[t]; ok {
			continue
		}
		seen[t] = struct{}{}
		for r := range c.counts {
			c.counts[r][sketchHash(t, r, c.bits)]++
		}
	}
}

func (c *IDFCounter) Merge(o *IDFCounter) {
	c.Docs += o.Docs
	c.Tokens += o.Tokens
	for r := range c.counts {
		for i, v := range o.counts[r] {
			c.counts[r][i] += v
		}
	}
}

func (c *IDFCounter) Build() *IDFSketch {
	s := &IDFSketch{Bits: c.bits, N: c.Docs}
	if c.Docs > 0 {
		s.AvgDL = float64(c.Tokens) / float64(c.Docs)
	}
	for r := range c.counts {
		s.rows[r] = make([]uint8, len(c.counts[r]))
		for i, v := range c.counts[r] {
			s.rows[r][i] = quantDF(v)
		}
	}
	return s
}

func quantDF(df uint32) uint8 {
	if df == 0 {
		return 0
	}
	q := 1 + math.Floor(math.Log(float64(df))/math.Log(idfBase))
	return uint8(min(q, 255))
}

// DF is the (over-)estimated document frequency of term.
func (s *IDFSketch) DF(term string) float64 {
	q := uint8(255)
	for r := range s.rows {
		q = min(q, s.rows[r][sketchHash(term, r, s.Bits)])
	}
	if q == 0 {
		return 0
	}
	return math.Pow(idfBase, float64(q-1)+0.5) // geometric middle of the bucket
}

// IDF is Lucene's BM25 idf, ln(1 + (N - df + 0.5)/(df + 0.5)).
func (s *IDFSketch) IDF(term string) float64 {
	df := min(s.DF(term), float64(s.N))
	return math.Log(1 + (float64(s.N)-df+0.5)/(df+0.5))
}

// Uniform drops the df table (keeps N and AvgDL): every term then gets the idf of an unseen term.
func (s *IDFSketch) Uniform() {
	s.Bits = 0
	s.rows = [2][]uint8{{0}, {0}}
}

func (s *IDFSketch) SizeBytes() int { return len(s.rows[0]) + len(s.rows[1]) }

// BM25 scores a decoded document against the (analysed, de-duplicated) query terms.
func (s *IDFSketch) BM25(query []string, doc []TermTF, docLen int, k1, b float64) float64 {
	if len(doc) == 0 {
		return 0
	}
	norm := k1 * (1 - b + b*float64(docLen)/math.Max(s.AvgDL, 1))
	score := 0.0
	for _, q := range query {
		for _, t := range doc {
			if t.Term == q {
				tf := float64(t.TF)
				score += s.IDF(q) * tf * (k1 + 1) / (tf + norm)
				break
			}
		}
	}
	return score
}
