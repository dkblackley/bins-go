package textcodec

import (
	"sort"
	"strings"
)

// EncodeDoc returns a document's code: Exp-Golomb term count, then (if any terms) one self-delimiting arithmetic code.
func (m *Model) EncodeDoc(terms []string) ([]byte, int) {
	var w BitWriter
	m.encodeDoc(&w, terms)
	return w.Bytes(), w.Bits()
}

func (m *Model) encodeDoc(w *BitWriter, terms []string) {
	items := prepare(m.Format, terms)
	w.WriteExpGolomb(uint64(len(items)), m.CountK)
	if len(items) == 0 {
		return
	}
	e := NewArithEncoder(w)
	prev := ""
	for i, it := range items {
		pre := 0
		if m.Format == FormatBOW && i > 0 {
			pre = sharedPrefix(prev, it.term)
			e.Encode(m.prefixCum[pre], m.prefixCum[pre+1], m.prefixCum[nPrefix])
		}
		for pos := pre; pos < len(it.term); pos++ {
			cum := m.charCum[contextOf(it.term, pos, m.Order)]
			b := it.term[pos]
			s := symESC
			if b >= 'a' && b <= 'z' {
				s = int(b - 'a')
			}
			e.Encode(cum[s], cum[s+1], cum[nCharSyms])
			if s == symESC {
				e.Encode(uint32(b), uint32(b)+1, 256)
			}
		}
		cum := m.charCum[contextOf(it.term, len(it.term), m.Order)]
		s := endSym(it.tf)
		e.Encode(cum[s], cum[s+1], cum[nCharSyms])
		if s == symEndN {
			e.EncodeExpGolomb(uint64(it.tf-4), 0)
		}
		prev = it.term
	}
	e.Finish()
}

// TermTF is one decoded term and its count in the document.
type TermTF struct {
	Term string
	TF   int
}

// DecodeDoc decodes one document starting at r's position and leaves r just after it. For seq the tokens are
// aggregated into term counts (BM25 only needs those); Len is the token count either way.
func (m *Model) DecodeDoc(r *BitReader) (terms []TermTF, length int) {
	n := int(r.ReadExpGolomb(m.CountK))
	if n == 0 {
		return nil, 0
	}
	if n > 1<<20 {
		n = 1 << 20 // corrupt input guard
	}
	d := NewArithDecoder(r)
	var sb strings.Builder
	prev := ""
	terms = make([]TermTF, 0, n)
	for i := 0; i < n; i++ {
		pre := 0
		if m.Format == FormatBOW && i > 0 {
			pre = d.DecodeCum(m.prefixCum)
			if pre > len(prev) {
				pre = len(prev)
			}
		}
		sb.Reset()
		sb.WriteString(prev[:pre])
		tf := 1
	chars:
		for guard := 0; guard < 1024; guard++ {
			cur := sb.String()
			s := d.DecodeCum(m.charCum[contextOf(cur, len(cur), m.Order)])
			if s < nLetters {
				sb.WriteByte(byte('a' + s))
				continue
			}
			if s == symESC {
				t := d.Target(256)
				d.Consume(t, t+1, 256)
				sb.WriteByte(byte(t))
				continue
			}
			switch s {
			case symEnd2:
				tf = 2
			case symEnd3:
				tf = 3
			case symEndN:
				tf = 4 + int(d.DecodeExpGolomb(0))
			}
			break chars
		}
		prev = sb.String()
		terms = append(terms, TermTF{prev, tf})
		length += tf
	}
	d.End()
	if m.Format == FormatSeq {
		agg := make(map[string]int, len(terms))
		for _, t := range terms {
			agg[t.Term]++
		}
		terms = terms[:0]
		for t, c := range agg {
			terms = append(terms, TermTF{t, c})
		}
	}
	return terms, length
}

// DocCode is one document's code, as returned by EncodeDoc.
type DocCode struct {
	Idx  uint32
	Bits []byte
	NBit int
}

// BinParams are the public parameters of the bin layout.
type BinParams struct {
	CountK uint // Exp-Golomb order of the number of docs in a bin
	GapK   uint // Exp-Golomb order of the gaps between (sorted) doc indices
}

// NewBinParams picks the Exp-Golomb orders for bins of about docsPerBin docs out of numDocs.
func NewBinParams(numDocs, docsPerBin int) BinParams {
	log2 := func(x int) uint {
		k := uint(0)
		for x > 1 {
			x >>= 1
			k++
		}
		return k
	}
	return BinParams{CountK: log2(max(docsPerBin, 1)), GapK: log2(max(numDocs/max(docsPerBin, 1), 1))}
}

// EncodeBin lays out a bin: doc count, then for each doc (sorted by index) the index gap and the doc's code.
// The order inside a bin does not matter to the client, which re-ranks everything it gets.
func EncodeBin(bp BinParams, docs []DocCode) []uint64 {
	sort.Slice(docs, func(i, j int) bool { return docs[i].Idx < docs[j].Idx })
	var w BitWriter
	w.WriteExpGolomb(uint64(len(docs)), bp.CountK)
	next := uint64(0)
	for _, d := range docs {
		w.WriteExpGolomb(uint64(d.Idx)-next, bp.GapK)
		next = uint64(d.Idx) + 1
		w.AppendBitString(d.Bits, d.NBit)
	}
	return w.Words()
}

// DecodedDoc is one document recovered from a bin.
type DecodedDoc struct {
	Idx   uint32
	Terms []TermTF
	Len   int
}

// DecodeBin reverses EncodeBin. An empty / missing entry decodes to no docs.
func (m *Model) DecodeBin(bp BinParams, entry []uint64) []DecodedDoc {
	if len(entry) == 0 {
		return nil
	}
	r := NewBitReader(entry)
	n := int(r.ReadExpGolomb(bp.CountK))
	out := make([]DecodedDoc, 0, n)
	next := uint64(0)
	for i := 0; i < n && !r.Exhausted(); i++ {
		idx := next + r.ReadExpGolomb(bp.GapK)
		next = idx + 1
		terms, l := m.DecodeDoc(r)
		out = append(out, DecodedDoc{uint32(idx), terms, l})
	}
	return out
}
