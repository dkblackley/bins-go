package textcodec

// A 32-bit binary arithmetic coder in the Witten-Neal-Cleary style (CACM 1987). Decoding is pure integer
// arithmetic over cumulative frequency counts: there is no codeword table.
//
// The code is self-delimiting: each renormalisation shift (E1/E2/E3) eventually produces exactly one output bit and
// Finish emits two more, so a decoder that mirrors the shifts knows the encoder wrote shifts+2 bits. That lets many
// documents' codes be concatenated with no length prefix; the decoder may read a little past the end of one code
// (into the next one), which WNC's termination is designed to tolerate.

const (
	codeBits = 32
	topValue = uint64(1)<<codeBits - 1
	firstQtr = topValue/4 + 1
	half     = 2 * firstQtr
	thirdQtr = 3 * firstQtr
	// MaxTotal is the largest frequency total a model may use (must stay well below 2^(codeBits-2)).
	MaxTotal = 1 << 16
)

type ArithEncoder struct {
	w         *BitWriter
	low, high uint64
	pending   int
	Shifts    int
}

func NewArithEncoder(w *BitWriter) *ArithEncoder {
	return &ArithEncoder{w: w, high: topValue}
}

func (e *ArithEncoder) bitPlusPending(b uint) {
	e.w.WriteBit(b)
	for ; e.pending > 0; e.pending-- {
		e.w.WriteBit(b ^ 1)
	}
}

// Encode narrows the interval to [cumLow, cumHigh) out of total.
func (e *ArithEncoder) Encode(cumLow, cumHigh, total uint32) {
	rng := e.high - e.low + 1
	e.high = e.low + rng*uint64(cumHigh)/uint64(total) - 1
	e.low = e.low + rng*uint64(cumLow)/uint64(total)
	for {
		switch {
		case e.high < half:
			e.bitPlusPending(0)
		case e.low >= half:
			e.bitPlusPending(1)
			e.low -= half
			e.high -= half
		case e.low >= firstQtr && e.high < thirdQtr:
			e.pending++
			e.low -= firstQtr
			e.high -= firstQtr
		default:
			return
		}
		e.Shifts++
		e.low <<= 1
		e.high = e.high<<1 | 1
	}
}

// EncodeBits codes the low n bits of v as n equiprobable binary symbols.
func (e *ArithEncoder) EncodeBits(v uint64, n int) {
	for i := n - 1; i >= 0; i-- {
		b := uint32(v>>i) & 1
		e.Encode(b, b+1, 2)
	}
}

func (e *ArithEncoder) EncodeExpGolomb(v uint64, k uint) {
	v += 1 << k
	n := 0
	for t := v; t > 1; t >>= 1 {
		n++
	}
	for i := 0; i < n-int(k); i++ {
		e.Encode(0, 1, 2)
	}
	e.EncodeBits(v, n+1)
}

func (e *ArithEncoder) Finish() {
	e.pending++
	if e.low < firstQtr {
		e.bitPlusPending(0)
	} else {
		e.bitPlusPending(1)
	}
}

type ArithDecoder struct {
	r                *BitReader
	start            int
	low, high, value uint64
	Shifts           int
}

// NewArithDecoder starts decoding a code that begins at r's current position.
func NewArithDecoder(r *BitReader) *ArithDecoder {
	d := &ArithDecoder{r: r, start: r.Pos(), high: topValue}
	d.value = r.ReadBits(codeBits)
	return d
}

// Target returns the cumulative count the next symbol's interval contains; find the symbol, then call Consume.
func (d *ArithDecoder) Target(total uint32) uint32 {
	rng := d.high - d.low + 1
	return uint32(((d.value-d.low+1)*uint64(total) - 1) / rng)
}

func (d *ArithDecoder) Consume(cumLow, cumHigh, total uint32) {
	rng := d.high - d.low + 1
	d.high = d.low + rng*uint64(cumHigh)/uint64(total) - 1
	d.low = d.low + rng*uint64(cumLow)/uint64(total)
	for {
		switch {
		case d.high < half:
		case d.low >= half:
			d.value -= half
			d.low -= half
			d.high -= half
		case d.low >= firstQtr && d.high < thirdQtr:
			d.value -= firstQtr
			d.low -= firstQtr
			d.high -= firstQtr
		default:
			return
		}
		d.Shifts++
		d.low <<= 1
		d.high = d.high<<1 | 1
		d.value = d.value<<1 | uint64(d.r.ReadBit())
	}
}

// DecodeCum decodes one symbol from a cumulative table cum (len = symbols+1, cum[0]=0, cum[last]=total).
func (d *ArithDecoder) DecodeCum(cum []uint32) int {
	total := cum[len(cum)-1]
	t := d.Target(total)
	lo, hi := 0, len(cum)-2
	for lo < hi { // largest s with cum[s] <= t
		m := (lo + hi + 1) / 2
		if cum[m] <= t {
			lo = m
		} else {
			hi = m - 1
		}
	}
	d.Consume(cum[lo], cum[lo+1], total)
	return lo
}

func (d *ArithDecoder) DecodeBits(n int) uint64 {
	var v uint64
	for i := 0; i < n; i++ {
		b := d.Target(2)
		d.Consume(b, b+1, 2)
		v = v<<1 | uint64(b)
	}
	return v
}

func (d *ArithDecoder) DecodeExpGolomb(k uint) uint64 {
	n := 0
	for d.DecodeBits(1) == 0 {
		n++
		if n > 63 {
			return 0
		}
	}
	v := uint64(1)<<(n+int(k)) | d.DecodeBits(n+int(k))
	return v - 1<<k
}

// End moves the reader to the first bit after this code (the encoder wrote Shifts+2 bits).
func (d *ArithDecoder) End() {
	d.r.Seek(d.start + d.Shifts + 2)
}
