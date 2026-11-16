package textcodec

import "encoding/binary"

// BitWriter appends bits MSB-first.
type BitWriter struct {
	buf  []byte
	nbit int
}

func (w *BitWriter) WriteBit(b uint) {
	if w.nbit&7 == 0 {
		w.buf = append(w.buf, 0)
	}
	if b != 0 {
		w.buf[w.nbit>>3] |= 0x80 >> (w.nbit & 7)
	}
	w.nbit++
}

// WriteBits writes the low n bits of v, most significant first.
func (w *BitWriter) WriteBits(v uint64, n int) {
	for i := n - 1; i >= 0; i-- {
		w.WriteBit(uint(v>>i) & 1)
	}
}

// AppendBitString appends the first nbit bits of src (as written by another BitWriter).
func (w *BitWriter) AppendBitString(src []byte, nbit int) {
	if w.nbit&7 == 0 { // byte aligned: plain copy
		w.buf = append(w.buf, src[:(nbit+7)>>3]...)
		w.nbit += nbit
		if w.nbit&7 != 0 { // clear bits past the end (src may have junk there)
			w.buf[len(w.buf)-1] &= 0xFF << (8 - w.nbit&7)
		}
		return
	}
	for i := 0; i < nbit; i += 8 {
		k := min(8, nbit-i)
		w.WriteBits(uint64(src[i>>3]>>(8-k)), k)
	}
}

// WriteExpGolomb writes v >= 0 with an order-k Exp-Golomb code (k+1+2*floor(log2(v/2^k+1)) bits).
func (w *BitWriter) WriteExpGolomb(v uint64, k uint) {
	v += 1 << k
	n := 0
	for t := v; t > 1; t >>= 1 {
		n++
	}
	for i := 0; i < n-int(k); i++ {
		w.WriteBit(0)
	}
	w.WriteBits(v, n+1)
}

func (w *BitWriter) Bits() int     { return w.nbit }
func (w *BitWriter) Bytes() []byte { return w.buf }

// Words packs the bits into uint64 words (byte i of the stream is byte i%8 of word i/8, little endian), the layout
// BitReader reads back. Trailing all-zero words are dropped: a reader treats bits past the end as zero.
func (w *BitWriter) Words() []uint64 {
	out := make([]uint64, (len(w.buf)+7)/8)
	var tmp [8]byte
	for i := range out {
		clear(tmp[:])
		copy(tmp[:], w.buf[i*8:])
		out[i] = binary.LittleEndian.Uint64(tmp[:])
	}
	for len(out) > 0 && out[len(out)-1] == 0 {
		out = out[:len(out)-1]
	}
	return out
}

// BitReader reads bits MSB-first from uint64 words packed by BitWriter.Words. Reads past the end return 0.
type BitReader struct {
	words []uint64
	pos   int
}

func NewBitReader(words []uint64) *BitReader { return &BitReader{words: words} }

func (r *BitReader) ReadBit() uint {
	p := r.pos
	r.pos++
	wi := p >> 6
	if wi >= len(r.words) {
		return 0
	}
	byteInWord := (p >> 3) & 7
	b := byte(r.words[wi] >> (8 * byteInWord))
	return uint(b>>(7-p&7)) & 1
}

func (r *BitReader) ReadBits(n int) uint64 {
	var v uint64
	for i := 0; i < n; i++ {
		v = v<<1 | uint64(r.ReadBit())
	}
	return v
}

func (r *BitReader) ReadExpGolomb(k uint) uint64 {
	n := 0
	for r.ReadBit() == 0 {
		n++
		if n > 63 {
			return 0 // corrupt / past the end
		}
	}
	v := uint64(1)<<(n+int(k)) | r.ReadBits(n+int(k))
	return v - 1<<k
}

func (r *BitReader) Pos() int     { return r.pos }
func (r *BitReader) Seek(pos int) { r.pos = pos }

// Exhausted reports whether the reader is past the last stored bit.
func (r *BitReader) Exhausted() bool { return r.pos >= len(r.words)*64 }
