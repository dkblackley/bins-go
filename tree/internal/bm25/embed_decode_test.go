package bm25

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestDecodeEmbedValue(t *testing.T) {
	f16 := map[uint16]float32{0x3c00: 1, 0xc000: -2, 0x3555: 0.33325195, 0x0001: 5.9604645e-08, 0x7bff: 65504, 0: 0}
	for bits, want := range f16 {
		b := make([]byte, 2)
		binary.LittleEndian.PutUint16(b, bits)
		if got := decodeEmbedValue(b, 2); got != want {
			t.Errorf("float16 %#04x: got %v, want %v", bits, got, want)
		}
	}
	if got := decodeEmbedValue([]byte{0x81}, 1); got != -127 {
		t.Errorf("int8 0x81: got %v, want -127", got)
	}
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, math.Float32bits(0.25))
	if got := decodeEmbedValue(b, 4); got != 0.25 {
		t.Errorf("float32: got %v", got)
	}
}

// A per-vector scale does not change cosine similarity, so int8 vectors
// quantized with their own absmax rank like the float originals.
func TestInt8CosineMatchesFloat(t *testing.T) {
	q := []float32{0.3, -0.2, 0.9, 0.1}
	doc := []float32{0.25, -0.1, 0.8, 0.05}
	quant := make([]float32, len(doc))
	for i, v := range doc {
		quant[i] = decodeEmbedValue([]byte{byte(int8(math.Round(float64(v / 0.8 * 127))))}, 1)
	}
	if d := math.Abs(float64(CosineSimilarity(q, doc) - CosineSimilarity(q, quant))); d > 0.01 {
		t.Fatalf("int8 cosine off by %v", d)
	}
}

func TestParseNPYHeaderSmallDtypes(t *testing.T) {
	for dtype, want := range map[string]int{"|i1": 1, "<f2": 2, "<f4": 4} {
		header := "{'descr': '" + dtype + "', 'fortran_order': False, 'shape': (5000, 192), }"
		data := append([]byte("\x93NUMPY\x01\x00"), byte(len(header)), 0)
		data = append(data, header...)
		docs, dim, bytesPerElem, _, err := ParseNPYHeader(data)
		if err != nil || docs != 5000 || dim != 192 || bytesPerElem != want {
			t.Errorf("%s: got %d %d %d %v, want 5000 192 %d", dtype, docs, dim, bytesPerElem, err, want)
		}
	}
}
