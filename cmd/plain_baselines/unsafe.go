package main

import "unsafe"

// unsafeFloats views b (from mmap, so page aligned + the .npy header, which numpy pads to 64 bytes) as n float32s
func unsafeFloats(b []byte, n int) []float32 {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n)
}
