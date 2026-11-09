package bm25

import (
	"os"
	"syscall"
)

// mmap creates a memory-mapped view of a file
func mmap(f *os.File, size int64) ([]byte, error) {
	return syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
}

// unmmap unmaps a memory-mapped file
func unmmap(data []byte) error {
	return syscall.Munmap(data)
}
