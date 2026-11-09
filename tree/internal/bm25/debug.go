package bm25

import "fmt"

// Debug controls whether debug logging is emitted.
var Debug bool

// StrictPIR verifies sampled and online returned rows against the local
// benchmark database. Production Cuckoo lookup always checks the returned key
// and never substitutes a plaintext row after a PIR failure.
var StrictPIR bool

// PIRConstantWork enables public fixed partition padding and dummy server work
// for client-cache hits. This makes online timings substantially less dependent
// on the queried terms, at the cost of additional PIR work.
var PIRConstantWork bool

// PIRThreads controls how many independent BatchPIR partitions are processed
// concurrently. Queries within a partition remain sequential because a Piano
// client mutates its hint and cache state on every query.
var PIRThreads = 1

// Debugf prints formatted debug output when Debug is true.
func Debugf(format string, args ...interface{}) {
	if Debug {
		fmt.Printf(format, args...)
	}
}
