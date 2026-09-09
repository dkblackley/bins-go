package bm25

import "fmt"

// Debug controls whether debug logging is emitted.
var Debug bool

// Debugf prints formatted debug output when Debug is true.
func Debugf(format string, args ...interface{}) {
	if Debug {
		fmt.Printf(format, args...)
	}
}
