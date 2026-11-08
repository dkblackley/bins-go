// simplepir_bench times the online server work of Tiptoe's two services without SEAL.
//
// In Tiptoe both the ranking answer and the URL answer are exactly SimplePIR's
// Server.Answer = matrix.MulVecPacked(squished DB, query) (simplepir/pir/server.go),
// so this builds a random DB with the real dimensions (L rows x M cols, values < p),
// squishes it like setupServer does, and times MulVecPacked. The hint is NOT computed
// (that is offline and would take hours at MS MARCO size).
//
// If the DB would exceed -max-gb, it times a row slice and scales linearly in rows
// (MulVecPacked does the same work for every row).
//
// Output: one JSON line {"seconds": per-answer seconds, ...}.
//
//	go run ./cmd/simplepir_bench -rows 62500 -cols 1340928 -logq 64 -p 131072
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/henrycg/simplepir/matrix"
	"github.com/henrycg/simplepir/rand"
)

func bench[T matrix.Elem](rows, cols, p uint64, reps int, maxBytes float64, bytesPer float64) (float64, uint64) {
	benchRows := rows
	if float64(rows)*float64(cols)*bytesPer > maxBytes {
		benchRows = uint64(maxBytes / (float64(cols) * bytesPer))
		if benchRows < 1 {
			benchRows = 1
		}
	}
	prg := rand.NewRandomBufPRG()
	db := matrix.Rand[T](prg, benchRows, cols, p)
	db.Squish()
	q := matrix.Rand[T](prg, cols, 1, 0)
	if cols%db.SquishRatio() != 0 {
		q.AppendZeros(db.SquishRatio() - cols%db.SquishRatio())
	}
	matrix.MulVecPacked(db, q) // warm-up
	start := time.Now()
	for i := 0; i < reps; i++ {
		matrix.MulVecPacked(db, q)
	}
	per := time.Since(start).Seconds() / float64(reps)
	return per * float64(rows) / float64(benchRows), benchRows
}

func main() {
	rows := flag.Uint64("rows", 0, "DB height L")
	cols := flag.Uint64("cols", 0, "DB width M")
	logq := flag.Int("logq", 64, "64 (ranking service) or 32 (URL service)")
	p := flag.Uint64("p", 1<<17, "plaintext modulus: 2^17 ranking, 256 URL")
	reps := flag.Int("reps", 3, "timed repetitions")
	maxGB := flag.Float64("max-gb", 8, "cap on the benchmark DB size before squishing")
	flag.Parse()
	if *rows == 0 || *cols == 0 {
		fmt.Fprintln(os.Stderr, "need -rows and -cols")
		os.Exit(2)
	}
	var sec float64
	var benched uint64
	if *logq == 64 {
		sec, benched = bench[matrix.Elem64](*rows, *cols, *p, *reps, *maxGB*(1<<30), 8)
	} else {
		sec, benched = bench[matrix.Elem32](*rows, *cols, *p, *reps, *maxGB*(1<<30), 4)
	}
	json.NewEncoder(os.Stdout).Encode(map[string]any{
		"seconds": sec, "rows": *rows, "cols": *cols, "benched_rows": benched, "logq": *logq,
	})
}
