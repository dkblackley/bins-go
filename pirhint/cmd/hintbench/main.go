// hintbench compares the client storage of PIR DB layouts for the tree and bins
// methods, at full dataset scale, without downloading anything.
//
// Sizes come from a synthetic Zipf corpus (an upper bound: real blocks are
// clustered, which shrinks the Cuckoo tables) or, for the tree, from a built
// artifact directory's id maps (exact). Client storage is computed with the
// same formulas the PIR code allocates with (checked in pirhint_test.go).
//
// -run additionally builds every layout for real with pir-old on random data
// at -run-docs documents, preprocesses it and makes one search's worth of PIR
// lookups, to measure preprocessing time and online bytes next to the storage.
//
//	go run ./pirhint/cmd/hintbench -dataset msmarco
//	go run ./pirhint/cmd/hintbench -dataset scifact -B 16 -r 8 -s 4
//	go run ./pirhint/cmd/hintbench -dataset msmarco -stage1-idmap .../stage1_idmap.bin -stage2-idmap .../stage2/idmap.bin
//	go run ./pirhint/cmd/hintbench -dataset msmarco -run -run-docs 200000
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"time"

	pirold "github.com/dkblackley/bins-go/pir-old"
	"github.com/dkblackley/bins-go/pirhint"
)

func main() {
	method := flag.String("method", "all", "tree, bins or all")
	dataset := flag.String("dataset", "msmarco", "msmarco or scifact: sets the corpus defaults below")
	docs := flag.Uint64("docs", 0, "documents (0 = dataset default)")
	tokens := flag.Float64("tokens", 0, "term draws per document (0 = dataset default)")
	vocab := flag.Uint64("vocab", 0, "vocabulary size (0 = dataset default)")
	zipf := flag.Float64("zipf", 1.0, "Zipf exponent of term frequencies")
	cluster := flag.Float64("cluster", 1.0, "keep this fraction of the random-assignment distinct terms per block (clustering)")

	t := pirhint.DefaultTree
	flag.IntVar(&t.B, "B", t.B, "tree block size")
	flag.IntVar(&t.R, "r", t.R, "tree fan-out")
	flag.IntVar(&t.S, "s", t.S, "tree sub-blocks: Stage2 values are B/s bytes")
	flag.Uint64Var(&t.Batch, "batch", t.Batch, "--pir-batch-size of Stage1/2")
	flag.Uint64Var(&t.Stage3Batch, "stage3-batch", t.Stage3Batch, "Stage3 batch size")
	flag.Float64Var(&t.LoadFactor, "load", t.LoadFactor, "Cuckoo load factor")
	flag.IntVar(&t.Hashes, "hashes", 2, "Cuckoo hash locations per key (2 or 3)")
	flag.IntVar(&t.KeyBytes, "key-bytes", 8, "Cuckoo key bytes kept per row")
	flag.IntVar(&t.UBBits, "ub-bits", 8, "bits per Stage1/2 upper bound")
	flag.IntVar(&t.EmbedDim, "dim", t.EmbedDim, "embedding dimension (tree Stage3 and bins)")
	flag.IntVar(&t.QueryWords, "query-words", t.QueryWords, "query words (online estimate)")
	flag.IntVar(&t.Beam, "beam", t.Beam, "nodes kept per level (online estimate)")
	stage1IDMap := flag.String("stage1-idmap", "", "real stage1_idmap.bin: exact Stage1 sizes instead of synthetic")
	stage2IDMap := flag.String("stage2-idmap", "", "real stage2/idmap.bin")

	bs := flag.Uint64("bs", 0, "bins hash table size (0 = dataset default)")
	dpb := flag.Uint64("dpb", 100, "bins docs per bin")

	detail := flag.Bool("detail", true, "print every DB of the current and the combined layout")
	jsonOut := flag.String("json", "", "also write every result to this JSON file")
	run := flag.Bool("run", false, "build each tree layout with pir-old on random data and measure it")
	runDocs := flag.Uint64("run-docs", 100000, "documents of the synthetic corpus for -run")
	runCap := flag.Float64("run-cap-gb", 4, "skip a -run layout whose DBs plus hints exceed this many GB")
	flag.Parse()

	corpus, binSize := pirhint.MSMarco, uint64(1_000_000)
	if *dataset == "scifact" {
		corpus, binSize = pirhint.SciFact, 5000
	}
	if *docs > 0 {
		corpus.Docs = *docs
	}
	if *tokens > 0 {
		corpus.Tokens = *tokens
	}
	if *vocab > 0 {
		corpus.Vocab = *vocab
	}
	corpus.ZipfExp, corpus.Clusters = *zipf, *cluster
	if *bs > 0 {
		binSize = *bs
	}

	out := map[string]any{}
	if *method == "tree" || *method == "all" {
		shape := pirhint.SyntheticTreeShape(corpus, t.B, t.R)
		if *stage1IDMap != "" {
			var err error
			if shape, err = pirhint.RealTreeShape(*stage1IDMap, *stage2IDMap, corpus.Docs, t.B, t.R); err != nil {
				log.Fatal(err)
			}
		}
		fmt.Printf("## Tree, %s (%s sizes): %d docs, B=%d r=%d s=%d, batch %d, Stage3 batch %d, dim %d\n\n",
			*dataset, shape.Source, shape.Docs, t.B, t.R, t.S, t.Batch, t.Stage3Batch, t.EmbedDim)
		fmt.Printf("Stage1 items per level: %v; Stage2 items: %d; leaves: %d\n", shape.Stage1Items, shape.Stage2Items, shape.Leaves)
		results := evaluate(pirhint.TreeLayouts(shape, t))
		pirhint.WriteMarkdown(os.Stdout, "Tree client storage", results)
		if *detail {
			pirhint.WriteDBDetail(os.Stdout, results[0])
			pirhint.WriteDBDetail(os.Stdout, results[len(results)-1])
		}
		out["tree"] = map[string]any{"shape": shape, "params": t, "layouts": results}

		if *run {
			small := corpus
			small.Docs = *runDocs
			runShape := pirhint.SyntheticTreeShape(small, t.B, t.R)
			fmt.Printf("\n## Measured with pir-old on random data, %d docs (Stage1 %v, Stage2 %d items)\n",
				small.Docs, runShape.Stage1Items, runShape.Stage2Items)
			out["tree_run"] = runLayouts(pirhint.TreeLayouts(runShape, t), *runCap*(1<<30))
		}
	}
	if *method == "bins" || *method == "all" {
		p := pirhint.BinsParams{Docs: corpus.Docs, BinSize: binSize, DPB: *dpb, Dim: t.EmbedDim, EmbedBytes: 4,
			Small: corpus.Docs < 10000, QueryTerms: 8}
		fmt.Printf("\n## Bins, %s: %d docs, bs=%d, dpb=%d, dim %d\n", *dataset, p.Docs, p.BinSize, p.DPB, p.Dim)
		results := evaluate(pirhint.BinsLayouts(p))
		pirhint.WriteMarkdown(os.Stdout, "Bins client storage", results)
		if *detail {
			pirhint.WriteDBDetail(os.Stdout, results[1])
		}
		out["bins"] = map[string]any{"params": p, "layouts": results}
	}
	if *jsonOut != "" {
		data, err := json.MarshalIndent(out, "", "  ")
		if err != nil {
			log.Fatal(err)
		}
		if err := os.WriteFile(*jsonOut, data, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}

func evaluate(layouts []pirhint.Layout) []pirhint.LayoutResult {
	rs := make([]pirhint.LayoutResult, len(layouts))
	for i, l := range layouts {
		rs[i] = pirhint.Evaluate(l)
	}
	return rs
}

// measured is what -run records for one layout.
type measured struct {
	Layout        string
	ClientBytes   float64 // summed LocalStorageSize of the real objects
	Predicted     float64 // pirhint's number for the same DBs
	OnlineBytes   float64 // bytes the server saw for one search's lookups
	PreprocessSec float64
	QuerySec      float64
	Skipped       string `json:",omitempty"`
}

// runLayouts builds each layout with pir-old on random data. A DB with
// Copies > 1 is built once and its costs multiplied, so the per-leaf layout
// stays affordable.
func runLayouts(layouts []pirhint.Layout, capBytes float64) []measured {
	defer func(c, q float64, d bool) {
		pirold.ChunkSizeFactor, pirold.QueryBudgetScale, pirold.DownloadIfCheaper = c, q, d
	}(pirold.ChunkSizeFactor, pirold.QueryBudgetScale, pirold.DownloadIfCheaper)

	var all []measured
	fmt.Println("\n| layout | client (measured) | client (pirhint) | online, one search | preprocess | query |")
	fmt.Println("|---|---|---|---|---|---|")
	rng := rand.New(rand.NewSource(1))
	for _, l := range layouts {
		res := pirhint.Evaluate(l)
		m := measured{Layout: l.Name, Predicted: res.Client}
		if built := builtBytes(res); built > capBytes {
			m.Skipped = fmt.Sprintf("over %s", pirhint.HumanBytes(capBytes))
			fmt.Printf("| %s | skipped (%s) | %s | | | |\n", l.Name, m.Skipped, pirhint.HumanBytes(res.Client))
			all = append(all, m)
			continue
		}
		pirold.ChunkSizeFactor, pirold.QueryBudgetScale, pirold.DownloadIfCheaper =
			l.Knobs.ChunkFactor, l.Knobs.QueryBudget, l.Download
		for _, db := range l.DBs {
			copies := float64(max(db.Copies, 1))
			words := db.EntryBytes / 8
			raw := make([]uint64, db.Rows*words)
			for i := range raw {
				raw[i] = rng.Uint64()
			}
			var queryErr error
			quiet(func() {
				pir := pirold.NewSimpleBatchPianoPIR(db.Rows, db.EntryBytes, db.Batch, raw, l.Knobs.FailBits)
				pir.SetThreadNum(8)
				start := time.Now()
				pir.Preprocessing()
				m.PreprocessSec += time.Since(start).Seconds() * copies
				idx := make([]uint64, db.Probes)
				for i := range idx {
					idx[i] = uint64(rng.Int63n(int64(db.Rows)))
				}
				start = time.Now()
				if _, err := pir.Query(idx); err != nil {
					queryErr = err
					return
				}
				m.QuerySec += time.Since(start).Seconds() * copies
				metrics := pir.GetMetrics()
				m.OnlineBytes += metrics["TotalByteSent"] * copies
				m.ClientBytes += pir.LocalStorageSize() * copies
			})
			if queryErr != nil {
				log.Fatalf("%s/%s: %v", l.Name, db.Name, queryErr)
			}
		}
		fmt.Printf("| %s | %s | %s | %s | %.2fs | %.3fs |\n", l.Name, pirhint.HumanBytes(m.ClientBytes),
			pirhint.HumanBytes(m.Predicted), pirhint.HumanBytes(m.OnlineBytes), m.PreprocessSec, m.QuerySec)
		all = append(all, m)
	}
	return all
}

// builtBytes is the memory -run needs: one copy of each DB plus its hints.
func builtBytes(res pirhint.LayoutResult) float64 {
	total := 0.0
	for _, r := range res.Results {
		copies := float64(max(r.Copies, 1))
		total += (r.DBBytes + r.HintBytes) / copies
	}
	return total
}

// quiet hides pir-old's progress printing while f runs.
func quiet(f func()) {
	stdout := os.Stdout
	if devnull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0); err == nil {
		os.Stdout = devnull
		defer func() { os.Stdout = stdout; devnull.Close() }()
	}
	log.SetOutput(io.Discard)
	defer log.SetOutput(os.Stderr)
	f()
}
