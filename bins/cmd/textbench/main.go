// textbench measures how small each text encoding makes a corpus, per document, so the options for the bins text
// mode (-text) can be compared on the real data. Train on the first -train docs, measure on the next -test docs.
//
//	go run ./bins/cmd/textbench -corpus ../datasets/msmarco/corpus.jsonl -train 200000 -test 200000 -dim 192
package main

import (
	"bufio"
	"container/heap"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/dkblackley/bins-go/bins"
	"github.com/dkblackley/bins-go/bins/textcodec"
)

type doc struct{ raw, terms []string }

func load(path string, n int) ([]string, [][]string) {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	an := bins.TextAnalyzer()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	var raws []string
	var terms [][]string
	for sc.Scan() && len(raws) < n {
		var d struct{ Title, Text, Abstract string }
		if err := json.Unmarshal(sc.Bytes(), &d); err != nil {
			continue
		}
		body := d.Text
		if body == "" {
			body = d.Abstract
		}
		raw := d.Title + " " + body
		if d.Title == "" {
			raw = body
		}
		raws = append(raws, raw)
		terms = append(terms, bins.AnalyzeTerms(an, raw))
	}
	return raws, terms
}

// huffman code lengths for the given counts
type node struct {
	w    uint64
	syms []int
}
type pq []node

func (p pq) Len() int           { return len(p) }
func (p pq) Less(i, j int) bool { return p[i].w < p[j].w }
func (p pq) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p *pq) Push(x any)        { *p = append(*p, x.(node)) }
func (p *pq) Pop() any          { o := *p; x := o[len(o)-1]; *p = o[:len(o)-1]; return x }

func huffLens(counts []uint64) []int {
	lens := make([]int, len(counts))
	h := &pq{}
	for s, c := range counts {
		heap.Push(h, node{c + 1, []int{s}})
	}
	for h.Len() > 1 {
		a, b := heap.Pop(h).(node), heap.Pop(h).(node)
		for _, s := range append(a.syms, b.syms...) {
			lens[s]++
		}
		heap.Push(h, node{a.w + b.w, append(append([]int{}, a.syms...), b.syms...)})
	}
	return lens
}

// symbol stream of the seq format for Huffman / 5-bit: a-z = 0..25, separator = 26, other byte = 27 (+8 raw bits)
func charSyms(terms []string, fn func(sym int)) {
	for _, t := range terms {
		for i := 0; i < len(t); i++ {
			if t[i] >= 'a' && t[i] <= 'z' {
				fn(int(t[i] - 'a'))
			} else {
				fn(27)
			}
		}
		fn(26)
	}
}

func main() {
	corpus := flag.String("corpus", "", "corpus.jsonl")
	nTrain := flag.Int("train", 100000, "docs to train models on")
	nTest := flag.Int("test", 100000, "docs to measure on (after the training docs)")
	dim := flag.Int("dim", 192, "embedding dims, for the comparison row (float32)")
	numDocs := flag.Int("num-docs", 8841823, "corpus size, for the doc-index cost in a bin")
	dpb := flag.Int("docsPerBin", 100, "docs per bin, for the doc-index cost in a bin")
	flag.Parse()

	raws, terms := load(*corpus, *nTrain+*nTest)
	if len(raws) <= *nTrain {
		*nTrain = len(raws) / 2
	}
	trR, trT, teR, teT := raws[:*nTrain], terms[:*nTrain], raws[*nTrain:], terms[*nTrain:]
	_ = trR
	n := float64(len(teT))
	fmt.Printf("corpus %s: trained on %d docs, measured on %d\n\n", *corpus, len(trT), len(teT))

	var rawB, words, toks, chars float64
	for i, r := range teR {
		rawB += float64(len(r))
		inWord := false
		for _, c := range r {
			if c == ' ' || c == '\n' || c == '\t' {
				inWord = false
			} else if !inWord {
				words++
				inWord = true
			}
		}
		for _, t := range teT[i] {
			toks++
			chars += float64(len(t))
		}
	}
	fmt.Printf("avg words/doc %.1f, stems/doc %.1f, chars/stem %.2f\n\n", words/n, toks/n, chars/toks)

	type row struct {
		name  string
		bytes float64
		note  string
	}
	rows := []row{
		{"embedding (float32)", float64(*dim * 4), fmt.Sprintf("%d dims", *dim)},
		{"raw text (UTF-8)", rawB / n, "lossless"},
		{"stems, ASCII + space", (chars + toks) / n, "stopwords dropped, Porter stems, digits dropped"},
	}
	// 5-bit reduced alphabet
	var fiveBits float64
	for _, d := range teT {
		charSyms(d, func(s int) {
			fiveBits += 5
			if s == 27 {
				fiveBits += 8
			}
		})
	}
	rows = append(rows, row{"stems, 5-bit alphabet", fiveBits / 8 / n, "27 symbols packed in 5 bits"})
	// static Huffman, order 0
	counts := make([]uint64, 28)
	for _, d := range trT {
		charSyms(d, func(s int) { counts[s]++ })
	}
	lens := huffLens(counts)
	var hBits float64
	for _, d := range teT {
		charSyms(d, func(s int) {
			hBits += float64(lens[s])
			if s == 27 {
				hBits += 8
			}
		})
	}
	rows = append(rows, row{"stems, static Huffman (order 0)", hBits / 8 / n, "28 code lengths"})

	for _, format := range []string{textcodec.FormatSeq, textcodec.FormatBOW} {
		for order := 0; order <= 3; order++ {
			tr := textcodec.NewTrainer(textcodec.Params{Format: format, Order: order})
			for _, d := range trT {
				tr.Add(d)
			}
			m := tr.Build()
			var bits float64
			for _, d := range teT {
				_, nb := m.EncodeDoc(d)
				bits += float64(nb)
			}
			what := "stems in order"
			if format == textcodec.FormatBOW {
				what = "sorted stems + tf, front-coded"
			}
			rows = append(rows, row{fmt.Sprintf("arith %s, order %d", format, order), bits / 8 / n,
				fmt.Sprintf("%s; model %.1f KB", what, float64(m.SizeBytes())/1024)})
		}
	}

	// Reference only: a word-level model needs the vocabulary on the client (a lookup table), which is not allowed.
	wc := map[string]uint64{}
	var tot uint64
	for _, d := range trT {
		for _, t := range d {
			wc[t]++
			tot++
		}
	}
	var vocabBytes float64
	for w := range wc {
		vocabBytes += float64(len(w) + 1)
	}
	var wBits float64
	oov := 0
	for _, d := range teT {
		for _, t := range d {
			if c, ok := wc[t]; ok {
				wBits += -math.Log2(float64(c) / float64(tot+uint64(len(wc))))
			} else {
				oov++
				wBits += 20 + 2.5*float64(len(t)) // escape + spell it out at ~order-2 cost
			}
		}
	}
	rows = append(rows, row{"(ref) word-ID entropy, needs vocab", wBits / 8 / n,
		fmt.Sprintf("NOT allowed: client needs a %.1f MB word list; %.1f%% OOV", vocabBytes/1e6, 100*float64(oov)/toks)})

	sort.SliceStable(rows[1:], func(i, j int) bool { return rows[1+i].bytes > rows[1+j].bytes })
	fmt.Println("| encoding | bytes/doc | vs embedding | notes |")
	fmt.Println("|---|---|---|---|")
	for _, r := range rows {
		fmt.Printf("| %s | %.1f | %.1f%% | %s |\n", r.name, r.bytes, 100*r.bytes/float64(*dim*4), r.note)
	}
	bp := textcodec.NewBinParams(*numDocs, *dpb)
	fmt.Printf("\nIn a bin each doc also costs its index: an Exp-Golomb(%d) gap, about %d bits for %d docs/bin of %d.\n",
		bp.GapK, bp.GapK+2, *dpb, *numDocs)
}
