// plain_baselines: non-private retrieval baselines for one dataset, written into a shared baselines.json.
//
//	go build -o bin/plain_baselines ./cmd/plain_baselines      (pure Go, no cgo/NGT needed)
//	./bin/plain_baselines -name scifact -dataset ../datasets -out baselines.json
//
// For every query with at least one relevant doc in qrels it computes MRR@k and Recall@k (same definitions as
// calcMRR / calcRecall in main.go) for:
//
//	BM25       exact BM25 over the whole corpus (k1=0.9, b=0.4 like Anserini/pyserini and the tree runs).
//	           Analyzer is Anserini-like: split on non letters/digits, lowercase, drop "'s", Lucene's English
//	           stop words, Porter stemmer. Doc text = title + " " + text. Lucene's idf and tf normalisation
//	           (exact doc lengths, Lucene stores them lossily, so numbers can differ from pyserini in the 3rd digit).
//	Dense      exact cosine similarity of each query embedding against every corpus embedding (the same .npy
//	           files the PIR methods and CosineReRank use), brute force, no ANN.
//	Hybrid     score fusion: union of the BM25 and dense top -depth, each score min-max normalised over that
//	           union, ranked by alpha*bm25 + (1-alpha)*dense. A doc missing from the BM25 list gets the list's
//	           lowest BM25 score (pyserini HybridSearcher convention); dense scores are exact for every candidate.
//	HybridRRF  reciprocal rank fusion of the same two lists (k_rrf = 60), no normalisation.
//
// Only the query terms' postings are kept (plus every doc's length), so memory is far below a full index.
// Both stages are parallel over GOMAXPROCS threads.
//
// baselines.json: {"<dataset>": {"BM25": {"MRR": .., "Recall": .., "MRR@100": .., ...}, "Dense": ..., "Hybrid":
// ..., "HybridRRF": ..., "NumQueries": .., ...}}. Updated under a file lock, so one job per dataset can run at once.
package main

import (
	"bufio"
	"bytes"
	"container/heap"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unicode"
	"unicode/utf8"

	porter "github.com/blevesearch/go-porterstemmer"
)

// ---------------------------------------------------------------------------------------------------------------
// dataset paths, mirrors GetDatasets() in main.go

type paths struct{ corpus, queries, qrels, corpusVec, queryVec string }

func datasetPaths(root, name string) paths {
	switch name {
	case "msmarco":
		return paths{root + "/msmarco/corpus.jsonl", root + "/msmarco/queries.dev.small.jsonl",
			root + "/msmarco/qrels/qrels.dev.tsv",
			root + "/msmarco/msmarco_corpus_embed_f32.npy", root + "/msmarco/msmarco_query_embed_f32.npy"}
	case "trec-covid":
		return paths{root + "/trec-covid/corpus.jsonl", root + "/trec-covid/queries.jsonl",
			root + "/trec-covid/qrels/test.tsv",
			root + "/trec-covid/trec_corpus_embed_f32.npy", root + "/trec-covid/trec_query_embed_f32.npy"}
	}
	d := root + "/" + name + "/"
	return paths{d + "corpus.jsonl", d + "queries.jsonl", d + "qrels/test.tsv",
		d + name + "_corpus_embed_f32.npy", d + name + "_query_embed_f32.npy"}
}

// ---------------------------------------------------------------------------------------------------------------
// analyzer

// Lucene's EnglishAnalyzer.ENGLISH_STOP_WORDS_SET (what Anserini's default analyzer uses)
var stopWords = map[string]bool{}

func init() {
	for _, w := range strings.Fields("a an and are as at be but by for if in into is it no not of on or such that " +
		"the their then there these they this to was will with") {
		stopWords[w] = true
	}
}

// tokens calls f with every lowercased, possessive-stripped, non-stop-word token of s (unstemmed).
func tokens(s string, buf []byte, f func(tok string)) []byte {
	emit := func() {
		if len(buf) == 0 {
			return
		}
		t := buf
		if n := len(t); n >= 2 && t[n-1] == 's' && t[n-2] == '\'' {
			t = t[:n-2]
		}
		t = bytes.Trim(t, "'")
		if len(t) > 0 && !stopWords[string(t)] {
			f(string(t))
		}
		buf = buf[:0]
	}
	for i := 0; i < len(s); {
		r, w := rune(s[i]), 1
		if r >= utf8.RuneSelf {
			r, w = utf8.DecodeRuneInString(s[i:])
		}
		i += w
		switch {
		case r < utf8.RuneSelf && (r >= 'a' && r <= 'z' || r >= '0' && r <= '9'):
			buf = append(buf, byte(r))
		case r >= 'A' && r <= 'Z':
			buf = append(buf, byte(r+32))
		case r == '\'' || r == '’':
			if len(buf) > 0 {
				buf = append(buf, '\'')
			}
		case r >= utf8.RuneSelf && (unicode.IsLetter(r) || unicode.IsDigit(r)):
			buf = utf8.AppendRune(buf, unicode.ToLower(r))
		default:
			emit()
		}
	}
	emit()
	return buf
}

func stem(t string) string { return porter.StemString(t) }

// ---------------------------------------------------------------------------------------------------------------
// queries / qrels

type query struct {
	ID, Text string
	row      int // row in the query .npy (= line order in queries.jsonl, as CosineReRank assumes)
}

func loadQueries(path string) []query {
	f, err := os.Open(path)
	check(err)
	defer f.Close()
	var out []query
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<28)
	for sc.Scan() {
		if len(bytes.TrimSpace(sc.Bytes())) == 0 {
			continue
		}
		var q struct {
			ID      any    `json:"_id"`
			AltID   any    `json:"id"`
			QueryID any    `json:"query_id"`
			Text    string `json:"text"`
		}
		check(json.Unmarshal(sc.Bytes(), &q))
		id := idString(q.ID)
		if id == "" {
			id = idString(q.AltID)
		}
		if id == "" {
			id = idString(q.QueryID)
		}
		if id == "" {
			continue // bins.LoadQueries skips these too
		}
		out = append(out, query{ID: id, Text: q.Text, row: len(out)})
	}
	check(sc.Err())
	return out
}

func idString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	}
	return ""
}

// qid -> set of relevant doc IDs (rel > 0), same parsing as bins.LoadQrels
func loadQrels(path string) map[string]map[string]bool {
	f, err := os.Open(path)
	check(err)
	defer f.Close()
	out := map[string]map[string]bool{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		p := strings.Fields(sc.Text())
		if len(p) < 3 {
			continue
		}
		qid, doc, score := p[0], p[1], p[2]
		if len(p) > 3 {
			doc, score = p[2], p[3]
		}
		if v, _ := strconv.Atoi(score); v <= 0 {
			continue
		}
		if out[qid] == nil {
			out[qid] = map[string]bool{}
		}
		out[qid][doc] = true
	}
	check(sc.Err())
	return out
}

// ---------------------------------------------------------------------------------------------------------------
// corpus pass: doc lengths for every doc, postings for the query terms only

type postings struct {
	docs []uint32
	tfs  []uint16
}

type corpusIndex struct {
	numDocs  int
	docLen   []uint32
	post     []postings     // by term id
	termID   map[string]int // stemmed query term -> id
	relRow   map[string]int // corpus row of every doc ID that appears in qrels
	totalLen uint64
}

type batch struct {
	seq   int
	lines [][]byte
}

type batchResult struct {
	seq    int
	n      int
	lens   []uint32
	termTF [][]termHit // per term: (doc within batch, tf)
	ids    []string    // every doc ID (only kept when checking the .ids file)
	rel    map[string]int
}

type termHit struct {
	doc uint32
	tf  uint16
}

func indexCorpus(path string, termID map[string]int, relDocs map[string]bool, idsPath string, threads int) *corpusIndex {
	f, err := os.Open(path)
	check(err)
	defer f.Close()

	idx := &corpusIndex{termID: termID, post: make([]postings, len(termID)), relRow: map[string]int{}}
	keepIDs := idsPath != ""

	batches := make(chan batch, threads*2)
	results := make(chan batchResult, threads*2)

	// reader: split into line batches
	go func() {
		r := bufio.NewReaderSize(f, 16<<20)
		const per = 4096
		seq := 0
		cur := make([][]byte, 0, per)
		for {
			line, err := r.ReadBytes('\n')
			if len(bytes.TrimSpace(line)) > 0 {
				cur = append(cur, line)
				if len(cur) == per {
					batches <- batch{seq, cur}
					seq++
					cur = make([][]byte, 0, per)
				}
			}
			if err != nil {
				break
			}
		}
		if len(cur) > 0 {
			batches <- batch{seq, cur}
		}
		close(batches)
	}()

	// workers: parse + tokenize
	var wg sync.WaitGroup
	for w := 0; w < threads; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			stemCache := map[string]int32{} // surface token -> term id or -1
			var buf []byte
			tf := map[int]int{}
			for b := range batches {
				res := batchResult{seq: b.seq, n: len(b.lines), lens: make([]uint32, len(b.lines)),
					termTF: make([][]termHit, len(termID))}
				if keepIDs {
					res.ids = make([]string, len(b.lines))
				}
				for i, line := range b.lines {
					var d struct {
						ID    any    `json:"_id"`
						Title string `json:"title"`
						Text  string `json:"text"`
					}
					if err := json.Unmarshal(line, &d); err != nil {
						log.Fatalf("corpus line in batch %d: %v", b.seq, err)
					}
					id := idString(d.ID)
					if keepIDs {
						res.ids[i] = id
					}
					if relDocs[id] {
						if res.rel == nil {
							res.rel = map[string]int{}
						}
						res.rel[id] = i
					}
					var n uint32
					clear(tf)
					count := func(t string) {
						n++
						tid, ok := stemCache[t]
						if !ok {
							tid = -1
							if v, ok := termID[stem(t)]; ok {
								tid = int32(v)
							}
							if len(stemCache) > 2_000_000 {
								clear(stemCache)
							}
							stemCache[t] = tid
						}
						if tid >= 0 {
							tf[int(tid)]++
						}
					}
					buf = tokens(d.Title, buf[:0], count)
					buf = tokens(d.Text, buf[:0], count)
					res.lens[i] = n
					for t, c := range tf {
						if c > math.MaxUint16 {
							c = math.MaxUint16
						}
						res.termTF[t] = append(res.termTF[t], termHit{uint32(i), uint16(c)})
					}
				}
				results <- res
			}
		}()
	}
	go func() { wg.Wait(); close(results) }()

	// assembler: append batches in file order so postings stay sorted by row
	var idsReader *bufio.Scanner
	if keepIDs {
		fi, err := os.Open(idsPath)
		check(err)
		defer fi.Close()
		idsReader = bufio.NewScanner(fi)
		idsReader.Buffer(make([]byte, 1<<20), 1<<24)
	}
	pending := map[int]batchResult{}
	next := 0
	last := time.Now()
	for res := range results {
		pending[res.seq] = res
		for {
			r, ok := pending[next]
			if !ok {
				break
			}
			delete(pending, next)
			next++
			base := uint32(idx.numDocs)
			for t, hits := range r.termTF {
				p := &idx.post[t]
				for _, h := range hits {
					p.docs = append(p.docs, base+h.doc)
					p.tfs = append(p.tfs, h.tf)
				}
			}
			for _, l := range r.lens {
				idx.totalLen += uint64(l)
			}
			idx.docLen = append(idx.docLen, r.lens...)
			for id, i := range r.rel {
				idx.relRow[id] = int(base) + i
			}
			for i, id := range r.ids {
				if !idsReader.Scan() {
					log.Fatalf("%s has fewer lines than the corpus (%d)", idsPath, int(base)+i)
				}
				if got := strings.TrimRight(idsReader.Text(), "\r"); got != id {
					log.Fatalf("%s line %d is %q but corpus row %d is %q: embedding rows are not in corpus order",
						idsPath, int(base)+i, got, int(base)+i, id)
				}
			}
			idx.numDocs += r.n
			if time.Since(last) > 30*time.Second {
				last = time.Now()
				log.Printf("corpus: %d docs", idx.numDocs)
			}
		}
	}
	if keepIDs && idsReader.Scan() {
		log.Fatalf("%s has more lines than the corpus (%d docs)", idsPath, idx.numDocs)
	}
	return idx
}

// ---------------------------------------------------------------------------------------------------------------
// BM25 search (Lucene's formula: idf = ln(1 + (N-df+0.5)/(df+0.5)), tf part tf/(tf + k1(1-b+b dl/avgdl)),
// each distinct query term weighted by its count, like pyserini's bag-of-words query)

type hit struct {
	row   uint32
	score float32
}

// minHeap of hits, worst on top
type minHeap []hit

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].score < h[j].score }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(hit)) }
func (h *minHeap) Pop() any          { o := *h; x := o[len(o)-1]; *h = o[:len(o)-1]; return x }

type topK struct {
	k int
	h minHeap
}

func (t *topK) add(row uint32, s float32) {
	if len(t.h) < t.k {
		heap.Push(&t.h, hit{row, s})
	} else if s > t.h[0].score {
		t.h[0] = hit{row, s}
		heap.Fix(&t.h, 0)
	}
}

// sorted returns the hits best first (ties: lower row first) and empties t
func (t *topK) sorted() []hit {
	out := append([]hit(nil), t.h...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].score != out[j].score {
			return out[i].score > out[j].score
		}
		return out[i].row < out[j].row
	})
	t.h = t.h[:0]
	return out
}

func bm25Search(idx *corpusIndex, qterms [][]qterm, depth int, k1, b float64, threads int) [][]hit {
	N := float64(idx.numDocs)
	avgdl := float64(idx.totalLen) / math.Max(N, 1)
	// per-doc length norm, float32 (4 B per doc)
	norm := make([]float32, idx.numDocs)
	parallelFor(idx.numDocs, threads, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			norm[i] = float32(k1 * (1 - b + b*float64(idx.docLen[i])/avgdl))
		}
	})
	out := make([][]hit, len(qterms))
	var next atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < threads; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			top := &topK{k: depth}
			for {
				qi := int(next.Add(1) - 1)
				if qi >= len(qterms) {
					return
				}
				type cursor struct {
					p   *postings
					pos int
					w   float32 // idf * query tf
				}
				var cs []cursor
				for _, t := range qterms[qi] {
					p := &idx.post[t.id]
					df := float64(len(p.docs))
					if df == 0 {
						continue
					}
					idf := math.Log(1 + (N-df+0.5)/(df+0.5))
					cs = append(cs, cursor{p, 0, float32(idf * float64(t.qtf))})
				}
				// document-at-a-time merge of the (row-sorted) posting lists
				for {
					minRow := uint32(math.MaxUint32)
					for i := range cs {
						if c := &cs[i]; c.pos < len(c.p.docs) && c.p.docs[c.pos] < minRow {
							minRow = c.p.docs[c.pos]
						}
					}
					if minRow == math.MaxUint32 {
						break
					}
					var s float32
					for i := range cs {
						c := &cs[i]
						if c.pos < len(c.p.docs) && c.p.docs[c.pos] == minRow {
							tf := float32(c.p.tfs[c.pos])
							s += c.w * tf / (tf + norm[minRow])
							c.pos++
						}
					}
					top.add(minRow, s)
				}
				out[qi] = top.sorted()
			}
		}()
	}
	wg.Wait()
	return out
}

type qterm struct{ id, qtf int }

// ---------------------------------------------------------------------------------------------------------------
// .npy (float32, C order, 2-D), memory mapped

type npy struct {
	data       []float32
	rows, cols int
	raw        []byte
}

var shapeRe = regexp.MustCompile(`'shape':\s*\((\d+),\s*(\d+)\)`)

func openNpy(path string) *npy {
	f, err := os.Open(path)
	check(err)
	defer f.Close()
	st, err := f.Stat()
	check(err)
	raw, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	check(err)
	if string(raw[:6]) != "\x93NUMPY" {
		log.Fatalf("%s: not a .npy file", path)
	}
	var hl, off int
	if raw[6] == 1 {
		hl, off = int(binary.LittleEndian.Uint16(raw[8:10])), 10
	} else {
		hl, off = int(binary.LittleEndian.Uint32(raw[8:12])), 12
	}
	hdr := string(raw[off : off+hl])
	if !strings.Contains(hdr, "'<f4'") || strings.Contains(hdr, "'fortran_order': True") {
		log.Fatalf("%s: need little-endian float32 in C order, header %s", path, hdr)
	}
	m := shapeRe.FindStringSubmatch(hdr)
	if m == nil {
		log.Fatalf("%s: need a 2-D array, header %s", path, hdr)
	}
	rows, _ := strconv.Atoi(m[1])
	cols, _ := strconv.Atoi(m[2])
	start := off + hl
	if len(raw)-start < rows*cols*4 {
		log.Fatalf("%s: truncated (%d bytes of data, want %d)", path, len(raw)-start, rows*cols*4)
	}
	data := unsafeFloats(raw[start:], rows*cols)
	return &npy{data, rows, cols, raw}
}

func (n *npy) row(i int) []float32 { return n.data[i*n.cols : (i+1)*n.cols] }

func normalised(v []float32) []float32 {
	out := make([]float32, len(v))
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if s == 0 {
		return out
	}
	inv := float32(1 / math.Sqrt(s))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

func cosineRow(docs *npy, row int, q []float32) float32 {
	d := docs.row(row)
	var dot, nn float32
	for i, x := range d {
		dot += x * q[i]
		nn += x * x
	}
	if nn == 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(nn)))
}

// denseSearch: exact top-depth cosine for every (normalised) query over every corpus row. Parallel over blocks of
// docs: each block is read from memory once and scored against all queries, each worker keeps its own heaps.
func denseSearch(docs *npy, qs [][]float32, depth, threads int) [][]hit {
	const block = 64
	D := docs.cols
	nq := len(qs)
	flatQ := make([]float32, 0, nq*D)
	for _, q := range qs {
		flatQ = append(flatQ, q...)
	}
	nBlocks := (docs.rows + block - 1) / block
	tops := make([][]topK, threads)
	var next, doneDocs atomic.Int64
	var wg sync.WaitGroup
	start := time.Now()
	for w := 0; w < threads; w++ {
		tops[w] = make([]topK, nq)
		for i := range tops[w] {
			tops[w][i].k = depth
		}
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			my := tops[w]
			inv := make([]float32, block)
			scores := make([]float32, 4)
			for {
				bi := int(next.Add(1) - 1)
				if bi >= nBlocks {
					return
				}
				lo := bi * block
				hi := min(lo+block, docs.rows)
				for r := lo; r < hi; r++ {
					var s float32
					for _, x := range docs.row(r) {
						s += x * x
					}
					inv[r-lo] = 0
					if s > 0 {
						inv[r-lo] = float32(1 / math.Sqrt(float64(s)))
					}
				}
				q := 0
				for ; q+4 <= nq; q += 4 {
					q0, q1 := flatQ[q*D:(q+1)*D], flatQ[(q+1)*D:(q+2)*D]
					q2, q3 := flatQ[(q+2)*D:(q+3)*D], flatQ[(q+3)*D:(q+4)*D]
					for r := lo; r < hi; r++ {
						dot4(docs.row(r), q0, q1, q2, q3, scores)
						iv := inv[r-lo]
						my[q].add(uint32(r), scores[0]*iv)
						my[q+1].add(uint32(r), scores[1]*iv)
						my[q+2].add(uint32(r), scores[2]*iv)
						my[q+3].add(uint32(r), scores[3]*iv)
					}
				}
				for ; q < nq; q++ {
					qq := flatQ[q*D : (q+1)*D]
					for r := lo; r < hi; r++ {
						d := docs.row(r)
						var s float32
						for i, x := range d {
							s += x * qq[i]
						}
						my[q].add(uint32(r), s*inv[r-lo])
					}
				}
				doneDocs.Add(int64(hi - lo))
			}
		}(w)
	}
	stop := make(chan struct{})
	go func() {
		tick := time.NewTicker(time.Minute)
		defer tick.Stop()
		for {
			select {
			case <-stop:
				return
			case <-tick.C:
				n := doneDocs.Load()
				el := time.Since(start).Seconds()
				log.Printf("dense: %d / %d docs (%.0fs, ~%.0fs left)", n, docs.rows, el,
					el*float64(int64(docs.rows)-n)/math.Max(float64(n), 1))
			}
		}
	}()
	wg.Wait()
	close(stop)
	// merge the workers' heaps
	out := make([][]hit, nq)
	parallelFor(nq, threads, func(lo, hi int) {
		for q := lo; q < hi; q++ {
			t := &topK{k: depth}
			for w := range tops {
				for _, h := range tops[w][q].h {
					t.add(h.row, h.score)
				}
				tops[w][q].h = nil
			}
			out[q] = t.sorted()
		}
	})
	return out
}

// dot4: d . q0..q3 into s[0..3], two accumulators per query so the adds pipeline
func dot4(d, q0, q1, q2, q3, s []float32) {
	n := len(d)
	q0, q1, q2, q3 = q0[:n], q1[:n], q2[:n], q3[:n]
	var a0, a1, a2, a3, b0, b1, b2, b3 float32
	i := 0
	for ; i+2 <= n; i += 2 {
		x, y := d[i], d[i+1]
		a0 += x * q0[i]
		b0 += y * q0[i+1]
		a1 += x * q1[i]
		b1 += y * q1[i+1]
		a2 += x * q2[i]
		b2 += y * q2[i+1]
		a3 += x * q3[i]
		b3 += y * q3[i+1]
	}
	for ; i < n; i++ {
		x := d[i]
		a0 += x * q0[i]
		a1 += x * q1[i]
		a2 += x * q2[i]
		a3 += x * q3[i]
	}
	s[0], s[1], s[2], s[3] = a0+b0, a1+b1, a2+b2, a3+b3
}

// ---------------------------------------------------------------------------------------------------------------
// fusion

func fuseScores(bm, de []hit, docs *npy, qv []float32, alpha float64, k int) []uint32 {
	bmScore := make(map[uint32]float32, len(bm))
	floor := float32(0)
	for i, h := range bm {
		bmScore[h.row] = h.score
		if i == 0 || h.score < floor {
			floor = h.score
		}
	}
	deScore := make(map[uint32]float32, len(de))
	for _, h := range de {
		deScore[h.row] = h.score
	}
	var cand []uint32
	seen := map[uint32]bool{}
	for _, l := range [][]hit{bm, de} {
		for _, h := range l {
			if !seen[h.row] {
				seen[h.row] = true
				cand = append(cand, h.row)
			}
		}
	}
	if len(cand) == 0 {
		return nil
	}
	b := make([]float64, len(cand))
	d := make([]float64, len(cand))
	for i, r := range cand {
		if s, ok := bmScore[r]; ok {
			b[i] = float64(s)
		} else {
			b[i] = float64(floor)
		}
		if s, ok := deScore[r]; ok {
			d[i] = float64(s)
		} else {
			d[i] = float64(cosineRow(docs, int(r), qv))
		}
	}
	minmax(b)
	minmax(d)
	order := make([]int, len(cand))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		return alpha*b[order[i]]+(1-alpha)*d[order[i]] > alpha*b[order[j]]+(1-alpha)*d[order[j]]
	})
	out := make([]uint32, 0, k)
	for _, i := range order[:min(k, len(order))] {
		out = append(out, cand[i])
	}
	return out
}

func minmax(x []float64) {
	lo, hi := x[0], x[0]
	for _, v := range x {
		lo, hi = math.Min(lo, v), math.Max(hi, v)
	}
	for i := range x {
		if hi == lo {
			x[i] = 1
		} else {
			x[i] = (x[i] - lo) / (hi - lo)
		}
	}
}

func rrf(bm, de []hit, k int) []uint32 {
	const kRRF = 60
	score := map[uint32]float64{}
	var order []uint32
	for _, l := range [][]hit{bm, de} {
		for rank, h := range l {
			if _, ok := score[h.row]; !ok {
				order = append(order, h.row)
			}
			score[h.row] += 1 / float64(kRRF+rank+1)
		}
	}
	sort.SliceStable(order, func(i, j int) bool { return score[order[i]] > score[order[j]] })
	return order[:min(k, len(order))]
}

// ---------------------------------------------------------------------------------------------------------------
// metrics: calcMRR / calcRecall from main.go, on corpus rows

func evaluate(ranked [][]uint32, rel []map[uint32]bool, numRel []int, ks []int) map[string]any {
	out := map[string]any{}
	for ki, k := range ks {
		var mrr, recall float64
		for q, list := range ranked {
			list = list[:min(k, len(list))]
			found := 0
			first := true
			for i, r := range list {
				if rel[q][r] {
					if first {
						mrr += 1 / float64(i+1)
						first = false
					}
					found++
				}
			}
			recall += float64(found) / float64(numRel[q])
		}
		n := float64(len(ranked))
		m, r := round4(mrr/n), round4(recall/n)
		out[fmt.Sprintf("MRR@%d", k)] = m
		out[fmt.Sprintf("Recall@%d", k)] = r
		if ki == 0 {
			out["MRR"], out["Recall"] = m, r
		}
	}
	return out
}

func round4(x float64) float64 { return math.Round(x*1e4) / 1e4 }

func rows(h []hit) []uint32 {
	out := make([]uint32, len(h))
	for i, x := range h {
		out[i] = x.row
	}
	return out
}

// ---------------------------------------------------------------------------------------------------------------
// baselines.json, read-modify-write under flock

func writeResult(path, name string, entry map[string]any) {
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o644)
	check(err)
	defer lock.Close()
	check(syscall.Flock(int(lock.Fd()), syscall.LOCK_EX))
	defer syscall.Flock(int(lock.Fd()), syscall.LOCK_UN)

	all := map[string]any{}
	if b, err := os.ReadFile(path); err == nil && len(bytes.TrimSpace(b)) > 0 {
		if err := json.Unmarshal(b, &all); err != nil {
			log.Fatalf("%s exists but is not valid JSON (%v): fix or move it, results are below", path, err)
		}
	}
	all[name] = entry
	b, err := json.MarshalIndent(all, "", "  ")
	check(err)
	tmp := fmt.Sprintf("%s.tmp%d", path, os.Getpid())
	check(os.WriteFile(tmp, append(b, '\n'), 0o644))
	check(os.Rename(tmp, path))
}

// ---------------------------------------------------------------------------------------------------------------

func main() {
	name := flag.String("name", "scifact", "dataset name, as -name in main.go")
	root := flag.String("dataset", "../datasets", "datasets root, as -dataset in main.go")
	out := flag.String("out", "baselines.json", "JSON file to add this dataset's results to")
	ksFlag := flag.String("k", "10,100", "cut-offs; the first one is reported as MRR/Recall, every one as MRR@k/Recall@k")
	depth := flag.Int("depth", 1000, "per-list depth (BM25 and dense) for the hybrid fusion")
	alpha := flag.Float64("alpha", 0.5, "BM25 weight in Hybrid (after min-max normalisation)")
	k1 := flag.Float64("k1", 0.9, "BM25 k1")
	b := flag.Float64("b", 0.4, "BM25 b")
	threads := flag.Int("threads", runtime.GOMAXPROCS(0), "worker threads")
	maxQueries := flag.Int("max-queries", 0, "only the first N scorable queries (0 = all), for a quick test")
	skipDense := flag.Bool("skip-dense", false, "BM25 only (Dense/Hybrid left out)")
	flag.Parse()
	log.SetFlags(log.Ltime)

	var ks []int
	for _, s := range strings.Split(*ksFlag, ",") {
		k, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || k <= 0 {
			log.Fatalf("bad -k %q", *ksFlag)
		}
		ks = append(ks, k)
	}
	maxK := slicesMax(ks)
	if *depth < maxK {
		*depth = maxK
	}

	p := datasetPaths(*root, *name)
	t0 := time.Now()

	// queries that can be scored
	qrels := loadQrels(p.qrels)
	var qs []query
	for _, q := range loadQueries(p.queries) {
		if len(qrels[q.ID]) > 0 {
			qs = append(qs, q)
		}
	}
	if *maxQueries > 0 && len(qs) > *maxQueries {
		qs = qs[:*maxQueries]
	}
	if len(qs) == 0 {
		log.Fatalf("%s: no query in %s has relevant docs in %s", *name, p.queries, p.qrels)
	}
	relDocs := map[string]bool{}
	for _, q := range qs {
		for d := range qrels[q.ID] {
			relDocs[d] = true
		}
	}
	log.Printf("%s: %d queries with qrels, %d threads", *name, len(qs), *threads)

	// query terms
	termID := map[string]int{}
	qterms := make([][]qterm, len(qs))
	for i, q := range qs {
		counts := map[int]int{}
		var order []int
		tokens(q.Text, nil, func(t string) {
			s := stem(t)
			id, ok := termID[s]
			if !ok {
				id = len(termID)
				termID[s] = id
			}
			if counts[id] == 0 {
				order = append(order, id)
			}
			counts[id]++
		})
		for _, id := range order {
			qterms[i] = append(qterms[i], qterm{id, counts[id]})
		}
	}

	// corpus
	var docs *npy
	idsPath := ""
	if !*skipDense {
		docs = openNpy(p.corpusVec)
		if _, err := os.Stat(p.corpusVec + ".ids"); err == nil {
			idsPath = p.corpusVec + ".ids" // checked line by line against the corpus while indexing
		}
	}
	idx := indexCorpus(p.corpus, termID, relDocs, idsPath, *threads)
	var nPost int
	for _, pl := range idx.post {
		nPost += len(pl.docs)
	}
	tIndex := time.Since(t0)
	log.Printf("corpus: %d docs, avg length %.1f, %d query terms, %d postings (%.0fs)", idx.numDocs,
		float64(idx.totalLen)/float64(idx.numDocs), len(termID), nPost, tIndex.Seconds())
	if docs != nil && docs.rows != idx.numDocs {
		log.Fatalf("%s has %d rows but the corpus has %d docs", p.corpusVec, docs.rows, idx.numDocs)
	}

	// relevance on corpus rows
	rel := make([]map[uint32]bool, len(qs))
	numRel := make([]int, len(qs))
	missing := 0
	for i, q := range qs {
		rel[i] = map[uint32]bool{}
		numRel[i] = len(qrels[q.ID]) // a relevant doc missing from the corpus still counts, as in main.go
		for d := range qrels[q.ID] {
			if r, ok := idx.relRow[d]; ok {
				rel[i][uint32(r)] = true
			} else {
				missing++
			}
		}
	}
	if missing > 0 {
		log.Printf("warning: %d qrels entries name docs that are not in the corpus", missing)
	}

	t1 := time.Now()
	bm := bm25Search(idx, qterms, *depth, *k1, *b, *threads)
	tBM25 := time.Since(t1)
	log.Printf("BM25 search: %.1fs", tBM25.Seconds())
	idx.post, idx.docLen = nil, nil
	runtime.GC()

	entry := map[string]any{
		"NumQueries": len(qs),
		"NumDocs":    idx.numDocs,
		"Params": map[string]any{"k": ks, "fusion_depth": *depth, "alpha": *alpha, "k1": *k1, "b": *b,
			"rrf_k": 60, "analyzer": "lowercase+lucene-english-stop+porter", "max_queries": *maxQueries},
		"BM25": evaluate(sliceRows(bm), rel, numRel, ks),
	}
	timings := map[string]float64{"corpus_pass_s": round1(tIndex.Seconds()), "bm25_search_s": round1(tBM25.Seconds())}

	if docs != nil {
		qv := openNpy(p.queryVec)
		if qv.cols != docs.cols {
			log.Fatalf("query dim %d != corpus dim %d", qv.cols, docs.cols)
		}
		nQ := len(loadQueries(p.queries))
		if qv.rows != nQ {
			log.Printf("warning: %s has %d rows but %s has %d queries (rows are matched by line order)",
				p.queryVec, qv.rows, p.queries, nQ)
		}
		qvecs := make([][]float32, len(qs))
		for i, q := range qs {
			if q.row >= qv.rows {
				log.Fatalf("query %s is line %d but %s only has %d rows", q.ID, q.row, p.queryVec, qv.rows)
			}
			qvecs[i] = normalised(qv.row(q.row))
		}
		syscall.Madvise(docs.raw, syscall.MADV_SEQUENTIAL)
		t2 := time.Now()
		de := denseSearch(docs, qvecs, *depth, *threads)
		tDense := time.Since(t2)
		log.Printf("dense search: %.1fs", tDense.Seconds())
		syscall.Madvise(docs.raw, syscall.MADV_RANDOM)

		hyb := make([][]uint32, len(qs))
		hrrf := make([][]uint32, len(qs))
		parallelFor(len(qs), *threads, func(lo, hi int) {
			for q := lo; q < hi; q++ {
				hyb[q] = fuseScores(bm[q], de[q], docs, qvecs[q], *alpha, maxK)
				hrrf[q] = rrf(bm[q], de[q], maxK)
			}
		})
		entry["Dense"] = evaluate(sliceRows(de), rel, numRel, ks)
		h := evaluate(hyb, rel, numRel, ks)
		h["alpha"] = *alpha
		entry["Hybrid"] = h
		entry["HybridRRF"] = evaluate(hrrf, rel, numRel, ks)
		timings["dense_search_s"] = round1(tDense.Seconds())
	}
	timings["total_s"] = round1(time.Since(t0).Seconds())
	entry["Timings"] = timings
	entry["Date"] = time.Now().Format(time.RFC3339)

	writeResult(*out, *name, entry)
	pretty, _ := json.MarshalIndent(entry, "", "  ")
	fmt.Printf("%s\n", pretty)
	log.Printf("wrote %s[%q] in %.0fs", *out, *name, time.Since(t0).Seconds())
}

func sliceRows(h [][]hit) [][]uint32 {
	out := make([][]uint32, len(h))
	for i := range h {
		out[i] = rows(h[i])
	}
	return out
}

func round1(x float64) float64 { return math.Round(x*10) / 10 }

func slicesMax(x []int) int {
	m := x[0]
	for _, v := range x {
		m = max(m, v)
	}
	return m
}

func parallelFor(n, threads int, f func(lo, hi int)) {
	var wg sync.WaitGroup
	chunk := (n + threads - 1) / max(threads, 1)
	for lo := 0; lo < n; lo += max(chunk, 1) {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func() { defer wg.Done(); f(lo, hi) }()
	}
	wg.Wait()
}

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
