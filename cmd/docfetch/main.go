// docfetch measures the step after a retrieval method has returned its top-k doc IDs: privately fetching those k
// documents' text. It builds a Piano PIR DB with one row per document (row i = doc i of corpus.jsonl, compressed),
// then for every query fetches k documents in one batch, checks that every row came back byte for byte and decodes
// to text, and writes the cost (bytes, LAN/WAN time, client storage, server time) to <out>/<run>/metadata.json.
//
// The DB is the same for every method (bins, tree, PACMANN, ...): they all end with k doc IDs. Which k docs are
// fetched does not change the cost (the PIR rows are shuffled, and every lookup touches the same number of rows), so
// by default each query fetches k distinct random docs. -ids <results.json> fetches a method's real answers instead
// ({"qid": ["docid", ...]}, what pir_app writes); a query with fewer than k answers is padded with random docs so
// every query is the same size on the wire.
//
// Row layout: uvarint(byte length) + the doc's bytes, zero-padded to 8 bytes. Rows have different lengths; the PIR
// stores them unpadded and charges each response its real length (see pianopir chargeQuery). A doc is
// title + "\n" + text. -compress flate (default) deflates each doc on its own with a preset dictionary of sample
// text (the client keeps the dictionary; it is added to ClientStorageMB). -max-doc-bytes truncates longer docs
// before compression (0 = never; the number truncated is in the metadata).
//
//	go run ./cmd/docfetch -name scifact -dataset ../../../datasets -k 10,100 -out ../../../datasets/results_docfetch
package main

import (
	"bufio"
	"bytes"
	"compress/flate"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/dkblackley/bins-go/pianopir"
	"github.com/sirupsen/logrus"
)

type doc struct {
	ID       string `json:"_id"`
	Title    string `json:"title"`
	Text     string `json:"text"`
	Abstract string `json:"abstract"`
}

// paths returns corpus.jsonl and the queries file for a dataset name (the same layout as GetDatasets in main.go).
func paths(root, name string) (corpus, queries string) {
	switch name {
	case "msmarco":
		return root + "/msmarco/corpus.jsonl", root + "/msmarco/queries.dev.small.jsonl"
	default: // scifact, trec-covid and the generic layout of datasets/prepare_dataset.py
		return root + "/" + name + "/corpus.jsonl", root + "/" + name + "/queries.jsonl"
	}
}

func docText(line []byte) (id, text string, err error) {
	var d doc
	if err = json.Unmarshal(line, &d); err != nil {
		return
	}
	body := d.Text
	if body == "" {
		body = d.Abstract
	}
	if d.Title == "" {
		return d.ID, body, nil
	}
	return d.ID, d.Title + "\n" + body, nil
}

// truncate cuts s to at most n bytes on a UTF-8 boundary.
func truncate(s string, n int) string {
	if n <= 0 || len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// buildDict is up to n bytes of text from the first docs of the corpus. Deflate looks back into it like into
// earlier text, and prefers the end, so the docs are in file order with the latest last.
func buildDict(corpus string, n int) []byte {
	f, err := os.Open(corpus)
	must(err)
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	var parts []string
	total := 0
	for sc.Scan() && total < n {
		if _, t, err := docText(sc.Bytes()); err == nil {
			parts = append(parts, t)
			total += len(t) + 1
		}
	}
	d := []byte(strings.Join(parts, "\n"))
	if len(d) > n {
		d = d[len(d)-n:]
	}
	return d
}

// codec turns doc text into a PIR row and back. One per worker (flate writers are not safe to share).
type codec struct {
	kind string
	dict []byte
	w    *flate.Writer
	buf  bytes.Buffer
}

func newCodec(kind string, dict []byte) *codec {
	c := &codec{kind: kind, dict: dict}
	if kind == "flate" {
		var err error
		c.w, err = flate.NewWriterDict(&c.buf, flate.BestCompression, dict)
		must(err)
	}
	return c
}

func (c *codec) encode(text string) []byte {
	if c.kind == "none" {
		return []byte(text)
	}
	c.buf.Reset()
	c.w.Reset(&c.buf) // keeps the dictionary
	_, err := c.w.Write([]byte(text))
	must(err)
	must(c.w.Close())
	return append([]byte(nil), c.buf.Bytes()...)
}

func (c *codec) decode(b []byte) (string, error) {
	if c.kind == "none" {
		return string(b), nil
	}
	r := flate.NewReaderDict(bytes.NewReader(b), c.dict)
	defer r.Close()
	out, err := io.ReadAll(r)
	return string(out), err
}

func toRow(enc []byte) []uint64 {
	var hdr [binary.MaxVarintLen64]byte
	h := binary.PutUvarint(hdr[:], uint64(len(enc)))
	b := make([]byte, (h+len(enc)+7)/8*8)
	copy(b, hdr[:h])
	copy(b[h:], enc)
	row := make([]uint64, len(b)/8)
	for i := range row {
		row[i] = binary.LittleEndian.Uint64(b[8*i:])
	}
	return row
}

// fromRow undoes toRow on a PIR answer (which may carry extra zero words up to the largest row).
func fromRow(row []uint64) ([]byte, error) {
	b := make([]byte, 8*len(row))
	for i, w := range row {
		binary.LittleEndian.PutUint64(b[8*i:], w)
	}
	n, h := binary.Uvarint(b)
	if h <= 0 || uint64(len(b)-h) < n {
		return nil, fmt.Errorf("bad row header")
	}
	return b[h : h+int(n)], nil
}

type corpusDB struct {
	ids       []string
	rows      [][]uint64
	rawBytes  uint64
	encBytes  uint64
	truncated int
	badJSON   int
	roundTrip int // docs decoded back at build time and compared with their text
}

// loadCorpus reads corpus.jsonl (line i = doc i) and encodes every doc into its row on GOMAXPROCS workers.
func loadCorpus(path, kind string, dict []byte, maxDocBytes int) *corpusDB {
	f, err := os.Open(path)
	must(err)
	defer f.Close()

	type job struct {
		idx  int
		line []byte
	}
	type result struct {
		idx       int
		id        string
		row       []uint64
		raw, enc  int
		truncated bool
		bad       bool
		checked   bool
	}
	nw := runtime.GOMAXPROCS(0)
	jobs := make(chan []job, 4*nw)
	results := make(chan []result, 4*nw)
	var wg sync.WaitGroup
	for w := 0; w < nw; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c := newCodec(kind, dict)
			for batch := range jobs {
				out := make([]result, len(batch))
				for i, j := range batch {
					id, text, err := docText(j.line)
					r := result{idx: j.idx, id: id, bad: err != nil, raw: len(text)}
					if t := truncate(text, maxDocBytes); len(t) < len(text) {
						text, r.truncated = t, true
					}
					enc := c.encode(text)
					r.enc, r.row = len(enc), toRow(enc)
					if j.idx%1000 == 0 { // spot-check the round trip
						got, err := fromRow(r.row)
						must(err)
						back, err := c.decode(got)
						if err != nil || back != text {
							logrus.Fatalf("doc %d (%s) does not decode back to its text (%v)", j.idx, id, err)
						}
						r.checked = true
					}
					out[i] = r
				}
				results <- out
			}
		}()
	}

	db := &corpusDB{}
	done := make(chan struct{})
	go func() {
		for batch := range results {
			for _, r := range batch {
				for r.idx >= len(db.rows) {
					db.rows = append(db.rows, nil)
					db.ids = append(db.ids, "")
				}
				db.rows[r.idx], db.ids[r.idx] = r.row, r.id
				db.rawBytes += uint64(r.raw)
				db.encBytes += uint64(r.enc)
				if r.truncated {
					db.truncated++
				}
				if r.bad {
					db.badJSON++
				}
				if r.checked {
					db.roundTrip++
				}
			}
		}
		close(done)
	}()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 256<<20)
	n := 0
	batch := make([]job, 0, 1024)
	start := time.Now()
	for sc.Scan() {
		batch = append(batch, job{n, append([]byte(nil), sc.Bytes()...)})
		n++
		if len(batch) == cap(batch) {
			jobs <- batch
			batch = make([]job, 0, 1024)
		}
		if n%5_000_000 == 0 {
			logrus.Infof("read %d docs (%s)", n, time.Since(start).Round(time.Second))
		}
	}
	must(sc.Err())
	if len(batch) > 0 {
		jobs <- batch
	}
	close(jobs)
	wg.Wait()
	close(results)
	<-done
	if db.badJSON > 0 {
		logrus.Warnf("%d corpus lines did not parse; they are empty rows", db.badJSON)
	}
	return db
}

func loadQueryIDs(path string) []string {
	f, err := os.Open(path)
	must(err)
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	var ids []string
	for sc.Scan() {
		var q struct {
			ID      string `json:"_id"`
			AltID   string `json:"id"`
			QueryID string `json:"query_id"`
		}
		if json.Unmarshal(sc.Bytes(), &q) != nil {
			continue
		}
		id := q.ID
		if id == "" {
			id = q.AltID
		}
		if id == "" {
			id = q.QueryID
		}
		ids = append(ids, id)
	}
	must(sc.Err())
	return ids
}

type query struct {
	id    string
	idx   []uint64 // doc rows to fetch, len k
	nReal int      // the first nReal are the method's answers (-ids), the rest random padding
}

// pickQueries gives every query k distinct doc rows: the method's top-k from -ids if given, padded with random docs.
func pickQueries(qids []string, answers map[string][]string, idOf map[string]int, n, k int, seed int64) []query {
	out := make([]query, 0, len(qids))
	missing := 0
	for _, qid := range qids {
		h := fnv.New64a()
		h.Write([]byte(qid))
		rng := rand.New(rand.NewSource(seed ^ int64(h.Sum64())))
		seen := make(map[uint64]struct{}, k)
		q := query{id: qid, idx: make([]uint64, 0, k)}
		for _, d := range answers[qid] {
			if len(q.idx) == k {
				break
			}
			i, ok := idOf[d]
			if !ok {
				missing++
				continue
			}
			if _, dup := seen[uint64(i)]; !dup {
				seen[uint64(i)] = struct{}{}
				q.idx = append(q.idx, uint64(i))
			}
		}
		q.nReal = len(q.idx)
		for len(q.idx) < min(k, n) {
			i := uint64(rng.Intn(n))
			if _, dup := seen[i]; !dup {
				seen[i] = struct{}{}
				q.idx = append(q.idx, i)
			}
		}
		out = append(out, q)
	}
	if missing > 0 {
		logrus.Warnf("%d answer doc IDs are not in the corpus; replaced by random docs", missing)
	}
	return out
}

func percentile(sorted []int, p float64) int {
	if len(sorted) == 0 {
		return 0
	}
	return sorted[min(len(sorted)-1, int(p/100*float64(len(sorted))))]
}

func main() {
	name := flag.String("name", "scifact", "dataset name (as for pir_app -name)")
	root := flag.String("dataset", "../datasets", "datasets root (as for pir_app -dataset)")
	ks := flag.String("k", "10,100", "comma-separated k: docs fetched per query (one run each, same DB)")
	outRoot := flag.String("out", "results_docfetch", "results root; each run writes <out>/docfetch_<name>_k<k>_<compress>/")
	compress := flag.String("compress", "flate", "'flate' (per-doc deflate with a preset dictionary) or 'none' (raw UTF-8)")
	dictBytes := flag.Int("dict-bytes", 32768, "flate: preset dictionary size (max 32768), kept by the client")
	maxDocBytes := flag.Int("max-doc-bytes", 0, "truncate doc text to this many bytes before compressing (0 = never)")
	idsFile := flag.String("ids", "", "optional results.json of a method ({qid: [docid...]}): fetch its top-k instead of random docs")
	maxQueries := flag.Int("max-queries", 0, "use only the first N queries (0 = all)")
	batchFlag := flag.Int("batch", 0, "PIR batch size (0 = k); partitions = batch/2")
	simulate := flag.Bool("simulate", false, "simulated PIR: same answers, bytes and client storage, no hints or XOR; timings -1")
	samples := flag.Int("samples", 3, "write the fetched text of the first N queries to sample_docs.json")
	seed := flag.Int64("seed", 42, "seed for the random docs")
	force := flag.Bool("force", false, "rerun k values whose metadata.json exists")
	flag.Parse()

	logrus.SetFormatter(&logrus.TextFormatter{FullTimestamp: true})
	if *compress != "flate" && *compress != "none" {
		logrus.Fatalf("-compress must be flate or none, not %q", *compress)
	}
	var kList []int
	for _, s := range strings.Split(*ks, ",") {
		k, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || k < 1 {
			logrus.Fatalf("bad -k %q", s)
		}
		kList = append(kList, k)
	}
	runName := func(k int) string {
		s := fmt.Sprintf("docfetch_%s_k%d_%s", *name, k, *compress)
		if *maxDocBytes > 0 {
			s += fmt.Sprintf("_max%d", *maxDocBytes)
		}
		return s
	}
	var todo []int
	for _, k := range kList {
		if _, err := os.Stat(filepath.Join(*outRoot, runName(k), "metadata.json")); err == nil && !*force {
			logrus.Infof("%s already has metadata.json, skipped (-force reruns)", runName(k))
			continue
		}
		todo = append(todo, k)
	}
	if len(todo) == 0 {
		return
	}
	if *simulate {
		pianopir.Simulate = true
		logrus.Warn("SIMULATED PIR: no hints, no XOR; server timings are written as -1")
	}

	corpusPath, queriesPath := paths(*root, *name)

	// ---- the DB
	var dict []byte
	if *compress == "flate" {
		dict = buildDict(corpusPath, min(*dictBytes, 32768))
	}
	start := time.Now()
	db := loadCorpus(corpusPath, *compress, dict, *maxDocBytes)
	buildTime := time.Since(start)
	n := len(db.rows)
	if n == 0 {
		logrus.Fatalf("no docs in %s", corpusPath)
	}
	sizes := make([]int, n)
	maxWords, totalWords := 0, 0
	for i, r := range db.rows {
		sizes[i] = 8 * len(r)
		maxWords = max(maxWords, len(r))
		totalWords += len(r)
	}
	sort.Ints(sizes)
	logrus.Infof("%d docs encoded in %s: text %.1f MB -> rows %.1f MB (%.1f B/doc avg, max %d B), %d truncated",
		n, buildTime.Round(time.Second), float64(db.rawBytes)/1e6, float64(totalWords*8)/1e6,
		float64(totalWords*8)/float64(n), maxWords*8, db.truncated)

	// ---- the queries
	qids := loadQueryIDs(queriesPath)
	var answers map[string][]string
	if *idsFile != "" {
		b, err := os.ReadFile(*idsFile)
		must(err)
		must(json.Unmarshal(b, &answers))
		qids = qids[:0]
		for q := range answers {
			qids = append(qids, q)
		}
		sort.Strings(qids)
	}
	if *maxQueries > 0 && len(qids) > *maxQueries {
		qids = qids[:*maxQueries]
	}
	if len(qids) == 0 {
		logrus.Fatalf("no queries (%s)", queriesPath)
	}
	var idOf map[string]int
	if answers != nil {
		idOf = make(map[string]int, n)
		for i, id := range db.ids {
			idOf[id] = i
		}
	}

	for _, k := range todo {
		run(k, runName(k), *outRoot, *name, *compress, *batchFlag, *simulate, *samples, *maxDocBytes,
			db, dict, sizes, maxWords, totalWords, buildTime, pickQueries(qids, answers, idOf, n, k, *seed), *idsFile)
		debug.FreeOSMemory()
	}
}

func run(k int, runName, outRoot, name, compress string, batchFlag int, simulate bool, samples, maxDocBytes int,
	db *corpusDB, dict []byte, sizes []int, maxWords, totalWords int, buildTime time.Duration, queries []query,
	idsFile string) {
	n := len(db.rows)
	batch := batchFlag
	if batch <= 0 {
		batch = k
	}
	batch = max(2, (batch+1)/2*2) // two lookups per partition
	for batch > 2 && n/(batch/2) < 64 {
		batch -= 2 // tiny corpora: keep partitions big enough for the Piano query budget
	}
	logrus.Infof("=== %s: %d queries x %d docs, PIR batch %d (%d partitions)", runName, len(queries), k, batch, batch/2)

	pir := pianopir.NewSimpleBatchPianoPIR(uint64(n), uint64(maxWords), uint64(maxWords)*8, uint64(batch),
		db.rows, 20, 1)
	prep := pir.Preprocessing()

	c := newCodec(compress, dict)
	var answerTime, decodeTime time.Duration
	fetched := make(map[string][]string, len(queries))
	sample := make(map[string][]map[string]string)
	ok, bad, real := 0, 0, 0
	start := time.Now()
	for qi, q := range queries {
		t0 := time.Now()
		ans, err := pir.Query(q.idx)
		answerTime += time.Since(t0)
		if err != nil {
			logrus.Fatalf("query %s: %v", q.id, err)
		}
		t0 = time.Now()
		got := make([]string, 0, q.nReal)
		for j, idx := range q.idx {
			want := db.rows[idx]
			a := ans[j]
			match := len(a) >= len(want)
			for w := 0; match && w < len(a); w++ {
				if w < len(want) {
					match = a[w] == want[w]
				} else {
					match = a[w] == 0
				}
			}
			var text string
			if match {
				enc, err := fromRow(a)
				if err == nil {
					text, err = c.decode(enc)
				}
				match = err == nil
			}
			if !match {
				bad++
				logrus.Errorf("query %s: doc %s (row %d) came back wrong", q.id, db.ids[idx], idx)
				continue
			}
			ok++
			if j < q.nReal || idsFile == "" {
				got = append(got, db.ids[idx])
				real++
			}
			if qi < samples {
				sample[q.id] = append(sample[q.id], map[string]string{"_id": db.ids[idx], "text": text})
			}
		}
		decodeTime += time.Since(t0)
		fetched[q.id] = got
		if (qi+1)%1000 == 0 {
			logrus.Infof("%d/%d queries (%s)", qi+1, len(queries), time.Since(start).Round(time.Second))
		}
	}

	nq := float64(len(queries))
	meta := pir.PrintInfo()
	fnum := func(key string) float64 { v, _ := strconv.ParseFloat(meta[key], 64); return v }
	pirMB := fnum("ClientStorageMB")
	meta["PIRClientStorageMB"] = meta["ClientStorageMB"]
	meta["ClientStorageMB"] = fmt.Sprint(pirMB + float64(len(dict))/1024/1024)
	set := func(key string, v any) { meta[key] = fmt.Sprint(v) }
	set("Method", "docfetch")
	set("Dataset", name)
	set("K", k)
	set("NumDocs", n)
	set("NumQueries", len(queries))
	set("BatchSize", batch)
	set("Compression", compress)
	set("DictBytes", len(dict))
	set("MaxDocBytes", maxDocBytes)
	set("TruncatedDocs", db.truncated)
	set("BadCorpusLines", db.badJSON)
	set("CorpusTextMB", float64(db.rawBytes)/1024/1024)
	set("CorpusEncodedMB", float64(db.encBytes)/1024/1024)
	set("DBUnpaddedMB", float64(totalWords*8)/1024/1024)
	set("DBPaddedMB", float64(n)*float64(maxWords*8)/1024/1024)
	set("AvgTextBytesPerDoc", float64(db.rawBytes)/float64(n))
	set("AvgRowBytes", float64(totalWords*8)/float64(n))
	for _, p := range []float64{50, 90, 99, 99.9} {
		set(fmt.Sprintf("RowBytesP%g", p), percentile(sizes, p))
	}
	set("MaxRowBytes", maxWords*8)
	set("BytesPerQuery", fnum("TotalByteSent")/nq)
	set("BytesPerQueryPadded", fnum("TotalByteSentPadded")/nq)
	set("LANTimePerQuery", fnum("TotalLANTime")/nq)
	set("WANTimePerQuery", fnum("TotalWANTime")/nq)
	set("DBBuildTime", buildTime.Seconds())
	set("PreprocessingTime", prep.Seconds())
	set("TotalAnswerTime", answerTime.Seconds())
	set("AnswerTimePerQueryMs", answerTime.Seconds()*1000/nq)
	set("ClientDecodeTimePerQueryMs", decodeTime.Seconds()*1000/nq)
	set("DocsFetched", ok)
	set("DocsFetchedFromIDs", real)
	set("DocsWrong", bad)
	set("RoundTripChecked", db.roundTrip)
	set("IDsFile", idsFile)
	set("Simulated", simulate)
	if simulate { // server work was not done: never let these pass for real timings
		for _, key := range []string{"PreprocessingTime", "MaintainenceTime", "TotalAnswerTime", "AnswerTimePerQueryMs"} {
			meta[key] = "-1"
		}
	}

	dir := filepath.Join(outRoot, runName)
	must(os.MkdirAll(dir, 0o755))
	writeJSON(filepath.Join(dir, "fetched.json"), fetched)
	writeJSON(filepath.Join(dir, "sample_docs.json"), sample)
	writeJSON(filepath.Join(dir, "metadata.json"), meta) // last: its presence means the run finished
	logrus.Infof("%s: %d docs fetched and verified, %d wrong; %.1f KB/query, client %.1f MB -> %s",
		runName, ok, bad, fnum("TotalByteSent")/nq/1024, pirMB, dir)
	if bad > 0 {
		os.Exit(1)
	}
}

func writeJSON(path string, v any) {
	f, err := os.Create(path)
	must(err)
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	must(enc.Encode(v))
	must(f.Close())
}

func must(err error) {
	if err != nil {
		logrus.Fatal(err)
	}
}
