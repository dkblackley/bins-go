package bins

// text_bins.go
// Text-retrieval mode for the bins (-text). Same single-DB layout as bins_old.go, but entry i holds the compressed
// text of every doc in bin i instead of their embeddings, and the client re-ranks what it gets with BM25.
//
// Each doc is analysed with the same analyzer the bins and queries use (letters only, lowercase, stopwords removed,
// Porter-stemmed), then coded with bins/textcodec: by default the sorted bag of stems with their tf (all BM25 needs),
// front-coded and arithmetic coded with a static order-2 character model. Decoding is integer arithmetic over the
// model's frequency counts (a few KB), not a word lookup. BM25's idf comes from a small count-min sketch of document
// frequencies (2 x 2^IDFBits bytes). Both are counted in the client storage.
//
// Bins have different sizes. Short entries are not padded in the DB, and the PIR accounting (pianopir) charges
// every response and hint its real length, since XORing the missing zero words is a no-op.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/blugelabs/bluge/analysis"
	"github.com/dkblackley/bins-go/bins/textcodec"
	"github.com/dkblackley/bins-go/globals"
	"github.com/dkblackley/bins-go/pianopir"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

// TextAnalyzer is the analyzer the bins use for docs and queries.
func TextAnalyzer() *analysis.Analyzer { return strictEnglishAnalyzer() }

// AnalyzeTerms returns the analysed terms of text, in order.
func AnalyzeTerms(an *analysis.Analyzer, text string) []string {
	toks := an.Analyze([]byte(text))
	out := make([]string, len(toks))
	for i, t := range toks {
		out[i] = string(t.Term)
	}
	return out
}

type TextBins struct {
	N           int
	DBEntrySize uint64 // bytes in the largest entry
	DBTotalSize uint64 // bytes over all entries (unpadded)
	Queries     map[string]globals.Query
	PIR         *pianopir.SimpleBatchPianoPIR

	Model  *textcodec.Model
	Bin    textcodec.BinParams
	IDF    *textcodec.IDFSketch
	stats  map[string]string
	config *globals.Args
}

func (v *TextBins) PIRPreprocess() time.Duration { return v.PIR.Preprocessing() }
func (v *TextBins) Preprocess()                  { v.PIR.Preprocessing() }
func (v *TextBins) GetBatchNums() (uint64, uint64, uint64) {
	return v.PIR.FinishedBatchNum, v.PIR.Config().BatchNumNeeded, v.PIR.SupportBatchNum
}

// GetMetaData adds the decoder model and IDF sketch to the client storage: the client has to keep them.
func (v *TextBins) GetMetaData() map[string]string {
	meta := v.PIR.PrintInfo()
	extra := float64(v.Model.SizeBytes()+v.IDF.SizeBytes()) / 1024 / 1024
	if pirMB, err := strconv.ParseFloat(meta["ClientStorageMB"], 64); err == nil {
		meta["PIRClientStorageMB"] = meta["ClientStorageMB"]
		meta["ClientStorageMB"] = fmt.Sprintf("%v", pirMB+extra)
	}
	if padMB, err := strconv.ParseFloat(meta["ClientStoragePaddedMB"], 64); err == nil {
		meta["ClientStoragePaddedMB"] = fmt.Sprintf("%v", padMB+extra)
	}
	for k, val := range v.stats {
		meta[k] = val
	}
	return meta
}

func (v *TextBins) DoSearch(QID string, _ int) (globals.Decodable, error) {
	query := v.Queries[QID]
	an := strictEnglishAnalyzer()
	terms := AnalyzeTerms(an, query.Text)

	indices := make([]uint64, len(terms))
	for i, t := range terms {
		indices[i] = hashTokenChoice(t, v.config.DChoice) % uint64(v.N)
	}
	if uint64(len(indices)) >= v.PIR.Config().BatchSize {
		logrus.Warnf("Too many indices in batch: %d for QID: %s - Possible corruption incoming", len(indices), QID)
	}
	results, err := v.PIR.Query(indices)
	return &TextEntry{entries: results, query: terms, db: v}, err
}

// TextEntry is one query's PIR answer: the raw bin entries. Decode returns the doc IDs in bin order, Rank re-ranks
// them with BM25 on the decoded text (client side).
type TextEntry struct {
	entries [][]uint64
	query   []string
	db      *TextBins
	docs    []textcodec.DecodedDoc // filled by decode
}

func (t *TextEntry) decode() {
	if t.docs != nil {
		return
	}
	seen := make(map[uint32]struct{})
	t.docs = make([]textcodec.DecodedDoc, 0)
	for _, e := range t.entries {
		if len(e) <= 1 && (len(e) == 0 || e[0] == 0) {
			continue // dropped lookup (batch collision) or empty bin
		}
		for _, d := range t.db.Model.DecodeBin(t.db.Bin, e) {
			if _, dup := seen[d.Idx]; dup {
				continue
			}
			seen[d.Idx] = struct{}{}
			t.docs = append(t.docs, d)
		}
	}
}

func (t *TextEntry) Decode(config *globals.Args) []string {
	t.decode()
	ids := make([]string, 0, len(t.docs))
	for _, d := range t.docs {
		if id, ok := config.DocIDMapPacmann[int(d.Idx)]; ok {
			ids = append(ids, id)
		} else {
			logrus.Warnf("Decoded doc index %d not in corpus map", d.Idx)
		}
	}
	return ids
}

// Rank returns the top-k doc IDs by BM25 against the query (unique analysed query terms).
func (t *TextEntry) Rank(config *globals.Args, k int) []string {
	t.decode()
	q := make([]string, 0, len(t.query))
	seenQ := make(map[string]struct{})
	for _, w := range t.query {
		if _, ok := seenQ[w]; !ok {
			seenQ[w] = struct{}{}
			q = append(q, w)
		}
	}
	type scored struct {
		idx   uint32
		score float64
	}
	sc := make([]scored, len(t.docs))
	for i, d := range t.docs {
		sc[i] = scored{d.Idx, t.db.IDF.BM25(q, d.Terms, d.Len, config.Text.K1, config.Text.B)}
	}
	sort.SliceStable(sc, func(i, j int) bool { return sc[i].score > sc[j].score })
	out := make([]string, 0, k)
	for _, s := range sc {
		if len(out) == k {
			break
		}
		if id, ok := config.DocIDMapPacmann[int(s.idx)]; ok {
			out = append(out, id)
		}
	}
	return out
}

// readCorpusText streams corpus.jsonl and calls fn(i, title+" "+text) for doc i (file order = doc index) on
// GOMAXPROCS workers. maxDocs <= 0 reads everything.
func readCorpusText(path string, maxDocs int, fn func(worker, idx int, text string)) int {
	f, err := os.Open(path)
	Must(err)
	defer f.Close()

	type job struct {
		idx  int
		line []byte
	}
	nw := runtime.GOMAXPROCS(0)
	jobs := make(chan []job, 4*nw)
	var wg sync.WaitGroup
	for w := 0; w < nw; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for batch := range jobs {
				for _, j := range batch {
					var d beirDoc
					if err := json.Unmarshal(j.line, &d); err != nil {
						logrus.Errorf("corpus line %d: %v", j.idx, err)
					}
					body := d.Text
					if body == "" {
						body = d.Abstract
					}
					fn(w, j.idx, d.Title+" "+body)
				}
			}
		}(w)
	}

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 64*1024*1024)
	n := 0
	batch := make([]job, 0, 1024)
	for sc.Scan() {
		if maxDocs > 0 && n >= maxDocs {
			break
		}
		batch = append(batch, job{n, append([]byte(nil), sc.Bytes()...)})
		n++
		if len(batch) == cap(batch) {
			jobs <- batch
			batch = make([]job, 0, 1024)
		}
	}
	Must(sc.Err())
	if len(batch) > 0 {
		jobs <- batch
	}
	close(jobs)
	wg.Wait()
	return n
}

// TrainTextModel trains the character model on the first trainDocs docs of the corpus.
func TrainTextModel(corpus string, p textcodec.Params, trainDocs int) *textcodec.Model {
	nw := runtime.GOMAXPROCS(0)
	trainers := make([][][]string, nw)
	ans := make([]*analysis.Analyzer, nw)
	for i := range ans {
		ans[i] = strictEnglishAnalyzer()
	}
	readCorpusText(corpus, trainDocs, func(w, _ int, text string) {
		trainers[w] = append(trainers[w], AnalyzeTerms(ans[w], text))
	})
	tr := textcodec.NewTrainer(p)
	for _, docs := range trainers {
		for _, d := range docs {
			tr.Add(d)
		}
	}
	return tr.Build()
}

// EncodeCorpusText codes every doc of the corpus (index = line number) and builds the IDF sketch.
func EncodeCorpusText(corpus string, m *textcodec.Model, idfBits uint, numDocs int) ([]textcodec.DocCode, *textcodec.IDFSketch) {
	nw := runtime.GOMAXPROCS(0)
	ans := make([]*analysis.Analyzer, nw)
	counters := make([]*textcodec.IDFCounter, nw)
	for i := range ans {
		ans[i] = strictEnglishAnalyzer()
		counters[i] = textcodec.NewIDFCounter(max(idfBits, 1))
	}
	codes := make([]textcodec.DocCode, numDocs)
	bar := progressbar.Default(int64(numDocs), "Compressing corpus text")
	readCorpusText(corpus, 0, func(w, idx int, text string) {
		terms := AnalyzeTerms(ans[w], text)
		counters[w].AddDoc(terms)
		if idx >= len(codes) {
			logrus.Errorf("corpus line %d past the %d docs in the ID map, skipped", idx, len(codes))
			return
		}
		b, n := m.EncodeDoc(terms)
		codes[idx] = textcodec.DocCode{Idx: uint32(idx), Bits: b, NBit: n} // each idx written by one worker only
		bar.Add(1)
	})
	bar.Finish()
	for _, c := range counters[1:] {
		counters[0].Merge(c)
	}
	return codes, counters[0].Build()
}

// MakeTextDb builds the text bins: same bins as MakeVecDbOld, each entry the compressed text of its docs.
func MakeTextDb(config *globals.Args) *TextBins {
	meta := config.DatasetMeta
	tc := config.Text
	params := textcodec.Params{Format: tc.Format, Order: tc.Order}
	Must(params.Validate())

	start := time.Now()
	model := TrainTextModel(meta.OriginalDir, params, tc.TrainDocs)
	logrus.Infof("Text model (%s, order %d) trained in %s, %d bytes", tc.Format, tc.Order, time.Since(start), model.SizeBytes())

	docMap, _ := MakeDocIDAndQueryIDMap(meta)
	codes, idf := EncodeCorpusText(meta.OriginalDir, model, tc.IDFBits, len(docMap))
	if tc.IDFBits == 0 {
		idf.Uniform() // no df table on the client: every term gets the same idf
	}
	var textBits uint64
	for _, c := range codes {
		textBits += uint64(c.NBit)
	}
	logrus.Infof("Compressed corpus: %d docs, %.2f MB, %.1f bytes/doc", len(codes), float64(textBits)/8/1024/1024,
		float64(textBits)/8/float64(max(len(codes), 1)))

	var DB [][]string
	var err error
	if config.Load {
		DB, err = ReadCSV(config.DataName + "_unigram_DB.csv")
		Must(err)
	} else {
		reader := OpenBinsIndex(meta)
		defer reader.Close()
		DB = MakeUnigramDB(reader, meta, config)
		if config.Save {
			Must(WriteCSV(config.DataName+"_unigram_DB.csv", DB))
		}
	}

	flipped := make(map[string]int, len(docMap))
	for key, value := range docMap {
		flipped[value] = key
	}
	docMap = nil

	bp := textcodec.NewBinParams(len(codes), int(config.DocsPerBin))
	sink := newRowSink("bins_text_db", len(DB)) // heap, or a file with -noram
	var next sync.WaitGroup
	nw := runtime.GOMAXPROCS(0)
	chunk := (len(DB) + nw - 1) / nw
	bar := progressbar.Default(int64(len(DB)), "Packing text bins into PIR entries")
	for w := 0; w < nw; w++ {
		lo, hi := min(w*chunk, len(DB)), min((w+1)*chunk, len(DB))
		next.Add(1)
		go func(lo, hi int) {
			defer next.Done()
			for i := lo; i < hi; i++ {
				if len(DB[i]) == 0 {
					continue
				}
				docs := make([]textcodec.DocCode, 0, len(DB[i]))
				for _, id := range DB[i] {
					idx, ok := flipped[id]
					if !ok || idx >= len(codes) {
						logrus.Warnf("doc %s not in corpus map", id)
						continue
					}
					docs = append(docs, codes[idx])
				}
				sink.Put(i, textcodec.EncodeBin(bp, docs))
				DB[i] = nil
				bar.Add(1)
			}
		}(lo, hi)
	}
	next.Wait()
	bar.Finish()
	codes = nil
	rawDB := sink.Finish()

	maxWords, totalWords, empty := 0, 0, 0
	for _, e := range rawDB {
		maxWords = max(maxWords, len(e))
		totalWords += len(e)
		if len(e) == 0 {
			empty++
		}
	}
	maxWords = max(maxWords, 1)
	DBSize := len(rawDB)

	BatchSize := OldBatchSize
	PartitionNum := BatchSize / 2
	PartitionSize := (DBSize + PartitionNum - 1) / PartitionNum
	if maxQ := uint64(math.Sqrt(float64(PartitionSize)) * math.Log(float64(PartitionSize))); maxQ < 4*2 {
		logrus.Errorf("BatchSize %d too large for DBSize %d", BatchSize, DBSize)
		BatchSize = int(maxQ)
	}
	pir := pianopir.NewSimpleBatchPianoPIR(uint64(DBSize), uint64(maxWords), uint64(maxWords)*8, uint64(BatchSize), rawDB, 20, 1)

	queries, err := LoadQueries(meta.Queries)
	Must(err)
	queryMap := make(map[string]globals.Query, len(queries))
	for _, q := range queries {
		queryMap[q.ID] = q
	}

	stats := map[string]string{
		"TextFormat":         tc.Format,
		"TextOrder":          strconv.Itoa(tc.Order),
		"TextBytesPerDoc":    fmt.Sprintf("%.2f", float64(textBits)/8/float64(max(len(flipped), 1))),
		"TextCorpusMB":       fmt.Sprintf("%.2f", float64(textBits)/8/1024/1024),
		"TextAvgEntryBytes":  fmt.Sprintf("%.1f", float64(totalWords*8)/float64(max(DBSize-empty, 1))),
		"TextMaxEntryBytes":  strconv.Itoa(maxWords * 8),
		"TextModelBytes":     strconv.Itoa(model.SizeBytes()),
		"TextIDFSketchBytes": strconv.Itoa(idf.SizeBytes()),
		"TextAvgDocLen":      fmt.Sprintf("%.2f", idf.AvgDL),
		"EmptyBins":          strconv.Itoa(empty),
	}
	for k, v := range stats {
		config.Metadata[k] = v
	}
	logrus.Infof("Text bins: %d bins (%d empty), entries avg %s B / max %d B, DB %.2f MB (unpadded)",
		DBSize, empty, stats["TextAvgEntryBytes"], maxWords*8, float64(totalWords*8)/1024/1024)

	return &TextBins{
		N:           DBSize,
		DBEntrySize: uint64(maxWords) * 8,
		DBTotalSize: uint64(totalWords) * 8,
		Queries:     queryMap,
		PIR:         pir,
		Model:       model,
		Bin:         bp,
		IDF:         idf,
		stats:       stats,
		config:      config,
	}
}
