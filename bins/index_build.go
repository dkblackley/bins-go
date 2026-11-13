package bins

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sync"

	"github.com/blugelabs/bluge"
	"github.com/blugelabs/bluge/index"
	"github.com/sirupsen/logrus"
)

// indexBatchDocs is how many documents go into one Bluge batch (one new segment), as in the serial builder.
const indexBatchDocs = 2000

// preAnalyzedField is a text field that is analyzed once, by our own worker goroutines. Bluge's Writer.Batch
// analyzes every document again on only 4 goroutines (index.Config.NumAnalysisWorkers, which bluge.Config does
// not expose); here its second Analyze call just returns the cached result.
type preAnalyzedField struct {
	*bluge.TermField
	done    bool
	lastPos int
}

func (f *preAnalyzedField) Analyze(startOffset int) int {
	if !f.done {
		f.lastPos = f.TermField.Analyze(startOffset)
		f.done = true
	}
	return f.lastPos
}

// beirLineToDoc turns one corpus.jsonl line into the document the serial builder made (nil if it has no _id).
func beirLineToDoc(line []byte) *bluge.Document {
	var d beirDoc
	if err := json.Unmarshal(line, &d); err != nil {
		logrus.Tracef("json unmarshal failed: %v", err)
	}
	if d.ID == "" {
		return nil
	}
	doc := bluge.NewDocument(d.ID)
	if d.Title != "" {
		doc.AddField(&preAnalyzedField{TermField: binsTextField("title", d.Title)})
	}
	if d.Text != "" {
		doc.AddField(&preAnalyzedField{TermField: binsTextField("body", d.Text)})
	}
	doc.AddField(bluge.NewKeywordField("_id", d.ID).StoreValue())
	return doc
}

// BuildBlugeIndexFromJSONL builds the bins' Bluge index from corpus.jsonl on every core (BINS_INDEX_THREADS or
// GOMAXPROCS). Lines are read in chunks of indexBatchDocs; workers parse and analyze the chunks in parallel; the
// batches go into the index strictly in file order, so documents get the same order (and the same tie-breaks in
// searches) as with the old serial builder.
//
// Writing each batch's segment (ice.New) is single-threaded in Bluge and is about half the work, so the ordered
// build tops out around 2x. BINS_INDEX_UNORDERED=1 lets the workers write their batches themselves, in parallel:
// much faster on many cores, but documents land in a different order, so searches can break score ties
// differently than an index built the ordered way. Only use it for a new index that every run will share.
func BuildBlugeIndexFromJSONL(jsonlPath, indexDir string) error {
	if err := os.RemoveAll(indexDir); err != nil {
		return err
	}
	if err := os.MkdirAll(indexDir, 0o755); err != nil {
		return err
	}
	w, err := bluge.OpenWriter(bluge.DefaultConfig(indexDir))
	if err != nil {
		return err
	}
	f, err := os.Open(jsonlPath)
	if err != nil {
		_ = w.Close()
		return err
	}
	defer f.Close()

	workers := runtime.GOMAXPROCS(0)
	if v := os.Getenv("BINS_INDEX_THREADS"); v != "" {
		fmt.Sscanf(v, "%d", &workers)
	}
	workers = max(workers, 1)

	type chunk struct {
		seq   int
		lines [][]byte
	}
	type built struct {
		seq   int
		batch *index.Batch
		n     int
	}
	jobs := make(chan chunk, workers)
	results := make(chan built, workers)
	inFlight := make(chan struct{}, 2*workers+2) // bounds the chunks read but not yet written (memory)
	var readErr, workerErr error
	var errOnce sync.Once

	go func() { // reader
		defer close(jobs)
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024) // 16MB max line
		seq, lines := 0, make([][]byte, 0, indexBatchDocs)
		for sc.Scan() {
			lines = append(lines, append([]byte(nil), sc.Bytes()...))
			if len(lines) == indexBatchDocs {
				inFlight <- struct{}{}
				jobs <- chunk{seq, lines}
				seq, lines = seq+1, make([][]byte, 0, indexBatchDocs)
			}
		}
		readErr = sc.Err()
		if len(lines) > 0 {
			inFlight <- struct{}{}
			jobs <- chunk{seq, lines}
		}
	}()

	unordered := os.Getenv("BINS_INDEX_UNORDERED") == "1"
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for c := range jobs {
				b, n := bluge.NewBatch(), 0
				for _, line := range c.lines {
					if doc := beirLineToDoc(line); doc != nil {
						doc.Analyze() // the expensive part, cached in the preAnalyzedFields
						b.Insert(doc)
						n++
					}
				}
				if unordered { // write it here, in parallel with the other workers
					if n > 0 {
						if err := w.Batch(b); err != nil {
							errOnce.Do(func() { workerErr = err })
						}
					}
					b = nil
				}
				results <- built{c.seq, b, n}
			}
		}()
	}
	go func() { wg.Wait(); close(results) }()

	// writer: introduce the batches in file order
	pending := map[int]built{}
	next, total := 0, 0
	var batchErr error
	for r := range results {
		pending[r.seq] = r
		for {
			b, ok := pending[next]
			if !ok {
				break
			}
			delete(pending, next)
			next++
			if b.batch != nil && b.n > 0 && batchErr == nil {
				batchErr = w.Batch(b.batch)
			}
			total += b.n
			<-inFlight
			if next%500 == 0 {
				logrus.Infof("indexed %d docs", total)
			}
		}
	}
	if batchErr == nil {
		batchErr = workerErr
	}
	if batchErr != nil {
		_ = w.Close()
		return batchErr
	}
	if readErr != nil {
		_ = w.Close()
		return readErr
	}
	if total == 0 {
		_ = w.Close()
		return fmt.Errorf("no documents indexed (temp jsonl produced zero parseable docs?)")
	}
	logrus.Infof("indexed %d docs with %d workers (unordered=%v)", total, workers, unordered)
	return w.Close() // snapshot persistence happens here as well
}
