package bins

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/blugelabs/bluge"
	"github.com/dkblackley/bins-go/globals"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

// Used to convert beir data into formate for go bm25
//
// For scifact doc ID is just an integer, like: "40584205" or "10608397", sometimes it's a smaller number though, like:
// "3845894" or probably even "1"
// For TREC-COVID its strings like "1hvihwkz" or "3jolt83r". Bodies are just sentences of text.
//func index_stuff() {
//	// 1) SCIFACT
//	LoadBeirJSONL("/home/yelnat/Nextcloud/10TB-STHDD/datasets/scifact/corpus.jsonl", "index_scifact")
//	logrus.SetLevel(logrus.InfoLevel)
//	// 2) TREC-COVID
//	LoadBeirJSONL("/home/yelnat/Nextcloud/10TB-STHDD/datasets/trec-covid/corpus.jsonl", "index_trec_covid")
//
//	// 3) MSMARCO passage
//	// loadMSMARCO("/home/yelnat/Nextcloud/10TB-STHDD/datasets/msmarco/collection.tsv", "index_msmarco")
//
//	log.Println("✅  All indices built.")
//}

// Takes in a mapping from QID to DOCID and loads the query text and document text. Then Re-ranks all the docIDs based
// Upon the BM25 search. Returns a mapping with only the top-k (from config) documents
func BasicReRank(results map[string][]string, config globals.Args) map[string][]string {

	qids := make([]string, 0, len(results))
	docIDs := make([]string, 0, len(results))
	new_results := make(map[string][]string, len(results))

	for k, v := range results {
		qids = append(qids, k)

		for _, id := range v {
			docIDs = append(docIDs, id)
		}
	}

	metaData := GetDatasets(config.DatasetsDirectory, config.DataName)

	err := FilterJSONLByIDs(metaData.OriginalDir, "./temp_doc.jsonl", docIDs)
	Must(err)
	err = FilterJSONLByIDs(metaData.Queries, "./temp_q.jsonl", qids)
	Must(err)

	// NEW: build a real Bluge index directory from temp_doc.jsonl
	err = BuildBlugeIndexFromJSONL("./temp_doc.jsonl", "./temp_doc")
	Must(err)

	// Now do BLUGE on the remaining items
	qs, err := LoadQueries("./temp_q.jsonl")
	Must(err)
	rels, err := loadQrels(metaData.Qrels)
	Must(err)

	bar := progressbar.Default(int64(len(qs)), fmt.Sprintf("BM25 eval %s", config.DataName))

	// NEW: open the DIRECTORY, not the jsonl file
	reader, err := bluge.OpenReader(bluge.DefaultConfig("./temp_doc"))
	Must(err)
	defer reader.Close()

	defer func(reader *bluge.Reader) {
		err := reader.Close()
		if err != nil {
			log.Fatal(err)
		}
	}(reader)

	var sumRR float64

	if len(qs) <= 0 {
		log.Fatal("No results found")
	}
	for _, q := range qs {

		// simple: match Query text against both title and body
		matchTitle := bluge.NewMatchQuery(q.Text).SetField("title")
		matchBody := bluge.NewMatchQuery(q.Text).SetField("body")
		boolean := bluge.NewBooleanQuery().
			AddShould(matchTitle).
			AddShould(matchBody)

		req := bluge.NewTopNSearch(int(config.K), boolean)
		it, err := reader.Search(context.Background(), req)

		Must(err)

		rr := 0.0
		for rank := 1; rank <= int(config.K); rank++ {
			match, err := it.Next()
			if err != nil {
				break
			}
			if match == nil {
				break
			}

			// pull out the stored "_id" field instead of match.ID()
			var docID string
			err = match.VisitStoredFields(func(field string, value []byte) bool {
				if field == "_id" {
					docID = string(value)
					return false // stop visiting as soon as we have the id
				}
				return true // keep scanning other stored fields
			})
			Must(err)

			new_results[q.ID] = append(new_results[q.ID], docID)

			if rels[q.ID][docID] > 0 {
				rr = 1.0 / float64(rank)
				break
			}
		}

		sumRR += rr
		err = bar.Add(1)
		if err != nil {
			log.Fatal(err)
		}

	}

	logrus.Infof("MRR (post BM25 search): %f", sumRR/float64(len(rels)))

	// old temp jsonl cleanup is fine
	Must(os.Remove("./temp_doc.jsonl"))
	Must(os.Remove("./temp_q.jsonl"))

	// TODO: Fix this
	// Must(os.Remove("./temp_doc"))

	return new_results

}

// Takes in two pahs and a list of docIDS/qIDs and then selects those elements from inputPath before outputting ONLY them
// to outputPath.
func FilterJSONLByIDs(inputPath, outputPath string, docIDs []string) error {
	// Build a set for O(1) lookups
	idSet := make(map[string]struct{}, len(docIDs))
	for _, id := range docIDs {
		idSet[id] = struct{}{}
	}

	inFile, err := os.Open(inputPath)
	if err != nil {
		return err
	}
	defer func(inFile *os.File) {
		err := inFile.Close()
		if err != nil {
			log.Fatal(err)
		}
	}(inFile)

	outFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer func(outFile *os.File) {
		err := outFile.Close()
		if err != nil {
			log.Fatal(err)
		}
	}(outFile)

	scanner := bufio.NewScanner(inFile)
	writer := bufio.NewWriter(outFile)
	defer func(writer *bufio.Writer) {
		err := writer.Flush()
		if err != nil {
			log.Fatal(err)
		}
	}(writer)

	for scanner.Scan() {
		line := scanner.Bytes()

		var obj struct {
			ID string `json:"_id"`
		}
		if err := json.Unmarshal(line, &obj); err != nil {
			return err
		}

		if _, ok := idSet[obj.ID]; ok {
			if _, err := writer.Write(line); err != nil {
				return err
			}
			if err := writer.WriteByte('\n'); err != nil {
				return err
			}
		}
	}

	return scanner.Err()
}

func BuildBlugeIndexFromJSONL(jsonlPath, indexDir string) error {
	// Start fresh (important if you re-run)
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
	// Close explicitly so you can catch errors
	defer func() {
		_ = w.Close()
	}()

	f, err := os.Open(jsonlPath)
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	// JSONL lines can be long; bump scanner limit.
	buf := make([]byte, 0, 1024*1024)
	sc.Buffer(buf, 16*1024*1024) // 16MB max line

	batch := bluge.NewBatch()
	const flushEvery = 2000
	batchCount := 0
	totalInserted := 0

	for sc.Scan() {
		var d beirDoc
		if err := json.Unmarshal(sc.Bytes(), &d); err != nil {

			logrus.Tracef("json unmarshal failed: %w", err)
		}
		if d.ID == "" {
			continue
		}

		doc := bluge.NewDocument(d.ID)
		if d.Title != "" {
			doc.AddField(bluge.NewTextField("title", d.Title))
		}
		if d.Text != "" {
			doc.AddField(bluge.NewTextField("body", d.Text))
		}
		// store _id so your VisitStoredFields logic still works
		doc.AddField(bluge.NewKeywordField("_id", d.ID).StoreValue())

		batch.Insert(doc)
		batchCount++
		totalInserted++

		if batchCount >= flushEvery {
			if err := w.Batch(batch); err != nil {
				return err
			}
			batch = bluge.NewBatch()
			batchCount = 0
		}
	}

	if err := sc.Err(); err != nil {
		return err
	}

	// Flush remainder
	if batchCount > 0 {
		if err := w.Batch(batch); err != nil {
			return err
		}
	}

	// Ensure we actually created an index snapshot
	if totalInserted == 0 {
		return fmt.Errorf("no documents indexed (temp jsonl produced zero parseable docs?)")
	}

	// Close writer and surface errors (snapshot persistence happens here as well)
	if err := w.Close(); err != nil {
		return err
	}

	return nil
}

// ----------------- evaluation ----------------------------------------------

//func MrrAtK(idxPath, qrelsPath string, args globals.Args, k int) float64 {
//
//	meta := GetDatasets(args.DatasetsDirectory, args.DataName)
//	qs, err := LoadQueries(meta.Queries)
//	Must(err)
//	rels, err := loadQrels(qrelsPath)
//	Must(err)
//
//	reader, err := bluge.OpenReader(bluge.DefaultConfig(idxPath))
//	Must(err)
//	defer reader.Close()
//
//	bar := progressbar.Default(int64(len(qs)), fmt.Sprintf("eval %s", idxPath))
//
//	var sumRR float64
//	for _, q := range qs {
//
//		// simple: match Query text against both title and body
//		matchTitle := bluge.NewMatchQuery(q.Text).SetField("title")
//		matchBody := bluge.NewMatchQuery(q.Text).SetField("body")
//		boolean := bluge.NewBooleanQuery().
//			AddShould(matchTitle).
//			AddShould(matchBody)
//
//		req := bluge.NewTopNSearch(k, boolean)
//		it, err := reader.Search(context.Background(), req)
//
//		Must(err)
//
//		rr := 0.0
//		for rank := 1; rank <= k; rank++ {
//			match, err := it.Next()
//			if err != nil {
//				break
//			}
//			if match == nil {
//				break
//			}
//
//			// pull out the stored "_id" field instead of match.ID()
//			var docID string
//			err = match.VisitStoredFields(func(field string, value []byte) bool {
//				if field == "_id" {
//					docID = string(value)
//					return false // stop visiting as soon as we have the id
//				}
//				return true // keep scanning other stored fields
//			})
//			Must(err)
//
//			if rels[q.ID][docID] > 0 {
//				rr = 1.0 / float64(rank)
//				break
//			}
//		}
//
//		sumRR += rr
//		bar.Add(1)
//	}
//
//	return sumRR / float64(len(rels))
//}

func DecodeEntryToVectors(entry []uint64, Dim int) ([][]float32, error) {
	if Dim <= 0 {
		return nil, errors.New("DecodeEntryToVectors: Dim must be > 0")
	}
	if len(entry) == 0 {
		return nil, errors.New("DecodeEntryToVectors: empty entry")
	}

	wordsPerVec := (Dim + 1) / 2 // 2 float32 per uint64
	if len(entry)%wordsPerVec != 0 {
		return nil, fmt.Errorf(
			"decodeEntryToVectors: len(entry)=%d not divisible by wordsPerVec=%d (Dim=%d). "+
				"Wrong Dim or PIR entry sizing mismatch",
			len(entry), wordsPerVec, Dim,
		)
	}

	maxRowSize := len(entry) / wordsPerVec

	// Trim trailing *all-zero vectors* (not trailing zero words)
	actualRows := maxRowSize
	for actualRows > 0 {
		start := (actualRows - 1) * wordsPerVec
		end := start + wordsPerVec

		allZero := true
		for _, w := range entry[start:end] {
			if w != 0 {
				allZero = false
				break
			}
		}
		if !allZero {
			break
		}
		actualRows--
	}

	// Decode only the non-padding vectors
	out := make([][]float32, actualRows)
	pos := 0
	for r := 0; r < actualRows; r++ {
		row := make([]float32, Dim)
		d := 0
		for d < Dim {
			w := entry[pos]
			pos++

			row[d] = math.Float32frombits(uint32(w))
			d++
			if d < Dim {
				row[d] = math.Float32frombits(uint32(w >> 32))
				d++
			}
		}
		out[r] = row
	}

	return out, nil
}

func HashFloat32s(xs []float32) string {
	buf := make([]byte, 4*len(xs))
	for i, f := range xs {
		bits := math.Float32bits(f)
		binary.LittleEndian.PutUint32(buf[i*4:], bits)
	}

	sum := sha256.Sum256(buf)
	return hex.EncodeToString(sum[:])
}

//// Takes in the original embeddings of the queries (assumed to be in order, i.e. first item has docID 1) and the answers
//// to the queries, assumed to be a mapping of qid to answer
//func FromEmbedToID(answers map[string][][]uint64, IDLookup map[string]int, dim int) map[string][]string {
//	// Result: qid -> list of DocIDs (as strings, unchanged)
//	queryIDstoDocIDS := make(map[string][]string, len(answers))
//
//	debugOnce := true
//
//	for qid, answer := range answers { // each answer = slices of entries in DB (per word)
//		// Small capacity hint to reduce reallocs; tune if you know more about average rows/entry.
//		dst := make([]string, 0, 8*len(answer))
//
//		for k := 0; k < len(answer); k++ {
//			entry := answer[k]
//
//			if debugOnce {
//				// bins_util.go, before DecodeEntryToVectors, inspect 'entry'
//				allZeroU64 := true
//				for i := 0; i < len(entry) && i < 8; i++ {
//					if entry[i] != 0 {
//						allZeroU64 = false
//						break
//					}
//				}
//				logrus.Debugf("first 8 uint64 words allZero=%t (len(entry)=%d)", allZeroU64, len(entry))
//			}
//
//			f32Entry, err := DecodeEntryToVectors(entry, dim)
//			Must(err)
//
//			if debugOnce {
//				// bins_util.go, right after DecodeEntryToVectors(...)
//				sum0 := 0.0
//				if len(f32Entry) > 0 {
//					for c := 0; c < dim && c < len(f32Entry[0]); c++ {
//						sum0 += float64(f32Entry[0][c])
//					}
//				}
//				logrus.Debugf("entry rows=%d firstRowSum=%.6f", len(f32Entry), sum0)
//			}
//
//			f32Entry = TrimZeroRows(f32Entry)
//
//			for q := 0; q < len(f32Entry); q++ {
//				key := HashFloat32s(f32Entry[q])
//				docID, ok := IDLookup[key]
//				if debugOnce {
//					if !ok { // This should never be the case
//						logrus.Errorf("BAD ID?? %d", docID)
//						logrus.Errorf("Key: %s", key)
//						logrus.Errorf("QueryID: %s", qid)
//						logrus.Errorf("IDLookup Length: %d", len(IDLookup))
//					}
//				}
//
//				dst = append(dst, strconv.Itoa(docID))
//			}
//
//			debugOnce = false
//
//		}
//
//		queryIDstoDocIDS[qid] = dst
//
//	}
//
//	return queryIDstoDocIDS
//}

func ReadCSV(path string) ([][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func(f *os.File) {
		err := f.Close()
		if err != nil {
			log.Fatal(err)
		}
	}(f)

	r := csv.NewReader(f)
	// Allow ragged rows if you don't know column count ahead of time:
	r.FieldsPerRecord = -1
	return r.ReadAll()
}

// WriteCSV writes a [][]string as CSV.
func WriteCSV(path string, data [][]string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func(f *os.File) {
		err := f.Close()
		if err != nil {
			log.Fatal(err)
		}
	}(f)

	w := csv.NewWriter(f)
	// Optional: TSV instead of CSV
	// w.Comma = '\t'
	err = w.WriteAll(data)
	if err != nil {
		log.Fatal(err)
	}
	return w.Error()
}
