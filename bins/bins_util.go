package bins

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"strings"

	"github.com/blugelabs/bluge"
	"github.com/dkblackley/bins-go/globals"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

func MakeDocIDAndQueryIDMap(meta globals.DatasetMetadata) (map[int]string, map[int]string) {
	docs, err := LoadCorpus(meta.OriginalDir)
	Must(err)

	docIDMap := make(map[int]string, len(docs))
	for i, doc := range docs {
		docIDMap[i] = doc.ID
	}

	qs, err := LoadQueries(meta.Queries)
	Must(err)

	queryIDMap := make(map[int]string, len(qs))
	for i, q := range qs {
		queryIDMap[i] = q.ID
	}

	return docIDMap, queryIDMap
}

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

	metaData := config.DatasetMeta

	err := FilterJSONLByIDs(metaData.OriginalDir, config.SearchType+"temp_doc.jsonl", docIDs)
	Must(err)
	err = FilterJSONLByIDs(metaData.Queries, config.SearchType+"temp_q.jsonl", qids)
	Must(err)

	// NEW: build a real Bluge index directory from temp_doc.jsonl
	err = BuildBlugeIndexFromJSONL(config.SearchType+"temp_doc.jsonl", config.SearchType+"temp_doc")
	Must(err)

	// Now do BLUGE on the remaining items
	qs, err := LoadQueries(config.SearchType + "temp_q.jsonl")
	Must(err)
	rels, err := LoadQrels(metaData.Qrels)
	Must(err)

	bar := progressbar.Default(int64(len(qs)), fmt.Sprintf("BM25 eval %s", config.DataName))

	// NEW: open the DIRECTORY, not the jsonl file
	reader, err := bluge.OpenReader(bluge.DefaultConfig(config.SearchType + "temp_doc"))
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
	Must(os.Remove(config.SearchType + "temp_doc.jsonl"))
	Must(os.Remove(config.SearchType + "temp_q.jsonl"))

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
		line := append([]byte(nil), scanner.Bytes()...)

		var obj struct {
			ID      string `json:"_id"`
			AltID   string `json:"id"`
			QueryID string `json:"query_id"`
		}

		if err := json.Unmarshal(line, &obj); err != nil {
			logrus.Errorf("Error unmarshalling JSON: %v", err)
			return err // I think this might be causing issues
		}

		id := obj.ID
		if id == "" {
			if obj.AltID != "" {
				id = obj.AltID
			} else {
				id = obj.QueryID
			}
		}
		if id == "" {
			continue
		}

		if _, ok := idSet[id]; ok {
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

		id := strings.Clone(d.ID)
		if id == "" {
			continue
		}

		title := strings.Clone(d.Title)
		text := strings.Clone(d.Text)

		doc := bluge.NewDocument(id)
		if title != "" {
			doc.AddField(bluge.NewTextField("title", title))
		}
		if text != "" {
			doc.AddField(bluge.NewTextField("body", text))
		}
		// store _id so your VisitStoredFields logic still works
		doc.AddField(bluge.NewKeywordField("_id", id).StoreValue())

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

func MakeLookup(meta globals.DatasetMetadata, dbsize, dimensions int) map[[32]byte]string {

	// Thankfull scifact embeddings are in the same order as vec... TODO: for marco
	docs, er := LoadCorpus(meta.OriginalDir)
	Must(er)

	IDLookup := make(map[[32]byte]string)
	vectors, err := globals.LoadFloat32MatrixFromNpy(meta.Vectors.CorpusVec, dbsize, dimensions)

	bar := progressbar.NewOptions64(
		int64(len(vectors)),
		progressbar.OptionSetDescription("Making map"),
		progressbar.OptionShowElapsedTimeOnFinish(),
	)
	Must(err)
	for i := 0; i < len(docs); i++ {
		docID := docs[i].ID
		ID := HashFloat32s(vectors[i])
		IDLookup[ID] = docID
		bar.Add(1)
	}

	bar.Finish()

	return IDLookup
}

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

func HashFloat32s(xs []float32) [32]byte {
	buf := make([]byte, 4*len(xs))
	for i, f := range xs {
		bits := math.Float32bits(f)
		binary.LittleEndian.PutUint32(buf[i*4:], bits)
	}

	sum := sha256.Sum256(buf)
	return sum
}
