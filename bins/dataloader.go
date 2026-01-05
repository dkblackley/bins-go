package bins

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/blugelabs/bluge"
	"github.com/sirupsen/logrus"
)

// ---------- small helpers ---------------------------------------------------

func Must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// ---------- simple BEIR JSONL loader ---------------------------------------

type beirDoc struct {
	ID    string `json:"_id"`
	Title string `json:"title"`
	Text  string `json:"text"`
	// These fields should always be empty, I just include it to avoid decoding errors
	Abstract string `json:"abstract"` // used by SciFact
	Metadata string `json:"metadata"`
}

//func LoadBeirJSONL(path, indexDir string) {
//	f, err := os.Open(path)
//	Must(err)
//	defer f.Close()
//
//	var counter = 0
//
//	cfg := bluge.DefaultConfig(indexDir)
//	w, err := bluge.OpenWriter(cfg)
//	Must(err)
//	defer w.Close()
//
//	bar := progressbar.Default(-1, "index "+indexDir) // unknown total
//
//	sc := bufio.NewScanner(f)
//	sc.Buffer(make([]byte, 1024*1024), 10*1024*1024) // 1 MiB buf, 10 MiB max
//
//	for sc.Scan() {
//		bar.Add64(int64(len(sc.Bytes()) + 1)) // +1 for '\n'
//
//		// use a Decoder so we can catch unknown fields
//		raw := sc.Bytes()
//		dec := json.NewDecoder(bytes.NewReader(sc.Bytes()))
//		dec.DisallowUnknownFields()
//
//		var d beirDoc
//		if err := dec.Decode(&d); err != nil {
//			// if it's an unknown‐field error, log it and continue
//			if strings.HasPrefix(err.Error(), "json: unknown field") {
//				logrus.Tracef("⚠️  unknown JSON field in line: %v", err)
//				logrus.Tracef("Raw JSON line: %s", raw)
//			}
//		}
//
//		// now index as before
//		doc := bluge.NewDocument(d.ID)
//		doc.AddField(bluge.NewTextField("title", d.Title))
//
//		body := d.Text
//		if body == "" {
//			body = d.Abstract
//		}
//		doc.AddField(bluge.NewTextField("body", body))
//		doc.AddField(bluge.NewKeywordField("dataset", indexDir))
//
//		Must(w.Insert(doc))
//		counter++
//	}
//	if err := sc.Err(); err != nil {
//		log.Fatal(err)
//	}
//	bar.Finish()
//
//	logrus.Debugf("Total documents: %d", counter)
//}

type qrels map[string]map[string]int

func LoadCorpus(path string) ([]beirDoc, error) {
	f, err := os.Open(path)
	Must(err)
	defer f.Close()

	var counter = 0

	var ds []beirDoc
	sc := bufio.NewScanner(f)

	sc.Buffer(make([]byte, 1024), 10*1024*1024) // max 10 mib, should be fine (I hope)
	for sc.Scan() {
		raw := sc.Bytes()
		dec := json.NewDecoder(bytes.NewReader(sc.Bytes()))
		dec.DisallowUnknownFields()

		var d beirDoc
		if err := dec.Decode(&d); err != nil {
			// if it's an unknown‐field error, log it and continue
			if strings.HasPrefix(err.Error(), "json: unknown field") {
				logrus.Errorf("⚠️  unknown JSON field in line: %v", err)
				logrus.Errorf("Raw JSON line: %s", raw)
			}
		}

		body := d.Text
		if body == "" {
			body = d.Abstract
		}

		document := beirDoc{
			ID:    d.ID,
			Title: d.Title,
			Text:  body,
		}

		ds = append(ds, document)
		counter++
	}
	return ds, sc.Err()
}

type Query struct {
	ID      string `json:"_id"`
	AltID   string `json:"id"`       // Add this to catch "id"
	QueryID string `json:"query_id"` // Add this to catch "query_id"
	Text    string `json:"text"`
	// Metadata string `json:"metadata"`
}

func LoadQueries(query_path string) ([]Query, error) {

	f, err := os.Open(query_path)

	if err != nil {
		return nil, err
	}
	defer f.Close()

	var counter = 0

	var qs []Query
	sc := bufio.NewScanner(f)
	// allow long queries
	sc.Buffer(make([]byte, 1024), 1024*1024)
	for sc.Scan() {
		var q Query

		if err := json.Unmarshal(sc.Bytes(), &q); err == nil {

			// LOGIC FIX: Normalize the ID
			if q.ID == "" {
				if q.AltID != "" {
					q.ID = q.AltID
				} else if q.QueryID != "" {
					q.ID = q.QueryID
				}
			}

			// SAFETY CHECK: Skip or Warn if still empty
			if q.ID == "" {
				logrus.Warnf("Skipping query with empty ID: %s", string(sc.Bytes()))
				continue
			}

			qs = append(qs, q)
			counter++
		} else {
			logrus.Errorf("Query JSON parse error: %v", err)
		}
	}
	return qs, sc.Err()
}

func loadQrels(path string) (qrels, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	rel := make(qrels)
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.Fields(sc.Text())
		if len(line) < 3 {
			continue
		}
		qid, docid, score := line[0], line[1], line[2]
		v, _ := strconv.Atoi(score)
		if v <= 0 {
			continue
		}
		if _, ok := rel[qid]; !ok {
			rel[qid] = make(map[string]int)
		}
		rel[qid][docid] = v
	}
	return rel, sc.Err()
}

// StringsToUint64Grid encodes []string -> [][]uint64. Strings are easy to make into bytes, but awkward to handle as
// Arrays of uint64, as a result I pack multiple bytes into a uint64 instead of casting bytes into uint64s. I'm not
// Sure if this is an easier or harder solution...
// Each row: [ length | packed bytes ... | zero padding ... ]
func StringsToUint64Grid(strs []string) ([][]uint64, int, error) {
	maxBytes := 0
	for _, s := range strs {
		n := len(s)
		//if n > 255 {
		//	return nil, 0, fmt.Errorf("string too long (%d bytes): %q", n, s)
		//}
		if n > maxBytes {
			maxBytes = n
		}
	}

	// How many uint64 blocks we need to store `maxBytes` bytes + 1 for length and made a multiple of 4 for
	// pacmann specific implementation
	//((n + 3) / 4) * 4
	blocksPerRow := (((1 + (maxBytes+7)/8) + 3) / 4) * 4

	grid := make([][]uint64, len(strs))
	for i, s := range strs {
		row := make([]uint64, blocksPerRow)
		row[0] = uint64(len(s))

		// Fill payload bytes into a temporary buffer
		tmp := make([]byte, (blocksPerRow-1)*8)
		copy(tmp, []byte(s))

		// Pack 8 bytes into each uint64 (big-endian)
		for j := 0; j < blocksPerRow-1; j++ {
			offset := j * 8
			row[j+1] = binary.BigEndian.Uint64(tmp[offset : offset+8])
		}
		grid[i] = row
	}

	return grid, blocksPerRow, nil
}

// Uint64GridToStrings decodes [][]uint64 -> []string.
func Uint64GridToStrings(grid [][]uint64) ([]string, error) {
	if len(grid) == 0 {
		return nil, nil
	}
	blocksPerRow := len(grid[0])
	for i, r := range grid {
		if len(r) != blocksPerRow {
			return nil, fmt.Errorf("row %d has mismatched length: got %d, want %d", i, len(r), blocksPerRow)
		}
	}

	out := make([]string, len(grid))
	// Make a buffer to start unpacking the bytes
	buf := make([]byte, (blocksPerRow-1)*8)

	for i, r := range grid {
		n := int(r[0])
		if n > (blocksPerRow-1)*8 {
			return nil, fmt.Errorf("row %d length %d exceeds capacity %d", i, n, (blocksPerRow-1)*8)
		}

		// Unpack bytes from uint64 blocks
		for j := 0; j < blocksPerRow-1; j++ {
			binary.BigEndian.PutUint64(buf[j*8:(j+1)*8], r[j+1])
		}

		out[i] = string(buf[:n])
	}

	return out, nil
}

// ----------------- Takes in the dataset and returns it in a format that is acceptable for PIR -----------------------

func PirPreprocessAndLoadData(idxPath string) [][]uint64 {
	// If you're not reading data from file and just have an array of strings, you can simply do:
	// bytesID, _, _ := StringsToUint64Grid(DB)

	// Open a reader on the index
	reader, err := bluge.OpenReader(bluge.DefaultConfig(idxPath))
	Must(err)
	defer reader.Close()

	// Match ALL documents, and ask for ALL matches (unbounded iterator)
	q := bluge.NewMatchAllQuery()
	req := bluge.NewAllMatches(q)

	it, err := reader.Search(context.Background(), req)
	Must(err)

	IDArray := make([]string, 0)

	for {
		dm, err := it.Next()
		Must(err)
		if dm == nil {
			break // no more docs
		}

		// Pull the stored "_id" for this document
		var docID string
		err = dm.VisitStoredFields(func(field string, value []byte) bool {
			if field == "_id" {
				docID = string(value)
				return false // stop visiting once we have the id
			}
			return true
		})
		Must(err)

		//fmt.Println(docID)

		IDArray = append(IDArray, docID)
	}

	bytesID, _, _ := StringsToUint64Grid(IDArray)

	return bytesID

}
