package bins

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/csv"
	"encoding/hex"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

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
func index_stuff() {
	// 1) SCIFACT
	LoadBeirJSONL("/home/yelnat/Nextcloud/10TB-STHDD/datasets/scifact/corpus.jsonl", "index_scifact")
	logrus.SetLevel(logrus.InfoLevel)
	// 2) TREC-COVID
	LoadBeirJSONL("/home/yelnat/Nextcloud/10TB-STHDD/datasets/trec-covid/corpus.jsonl", "index_trec_covid")

	// 3) MSMARCO passage
	// loadMSMARCO("/home/yelnat/Nextcloud/10TB-STHDD/datasets/msmarco/collection.tsv", "index_msmarco")

	log.Println("✅  All indices built.")
}

// ----------------- evaluation ----------------------------------------------

func MrrAtK(idxPath, qrelsPath string, args globals.Args, k int) float64 {

	qs, err := LoadQueries(args)
	Must(err)
	rels, err := loadQrels(qrelsPath)
	Must(err)

	reader, err := bluge.OpenReader(bluge.DefaultConfig(idxPath))
	Must(err)
	defer reader.Close()

	bar := progressbar.Default(int64(len(qs)), fmt.Sprintf("eval %s", idxPath))

	var sumRR float64
	for _, q := range qs {

		// simple: match Query text against both title and body
		matchTitle := bluge.NewMatchQuery(q.Text).SetField("title")
		matchBody := bluge.NewMatchQuery(q.Text).SetField("body")
		boolean := bluge.NewBooleanQuery().
			AddShould(matchTitle).
			AddShould(matchBody)

		req := bluge.NewTopNSearch(k, boolean)
		it, err := reader.Search(context.Background(), req)

		Must(err)

		rr := 0.0
		for rank := 1; rank <= k; rank++ {
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

			if rels[q.ID][docID] > 0 {
				rr = 1.0 / float64(rank)
				break
			}
		}

		sumRR += rr
		bar.Add(1)
	}

	return sumRR / float64(len(rels))
}

func DecodeEntryToVectors(entry []uint64, Dim int) ([][]float32, error) {
	if Dim <= 0 {
		return nil, errors.New("DecodeEntryToVectors: Dim must be > 0")
	}
	if len(entry) == 0 {
		return nil, errors.New("DecodeEntryToVectors: empty entry")
	}

	bytesInEntry := len(entry) * 8
	bytesPerRow := Dim * 4
	if bytesPerRow == 0 {
		return nil, errors.New("DecodeEntryToVectors: invalid bytesPerRow (Dim?)")
	}

	// If caller didn't provide maxRowSize (or it's wrong), try to infer from entry size.
	rowsInEntry := bytesInEntry / bytesPerRow
	if bytesInEntry%bytesPerRow != 0 {
		return nil, fmt.Errorf(
			"DecodeEntryToVectors: entry size (%d bytes) is not a multiple of row size (%d bytes). "+
				"Dim mismatch? len(entry)=%d",
			bytesInEntry, bytesPerRow, len(entry),
		)
	}

	maxRowSize := rowsInEntry

	// Sanity check: expected words given (Dim, maxRowSize)
	expectedWords := (Dim * 4 * maxRowSize) / 8
	if expectedWords != len(entry) {
		// If the full entry contains fewer/more rows than declared maxRowSize, prefer the
		// rows actually present to avoid out-of-range.
		maxRowSize = rowsInEntry
		expectedWords = (Dim * 4 * maxRowSize) / 8
		if expectedWords != len(entry) {
			return nil, fmt.Errorf(
				"DecodeEntryToVectors: size mismatch. expectedWords=%d (Dim=%d, rows=%d), got len(entry)=%d. "+
					"Use the same (Dim,maxRowSize) used at encode time.",
				expectedWords, Dim, maxRowSize, len(entry),
			)
		}
	}

	// Rebuild the raw bytes from uint64 words (little-endian)
	entryBytes := make([]byte, len(entry)*8)
	for k := 0; k < len(entry); k++ {
		binary.LittleEndian.PutUint64(entryBytes[k*8:], entry[k])
	}

	// Slice back into rows and floats
	out := make([][]float32, maxRowSize)
	for r := 0; r < maxRowSize; r++ {
		start := r * bytesPerRow
		end := start + bytesPerRow
		rowBytes := entryBytes[start:end]

		row := make([]float32, Dim)
		for c := 0; c < Dim; c++ {
			off := c * 4
			bits := binary.LittleEndian.Uint32(rowBytes[off : off+4])
			row[c] = math.Float32frombits(bits)
		}
		out[r] = row
	}

	return out, nil
}

// TrimZeroRows removes rows that are entirely 0.0 (from padding).
func TrimZeroRows(vv [][]float32) [][]float32 {
	out := vv[:0]
RowLoop:
	for _, row := range vv {
		for _, x := range row {
			if x != 0 {
				out = append(out, row)
				continue RowLoop
			}
		}
		// all zeros -> skip
	}
	return out
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

// Takes in the original embeddings of the queries (assumed to be in order, i.e. first item has docID 1) and the answers
// to the queries, assumed to be a mapping of qid to answer
func FromEmbedToID(answers map[string][][]uint64, IDLookup map[string]int, dim int) map[string][]string {
	// Result: qid -> list of DocIDs (as strings, unchanged)
	queryIDstoDocIDS := make(map[string][]string, len(answers))

	debugOnce := true

	for qid, answer := range answers { // each answer = slices of entries in DB (per word)
		// Small capacity hint to reduce reallocs; tune if you know more about average rows/entry.
		dst := make([]string, 0, 8*len(answer))

		for k := 0; k < len(answer); k++ {
			entry := answer[k]

			if debugOnce {
				// util.go, before DecodeEntryToVectors, inspect 'entry'
				allZeroU64 := true
				for i := 0; i < len(entry) && i < 8; i++ {
					if entry[i] != 0 {
						allZeroU64 = false
						break
					}
				}
				logrus.Debugf("first 8 uint64 words allZero=%t (len(entry)=%d)", allZeroU64, len(entry))
			}

			f32Entry, err := DecodeEntryToVectors(entry, dim)
			Must(err)

			if debugOnce {
				// util.go, right after DecodeEntryToVectors(...)
				sum0 := 0.0
				if len(f32Entry) > 0 {
					for c := 0; c < dim && c < len(f32Entry[0]); c++ {
						sum0 += float64(f32Entry[0][c])
					}
				}
				logrus.Debugf("entry rows=%d firstRowSum=%.6f", len(f32Entry), sum0)
			}

			f32Entry = TrimZeroRows(f32Entry)

			for q := 0; q < len(f32Entry); q++ {
				key := HashFloat32s(f32Entry[q])
				docID, ok := IDLookup[key]
				if debugOnce {
					if !ok { // This should never be the case
						logrus.Errorf("BAD ID?? %d", docID)
						logrus.Errorf("Key: %s", key)
						logrus.Errorf("QueryID: %s", qid)
						logrus.Errorf("IDLookup Length: %d", len(IDLookup))
					}
				}

				dst = append(dst, strconv.Itoa(docID))
			}

			debugOnce = false

		}

		queryIDstoDocIDS[qid] = dst

	}

	return queryIDstoDocIDS
}

func ReadCSV(path string) ([][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

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
	defer f.Close()

	w := csv.NewWriter(f)
	// Optional: TSV instead of CSV
	// w.Comma = '\t'
	w.WriteAll(data)
	return w.Error()
}
