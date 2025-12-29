package bins

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"

	"github.com/blugelabs/bluge"
	"github.com/blugelabs/bluge/analysis"
	"github.com/dkblackley/bins-go/globals"
	"github.com/dkblackley/bins-go/pianopir"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

func GetDatasets(root, name string) DatasetMetadata {

	if name == "msmarco" {
		return DatasetMetadata{
			"Marco",
			"index_marco",
			root + "/msmarco/corpus.jsonl",
			root + "/msmarco/queries.dev.small.jsonl",
			root + "/msmarco/qrels/dev.tsv",
			root + "/Son/my_vectors_192.npy", // Change this when debugging as it's a big file
		}
	} else if name == "scifact" {
		return DatasetMetadata{
			"SciFact",
			"index_scifact", // index folders created earlier
			root + "/scifact/corpus.jsonl",
			root + "/scifact/queries.jsonl",
			root + "/scifact/qrels/test.tsv",
			"",
		}
	} else if name == "debug" {
		logrus.Debugf("Using debug dataset")
		return DatasetMetadata{
			"Marco",
			"index_marco",
			root + "/msmarco/corpus_debug.jsonl",
			root + "/msmarco/queries.dev.small.jsonl",
			root + "/msmarco/qrels/dev.tsv",
			root + "/Son/my_vectors_192_debug.npy",
		}
	} else {
		return DatasetMetadata{
			"TREC-COVID",
			"index_trec_covid",
			root + "/trec-covid/corpus.jsonl",
			root + "/trec-covid/queries.jsonl",
			root + "/trec-covid/qrels/test.tsv",
			"",
		}
	}
}

type VecBins struct {
	N                    int              // Number of Bins
	Dimensions           int              // Dimension of vectors
	EntrySize            int              // number of vectors in a row (Size of one entry)
	DBEntrySize          uint64           // Number of bytes in an entry
	DBTotalSize          uint64           // in bytes
	Queries              map[string]Query // A mapping from QID to query
	EnglishTokenAnalyzer *analysis.Analyzer
	PIR                  *pianopir.SimpleBatchPianoPIR
	MaxRowSize           uint

	rawDB  [][]uint64
	config globals.Args
}

func (v VecBins) Decode(answers map[string][][]uint64, config globals.Args) map[string][]string {
	// Each input here is going to be a map of QID to a 2d array of uint64s. We want to produce a map of QID to top-k
	// (larger than k in our case) docIDs.

	metaData := GetDatasets(config.DatasetsDirectory, config.DataName)

	IDLookup := make(map[string]int)
	bm25Vectors, err := LoadFloat32MatrixFromNpy(metaData.Vectors, int(config.DBSize), int(config.Dimensions))
	Must(err)
	for i := 0; i < len(bm25Vectors); i++ {
		ID := HashFloat32s(bm25Vectors[i])
		IDLookup[ID] = i
	}

	docIDs := make(map[string][]string)
	empty := 0

	for qid, results := range answers {
		for i := 0; i < len(results); i++ {
			singleResult := results[i]
			if len(singleResult) == 1 {
				logrus.Warnf("Got an empty result: %v - Possibly missed and entry", singleResult)
				empty++
				if empty == len(results) {
					logrus.Errorf("All results were empty!!!!")

				}
				continue
			}
			multipleVectors, err := DecodeEntryToVectors(singleResult, 192)
			Must(err)

			// TODO: Remove this when not debug
			if len(multipleVectors) > 0 {
				// Check if the first vector is all zeros
				isZero := true
				for _, val := range multipleVectors[0] {
					if val != 0 {
						isZero = false
						break
					}
				}
				if isZero {
					logrus.Warnf("WARNING: Decoded vector is ALL ZEROS for QID %s", qid)
				}
			}

			for j := 0; j < len(multipleVectors); j++ {
				ID := HashFloat32s(multipleVectors[j])
				docID, ok := IDLookup[ID]
				if !ok {
					logrus.Warnf("Vector hash not found: %s", ID)
					continue
				}
				docIDs[qid] = append(docIDs[qid], strconv.Itoa(docID))

			}
		}
	}

	logrus.Debugf("Number of empty: %d over all  %d", empty, len(answers))
	return docIDs

}

func (v VecBins) GetBatchPIRInfo() *pianopir.SimpleBatchPianoPIR {
	return v.PIR
}

func (v VecBins) Preprocess() {
	v.PIR.Preprocessing()
}

func (v VecBins) DoSearch(QID string, _ int) ([][]uint64, error) {
	indices := v.MakeIndices(QID)

	if uint64(len(indices)) >= 32 { // TODO: pass batchsize in args to checl
		logrus.Warnf("Too many indices in batch: %d for QID: %s - Possible corruption incoming", len(indices), QID)
	}
	results, err := v.PIR.Query(indices)

	//TODO: something with K
	return results, err
}

func (v VecBins) MakeIndices(QID string) []uint64 {

	query := v.Queries[QID]

	tokeniser := strictEnglishAnalyzer()
	tokens := tokeniser.Analyze([]byte(query.Text))

	indices := make([]uint64, len(tokens))
	for i, t := range tokens {
		indices[i] = hashTokenChoice(fmt.Sprintf("%s", t.Term), v.config.DChoice) % uint64(v.N)
	}

	return indices

}

// MakeVecDb Takes in args from command line and then outputs a 'VecBins' object that implements the functions required for
// binsDB.
func MakeVecDb(config globals.Args) VecBins {

	metaData := GetDatasets(config.DatasetsDirectory, config.DataName)

	logrus.Debugf("Loading data from: %s and %s", metaData.Vectors, metaData.IndexDir)

	// TODO: Uncomment when back
	//if config.Vectors { // If we want to lead npy vectors
	bm25Vectors, err := LoadFloat32MatrixFromNpy(metaData.Vectors, int(config.DBSize), int(config.Dimensions))
	logrus.Infof("Size of vectors: %d", len(bm25Vectors))
	Must(err)
	var DB [][]string
	if config.Load {
		// TODO: make this dynamic
		DB, err = ReadCSV(config.DataName + "_unigram_DB.csv")
		Must(err)
		logrus.Debugf("Loaded DB from %s", config.DataName+"_unigram_DB.csv")

	} else {
		reader, _ := bluge.OpenReader(bluge.DefaultConfig(metaData.IndexDir))
		defer reader.Close()
		DB = MakeUnigramDB(reader, metaData, config)
		Must(err)

		if config.Save {
			err = WriteCSV(config.DataName+"_unigram_DB.csv", DB)
			Must(err)
			logrus.Debugf("Saved DB to %s", config.DataName+"_unigram_DB.csv")
		}

	}

	if config.DebugLevel >= 1 {
		nonEmpty, empty := 0, 0
		for i := range DB {
			if len(DB[i]) == 0 {
				empty++
			} else {
				nonEmpty++
			}
		}

		logrus.Debugf("CSV bins: non-empty=%d empty=%d total=%d", nonEmpty, empty, len(DB))
	}

	// Padding is now done dynamically...
	//pad := make([]float32, config.Dimensions)
	maxRowSize := 0
	redundancy := 0
	for _, e := range DB {
		if len(e) > maxRowSize {
			maxRowSize = len(e)
		}
	}

	newDb := make([][][]float32, 0, len(DB))
	for _, entry := range DB {
		row := make([][]float32, 0, len(entry))
		// Add the vectors to the row
		for j := 0; j < len(entry); j++ {
			id, err := strconv.ParseUint(entry[j], 10, 32)
			id64 := uint(id)
			Must(err)
			// This shouldn't do anything unless you're debugging!
			id64 = id64 % config.DBSize
			row = append(row, bm25Vectors[id64]) // shares the row slice; no copy
		}
		// Pad the row for all the missing vectors
		//for len(row) < maxRowSize {
		//	redundancy++
		//	row = append(row, pad) // shared, no per-cell alloc
		//}
		newDb = append(newDb, row)
	}

	if config.DebugLevel >= 1 {
		wordsPerEntry := (uint64(config.Dimensions) * 4 * uint64(maxRowSize)) / 8
		logrus.Debugf("Row layout: config.Dimensions=%d, maxRowSize=%d, wordsPerEntry=%d", config.Dimensions, maxRowSize, wordsPerEntry)

		b := uint64(len(newDb)) * uint64(maxRowSize) * uint64(config.Dimensions) * 4
		logrus.Debugf("New DB size: %.2f MiB (%d bytes)", float64(b)/(1<<20), b)

		logrus.Debugf("Marco vectors: %.2f GiB", float64(config.DBSize*config.Dimensions*4)/(1<<30))
		logrus.Debugf("Max row size: %d", maxRowSize)
		logrus.Debugf("Padded files %d", redundancy)
	}

	// PIR setup
	// start := time.Now()

	binPir := ProcessVecDB(config, uint(maxRowSize), newDb)
	//end := time.Now()
	//
	//logrus.Infof("Preprocessing took %s", end.Sub(start))

	meta := GetDatasets(config.DatasetsDirectory, config.DataName)
	queires, err := LoadQueries(meta.Queries)
	Must(err)
	queryMap := make(map[string]Query)
	for q := range len(queires) {
		qid := queires[q].ID
		queryMap[qid] = queires[q]

	}
	binPir.Queries = queryMap
	binPir.EnglishTokenAnalyzer = strictEnglishAnalyzer()

	return binPir

}

func ProcessVecDB(config globals.Args, maxRowSize uint, vectorsInBins [][][]float32) VecBins {
	//DBEntrySize := config.Dimensions * 4 * maxRowSize // bytes per DB entry (maxRowSize vectors × config.Dimensions float32s)
	DBSize := len(vectorsInBins)

	// A single 'word' should be how many uint64s are required to re-make the entry (divide by 8 because uint64 is 8 bytes)
	//wordsPerEntry := DBEntrySize / 8

	// I think just DBsize is big enough but I might need to multiply by wordsPerEntry
	rawDB := make([][]uint64, DBSize)

	//TODO: remove bar for efficiency
	bar := progressbar.Default(int64(len(vectorsInBins)), fmt.Sprintf("Preprocessing"))

	for i := 0; i < len(vectorsInBins); i++ {

		vectorBytesArray := make([][]byte, 0, len(vectorsInBins[i]))

		for j := 0; j < len(vectorsInBins[i]); j++ {
			vector := vectorsInBins[i][j]
			vectorBytes := make([]byte, config.Dimensions*4)
			for k := 0; k < int(config.Dimensions) && k < len(vector); k++ {
				binary.LittleEndian.PutUint32(vectorBytes[k*4:], math.Float32bits(vector[k]))
			}
			vectorBytesArray = append(vectorBytesArray, vectorBytes)
		}

		// Flatten the array of byte arrays into a single byte array
		//TODO: this might be wrong....
		entryBytes := make([]byte, 0, len(vectorBytesArray)*int(config.Dimensions)*4)
		for _, vb := range vectorBytesArray {
			// A byte array that is exactly the size of 1 entry.
			entryBytes = append(entryBytes, vb...)
		}

		wordsPerEntry := (len(entryBytes) + 7) / 8 // ceil(bytes/8)

		entry := make([]uint64, wordsPerEntry)
		for k := 0; k < wordsPerEntry; k++ {
			off := k * 8
			if off+8 <= len(entryBytes) {
				entry[k] = binary.LittleEndian.Uint64(entryBytes[off : off+8])
			} else {
				// last partial word (only happens if total bytes not divisible by 8)
				var tmp [8]byte
				copy(tmp[:], entryBytes[off:])
				entry[k] = binary.LittleEndian.Uint64(tmp[:])
			}
		}

		// Copy into rawDB at the right offset
		//copy(rawDB[i*int(wordsPerEntry):], entry)

		// We just directly set the entry in rawdb:
		rawDB[i] = entry

		bar.Add(1)
	}

	bar.Finish()

	// TODO: Get average size instead of worst-case
	DBEntrySize := config.Dimensions * 4 * maxRowSize // bytes per DB entry (maxRowSize vectors × config.Dimensions float32s)
	maxWordsPerEntry := (uint64(DBEntrySize) + 7) / 8

	// Now that we have the rawDB, set up the PIR
	// pir := pianopir.NewSimpleBatchPianoPIR(uint64(len(vectorsInBins)), uint64(DBEntrySize), uint64(DBEntrySize), 16, rawDB, 8)
	pir := pianopir.NewSimpleBatchPianoPIR(
		uint64(len(vectorsInBins)),
		maxWordsPerEntry,
		uint64(DBEntrySize),
		16,
		rawDB,
		8,
	)

	//TODO: Remove this when not debugging
	if len(rawDB) > 0 {
		logrus.Debugf("DEBUG: rawDB[0] length (uint64s): %d", len(rawDB[0]))
		// Print first few uint64s to see if they are 0
		if len(rawDB[0]) > 5 {
			logrus.Debugf("DEBUG: rawDB[0] head: %v", rawDB[0][:5])
		}
	}

	logrus.Info("PIR Ready for preprocessing")

	// pir.Preprocessing()

	ret := VecBins{
		N:          len(vectorsInBins),
		Dimensions: int(config.Dimensions),
		EntrySize:  int(maxRowSize),
		rawDB:      rawDB,

		PIR:         pir,
		DBTotalSize: uint64(len(vectorsInBins) * int(DBEntrySize)),
		DBEntrySize: uint64(DBEntrySize),
	}

	if config.DebugLevel >= 1 {
		logrus.Infof("%d, %d, %d, %d", ret.N, ret.DBTotalSize, ret.DBEntrySize, ret.EntrySize)
	}

	return ret

}
