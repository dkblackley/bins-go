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

func getDatasets(root, name string) DatasetMetadata {

	if name == "msmarco" {
		return DatasetMetadata{
			"Marco",
			"index_marco",
			root + "/msmarco/corpus.jsonl",
			root + "/msmarco/queries.jsonl",
			root + "/msmarco/qrels/test.tsv",
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

	rawDB  [][]uint64
	config globals.Args
}

func (v VecBins) GetRawDB() [][]uint64 {
	//TODO implement me
	panic("implement me")
}

func (v VecBins) GetBatchPIRInfo() *pianopir.SimpleBatchPianoPIR {
	return v.PIR
}

func (v VecBins) GetNumQueries() int {
	//TODO implement me
	return len(v.Queries)
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

	metaData := getDatasets(config.DatasetsDirectory, config.DataName)

	// TODO: Uncomment when back
	//if config.Vectors { // If we want to lead npy vectors
	bm25Vectors, err := LoadFloat32MatrixFromNpy(metaData.Vectors, int(config.DBSize), int(config.Dimensions))
	logrus.Infof("Size of vectors: %d", len(bm25Vectors))
	Must(err)
	var DB [][]string
	if config.Load {

		DB, err = ReadCSV("debug_marco.csv")
		Must(err)

	} else {
		reader, _ := bluge.OpenReader(bluge.DefaultConfig(metaData.IndexDir))
		defer reader.Close()
		DB = MakeUnigramDB(reader, metaData, config)
		Must(err)

		if config.Save {
			err = WriteCSV("marco.csv", DB)
			Must(err)
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

	// Right now I pad all entries to be of the same size. This might not be good? Not sure how else to deal with this
	// TODO: maybe pad in the pianoPIR dynamically? Then how do we work out when the entry begins/e// This does nothing??nds...
	pad := make([]float32, config.Dimensions)
	maxRowSize := 0
	redundancy := 0
	for _, e := range DB {
		if len(e) > maxRowSize {
			maxRowSize = len(e)
		}
	}

	newDb := make([][][]float32, 0, len(DB))
	for _, entry := range DB {
		row := make([][]float32, 0, maxRowSize)
		// cap columns
		upto := len(entry)
		if upto > maxRowSize { // This should never occur? TODO: remove
			logrus.Warnf("Row exceeded the maximum row size!!")
			upto = maxRowSize
		}
		// Add the vectors to the row
		for j := 0; j < upto; j++ {
			id64, err := strconv.ParseUint(entry[j], 10, 32)
			Must(err)
			row = append(row, bm25Vectors[id64]) // shares the row slice; no copy
		}
		// Pad the row for all the missing vectors
		for len(row) < maxRowSize {
			redundancy++
			row = append(row, pad) // shared, no per-cell alloc
		}
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
	binPir := PreprocessVecDB(config, uint(maxRowSize), newDb)
	//end := time.Now()
	//
	//logrus.Infof("Preprocessing took %s", end.Sub(start))

	queires, err := LoadQueries(config)
	queryMap := make(map[string]Query)
	for q := range len(queires) {
		qid := queires[q].ID
		queryMap[qid] = queires[q]

	}
	binPir.Queries = queryMap
	binPir.EnglishTokenAnalyzer = strictEnglishAnalyzer()

	return binPir

}

func PreprocessVecDB(config globals.Args, maxRowSize uint, vectorsInBins [][][]float32) VecBins {
	DBEntrySize := config.Dimensions * 4 * maxRowSize // bytes per DB entry (maxRowSize vectors × config.Dimensions float32s)
	DBSize := len(vectorsInBins)

	// A single 'word' should be how many uint64s are required to re-make the entry (divide by 8 because uint64 is 8 bytes)
	wordsPerEntry := DBEntrySize / 8

	// I think just DBsize is big enough but I might need to multiply by wordsPerEntry
	rawDB := make([][]uint64, DBSize)

	//TODO: remove bar for efficiency
	bar := progressbar.Default(int64(len(vectorsInBins)), fmt.Sprintf("Preprocessing"))

	for i := 0; i < len(vectorsInBins); i++ {
		// 1) Build byte-slices for up to maxRowSize vectors and init to all 0's
		vectorBytesArray := make([][]byte, 0, maxRowSize)

		for j := 0; j < len(vectorsInBins[i]) && len(vectorBytesArray) < int(maxRowSize); j++ {
			vector := vectorsInBins[i][j]
			vectorBytes := make([]byte, config.Dimensions*4)
			for k := 0; k < int(config.Dimensions) && k < len(vector); k++ {
				binary.LittleEndian.PutUint32(vectorBytes[k*4:], math.Float32bits(vector[k]))
			}
			vectorBytesArray = append(vectorBytesArray, vectorBytes)
		}
		for len(vectorBytesArray) < int(maxRowSize) {
			vectorBytesArray = append(vectorBytesArray, make([]byte, config.Dimensions*4)) // zero-pad rows
		}

		// Concatenate the row into one byte slice of size DBEntrySize
		entryBytes := make([]byte, 0, DBEntrySize)
		for _, vb := range vectorBytesArray {
			// A byte array that is exactly the size of 1 entry.
			entryBytes = append(entryBytes, vb...)
		}

		// Convert bytes → uint64s (exact 8-byte windows)
		entry := make([]uint64, wordsPerEntry)
		for k := 0; k < int(wordsPerEntry); k++ {
			off := k * 8
			entry[k] = binary.LittleEndian.Uint64(entryBytes[off : off+8])
		}

		// Copy into rawDB at the right offset
		//copy(rawDB[i*int(wordsPerEntry):], entry)

		// We just directly set the entry in rawdb:
		rawDB[i] = entry

		bar.Add(1)
	}

	bar.Finish()

	// Now that we have the rawDB, set up the PIR
	pir := pianopir.NewSimpleBatchPianoPIR(uint64(len(vectorsInBins)), uint64(maxRowSize), uint64(DBEntrySize), 16, rawDB, 8)

	logrus.Info("PIR Ready for preprocessing")

	pir.Preprocessing()

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
