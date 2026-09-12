package bins

// pir_bins_old.go
// The ORIGINAL single-DB bins method: one PIR database where entry i is the
// *concatenated embeddings* of every doc in bin i. One round of PIR, one Query
// call, no ID stage and no decode stage - the vectors come straight back and
// Decode just hashes them to get the doc IDs.
//
// Everything is suffixed with "Old" so this can sit next to PIR_bins.go without
// clashing. Control flow is the same as it was: MakeVecDbOld builds the bins and
// hands them to ProcessVecDBOld, which serialises them into rawDB and wires up a
// single SimpleBatchPianoPIR.
//
// Run it with -t bins-old (see the main.go note at the bottom of this file).

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/blugelabs/bluge"
	"github.com/blugelabs/bluge/analysis"
	"github.com/dkblackley/bins-go/globals"
	"github.com/dkblackley/bins-go/pianopir"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

// OldBatchSize is the PIR batch size for the single DB. The batch PIR always
// makes PartitionNum*QueryPerPartition sub-queries whatever you pass it, so this
// is *also* the number of query terms we can hide - it has to be comfortably
// bigger than the token count of a query or terms get dropped on collisions.
const OldBatchSize = 20

type VecBinsOld struct {
	N                    int                      // Number of Bins
	Dimensions           int                      // Dimension of vectors
	EntrySize            int                      // number of vectors in a row (Size of one entry)
	DBEntrySize          uint64                   // Number of bytes in an entry
	DBTotalSize          uint64                   // in bytes
	Queries              map[string]globals.Query // A mapping from QID to query
	EnglishTokenAnalyzer *analysis.Analyzer
	PIR                  *pianopir.SimpleBatchPianoPIR
	MaxRowSize           uint

	rawDB  [][]uint64
	config *globals.Args
}

func (v VecBinsOld) PIRPreprocess() time.Duration {
	return v.PIR.Preprocessing()
}

func (v VecBinsOld) Preprocess() {
	v.PIR.Preprocessing()
}

func (v VecBinsOld) GetBatchNums() (uint64, uint64, uint64) {
	pir := v.PIR
	return pir.FinishedBatchNum, pir.Config().BatchNumNeeded, pir.SupportBatchNum
}

func (v VecBinsOld) GetMetaData() map[string]string {
	return v.PIR.PrintInfo()
}

type DBentryOld struct {
	entry [][]uint64
}

// Decode turns the raw PIR entries back into doc IDs. Each entry holds every
// vector in the bin end-to-end, so we split it up and hash each vector against
// config.IDLookup (built by MakeLookup in main).
func (d DBentryOld) Decode(config *globals.Args) []string {

	results := d.entry
	empty := 0

	docIDs := make([]string, 0)

	IDLookup := config.IDLookup

	for i := 0; i < len(results); i++ {
		singleResult := results[i]
		if len(singleResult) <= 1 {
			logrus.Warnf("Got an empty result: %v - Possibly missed an entry", singleResult)
			empty++
			if empty == len(results) {
				logrus.Errorf("All results were empty!!!!")
			}
			continue
		}

		multipleVectors, err := DecodeEntryToVectors(singleResult, int(config.Dimensions))
		Must(err)

		for j := 0; j < len(multipleVectors); j++ {
			ID := HashFloat32s(multipleVectors[j])
			docID, ok := IDLookup[ID]
			if !ok {
				logrus.Warnf("Vector hash not found: %x", ID)
				continue
			}
			docIDs = append(docIDs, docID)
		}
	}
	return docIDs
}

// DoSearch is the whole online protocol: one batched PIR call, one round.
func (v VecBinsOld) DoSearch(QID string, _ int) (globals.Decodable, error) {
	indices := v.MakeIndices(QID)

	if uint64(len(indices)) >= v.PIR.Config().BatchSize {
		logrus.Warnf("Too many indices in batch: %d for QID: %s - Possible corruption incoming", len(indices), QID)
	}
	results, err := v.PIR.Query(indices)

	return DBentryOld{
		results,
	}, err
}

func (v VecBinsOld) MakeIndices(QID string) []uint64 {

	query := v.Queries[QID]

	tokeniser := strictEnglishAnalyzer()
	tokens := tokeniser.Analyze([]byte(query.Text))

	indices := make([]uint64, len(tokens))
	for i, t := range tokens {
		indices[i] = hashTokenChoice(fmt.Sprintf("%s", t.Term), v.config.DChoice) % uint64(v.N)
	}

	return indices

}

// MakeVecDbOld Takes in args from command line and then outputs a 'VecBinsOld' object that implements the functions
// required for binsDB. Same as MakeVecDb up to the point where the bins are built - after that, instead of splitting
// into an ID DB and a vector DB, every bin becomes one entry of stacked embeddings.
func MakeVecDbOld(config *globals.Args) VecBinsOld {

	metaData := config.DatasetMeta

	logrus.Debugf("Loading data from: %s and %s", metaData.Vectors.CorpusVec, metaData.IndexDir)

	bm25Vectors, err := globals.LoadFloat32MatrixFromNpy(metaData.Vectors.CorpusVec, int(config.DBSize), int(config.Dimensions))
	logrus.Infof("Size of vectors: %d", len(bm25Vectors))
	if len(bm25Vectors) == 0 {
		logrus.Errorf("loaded vects from %s: DBSize: %d and dims: %d ", metaData.Vectors.CorpusVec, config.DBSize, config.Dimensions)
	}
	Must(err)

	var DB [][]string
	if config.Load {
		DB, err = ReadCSV(config.DataName + "_unigram_DB.csv")
		Must(err)
		logrus.Debugf("Loaded DB with %d items from %s", len(DB), config.DataName+"_unigram_DB.csv")

	} else {
		logrus.Debugf("About to laod data from %s", metaData.IndexDir)
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
		config.Metadata["EmptyBins"] = strconv.Itoa(empty)
	}

	// Padding is now done dynamically, so a short bin just makes a short entry. Note that this only saves
	// server-side rawDB memory: every PIR *response* and every client hint is still MaxDBEntrySize words.
	maxRowSize := 0
	redundancy := 0
	for _, e := range DB {
		if len(e) > maxRowSize {
			maxRowSize = len(e)
		}
	}

	docMap, _ := MakeDocIDAndQueryIDMap(config.DatasetMeta)
	flipped := make(map[string]int, len(docMap))

	for key, value := range docMap {
		flipped[value] = key
	}

	newDb := make([][][]float32, 0, len(DB))
	for _, entry := range DB {
		row := make([][]float32, 0, len(entry))
		// Add the vectors to the row
		for j := 0; j < len(entry); j++ {
			id, ok := flipped[entry[j]]
			if !ok {
				logrus.Warnf("doc %s not in corpus map", entry[j])
				continue
			}
			id64 := uint(id)
			if id64 >= config.DBSize {
				logrus.Errorf("ERROR: ID is larger that the database size!!")
				panic("ERROR: ID is larger that the database size!!")
			}
			row = append(row, bm25Vectors[id64]) // shares the row slice; no copy
		}
		newDb = append(newDb, row)
	}

	if config.DebugLevel >= 1 {
		wordsPerEntry := (uint64(config.Dimensions) * 4 * uint64(maxRowSize)) / 8
		logrus.Debugf("Row layout: config.Dimensions=%d, maxRowSize=%d, wordsPerEntry=%d", config.Dimensions, maxRowSize, wordsPerEntry)

		b := uint64(len(newDb)) * uint64(maxRowSize) * uint64(config.Dimensions) * 4
		logrus.Debugf("New DB size (worst case): %.2f MiB (%d bytes)", float64(b)/(1<<20), b)

		logrus.Debugf("Marco vectors: %.2f GiB", float64(config.DBSize*config.Dimensions*4)/(1<<30))
		logrus.Debugf("Max row size: %d", maxRowSize)
		logrus.Debugf("Padded files %d", redundancy)
	}

	binPir := ProcessVecDBOld(config, uint(maxRowSize), newDb)

	meta := config.DatasetMeta
	queires, err := LoadQueries(meta.Queries)
	Must(err)
	queryMap := make(map[string]globals.Query)
	for q := range len(queires) {
		qid := queires[q].ID
		queryMap[qid] = queires[q]
	}
	binPir.Queries = queryMap
	binPir.EnglishTokenAnalyzer = strictEnglishAnalyzer()

	return binPir

}

// ProcessVecDBOld flattens each bin of vectors into one []uint64 entry and builds the single PIR over them.
func ProcessVecDBOld(config *globals.Args, maxRowSize uint, vectorsInBins [][][]float32) VecBinsOld {

	DBSize := len(vectorsInBins)

	rawDB := make([][]uint64, DBSize)

	bar := progressbar.Default(int64(len(vectorsInBins)), "Packing bins into PIR entries")

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

		// We just directly set the entry in rawdb:
		rawDB[i] = entry
		vectorsInBins[i] = nil // free the bin as we go, ProcessVecDBOld owns the data now

		bar.Add(1)
	}

	bar.Finish()

	// Worst-case entry: maxRowSize vectors of config.Dimensions float32s. Every response and every client hint
	// is this size, so this single number drives both the comm cost and the hint storage.
	DBEntrySize := config.Dimensions * 4 * maxRowSize
	maxWordsPerEntry := (uint64(DBEntrySize) + 7) / 8

	pir := pianopir.NewSimpleBatchPianoPIR(
		uint64(DBSize),
		maxWordsPerEntry,
		uint64(DBEntrySize),
		OldBatchSize,
		rawDB,
		20,
		OldBatchSize,
	)

	if config.DebugLevel >= 1 && len(rawDB) > 0 {
		logrus.Debugf("DEBUG: rawDB[0] length (uint64s): %d", len(rawDB[0]))
		if len(rawDB[0]) > 5 {
			logrus.Debugf("DEBUG: rawDB[0] head: %v", rawDB[0][:5])
		}
	}

	logrus.Info("PIR Ready for preprocessing")

	ret := VecBinsOld{
		N:          DBSize,
		Dimensions: int(config.Dimensions),
		EntrySize:  int(maxRowSize),
		MaxRowSize: maxRowSize,
		rawDB:      rawDB,

		PIR:         pir,
		DBTotalSize: uint64(DBSize) * uint64(DBEntrySize),
		DBEntrySize: uint64(DBEntrySize),
		config:      config, // the old version forgot this, so MakeIndices always hashed with DChoice=0
	}

	if config.DebugLevel >= 1 {
		logrus.Infof("%d, %d, %d, %d", ret.N, ret.DBTotalSize, ret.DBEntrySize, ret.EntrySize)
		logrus.Debugf("Single DB: %d bins, %d words/entry (%d B), maxRowSize=%d",
			DBSize, maxWordsPerEntry, DBEntrySize, maxRowSize)
	}

	return ret

}

// To use this, add a branch next to the other search types in main.go:
//
//	} else if *searchType == "bins-old" {
//		PIRImplemented = bins.MakeVecDbOld(&config)
//	}
//
// config.IDLookup is already built in main before the decode loop, so Decode works unchanged.
