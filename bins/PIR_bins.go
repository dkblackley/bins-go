package bins

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/blugelabs/bluge"
	"github.com/blugelabs/bluge/analysis"
	"github.com/dkblackley/bins-go/globals"
	"github.com/dkblackley/bins-go/pianopir"
	"github.com/sirupsen/logrus"
)

type VecBins struct {
	N                    int                      // Number of Bins
	Dimensions           int                      // Dimension of vectors
	EntrySize            int                      // number of vectors in a row (Size of one entry)
	DBEntrySize          uint64                   // Number of bytes in an entry
	DBTotalSize          uint64                   // in bytes
	Queries              map[string]globals.Query // A mapping from QID to query
	EnglishTokenAnalyzer *analysis.Analyzer
	// PIR                  *pianopir.SimpleBatchPianoPIR
	MaxRowSize uint
	T          int
	idPIR      *pianopir.SimpleBatchPianoPIR
	vecPIR     *pianopir.SimpleBatchPianoPIR
	docMap     map[int]string

	rawDB  [][]uint64
	config *globals.Args
}

func (v VecBins) PIRPreprocess() time.Duration {
	if (v.vecPIR.FinishedBatchNum+v.vecPIR.Config().BatchNumNeeded >= v.vecPIR.SupportBatchNum) && (v.idPIR.FinishedBatchNum+v.idPIR.Config().BatchNumNeeded >= v.idPIR.SupportBatchNum) {
		return v.idPIR.Preprocessing() + v.vecPIR.Preprocessing()
	}
	if v.vecPIR.FinishedBatchNum+v.vecPIR.Config().BatchNumNeeded >= v.vecPIR.SupportBatchNum {
		return v.vecPIR.Preprocessing()
	}
	return v.idPIR.Preprocessing() + v.vecPIR.Preprocessing()
}

func (v VecBins) Preprocess() {
	v.idPIR.Preprocessing()
	v.vecPIR.Preprocessing()
}

// report whichever PIR is closer to running out of hints
func (v VecBins) GetBatchNums() (uint64, uint64, uint64) {
	if v.vecPIR.FinishedBatchNum+v.vecPIR.Config().BatchNumNeeded >= v.vecPIR.SupportBatchNum {
		return v.vecPIR.FinishedBatchNum, v.vecPIR.Config().BatchNumNeeded, v.vecPIR.SupportBatchNum
	}
	return v.idPIR.FinishedBatchNum, v.idPIR.Config().BatchNumNeeded, v.idPIR.SupportBatchNum
}

func (v VecBins) GetMetaData() map[string]string {
	meta := v.idPIR.PrintInfo()
	for k, val := range v.vecPIR.PrintInfo() {
		meta["Vec"+k] = val
	}
	return meta
}

type DBentry struct {
	// entry []string
	entry []uint64
}

func (d DBentry) Decode(config *globals.Args) []string {
	docIDs := make([]string, 0, len(d.entry))
	for _, idx := range d.entry {
		docID, ok := config.DocIDMapPacmann[int(idx)]
		if !ok {
			logrus.Warnf("Doc index not in corpus map: %d", idx)
			continue
		}
		docIDs = append(docIDs, docID)
	}
	return docIDs
}

func (v VecBins) DoSearch(QID string, _ int) (globals.Decodable, error) {
	indices := v.MakeIndices(QID)

	if uint64(len(indices)) >= v.idPIR.Config().BatchSize {
		logrus.Warnf("Too many indices in batch: %d for QID: %s - Possible corruption incoming", len(indices), QID)
	}

	idResults, err := v.idPIR.Query(indices)
	if err != nil {
		return DBentry{nil}, err
	}

	docIdx := v.collectDocIdx(idResults)
	if len(docIdx) == 0 {
		logrus.Errorf("Empty response ")
		os.Exit(1)
		// return DBentry{nil}, nil
	}

	batch := int(v.vecPIR.Config().BatchSize)
	// docIDs := make([]string, 0, len(docIdx))
	// Because Batch size is T, there should be exactly one batched query per term
	// TODO: FIX!!

	//for start := 0; start < len(docIdx); start += batch {
	start := 0
	end := start + batch
	if end > len(docIdx) {
		end = len(docIdx)
	}
	// vecResults, err := v.vecPIR.Query(docIdx[start:end])
	// Do this for completeness, in a real 'run' we would just return this, but the main decoding loop requires strings
	_, err = v.vecPIR.Query(docIdx[start:end])
	if err != nil {
		// return DBentry{docIDs}, err
		return DBentry{docIdx[:start]}, err
	}
	// docIDs = append(docIDs, v.preDecode(docIdx[start:end], vecResults)...)
	//}

	// return DBentry{docIDs}, nil
	return DBentry{docIdx}, nil
}

func (v VecBins) collectDocIdx(results [][]uint64) []uint64 {
	docIdx := make([]uint64, 0, len(results)*v.T)
	seen := make(map[uint32]struct{}, len(results)*v.T)

	for i := 0; i < len(results); i++ {
		for _, w := range results[i] {
			for _, half := range [...]uint32{uint32(w), uint32(w >> 32)} {
				if half == 0 {
					continue // padding, not doc 0
				}
				if _, dup := seen[half]; dup {
					continue
				}
				seen[half] = struct{}{}
				docIdx = append(docIdx, uint64(half-1))
			}
		}
	}

	if len(docIdx) == 0 {
		logrus.Errorf("All results were empty!!!!")
	}
	return docIdx
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
func MakeVecDb(config *globals.Args) VecBins {

	metaData := config.DatasetMeta

	logrus.Debugf("Loading data from: %s and %s", metaData.Vectors.CorpusVec, metaData.IndexDir)

	// TODO: Uncomment when back
	//if config.CorpusVec { // If we want to lead npy vectors
	bm25Vectors, err := globals.LoadFloat32MatrixFromNpy(metaData.Vectors.CorpusVec, int(config.DBSize), int(config.Dimensions))
	logrus.Infof("Size of vectors: %d", len(bm25Vectors))
	if len(bm25Vectors) == 0 {
		logrus.Errorf("loaded vects from %s: DBSize: %d and dims: %d ", metaData.Vectors.CorpusVec, config.DBSize, config.Dimensions)
	}
	Must(err)
	var DB [][]string
	if config.Load {
		// TODO: make this dynamic
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

	// Padding is now done dynamically...
	//pad := make([]float32, config.Dimensions)
	maxRowSize := 0
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

	idRaw, idWords := BuildIndexDB(DB, flipped, maxRowSize)
	vecRaw, vecWords := BuildVectorDB(bm25Vectors, int(config.Dimensions))
	bm25Vectors = nil // vecRaw owns the data now

	T := maxRowSize

	idPIR := pianopir.NewSimpleBatchPianoPIR(
		uint64(len(idRaw)), idWords, idWords*8, 20, idRaw, 20, 20)

	stage2Batch := T * 10
	vecPIR := pianopir.NewSimpleBatchPianoPIR(
		uint64(len(vecRaw)), vecWords, vecWords*8, uint64(stage2Batch), vecRaw, 20, 1)

	meta := config.DatasetMeta
	queires, err := LoadQueries(meta.Queries)
	Must(err)
	queryMap := make(map[string]globals.Query)
	for q := range len(queires) {
		qid := queires[q].ID
		queryMap[qid] = queires[q]
	}

	binPir := VecBins{
		N:                    len(idRaw),
		Dimensions:           int(config.Dimensions),
		EntrySize:            maxRowSize,
		MaxRowSize:           uint(maxRowSize),
		T:                    T,
		idPIR:                idPIR,
		vecPIR:               vecPIR,
		docMap:               docMap,
		rawDB:                idRaw,
		DBEntrySize:          idWords * 8,
		DBTotalSize:          uint64(len(idRaw)) * idWords * 8,
		Queries:              queryMap,
		EnglishTokenAnalyzer: strictEnglishAnalyzer(),
		config:               config,
	}

	if config.DebugLevel >= 1 {
		logrus.Infof("%d, %d, %d, %d", binPir.N, binPir.DBTotalSize, binPir.DBEntrySize, binPir.EntrySize)
		logrus.Debugf("Stage 1: %d bins, %d words/entry (%d B). Stage 2: %d vectors, %d words/entry (%d B). T=%d",
			len(idRaw), idWords, idWords*8, len(vecRaw), vecWords, vecWords*8, T)
	}

	return binPir

}

// two uint32 per uint64; ids stored as idx+1, 0 means empty
func BuildIndexDB(DB [][]string, flipped map[string]int, maxRowSize int) ([][]uint64, uint64) {
	rawDB := make([][]uint64, len(DB))
	for i, bin := range DB {
		if len(bin) == 0 {
			continue // leave nil; EntryXor already skips n==0
		}
		entry := make([]uint64, (len(bin)+1)/2)
		for j, id := range bin {
			idx, ok := flipped[id]
			if !ok {
				logrus.Warnf("doc %s not in corpus map", id)
				continue
			}
			v := uint64(uint32(idx) + 1)
			if j%2 == 0 {
				entry[j/2] |= v
			} else {
				entry[j/2] |= v << 32
			}
		}
		rawDB[i] = entry
		DB[i] = nil // free the strings as you go
	}
	return rawDB, uint64((maxRowSize + 1) / 2)
}

func BuildVectorDB(vectors [][]float32, dim int) ([][]uint64, uint64) {
	wordsPerVec := (dim + 1) / 2
	rawDB := make([][]uint64, len(vectors))
	for i, v := range vectors {
		e := make([]uint64, wordsPerVec)
		for k := 0; k+1 < dim; k += 2 {
			e[k/2] = uint64(math.Float32bits(v[k])) | uint64(math.Float32bits(v[k+1]))<<32
		}
		rawDB[i] = e
	}
	return rawDB, uint64(wordsPerVec)
}
