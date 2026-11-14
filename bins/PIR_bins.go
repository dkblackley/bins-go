package bins

import (
	"math"
	"math/rand"
	"sort"
	"strconv"
	"sync/atomic"
	"time"

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
	MaxRowSize    uint
	T             int
	idPIR         *pianopir.SimpleBatchPianoPIR
	vecPIR        *pianopir.SimpleBatchPianoPIR
	docMap        map[int]string
	maxQueryTerms int
	// vecDocs maps an embedding-DB row to its corpus index when docs that are in
	// no bin were left out (-vecPrune); nil means row i is corpus doc i.
	vecDocs []uint32
	// roundRobin: candidates go to the embedding DB rank by rank across the query's bins (-vecOrder rr)
	roundRobin bool
	stats      *vecStats

	rawDB  [][]uint64
	config *globals.Args
}

// vecStats counts, over every query, the stage-1 candidates and how many of them the embedding-DB rounds
// had no room for (batch * rounds < candidates), so a small -vecBatch / -vecRounds shows what it cost.
type vecStats struct {
	queries, candidates, dropped, truncated atomic.Int64
}

func (v VecBins) PIRPreprocess() time.Duration {
	var t time.Duration
	if v.idPIR.FinishedBatchNum+v.idPIR.Config().BatchNumNeeded >= v.idPIR.SupportBatchNum {
		t += v.idPIR.Preprocessing()
	}
	if v.vecPIR.FinishedBatchNum+v.vecPIR.Config().BatchNumNeeded >= v.vecPIR.SupportBatchNum {
		t += v.vecPIR.Preprocessing()
	}
	return t
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
	meta["IDBatchSize"] = strconv.FormatUint(v.idPIR.Config().BatchSize, 10)
	meta["VecBatchSize"] = strconv.FormatUint(v.vecPIR.Config().BatchSize, 10)
	meta["VecRounds"] = strconv.Itoa(v.maxQueryTerms)
	meta["VecOrder"] = map[bool]string{true: "rr", false: "bin"}[v.roundRobin]
	addQueryTermMeta(meta, v.config.MaxTerms)
	if q := v.stats.queries.Load(); q > 0 {
		c, d := v.stats.candidates.Load(), v.stats.dropped.Load()
		meta["VecCandidatesPerQuery"] = strconv.FormatFloat(float64(c)/float64(q), 'f', 2, 64)
		meta["VecDroppedPerQuery"] = strconv.FormatFloat(float64(d)/float64(q), 'f', 2, 64)
		meta["VecQueriesTruncatedFrac"] = strconv.FormatFloat(float64(v.stats.truncated.Load())/float64(q), 'f', 4, 64)
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
	return v.searchBins(QID, v.MakeIndices(QID))
}

// searchBins runs both PIR rounds for the given bin indices. The ID DB holds
// embedding-DB rows; the result is corpus indices.
func (v VecBins) searchBins(QID string, indices []uint64) (globals.Decodable, error) {
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
		// os.Exit(1)
		// return DBentry{nil}, nil
		// Add a random doc ID for security (so server doesn't realise we got nothing)
		docIdx = append(docIdx, v.rowOfCorpus(10))
	}

	batch := int(v.vecPIR.Config().BatchSize)
	dbSize := int(v.vecPIR.Config().DBSize)
	if drop := len(docIdx) - batch*v.maxQueryTerms; drop > 0 {
		v.stats.dropped.Add(int64(drop))
		v.stats.truncated.Add(1)
	}
	v.stats.candidates.Add(int64(len(docIdx)))
	v.stats.queries.Add(1)
	retrieved := make([]uint64, 0, len(docIdx))
	// Always MaxQueryTerms batches of exactly T, whatever the real term count
	for it := 0; it < v.maxQueryTerms; it++ {
		start := min(it*batch, len(docIdx))
		end := min(start+batch, len(docIdx))

		batchIdx := append(make([]uint64, 0, batch), docIdx[start:end]...)
		for len(batchIdx) < batch {
			batchIdx = append(batchIdx, uint64(rand.Intn(dbSize))) // dummy doc
		}

		vecResults, err := v.vecPIR.Query(batchIdx)
		if err != nil {
			return DBentry{retrieved}, err
		}
		for k := 0; k < end-start; k++ {
			if len(vecResults[k]) > 1 { // 1-word response means the lookup was dropped
				retrieved = append(retrieved, v.corpusIdx(docIdx[start+k]))
			}
		}
	}

	return DBentry{retrieved}, nil
}

// corpusIdx turns an embedding-DB row into the corpus index of its doc.
func (v VecBins) corpusIdx(row uint64) uint64 {
	if v.vecDocs == nil {
		return row
	}
	return uint64(v.vecDocs[row])
}

// rowOfCorpus is the embedding-DB row of a corpus index (row 0 when that doc
// was left out of a pruned DB).
func (v VecBins) rowOfCorpus(idx uint64) uint64 {
	if v.vecDocs == nil {
		return idx
	}
	row := sort.Search(len(v.vecDocs), func(i int) bool { return uint64(v.vecDocs[i]) >= idx })
	if row < len(v.vecDocs) && uint64(v.vecDocs[row]) == idx {
		return uint64(row)
	}
	return 0
}

// collectDocIdx turns the ID-DB answers (one per query term, in query order) into the distinct embedding-DB
// rows to fetch. A bin lists its docs in BM25 rank order (searchTopDocs, mergeIntoBin), so with roundRobin the
// list is rank 1 of every term's bin, then rank 2, ...: when batch*rounds cuts the list short, each term loses
// its lowest-ranked docs instead of the last terms losing all of theirs.
func (v VecBins) collectDocIdx(results [][]uint64) []uint64 {
	return collectDocIdx(results, v.T, v.roundRobin)
}

func collectDocIdx(results [][]uint64, T int, roundRobin bool) []uint64 {
	docIdx := make([]uint64, 0, len(results)*T)
	seen := make(map[uint32]struct{}, len(results)*T)
	push := func(half uint32) {
		if half == 0 {
			return // padding, not doc 0
		}
		if _, dup := seen[half]; dup {
			return
		}
		seen[half] = struct{}{}
		docIdx = append(docIdx, uint64(half-1))
	}
	// doc j of a bin is half j%2 of word j/2
	at := func(r []uint64, j int) uint32 {
		if j/2 >= len(r) {
			return 0
		}
		if j%2 == 0 {
			return uint32(r[j/2])
		}
		return uint32(r[j/2] >> 32)
	}

	if roundRobin {
		longest := 0
		for _, r := range results {
			longest = max(longest, 2*len(r))
		}
		for j := 0; j < longest; j++ {
			for _, r := range results {
				push(at(r, j))
			}
		}
	} else {
		for _, r := range results {
			for j := 0; j < 2*len(r); j++ {
				push(at(r, j))
			}
		}
	}

	if len(docIdx) == 0 {
		logrus.Errorf("All results were empty!!!!")
	}
	return docIdx
}

func (v VecBins) MakeIndices(QID string) []uint64 {
	return queryBinIndices(v.Queries[QID].Text, v.config.DChoice, v.N, v.config.MaxTerms)
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
		reader := OpenBinsIndex(metaData)
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

	docMap, _ := MakeDocIDAndQueryIDMap(config.DatasetMeta)
	meta := config.DatasetMeta
	queires, err := LoadQueries(meta.Queries)
	Must(err)
	queryMap := make(map[string]globals.Query)
	for q := range len(queires) {
		qid := queires[q].ID
		queryMap[qid] = queires[q]
	}

	binPir := buildVecBins(config, DB, docMap, bm25Vectors)
	binPir.Queries = queryMap
	return binPir

}

// vecBatchAndRounds applies -vecBatch / -vecRounds over the automatic Stage2
// setting. Each query makes `rounds` sequential embedding-DB rounds of exactly
// `batch` lookups, so it can fetch batch*rounds candidate embeddings; the rest
// are dropped. The client hint grows with sqrt(batch/2 partitions), so a
// smaller batch saves storage and, to keep the same coverage, costs rounds.
// With only -vecBatch set, rounds are chosen to cover what the automatic
// setting covered.
func vecBatchAndRounds(autoBatch, autoRounds, batch, rounds int) (int, int) {
	if batch <= 0 && rounds <= 0 {
		return autoBatch, autoRounds
	}
	coverage := autoBatch * autoRounds
	if batch <= 0 {
		batch = (coverage + rounds - 1) / rounds
	}
	batch = max(batch, 2) // one partition at least
	if rounds <= 0 {
		rounds = (coverage + batch - 1) / batch
	}
	return batch, max(rounds, 1)
}

// binnedDocs lists the corpus indices that appear in at least one bin, in
// corpus order (vecDocs: embedding-DB row -> corpus index), and the inverse
// (rowOf: corpus index -> row, -1 for a doc in no bin).
func binnedDocs(DB [][]string, flipped map[string]int, numDocs int) (vecDocs []uint32, rowOf []int32) {
	rowOf = make([]int32, numDocs)
	for i := range rowOf {
		rowOf[i] = -1
	}
	for _, bin := range DB {
		for _, id := range bin {
			if idx, ok := flipped[id]; ok && idx < numDocs {
				rowOf[idx] = -2 // in a bin; numbered below
			}
		}
	}
	for idx, r := range rowOf {
		if r == -2 {
			rowOf[idx] = int32(len(vecDocs))
			vecDocs = append(vecDocs, uint32(idx))
		}
	}
	if len(vecDocs) == 0 { // keep one row so the PIR DB is never empty
		vecDocs = []uint32{0}
		rowOf[0] = 0
	}
	return vecDocs, rowOf
}

// two uint32 per uint64; ids stored as row+1, 0 means empty. The row is the
// corpus index, or rowOf[corpus index] when the embedding DB was pruned.
func BuildIndexDB(DB [][]string, flipped map[string]int, rowOf []int32, maxRowSize int) ([][]uint64, uint64) {
	sink := newRowSink("bins_vec0_id_db", len(DB)) // heap, or a file with -noram
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
			if rowOf != nil {
				if idx >= len(rowOf) || rowOf[idx] < 0 {
					logrus.Warnf("doc %s has no embedding-DB row", id)
					continue
				}
				idx = int(rowOf[idx])
			}
			v := uint64(uint32(idx) + 1)
			if j%2 == 0 {
				entry[j/2] |= v
			} else {
				entry[j/2] |= v << 32
			}
		}
		sink.Put(i, entry)
		DB[i] = nil // free the strings as you go
	}
	return sink.Finish(), uint64((maxRowSize + 1) / 2)
}

// BuildVectorDB stores one embedding per row: every corpus doc, or with vecDocs
// only those docs (row i = vectors[vecDocs[i]]).
func BuildVectorDB(vectors [][]float32, dim int, vecDocs []uint32) ([][]uint64, uint64) {
	wordsPerVec := (dim + 1) / 2
	rows := len(vectors)
	if vecDocs != nil {
		rows = len(vecDocs)
	}
	sink := newRowSink("bins_vec0_vec_db", rows) // heap, or a file with -noram
	e := make([]uint64, wordsPerVec)
	for i := 0; i < rows; i++ {
		v := vectors[i]
		if vecDocs != nil {
			v = vectors[vecDocs[i]]
		}
		if !sink.onDisk() {
			e = make([]uint64, wordsPerVec) // the heap sink keeps the row; the disk sink copies it
		}
		for k := 0; k+1 < dim; k += 2 {
			e[k/2] = uint64(math.Float32bits(v[k])) | uint64(math.Float32bits(v[k+1]))<<32
		}
		sink.Put(i, e)
	}
	return sink.Finish(), uint64(wordsPerVec)
}

// buildVecBins builds both PIR DBs of the two-round (vec0) layout from the bins
// (doc ID strings), the corpus index -> doc ID map and the corpus embeddings.
func buildVecBins(config *globals.Args, DB [][]string, docMap map[int]string, bm25Vectors [][]float32) VecBins {
	// Padding is now done dynamically...
	//pad := make([]float32, config.Dimensions)
	maxRowSize := 0
	for _, e := range DB {
		if len(e) > maxRowSize {
			maxRowSize = len(e)
		}
	}

	flipped := make(map[string]int, len(docMap))

	for key, value := range docMap {
		flipped[value] = key
	}

	// Docs that are in no bin can never be fetched, so with -vecPrune the
	// embedding DB keeps only the binned ones and the ID DB stores their new rows.
	var vecDocs []uint32
	var rowOf []int32
	if config.VecPrune {
		vecDocs, rowOf = binnedDocs(DB, flipped, len(bm25Vectors))
		logrus.Infof("Embedding DB: %d of %d docs are in some bin, %d left out", len(vecDocs), len(bm25Vectors), len(bm25Vectors)-len(vecDocs))
	}
	config.Metadata["VecPrune"] = strconv.FormatBool(config.VecPrune)
	config.Metadata["VecDocsStored"] = strconv.Itoa(len(bm25Vectors))
	if vecDocs != nil {
		config.Metadata["VecDocsStored"] = strconv.Itoa(len(vecDocs))
	}
	config.Metadata["VecDocsTotal"] = strconv.Itoa(len(bm25Vectors))

	idRaw, idWords := BuildIndexDB(DB, flipped, rowOf, maxRowSize)
	vecRaw, vecWords := BuildVectorDB(bm25Vectors, int(config.Dimensions), vecDocs)
	bm25Vectors = nil // vecRaw owns the data now (the caller passes its last reference)

	T := maxRowSize
	BatchSize := 24
	if config.IDBatch > 0 {
		BatchSize = int(config.IDBatch)
	}

	PartitionNum := BatchSize / 2
	PartitionSize := (len(idRaw) + PartitionNum - 1) / PartitionNum
	PartitionNum = (len(idRaw) + PartitionSize - 1) / PartitionSize // drop partitions that would start past DBSize
	if maxQ := uint64(math.Sqrt(float64(PartitionSize)) * math.Log(float64(PartitionSize))); maxQ < 4*2 {
		logrus.Errorf("BatchSize %d too large for DBSize %d: %d-row partitions allow only %d lookups between preprocessings",
			BatchSize, len(idRaw), PartitionSize, maxQ)
		BatchSize = int(maxQ)
	}

	idPIR := pianopir.NewSimpleBatchPianoPIR(
		uint64(len(idRaw)), idWords, idWords*8, uint64(BatchSize), idRaw, 20, 1)

	//SCifact is such a small DB that if we make the batch size big enough, then PIR crashes (it cant
	// make batches of a size big enough) so we have it do a fixed/globally known number of rounds.
	stage2Batch := T * 3
	maxQuery := 1
	if stage2Batch > 3000 {
		stage2Batch = 3000
		maxQuery = 2
	}
	if len(docMap) < 10000 {
		// The second DB crashes because scifact is so small, as a result we reduce the 'maxquery' so there are less
		// rounds of PIR for stage2.
		maxQuery = 6
		stage2Batch = T
		if stage2Batch > 500 {
			stage2Batch = 500
			maxQuery = 15
		}
	}

	stage2Batch, maxQuery = vecBatchAndRounds(stage2Batch, maxQuery, int(config.VecBatch), int(config.VecRounds))

	// A batch of B makes B/2 partitions, and a Piano partition needs enough rows for 8 lookups between
	// preprocessings (sqrt(P) ln P >= 8, i.e. P >= 16 to be safe): a -vecBatch too big for this (pruned)
	// embedding DB is cut to the largest that fits; metadata.json's VecBatchSize is what was used.
	if maxBatch := 2 * max(len(vecRaw)/16, 1); stage2Batch > maxBatch {
		logrus.Warnf("Stage2 batch %d too large for the %d-row embedding DB, using %d", stage2Batch, len(vecRaw), maxBatch)
		stage2Batch = maxBatch
	}

	logrus.Infof("Stage2 Max query size: %d and Stage2 Batch size is %d", maxQuery, stage2Batch)
	vecPIR := pianopir.NewSimpleBatchPianoPIR(
		uint64(len(vecRaw)), vecWords, vecWords*8, uint64(stage2Batch), vecRaw, 20, uint64(maxQuery))

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
		EnglishTokenAnalyzer: strictEnglishAnalyzer(),
		config:               config,
		maxQueryTerms:        maxQuery,
		vecDocs:              vecDocs,
		roundRobin:           config.VecOrder == "rr",
		stats:                &vecStats{},
	}
	if config.VecOrder != "" && config.VecOrder != "rr" && config.VecOrder != "bin" {
		logrus.Fatalf("-vecOrder %q: use bin or rr", config.VecOrder)
	}

	if config.DebugLevel >= 1 {
		logrus.Infof("%d, %d, %d, %d", binPir.N, binPir.DBTotalSize, binPir.DBEntrySize, binPir.EntrySize)
		logrus.Debugf("Stage 1: %d bins, %d words/entry (%d B). Stage 2: %d vectors, %d words/entry (%d B). T=%d",
			len(idRaw), idWords, idWords*8, len(vecRaw), vecWords, vecWords*8, T)
	}

	return binPir
}
