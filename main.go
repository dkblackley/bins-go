package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"time"

	"github.com/dkblackley/bins-go/Pacmann"
	"github.com/dkblackley/bins-go/bins"
	"github.com/dkblackley/bins-go/globals"
	"github.com/dkblackley/bins-go/tree/cmd"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

type StackHook struct{}

func (h *StackHook) Levels() []logrus.Level {
	// Only trigger for Errors and above
	return []logrus.Level{logrus.ErrorLevel, logrus.FatalLevel, logrus.PanicLevel}
}

func (h *StackHook) Fire(entry *logrus.Entry) error {
	pcs := make([]uintptr, 10)
	n := runtime.Callers(6, pcs) // Skip internal logrus frames
	if n == 0 {
		return nil
	}

	frames := runtime.CallersFrames(pcs[:n])
	stack := ""
	for {
		frame, more := frames.Next()
		stack += frame.Function + "\n\t" + frame.File + ":" + string(rune(frame.Line)) + "\n"
		if !more {
			break
		}
	}
	entry.Data["stacktrace"] = stack
	return nil
}

// const MAX_UINT32 = ^uint32(0)

type PIRImplement interface {
	GetMetaData() map[string]string
	DoSearch(QID string, k int) (globals.Decodable, error)
	Preprocess()
	PIRPreprocess() time.Duration
	GetBatchNums() (uint64, uint64, uint64)
}

func GetDatasets(root, name string) globals.DatasetMetadata {

	if name == "msmarco" {
		vectors := globals.Vectors{

			root + "/msmarco/msmarco_corpus_embed_f32.npy",
			root + "/msmarco/msmarco_query_embed_f32.npy",
			root + "/msmarco/marco_graph.npy"}

		return globals.DatasetMetadata{
			"Marco",
			root + "/index_marco",
			root + "/msmarco/corpus.jsonl",
			root + "/msmarco/queries.dev.small.jsonl",
			root + "/msmarco/qrels/qrels.dev.tsv",
			vectors,
		}
	} else if name == "scifact" {
		vectors := globals.Vectors{

			root + "/scifact/scifact_corpus_embed_f32.npy",
			root + "/scifact/scifact_query_embed_f32.npy",
			root + "/scifact/scifact_graph.npy"}
		return globals.DatasetMetadata{
			"SciFact",
			root + "/index_scifact", // index folders created earlier
			root + "/scifact/corpus.jsonl",
			root + "/scifact/queries.jsonl",
			root + "/scifact/qrels/test.tsv",
			vectors,
		}
	} else if name == "debug" {
		logrus.Debugf("Using debug dataset")
		vectors := globals.Vectors{

			root + "/Son/my_vectors_192_debug.npy",
			root + "/Son/query_192_float32_debug.npy",
			root + "/Son/debug_graph.npy"}

		return globals.DatasetMetadata{
			"Marco",
			root + "/index_marco",
			root + "/msmarco/corpus_debug.jsonl",
			root + "/msmarco/queries.dev.small.jsonl",
			root + "/msmarco/qrels/qrels.dev.tsv",
			vectors,
		}
	} else {
		vectors := globals.Vectors{

			root + "/trec-covid/trec_corpus_embed_f32.npy",
			root + "/trec-covid/trec_query_embed_f32.npy",
			root + "/trec-covid/trec_graph.npy"}
		return globals.DatasetMetadata{
			"TREC-COVID",
			root + "/index_trec_covid",
			root + "/trec-covid/corpus.jsonl",
			root + "/trec-covid/queries.jsonl",
			root + "/trec-covid/qrels/test.tsv",
			vectors,
		}
	}
}

func main() {

	DBSize := flag.Uint("n", 8841823, "Number of items/vectors in DB")
	searchType := flag.String("t", "bins", "Search type, current options are 'bins'|'pacmann'")
	dbFileName := flag.String("name", "msmarco", "Identifier for the dataset to be loaded")
	datasetsDirectory := flag.String("dataset", "../datasets", "Where to look for the dataset/data")
	topK := flag.Uint("k", 5, "K many items to return in search")
	vectors := flag.Bool("vectors", true, "Use npy vectors for retrieval or raw text")
	dimensions := flag.Uint("dim", 4, "Dimension of vectors (if being used)")
	thresh := flag.Uint("thresh", 0, "Threshold to start dropping items from bins")
	dChoice := flag.Uint("d", 1, "Number of bins to choose from")
	binSize := flag.Float64("binSize", 1.0, "How many total bins to use, it's vocab size times this number")
	docsPerBin := flag.Uint("docsPerBin", 1000, "How many documents to put into each bin.")
	save := flag.Bool("save", false, "Whether or not to save data")
	load := flag.Bool("load", false, "Whether or not to load data")
	debugLevel := flag.Int("debug", 0, "Debug level, 0 for info, 1 for debug, 2 for trace and -1 for no debug")
	checkPointFolder := flag.String("checkpoint", "checkPoint", "Where to look for the checkpoint data")
	//RTT := flag.Uint("RTT", 50, "RTT for the network")
	outFile := flag.String("outFile", "out", "Where to save the answers")
	outDir := flag.String("outDir", ".", "Directory to save the answers to")

	// Flags for Pacmann method
	stepN := flag.Uint("steps", 15, "How many steps to take in PACMANN/NN search")
	neighbhourNum := flag.Uint("neighb", 32, "How many neighbours to retrieve at each step in pacmann")

	// Flags for tree method
	index := flag.String("index", "", "Path to Lucene index")
	queries := flag.String("queries", "", "Path to queries.tsv file")
	B := flag.Int("B", 32, "Block size")
	r := flag.Int("r", 128, "Arity of the conceptual tree for Round 1")
	L := flag.Int("L", 200, "Beam width for Round 1")
	s := flag.Int("s", 8, "Sub-block size for Round 2")
	kCandidates := flag.Int("k_candidates", 200, "Number of candidate sub-blocks to select in Round 2")
	maxQueries := flag.Int("max-queries", 25, "Maximum number of queries to run")
	k1 := flag.Float64("k1", 0.9, "BM25 k1 parameter")
	bParam := flag.Float64("b", 0.4, "BM25, b parameter")
	qrels := flag.String("qrels", "", "Path to qrels file for evaluation")
	precompute := flag.Bool("precompute", false, "If set, load precomputed layout and stats from storage")
	stage1DataBin := flag.String("stage1-data-bin", "", "Path to Stage-1, data.bin")
	stage1IdmapBin := flag.String("stage1-idmap-bin", "", "Path to Stage-1, idmap.bin")
	vocab := flag.String("vocab", "", "Path to vocab.json")
	scale := flag.Float64("scale", 10.0, "Scale factor")
	pirBatchSize := flag.Int64("pir-batch-size", 1024, "Batch size for PIR queries (larger = fewer partitions but more memory)")

	// Stage3 (dense rerank) flags
	// docEmbed := flag.String("doc-embed", "", "Path to document embeddings (npy float32 file)")
	docIDMap := flag.String("doc-id-map", "", "Path to document ID map (one ID per line)")
	// queryEmbed := flag.String("query-embed", "", "Path to query embeddings (npy float32 file)")
	embedDim := flag.Int("embed-dim", 192, "Embedding dimensionality for Stage3 reranker")
	queryWords := flag.Int("query-words", 8, "Fixed number of words in query (pad with dummy if needed)")

	// Stage2 hit load/save flags for Stage3-only runs
	loadStage2Hits := flag.String("load-stage2-hits", "", "Path to JSON file containing saved Stage2 hitSubs (map qid->[]HitSubBlock)")
	saveStage2Hits := flag.String("save-stage2-hits", "", "If set, save computed Stage2 hitSubs to this JSON file")
	skipStage2 := flag.Bool("skip-stage2", false, "If set, do not initialize Stage2 DB (use --load-stage2-hits to provide hits)")

	flag.Parse()

	meta := GetDatasets(*datasetsDirectory, *dbFileName)

	docEmbed := meta.Vectors.CorpusVec
	queryEmbed := meta.Vectors.QueryVec

	IDLookup := make(map[[32]byte]string) // empty lookup

	config := globals.Args{
		DatasetsDirectory: *datasetsDirectory,
		K:                 *topK,
		SearchType:        *searchType,
		DataName:          *dbFileName,
		Vectors:           *vectors,
		Threshold:         *thresh,
		DChoice:           *dChoice,
		BinSize:           *binSize,
		DocsPerBin:        *docsPerBin,
		DBSize:            *DBSize,
		Save:              *save,
		Load:              *load,
		DebugLevel:        *debugLevel,
		CheckPointFolder:  *checkPointFolder,
		// RTT:               *RTT,
		Dimensions:    *dimensions,
		OutFile:       *outFile,
		OutDir:        *outDir,
		QueryNum:      0,
		DatasetMeta:   meta,
		IDLookup:      IDLookup,
		StepN:         *stepN,
		NeighbhourNum: *neighbhourNum,
		Metadata:      make(map[string]string),

		BinsConf: globals.BinsConf{
			Index:   *index,
			Queries: *queries,
			B:       *B,
			R:       *r,
			L:       *L,
			S:       *s,
			K:       int(*topK),

			KCandidates: *kCandidates,
			MaxQueries:  *maxQueries,

			K1:     *k1,
			BParam: *bParam,

			Qrels: *qrels,

			Precompute:     *precompute,
			Stage1DataBin:  *stage1DataBin,
			Stage1IdmapBin: *stage1IdmapBin,

			Vocab: *vocab,
			Scale: *scale,

			PIRBatchSize: *pirBatchSize,

			DocEmbed:   docEmbed,
			DocIDMap:   *docIDMap,
			QueryEmbed: queryEmbed,
			EmbedDim:   *embedDim,

			Debug:      *debugLevel,
			QueryWords: *queryWords,

			LoadStage2Hits: *loadStage2Hits,
			SaveStage2Hits: *saveStage2Hits,
			SkipStage2:     *skipStage2,
		},
	}

	switch *debugLevel {
	case 0:
		logrus.SetLevel(logrus.InfoLevel)
	case 1:
		logrus.SetLevel(logrus.DebugLevel)
	case 2:
		logrus.SetLevel(logrus.TraceLevel)
	default:
		logrus.SetLevel(logrus.ErrorLevel)
	}
	logrus.SetReportCaller(true)
	// logrus.AddHook(&StackHook{})

	logrus.SetFormatter(&logrus.TextFormatter{
		FullTimestamp: true,
	})

	logrus.Debugf("Config: %v", config)

	qids := getQIDS(config)
	config.QueryNum = uint(len(qids))

	if config.QueryNum <= 19 {
		logrus.Errorf("Only %d queries, skipping loaded from %s", config.QueryNum, config.DatasetMeta.Queries)
		return
	}

	var PIRImplemented PIRImplement
	// TODO: is it sensible to start the 'pre-processing' timer here? If so replace if with switch case!

	if *searchType == "bins" {
		PIRImplemented = bins.MakeVecDb(&config)
	} else if *searchType == "pacmann" {
		PIRImplemented = Pacmann.PacmannMain(&config)
	} else if *searchType == "tree" {
		PIRImplemented = run_tree.Runtree(&config)
	} else {
		logrus.Errorf("Invalid search type: %s", *searchType)
		return
	}
	config.DocIDMapPacmann, _ = bins.MakeDocIDAndQueryIDMap(config.DatasetMeta)
	start_pre := time.Now()
	PIRImplemented.Preprocess()
	end_pre := time.Now()
	logrus.Infof("Preprocessing finished in %s seconds", end_pre.Sub(start_pre))

	for key, value := range PIRImplemented.GetMetaData() {
		config.Metadata[key] = value
	}

	keys := make([]string, 0, len(config.Metadata))
	for k := range config.Metadata {
		keys = append(keys, k)
	}

	for _, k := range keys {
		newKey := "Pre" + k
		config.Metadata[newKey] = config.Metadata[k]
	}

	start := time.Now()
	encodedAnswers := doPIRSearch(PIRImplemented, qids, int(config.K), &config)
	end := time.Now()

	for key, value := range PIRImplemented.GetMetaData() {
		config.Metadata[key] = value
	}

	config.Metadata["PreprocessingTime"] = end_pre.Sub(start_pre).String()
	config.Metadata["NumQueries"] = strconv.Itoa(int(config.QueryNum))

	logrus.Infof("Answers finished in %s seconds", end.Sub(start))
	config.Metadata["TotalAnswerTime"] = end.Sub(start).String()

	//answers := make(map[string][][]uint64, config.QueryNum)
	answers := make(map[string][]string, config.QueryNum)

	IDLookup = bins.MakeLookup(meta, int(*DBSize), int(*dimensions))
	config.IDLookup = IDLookup

	bar := progressbar.NewOptions64(
		int64(len(encodedAnswers)),
		progressbar.OptionSetDescription("Decoding stuff"),
		progressbar.OptionShowElapsedTimeOnFinish(),
	)
	for qid, encodedAnswer := range encodedAnswers {
		answers[qid] = encodedAnswer.Decode(&config)
		bar.Add(1)
	}

	bar.Finish()

	start = time.Now()
	reRanked := CosineReRank(answers, &config)
	end = time.Now()

	config.Metadata["ReRankTime"] = end.Sub(start).String()

	oldOut := config.OutFile
	config.OutFile = fmt.Sprintf("%s/%s.json", config.OutDir, config.OutFile)
	writeAnswers(answers, config)
	config.OutFile = fmt.Sprintf("%s/%s_reRank.json", config.OutDir, oldOut)
	writeAnswers(reRanked, config)

	//stringAnwsers := Decode(answers, config)

	//if config.DataName != "debug" {

	//}

}

func writeAnswers(answers map[string][]string, config globals.Args) {
	f, err := os.Create(config.OutFile)
	if err != nil {
		panic(err)
	}
	defer func(f *os.File) {
		err := f.Close()
		if err != nil {
			log.Fatal(err)
		}
	}(f)

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ") // optional

	if err := enc.Encode(answers); err != nil {
		panic(err)
	}

	logrus.Infof("Wrote answers to %s", config.OutFile)

	metaFile := fmt.Sprintf("%s/metadata.json", config.OutDir)

	f, err = os.Create(metaFile)
	if err != nil {
		panic(err)
	}
	defer func(f *os.File) {
		err := f.Close()
		if err != nil {
			log.Fatal(err)
		}
	}(f)

	enc = json.NewEncoder(f)
	enc.SetIndent("", "  ") // optional

	if err := enc.Encode(config.Metadata); err != nil {
		panic(err)
	}

	logrus.Infof("Wrote answers to %s", metaFile)
}

func getQIDS(config globals.Args) []string {

	meta := config.DatasetMeta
	queries, _ := bins.LoadQueries(meta.Queries)

	ids := make([]string, len(queries))
	for i, q := range queries {
		ids[i] = q.ID
	}
	return ids

}

func doPIRSearch(PIRImplemented PIRImplement, qids []string, k int, config *globals.Args) map[string]globals.Decodable {

	logrus.Infof("Starting PIR search on %d queries", config.QueryNum)

	numQueries := config.QueryNum
	//numQueries := 30

	decodables := make(map[string]globals.Decodable)
	maintainenceTime := time.Duration(0)

	finishedBatchNum, batchNumNeeded, supportBatchNum := PIRImplemented.GetBatchNums()

	//start := time.Now()

	// TODO REMOVE THIS (?)
	bar := progressbar.NewOptions64(
		int64(numQueries),
		progressbar.OptionSetDescription("Answering Queries"),
		progressbar.OptionShowElapsedTimeOnFinish(),
	)
	for i := 0; i < int(numQueries); i++ {

		err := bar.Add(1)
		if err != nil {
			log.Fatal(err)
		}
		q := qids[i]

		if finishedBatchNum+batchNumNeeded >= supportBatchNum {
			// re-run the preprocessing
			maintainenceTime += PIRImplemented.PIRPreprocess()
		}

		// Results should be a 2d array, each item in the first dimension should be a single result and then the lower
		//dimension is an item in the DB
		results, err := PIRImplemented.DoSearch(q, k)

		if err != nil {
			logrus.Errorf("Error querying PIR: %v", err)
			continue
		}

		decodables[q] = results

	}
	err := bar.Finish()

	logrus.Infof("Total maintainence time: %s", maintainenceTime)
	config.Metadata["MaintainenceTime"] = maintainenceTime.String()

	if err != nil {
		log.Fatal(err)
	}

	return decodables
}

// For degugging, return the first n elements.
//func FirstN[T any](xs []T, n int) []T {
//	if len(xs) <= n {
//		return xs
//	}
//	return xs[:n]
//}

// Define a helper struct to hold the ID and Score together
type ScoredDoc struct {
	ID    string
	Score float32
}

func CosineReRank(results map[string][]string, config *globals.Args) map[string][]string {

	// docIDMap, queryIDMap := bins.MakeDocIDAndQueryIDMap(config.DatasetMeta)

	// First, load qrels and queries
	queries, err := bins.LoadQueries(config.DatasetMeta.Queries)
	if err != nil {
		logrus.Errorf("Error loading queries: %v", err)
		logrus.Errorf("from file: %s", config.DatasetMeta.Queries)
		return results
	}
	qrels, err := bins.LoadQrels(config.DatasetMeta.Qrels)
	if err != nil {
		logrus.Errorf("Error loading qrels: %v", err)
		logrus.Errorf("from file: %s", config.DatasetMeta.Qrels)
		return results
	}
	bins.Must(err)

	config.Metadata["MRRPreReRank"] = fmt.Sprintf("%.4f", calcMRR(results, qrels))
	config.Metadata["RecallPreReRank"] = fmt.Sprintf("%.4f", calcRecall(results, qrels))

	// Now load embeddings
	docEmbedPath := config.DatasetMeta.Vectors.CorpusVec
	queryEmbedPath := config.DatasetMeta.Vectors.QueryVec

	docEmbed, err := globals.LoadFloat32MatrixFromNpy(docEmbedPath, int(config.DBSize), int(config.Dimensions))
	if err != nil {
		logrus.Errorf("Error loading doc embeddings: %v from file %s", err, docEmbedPath)
		return results
	}
	logrus.Debugf("Loaded doc embeddings from %s", docEmbedPath)

	queryEmbed, err := globals.LoadFloat32MatrixFromNpy(queryEmbedPath, int(config.QueryNum), int(config.Dimensions))
	if err != nil {
		logrus.Errorf("Error loading doc embeddings: %vfrom file %s", err, queryEmbedPath)
		return results
	}
	logrus.Debugf("Loaded query embeddings from %s", queryEmbedPath)

	qidEmbedMap := make(map[string][]float32)
	docEmbedMap := make(map[string][]float32)

	for i, q := range queries {
		// queryIndex = queryIDMap[i]
		qidEmbedMap[q.ID] = queryEmbed[i]
	}

	for i := 0; i < len(docEmbed); i++ {
		embedHash := bins.HashFloat32s(docEmbed[i])
		docID := config.IDLookup[embedHash]
		docEmbedMap[docID] = docEmbed[i]
	}

	new_results := make(map[string][]string, len(results))
	missed := 0

	for qid, docIds := range results {
		queryEmb := qidEmbedMap[qid]

		scoredDocs := make([]ScoredDoc, 0, len(docIds))

		for _, docId := range docIds {

			if docId == "-1" { // debug signal/not found for something like pacmann
				continue
			}

			// Safety check: ensure doc has an embedding
			if docEmb, ok := docEmbedMap[docId]; ok {
				similarity := CosineSimilarity(queryEmb, docEmb)
				scoredDocs = append(scoredDocs, ScoredDoc{ID: docId, Score: similarity})
			} else {
				logrus.Errorf("Doc %s has no embedding", docId)
				missed++
			}
		}

		// 2. Sort by Score (Descending)
		sort.Slice(scoredDocs, func(i, j int) bool {
			return scoredDocs[i].Score > scoredDocs[j].Score
		})

		// 3. Slice to Top K
		// Handle case where we have fewer results than K
		limit := int(config.K)
		if len(scoredDocs) < limit {
			limit = len(scoredDocs)
		}

		// 4. Extract just the IDs for the final result
		finalDocs := make([]string, limit)
		for i := 0; i < limit; i++ {
			finalDocs[i] = scoredDocs[i].ID
		}

		new_results[qid] = finalDocs
	}

	logrus.Infof("Missed %d docs", missed)

	config.Metadata["MRR"] = fmt.Sprintf("%.4f", calcMRR(new_results, qrels))
	config.Metadata["Recall"] = fmt.Sprintf("%.4f", calcMRR(new_results, qrels))
	logrus.Infof("MRR Pre re-rank: %s, Post re-rank: %s", config.Metadata["MRRPreReRank"], config.Metadata["MRR"])
	logrus.Infof("Recall Pre re-rank: %s, Post re-rank: %s", config.Metadata["RecallPreReRank"], config.Metadata["Recall"])

	return new_results

}

func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProd, normA, normB float32
	for i := 0; i < len(a); i++ {
		dotProd += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProd / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func calcMRR(results map[string][]string, qrels map[string]map[string]int) float64 {
	sumMRR := 0.0
	queryCount := 0

	for qid, rankedDocs := range results {
		// 1. Get the map of relevant docs (and their scores) for this query
		relDocs, exists := qrels[qid]
		if !exists || len(relDocs) == 0 {
			continue // Skip queries that have no ground truth data
		}

		// 2. Iterate through the ranked results to find the first relevant match
		for i, docID := range rankedDocs {
			// Check if this docID exists in the qrels map for this query
			// We assume existence implies relevance because LoadQrels filters v <= 0
			if _, isRelevant := relDocs[docID]; isRelevant {
				rank := i + 1 // Convert 0-based index to 1-based rank
				sumMRR += 1.0 / float64(rank)
				break // Found the first relevant doc; stop looking for this query
			}
		}

		queryCount++
	}

	if queryCount == 0 {
		return 0.0
	}

	return sumMRR / float64(queryCount)
}

func calcRecall(results map[string][]string, qrels map[string]map[string]int) float64 {
	sumRecall := 0.0
	queryCount := 0

	for qid, rankedDocs := range results {
		// 1. Get the map of relevant docs for this query
		relDocs, exists := qrels[qid]
		if !exists || len(relDocs) == 0 {
			continue // Skip queries that have no ground truth data
		}

		relevantRetrievedCount := 0

		// 2. Iterate through all ranked results to count total relevant matches
		for _, docID := range rankedDocs {
			if _, isRelevant := relDocs[docID]; isRelevant {
				relevantRetrievedCount++
			}
		}

		// 3. Calculate recall for this specific query and add to the sum
		// Recall = (relevant docs retrieved) / (total known relevant docs)
		queryRecall := float64(relevantRetrievedCount) / float64(len(relDocs))
		sumRecall += queryRecall

		queryCount++
	}

	if queryCount == 0 {
		return 0.0
	}

	return sumRecall / float64(queryCount)
}
