package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
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

type PIRImpliment interface {
	GetMetaData() map[string]string
	DoSearch(QID string, k int) (globals.Decodable, error)
	Preprocess()
	PIRPreprocess() time.Duration
	GetBatchNums() (uint64, uint64, uint64)
}

func GetDatasets(root, name string) globals.DatasetMetadata {
	vectors := globals.Vectors{

		root + "/Son/my_vectors_192.npy",
		root + "/Son/query_192_float32.npy",
		root + "/Son/my_vectors_192_8841823_192_32_graph.npy"}

	if name == "msmarco" {
		return globals.DatasetMetadata{
			"Marco",
			root + "/index_marco",
			root + "/msmarco/corpus.jsonl",
			root + "/msmarco/queries.dev.small.jsonl",
			root + "/msmarco/qrels/qrels.dev.tsv",
			vectors,
		}
	} else if name == "scifact" {
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

		vectors.CorpusVec = root + "/Son/my_vectors_192_debug.npy"
		vectors.QueryVec = root + "/Son/query_192_float32.npy"
		vectors.Graph = root + "/Son/debug_graph.npy"

		return globals.DatasetMetadata{
			"Marco",
			root + "/index_marco",
			root + "/msmarco/corpus_debug.jsonl",
			root + "/msmarco/queries.dev.small_debug.jsonl",
			root + "/msmarco/qrels/qrels.dev.tsv",
			vectors,
		}
	} else {
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
	dimensions := flag.Uint("dim", 192, "Dimension of vectors (if being used)")
	thresh := flag.Uint("thresh", 0, "Threshold to start dropping items from bins")
	dChoice := flag.Uint("d", 1, "Number of bins to choose from")
	binSize := flag.Uint("binSize", 8841823/100, "The number of bins to use")
	save := flag.Bool("save", false, "Whether or not to save data")
	load := flag.Bool("load", false, "Whether or not to load data")
	debugLevel := flag.Int("debug", 0, "Debug level, 0 for info, 1 for debug, 2 for trace and -1 for no debug")
	checkPointFolder := flag.String("checkpoint", "checkPoint", "Where to look for the checkpoint data")
	RTT := flag.Uint("RTT", 50, "RTT for the network")
	outFile := flag.String("outFile", "out.json", "Where to save the answers")

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
	docEmbed := flag.String("doc-embed", "", "Path to document embeddings (npy float32 file)")
	docIDMap := flag.String("doc-id-map", "", "Path to document ID map (one ID per line)")
	queryEmbed := flag.String("query-embed", "", "Path to query embeddings (npy float32 file)")
	embedDim := flag.Int("embed-dim", 192, "Embedding dimensionality for Stage3 reranker")
	queryWords := flag.Int("query-words", 8, "Fixed number of words in query (pad with dummy if needed)")

	// Stage2 hit load/save flags for Stage3-only runs
	loadStage2Hits := flag.String("load-stage2-hits", "", "Path to JSON file containing saved Stage2 hitSubs (map qid->[]HitSubBlock)")
	saveStage2Hits := flag.String("save-stage2-hits", "", "If set, save computed Stage2 hitSubs to this JSON file")
	skipStage2 := flag.Bool("skip-stage2", false, "If set, do not initialize Stage2 DB (use --load-stage2-hits to provide hits)")

	flag.Parse()

	meta := GetDatasets(*datasetsDirectory, *dbFileName)

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
		DBSize:            *DBSize,
		Save:              *save,
		Load:              *load,
		DebugLevel:        *debugLevel,
		CheckPointFolder:  *checkPointFolder,
		RTT:               *RTT,
		Dimensions:        *dimensions,
		OutFile:           *outFile,
		QueryNum:          0,
		DatasetMeta:       meta,
		IDLookup:          IDLookup,
		Metadata:          make(map[string]string),

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

			DocEmbed:   *docEmbed,
			DocIDMap:   *docIDMap,
			QueryEmbed: *queryEmbed,
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

	var PIRImplemented PIRImpliment
	// TODO: is it sensible to start the 'pre-processing' timer here? If so replace if with switch case!

	if *searchType == "bins" {
		PIRImplemented = bins.MakeVecDb(config)
	} else if *searchType == "pacmann" {
		PIRImplemented = Pacmann.PacmannMain(config)
	} else if *searchType == "tree" {
		PIRImplemented = run_tree.Runtree(config)
	} else {
		logrus.Errorf("Invalid search type: %s", *searchType)
		return
	}

	start := time.Now()
	PIRImplemented.Preprocess()
	end := time.Now()
	logrus.Infof("Preprocessing finished in %s seconds", end.Sub(start))
	config.Metadata = PIRImplemented.GetMetaData()
	config.Metadata["PreprocessingTime"] = end.Sub(start).String()
	config.Metadata["NumQueries"] = strconv.Itoa(int(config.QueryNum))

	start = time.Now()
	encodedAnswers := doPIRSearch(PIRImplemented, qids, int(config.K), config)
	end = time.Now()
	logrus.Infof("Answers finished in %s seconds", end.Sub(start))
	config.Metadata["TotalAnswerTime"] = end.Sub(start).String()

	//answers := make(map[string][][]uint64, config.QueryNum)
	answers := make(map[string][]string, config.QueryNum)

	if *searchType == "bins" {
		IDLookup = bins.MakeLookup(meta, int(*DBSize), int(*dimensions))
		config.IDLookup = IDLookup
	}

	bar := progressbar.NewOptions64(
		int64(len(encodedAnswers)),
		progressbar.OptionSetDescription("Decoding stuff"),
		progressbar.OptionShowElapsedTimeOnFinish(),
	)
	for qid, encodedAnswer := range encodedAnswers {
		answers[qid] = encodedAnswer.Decode(config)
		bar.Add(1)
	}

	bar.Finish()

	writeAnswers(answers, config)

	//stringAnwsers := Decode(answers, config)

	//if config.DataName != "debug" {
	bins.BasicReRank(answers, config)
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

	f, err = os.Create(fmt.Sprintf("%s_%d_metadata.json", config.SearchType, config.K))
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

	logrus.Infof("Wrote answers to metadata.json")
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

func doPIRSearch(PIRImplimented PIRImpliment, qids []string, k int, config globals.Args) map[string]globals.Decodable {

	numQueries := len(qids)
	//numQueries := 300

	decodables := make(map[string]globals.Decodable)
	maintainenceTime := time.Duration(0)

	finishedBatchNum, batchNumNeeded, supportBatchNum := PIRImplimented.GetBatchNums()

	//start := time.Now()

	// TODO REMOVE THIS (?)
	bar := progressbar.NewOptions64(
		int64(numQueries),
		progressbar.OptionSetDescription("Answering Queries"),
		progressbar.OptionShowElapsedTimeOnFinish(),
	)
	for i := 0; i < numQueries; i++ {

		err := bar.Add(1)
		if err != nil {
			log.Fatal(err)
		}
		q := qids[i]

		if finishedBatchNum+batchNumNeeded >= supportBatchNum {
			// re-run the preprocessing
			maintainenceTime += PIRImplimented.PIRPreprocess()
		}

		// Results should be a 2d array, each item in the first dimension should be a single result and then the lower
		//dimension is an item in the DB
		results, err := PIRImplimented.DoSearch(q, k)

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
