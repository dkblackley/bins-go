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
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/dkblackley/bins-go/Pacmann"
	"github.com/dkblackley/bins-go/bins"
	"github.com/dkblackley/bins-go/diskstore"
	"github.com/dkblackley/bins-go/globals"
	"github.com/dkblackley/bins-go/pianopir"
	pirold "github.com/dkblackley/bins-go/pir-old"
	testpir "github.com/dkblackley/bins-go/testPIR"
	"github.com/dkblackley/bins-go/tiptoe"
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
	} else if name == "trec-covid" {
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

	// Generic layout written by datasets/prepare_dataset.py + datasets/embed.slurm. Covers nq, hotpotqa, fever,
	// msmarco-v2 and its subsamples (e.g. msmarco-v2-100m), or any other name you prepare the same way:
	//   <root>/<name>/corpus.jsonl, queries.jsonl, qrels/test.tsv,
	//   <root>/<name>/<name>_corpus_embed_f32.npy, <name>_query_embed_f32.npy
	pretty, ok := datasetDisplayNames[name]
	if !ok {
		pretty = name
	}
	if _, err := os.Stat(root + "/" + name + "/corpus.jsonl"); err != nil {
		logrus.Warnf("Dataset %q: %s/%s/corpus.jsonl not found (%v)", name, root, name, err)
	}
	vectors := globals.Vectors{
		root + "/" + name + "/" + name + "_corpus_embed_f32.npy",
		root + "/" + name + "/" + name + "_query_embed_f32.npy",
		root + "/" + name + "/" + name + "_graph.npy"}
	return globals.DatasetMetadata{
		pretty,
		root + "/index_" + name,
		root + "/" + name + "/corpus.jsonl",
		root + "/" + name + "/queries.jsonl",
		root + "/" + name + "/qrels/test.tsv",
		vectors,
	}
}

// Display names for the datasets that use the generic layout (unknown names just use the -name flag as-is)
var datasetDisplayNames = map[string]string{
	"nq":              "NQ",
	"hotpotqa":        "HotpotQA",
	"fever":           "FEVER",
	"dbpedia-entity":  "DBPedia",
	"miracl-en":       "MIRACL-en",
	"msmarco-v2":      "MSMARCO-v2",
	"msmarco-v2-25m":  "MSMARCO-v2-25M",
	"msmarco-v2-50m":  "MSMARCO-v2-50M",
	"msmarco-v2-100m": "MSMARCO-v2-100M",
}

func main() {

	DBSize := flag.Uint("n", 8841823, "Number of items/vectors in DB")
	searchType := flag.String("t", "bins", "Search type: 'bins'|'pacmann'|'tree', or 'index' to build the bins' Bluge index for -name and exit")
	dbFileName := flag.String("name", "msmarco", "Identifier for the dataset to be loaded")
	datasetsDirectory := flag.String("dataset", "../datasets", "Where to look for the dataset/data")
	topK := flag.Uint("k", 5, "K many items to return in search")
	vectors := flag.Bool("vectors", true, "Use npy vectors for retrieval or raw text")
	// Text retrieval mode for the bins: bins hold compressed doc text, the client re-ranks with BM25 (no embeddings)
	text := flag.Bool("text", false, "bins: store compressed document text in the bins instead of embeddings and re-rank with BM25 on the client")
	textFormat := flag.String("text-format", "bow", "bins -text: 'bow' = sorted stems + tf (smallest, all BM25 needs) or 'seq' = stems in document order")
	textOrder := flag.Int("text-order", 3, "bins -text: characters of context in the arithmetic coder's model (0-3); 3 is smallest, model ~1.3 MB")
	textIDFBits := flag.Uint("text-idf-bits", 20, "bins -text: client IDF sketch has 2 x 2^bits one-byte cells (0 = no idf)")
	textTrainDocs := flag.Int("text-train-docs", 200000, "bins -text: docs used to train the coder's model")
	dimensions := flag.Uint("dim", 4, "Dimension of vectors (if being used)")
	thresh := flag.Uint("thresh", 0, "Threshold to start dropping items from bins")
	dChoice := flag.Uint("d", 1, "Number of bins to choose from")
	binSize := flag.Uint("binSize", 1000000, "How many total bins to use, it's vocab size times this number")
	docsPerBin := flag.Uint("docsPerBin", 100, "How many documents to put into each bin.")
	idBatch := flag.Uint("idBatch", 0, "bins vec0: batch size of the ID DB, partitions = batch/2 (0 = 24)")
	vecBatch := flag.Uint("vecBatch", 0, "bins vec0: embeddings fetched per embedding-DB round; partitions = batch/2, so smaller = less client storage (0 = automatic)")
	vecRounds := flag.Uint("vecRounds", 0, "bins vec0: embedding-DB rounds per query (0 = automatic: enough rounds to cover as many docs as the automatic setting)")
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
	docEmbedFlag := flag.String("doc-embed", "", "Path to Stage3 document embeddings (npy float32 file), defaults to the dataset corpus vectors")
	docIDMap := flag.String("doc-id-map", "", "Path to document ID map (one ID per line)")
	queryEmbedFlag := flag.String("query-embed", "", "Path to Stage3 query embeddings (npy float32 file), defaults to the dataset query vectors")
	embedDim := flag.Int("embed-dim", 192, "Embedding dimensionality for Stage3 reranker")
	queryWords := flag.Int("query-words", 8, "Fixed number of words in query (pad with dummy if needed)")

	// Stage2 hit load/save flags for Stage3-only runs
	loadStage2Hits := flag.String("load-stage2-hits", "", "Path to JSON file containing saved Stage2 hitSubs (map qid->[]HitSubBlock)")
	saveStage2Hits := flag.String("save-stage2-hits", "", "If set, save computed Stage2 hitSubs to this JSON file")
	skipStage2 := flag.Bool("skip-stage2", false, "If set, do not initialize Stage2 DB (use --load-stage2-hits to provide hits)")

	// Extra flags for the updated tree method (collaborator's main-updated, runs on pir-old)
	stage2DataBin := flag.String("stage2-data-bin", "", "Path to Stage-2 data.bin (defaults to <dataset>/tree/stage2/data.bin)")
	stage2IdmapBin := flag.String("stage2-idmap-bin", "", "Path to Stage-2 idmap.bin (defaults to <dataset>/tree/stage2/idmap.bin)")
	layoutPath := flag.String("layout-path", "", "Path to layout_with_stats.json (defaults to <dataset>/tree/layout_with_stats.json)")
	blockPerm := flag.String("block-permutation", "", "Path to block permutation JSON (must match this corpus's block count)")
	disablePIR := flag.Bool("disable-pir", false, "Load precomputed tree artifacts but use direct row access (diagnostic only)")
	strictPIR := flag.Bool("strict-pir", true, "Verify tree PIR responses against the local plaintext database (diagnostic)")
	pirConstantWork := flag.Bool("pir-constant-work", false, "Use public fixed BatchPIR padding and dummy work for cache hits (tree)")
	pirThreads := flag.Int("pir-threads", 15, "Maximum number of BatchPIR partitions processed concurrently (tree)")
	cuckooLoadFactor := flag.Float64("cuckoo-load-factor", 0.4, "Cuckoo table load factor (must be below 0.5 for two hashes)")
	// client-storage knobs (tree, pir-old). Defaults reproduce the original runs.
	pirDownloadSmall := flag.Bool("pir-download-small", false, "Download a PIR DB in the clear when its Piano hints would be at least as large as the DB (tree and bins)")
	pirChunkFactor := flag.Float64("pir-chunk-factor", 2.0, "Piano ChunkSize = next pow2 >= factor*sqrt(partition rows); smaller = smaller primary hints, larger uploads (tree)")
	pirQueryBudget := flag.Float64("pir-query-budget", 1.0, "Scale on the sqrt(n)ln(n) queries between hint refreshes; smaller = smaller backup hints, more re-preprocessing (tree)")
	cuckooHashes := flag.Int("cuckoo-hashes", 2, "tree: hash locations per Cuckoo key, 2 or 3 (3 allows load factors up to 0.91)")
	cuckooKeyBytes := flag.Int("cuckoo-key-bytes", 8, "tree: bytes of the (term,node) key kept per Cuckoo row, 1..8 (below 8 = fingerprint)")
	cuckooUBBits := flag.Int("ub-bits", 8, "tree: bits per Stage1/2 upper bound, 1..8 (rounded up, so still upper bounds)")
	stage3BatchSize := flag.Int64("stage3-batch-size", 32, "BatchPIR batch size of the Stage3 embedding DB, partitions = batch/2 (tree)")
	cuckooMaxKicks := flag.Int("cuckoo-max-kicks", 1000, "Maximum displacement steps per Cuckoo insertion")
	cuckooBuildAttempts := flag.Int("cuckoo-build-attempts", 8, "Maximum full-table Cuckoo rehash attempts")
	cuckooCacheDir := flag.String("cuckoo-cache-dir", "", "Optional directory for persistent physical Cuckoo tables")

	// Flags for the Tiptoe baseline (needs a build with -tags tiptoe)
	tiptoeDir := flag.String("tiptoe-dir", "", "Tiptoe corpus dir made by baselines/tiptoe/preprocess_tiptoe.py")
	tiptoeEmbHint := flag.Int("tiptoe-emb-hint-mb", 500, "Tiptoe ranking-service hint size target in MB (upstream default 500)")
	tiptoeURLHint := flag.Int("tiptoe-url-hint-mb", 100, "Tiptoe URL-service hint size target in MB (upstream default 100)")
	tiptoeMaxQueries := flag.Int("tiptoe-max-queries", 0, "Tiptoe: answer only the first N queries (0 = all)")

	// Simulated PIR (bins, pacmann, tree): lookups return the row directly, no hints are built, no XOR is done.
	// Answers, MRR/Recall, rounds, bytes, LAN/WAN and client storage stay as in a real run (see
	// pianopir/simulate.go); every server timing is written as -1.
	simulate := flag.Bool("simulate", false, "Simulate the PIR: same answers, bytes, LAN/WAN and client storage, no hints or XOR (timings are written as -1)")
	// Low-RAM mode: big read-only arrays go to files under <noram-dir>/<unique run folder>/ and are mmap'd
	noram := flag.Bool("noram", false, "Low-RAM mode: keep the PIR databases, client hint tables and corpus embeddings on disk (mmap) instead of in RAM. Same results, slower; metadata gets \"noram\": \"true\" so the timings can be filtered out")
	noramDir := flag.String("noram-dir", "temp_storage", "Where -noram puts its files; each run gets its own random subfolder, deleted at the end")
	noramBlockMB := flag.Uint64("noram-hint-block-mb", 512, "-noram: MB of a client hint table filled at once per partition (up to 30 partitions at a time); bigger = fewer passes over the DB, more RAM")

	flag.Parse()
	if *noram {
		if _, err := diskstore.Init(*noramDir, *dbFileName+"_"+*searchType); err != nil {
			logrus.Fatalf("%v", err)
		}
		defer diskstore.Cleanup()
		pianopir.HintStore = diskstore.ZeroTable // client hint tables (as big as the DB for the bins) on disk too
		pianopir.HintBlockBytes = *noramBlockMB << 20
		if *searchType != "bins" {
			logrus.Warnf("-noram: only the bins keep their PIR DB on disk; for -t %s just the .npy embeddings are mmap'd", *searchType)
		}
	}
	pianopir.DownloadIfCheaper = *pirDownloadSmall // bins; the tree sets pir-old's copy from BinsConf
	if *simulate {
		switch *searchType {
		case "bins", "pacmann", "tree":
		default:
			logrus.Fatalf("-simulate only applies to the Piano PIR methods (bins, pacmann, tree), not %q", *searchType)
		}
		pianopir.Simulate = true
		pirold.Simulate = true
		logrus.Warnf("SIMULATED PIR: no hints are built and no XOR is done; every timing in metadata.json is -1")
	}

	meta := GetDatasets(*datasetsDirectory, *dbFileName)

	docEmbed := meta.Vectors.CorpusVec
	queryEmbed := meta.Vectors.QueryVec
	if *docEmbedFlag != "" {
		docEmbed = *docEmbedFlag
	}
	if *queryEmbedFlag != "" {
		queryEmbed = *queryEmbedFlag
	}
	if *stage2DataBin == "" {
		*stage2DataBin = *datasetsDirectory + "/tree/stage2/data.bin"
	}
	if *stage2IdmapBin == "" {
		*stage2IdmapBin = *datasetsDirectory + "/tree/stage2/idmap.bin"
	}
	if *layoutPath == "" {
		*layoutPath = *datasetsDirectory + "/tree/layout_with_stats.json"
	}

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
		IDBatch:           *idBatch,
		VecBatch:          *vecBatch,
		VecRounds:         *vecRounds,
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
		Text: globals.TextConf{
			Enabled:   *text,
			Format:    *textFormat,
			Order:     *textOrder,
			IDFBits:   *textIDFBits,
			TrainDocs: *textTrainDocs,
			K1:        *k1,
			B:         *bParam,
		},

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
			SearchType:     *searchType,

			Stage2DataBin:       *stage2DataBin,
			Stage2IdmapBin:      *stage2IdmapBin,
			LayoutPath:          *layoutPath,
			BlockPerm:           *blockPerm,
			DisablePIR:          *disablePIR,
			StrictPIR:           *strictPIR,
			PIRConstantWork:     *pirConstantWork,
			PIRThreads:          *pirThreads,
			CuckooLoadFactor:    *cuckooLoadFactor,
			CuckooHashes:        *cuckooHashes,
			CuckooKeyBytes:      *cuckooKeyBytes,
			CuckooUBBits:        *cuckooUBBits,
			PIRDownloadSmall:    *pirDownloadSmall,
			PIRChunkFactor:      *pirChunkFactor,
			PIRQueryBudget:      *pirQueryBudget,
			Stage3BatchSize:     *stage3BatchSize,
			CuckooMaxKicks:      *cuckooMaxKicks,
			CuckooBuildAttempts: *cuckooBuildAttempts,
			CuckooCacheDir:      *cuckooCacheDir,

			TiptoeDir:        *tiptoeDir,
			TiptoeEmbHintMB:  *tiptoeEmbHint,
			TiptoeURLHintMB:  *tiptoeURLHint,
			TiptoeMaxQueries: *tiptoeMaxQueries,
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

	if *searchType == "index" {
		logrus.Infof("Building bins index %s from %s", meta.IndexDir, meta.OriginalDir)
		start := time.Now()
		bins.Must(bins.BuildBinsIndex(meta))
		logrus.Infof("Index built in %s", time.Since(start))
		return
	}

	qids := getQIDS(config)
	config.QueryNum = uint(len(qids))

	if config.QueryNum <= 19 {
		// exit non-zero: returning here wrote no metadata.json but looked like a success to the slurm scripts
		logrus.Errorf("Only %d queries, skipping loaded from %s", config.QueryNum, config.DatasetMeta.Queries)
		diskstore.Cleanup()
		os.Exit(3)
	}

	var PIRImplemented PIRImplement
	// TODO: is it sensible to start the 'pre-processing' timer here? If so replace if with switch case!

	if *searchType == "bins" {
		if config.Text.Enabled {
			PIRImplemented = bins.MakeTextDb(&config)
		} else if config.Vectors {
			// Directly store the embeddings
			PIRImplemented = bins.MakeVecDbOld(&config)
		} else {
			PIRImplemented = bins.MakeVecDb(&config)
		}
	} else if *searchType == "pacmann" {
		PIRImplemented = Pacmann.PacmannMain(&config)
	} else if *searchType == "tree" {
		PIRImplemented = run_tree.Runtree(&config)
	} else if *searchType == "test" {
		PIRImplemented = testpir.MakeTextDb(&config)
	} else if *searchType == "tiptoe" {
		PIRImplemented = tiptoe.MakeTiptoe(&config)
	} else {
		logrus.Errorf("Invalid search type: %s", *searchType)
		return
	}
	config.DocIDMapPacmann, _ = bins.MakeDocIDAndQueryIDMap(config.DatasetMeta)

	// Some methods (tree) only answer a subset of the queries, e.g. the ones in their analyzed-queries file
	if f, ok := PIRImplemented.(interface{ FilterQIDs([]string) []string }); ok {
		qids = f.FilterQIDs(qids)
		logrus.Infof("%s answers %d of %d queries", *searchType, len(qids), config.QueryNum)
	}
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
	config.Metadata["NumQueries"] = strconv.Itoa(len(qids))
	if *noram {
		// Same answers and byte counts, but the timings include disk reads: filter these out of compute-time plots
		config.Metadata["noram"] = "true"
	}

	logrus.Infof("Answers finished in %s seconds", end.Sub(start))
	config.Metadata["TotalAnswerTime"] = end.Sub(start).String()

	//answers := make(map[string][][]uint64, config.QueryNum)
	answers := make(map[string][]string, config.QueryNum)

	if !config.Text.Enabled { // text mode decodes doc indices directly, no embedding hash lookup needed
		IDLookup = bins.MakeLookup(meta, int(*DBSize), int(*dimensions))
		config.IDLookup = IDLookup
	}

	bar := progressbar.NewOptions64(
		int64(len(encodedAnswers)),
		progressbar.OptionSetDescription("Decoding stuff"),
		progressbar.OptionShowElapsedTimeOnFinish(),
	)
	// Text mode ranks each answer right after decoding it and then drops its decoded docs (TextEntry.Release), so only
	// one answer's docs are in memory at a time. ReRankTime still counts only the BM25 ranking, as before.
	var reRanked map[string][]string
	var textRankTime time.Duration
	if config.Text.Enabled {
		reRanked = make(map[string][]string, len(encodedAnswers))
	}
	// Answers are decoded on GOMAXPROCS workers: one at a time this was the slowest part of a simulated run (text
	// bins on fever: ~0.27 s per query, 30 min for 6666 queries, longer than the PIR itself). Decode and Rank only
	// read shared state (the text model, the IDF sketch, the doc-ID maps), so the results are the same. ReRankTime
	// stays the sum of the per-query ranking times, as before.
	decodeQIDs := make([]string, 0, len(encodedAnswers))
	for qid := range encodedAnswers {
		decodeQIDs = append(decodeQIDs, qid)
	}
	var (
		decodeMu   sync.Mutex
		decodeWG   sync.WaitGroup
		rankNanos  atomic.Int64
		nextDecode atomic.Int64
	)
	for w := 0; w < runtime.GOMAXPROCS(0); w++ {
		decodeWG.Add(1)
		go func() {
			defer decodeWG.Done()
			for {
				i := int(nextDecode.Add(1) - 1)
				if i >= len(decodeQIDs) {
					return
				}
				qid := decodeQIDs[i]
				encodedAnswer := encodedAnswers[qid] // read-only while the workers run
				ids := encodedAnswer.Decode(&config)
				var ranked []string
				if config.Text.Enabled {
					if te, ok := encodedAnswer.(*bins.TextEntry); ok {
						t0 := time.Now()
						ranked = te.Rank(&config, int(config.K))
						rankNanos.Add(int64(time.Since(t0)))
						te.Release()
					} else {
						logrus.Errorf("text re-rank: answer for %s is %T, not a text entry", qid, encodedAnswer)
					}
				}
				decodeMu.Lock()
				answers[qid] = ids
				if config.Text.Enabled && ranked != nil {
					reRanked[qid] = ranked
				}
				decodeMu.Unlock()
				_ = bar.Add(1)
			}
		}()
	}
	decodeWG.Wait()
	textRankTime = time.Duration(rankNanos.Load())
	encodedAnswers = nil

	bar.Finish()

	start = time.Now()
	if config.Text.Enabled {
		TextBM25Metrics(answers, reRanked, &config)
	} else {
		reRanked = CosineReRank(answers, &config)
	}
	end = time.Now()

	config.Metadata["ReRankTime"] = (end.Sub(start) + textRankTime).String()
	markSimulated(config.Metadata, *simulate)

	if *noram {
		// what the run really needed (heap etc.), without the mmap'd file pages that peak RSS counts (slurm_common.sh)
		config.Metadata["NoRAMPeakAnonMB"] = fmt.Sprintf("%.0f", diskstore.PeakAnonMB())
	}
	oldOut := config.OutFile
	config.OutFile = fmt.Sprintf("%s/%s.json", config.OutDir, config.OutFile)
	writeAnswers(answers, config)
	config.OutFile = fmt.Sprintf("%s/%s_reRank.json", config.OutDir, oldOut)
	writeAnswers(reRanked, config)

	//stringAnwsers := Decode(answers, config)

	//if config.DataName != "debug" {

	//}

}

// markSimulated records whether the PIR was simulated and, if it was, overwrites every server timing
// (TotalAnswerTime, PreprocessingTime, MaintainenceTime, the tree's round*_seconds, their Vec/Pre copies, ...)
// with -1, so a simulated run can never pass for a real computation-time point on a plot. The network times
// (LAN/WAN) are computed from bytes and rounds, so they stay. ReRankTime is the client's local re-rank, which
// is not simulated, and GraphGenerationTime is Pacmann's graph build, which is not PIR.
func markSimulated(meta map[string]string, simulated bool) {
	meta["Simulated"] = strconv.FormatBool(simulated)
	if !simulated {
		return
	}
	for k := range meta {
		if !strings.Contains(k, "Time") && !strings.Contains(k, "_seconds") {
			continue
		}
		if strings.Contains(k, "LAN") || strings.Contains(k, "WAN") ||
			strings.Contains(k, "ReRank") || strings.Contains(k, "GraphGeneration") {
			continue
		}
		meta[k] = "-1"
	}
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

	logrus.Infof("Starting PIR search on %d queries", len(qids))

	numQueries := len(qids)
	//numQueries := 30

	decodables := make(map[string]globals.Decodable)
	maintainenceTime := time.Duration(0)

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

		finishedBatchNum, batchNumNeeded, supportBatchNum := PIRImplemented.GetBatchNums()

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

		if config.SearchType == "bins" {

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
		seen := make(map[string]bool, len(docIds))

		for _, docId := range docIds {

			if docId == "-1" { // debug signal/not found for something like pacmann
				continue
			}
			if seen[docId] {
				continue
			}
			seen[docId] = true

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
	config.Metadata["Recall"] = fmt.Sprintf("%.4f", calcRecall(new_results, qrels))
	logrus.Infof("MRR Pre re-rank: %s, Post re-rank: %s", config.Metadata["MRRPreReRank"], config.Metadata["MRR"])
	logrus.Infof("Recall Pre re-rank: %s, Post re-rank: %s", config.Metadata["RecallPreReRank"], config.Metadata["Recall"])

	return new_results

}

// TextBM25Metrics: MRR / Recall before and after the bins text mode's client-side BM25 re-rank.
// The ranking itself (TextEntry.Rank) runs in main's decode loop, one answer at a time; this records the metrics.
func TextBM25Metrics(results, reRanked map[string][]string, config *globals.Args) {
	qrels, err := bins.LoadQrels(config.DatasetMeta.Qrels)
	if err != nil {
		logrus.Errorf("Error loading qrels from %s: %v", config.DatasetMeta.Qrels, err)
	}
	config.Metadata["MRRPreReRank"] = fmt.Sprintf("%.4f", calcMRR(results, qrels))
	config.Metadata["RecallPreReRank"] = fmt.Sprintf("%.4f", calcRecall(results, qrels))

	config.Metadata["MRR"] = fmt.Sprintf("%.4f", calcMRR(reRanked, qrels))
	config.Metadata["Recall"] = fmt.Sprintf("%.4f", calcRecall(reRanked, qrels))
	logrus.Infof("MRR Pre re-rank: %s, Post BM25 re-rank: %s", config.Metadata["MRRPreReRank"], config.Metadata["MRR"])
	logrus.Infof("Recall Pre re-rank: %s, Post BM25 re-rank: %s", config.Metadata["RecallPreReRank"], config.Metadata["Recall"])
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

		seen := make(map[string]bool, len(rankedDocs))
		for _, docID := range rankedDocs {
			if seen[docID] {
				continue
			}
			seen[docID] = true
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
