package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/dkblackley/bins-go/Pacmann"
	"github.com/dkblackley/bins-go/bins"
	"github.com/dkblackley/bins-go/globals"
	"github.com/dkblackley/bins-go/pianopir"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

// const MAX_UINT32 = ^uint32(0)

type PIRImpliment interface {
	GetBatchPIRInfo() *pianopir.SimpleBatchPianoPIR
	DoSearch(QID string, k int) (globals.Decodable, error)
	Preprocess()
}

func GetDatasets(root, name string) globals.DatasetMetadata {
	vectors := globals.Vectors{
		root + "/Son/my_vectors_192.npy",
		root + "/Son/my_vectors_192_f64.npy",
		root + "/Son/query_192_float32.npy",
		root + "/Son/query_192_f64.npy",
		root + "my_vectors_192_f64_8841823_192_32_graph.npy"}

	if name == "msmarco" {
		return globals.DatasetMetadata{
			"Marco",
			"index_marco",
			root + "/msmarco/corpus.jsonl",
			root + "/msmarco/queries.dev.small.jsonl",
			root + "/msmarco/qrels/dev.tsv",
			vectors,
		}
	} else if name == "scifact" {
		return globals.DatasetMetadata{
			"SciFact",
			"index_scifact", // index folders created earlier
			root + "/scifact/corpus.jsonl",
			root + "/scifact/queries.jsonl",
			root + "/scifact/qrels/test.tsv",
			vectors,
		}
	} else if name == "debug" {
		logrus.Debugf("Using debug dataset")

		vectors.CorpusVec = root + "/Son/my_vectors_192_debug.npy"
		vectors.CorpusVec64 = root + "/Son/my_vectors_192_f64_debug.npy"
		vectors.QueryVec = root + "/Son/query_192_float32.npy"
		vectors.QueryVec64 = root + "/Son/query_192_f64_debug.npy"
		vectors.Graph = root + "/Son/debug_graph.npy"

		return globals.DatasetMetadata{
			"Marco",
			"index_marco",
			root + "/msmarco/corpus_debug.jsonl",
			root + "/msmarco/queries.dev.small_debug.jsonl",
			root + "/msmarco/qrels/dev.tsv",
			vectors,
		}
	} else {
		return globals.DatasetMetadata{
			"TREC-COVID",
			"index_trec_covid",
			root + "/trec-covid/corpus.jsonl",
			root + "/trec-covid/queries.jsonl",
			root + "/trec-covid/qrels/test.tsv",
			vectors,
		}
	}
}

func main() {

	DBSize := flag.Uint("n", 8841823, "Number of items/vectors in DB")
	searchType := flag.String("t", "bins", "Search type, current options are 'bins'|'Pacmann'")
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
	debugLevel := flag.Int("debug", 1, "Debug level, 0 for info, 1 for debug, 2 for trace and -1 for no debug")
	checkPointFolder := flag.String("checkpoint", "checkPoint", "Where to look for the checkpoint data")
	RTT := flag.Uint("RTT", 50, "RTT for the network")
	outFile := flag.String("outFile", "out.json", "Where to save the answers")

	flag.Parse()

	meta := GetDatasets(*datasetsDirectory, *dbFileName)

	IDLookup := make(map[string]int) // empty lookup

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

	logrus.SetFormatter(&logrus.TextFormatter{
		FullTimestamp: true,
	})

	logrus.Debugf("Config: %v", config)

	if *searchType == "bins" {
		IDLookup = bins.MakeLookup(meta, int(*DBSize), int(*dimensions))
		config.IDLookup = IDLookup
	}

	qids := getQIDS(config)
	config.QueryNum = uint(len(qids))

	var PIRImplemented PIRImpliment
	// TODO: is it sensible to start the 'pre-processing' timer here? If so replace if with switch case!

	if *searchType == "bins" {
		PIRImplemented = bins.MakeVecDb(config)
	} else if *searchType == "Pacmann" {
		PIRImplemented = Pacmann.PacmannMain(config)
	} else {
		logrus.Errorf("Invalid search type: %s", *searchType)
		return
	}
	start := time.Now()
	PIRImplemented.Preprocess()
	end := time.Now()
	logrus.Infof("Preprocessing finished in %s seconds", end.Sub(start))

	start = time.Now()
	encodedAnswers := doPIRSearch(PIRImplemented, qids, int(config.K))
	end = time.Now()
	logrus.Infof("Answers finished in %s seconds", end.Sub(start))

	//answers := make(map[string][][]uint64, config.QueryNum)
	answers := make(map[string][]string, config.QueryNum)

	bar := progressbar.Default(int64(len(encodedAnswers)), "Decoding stuff")
	for qid, encodedAnswer := range encodedAnswers {
		answers[qid] = encodedAnswer.Decode(config)
		bar.Add(1)
	}

	bar.Finish()

	//stringAnwsers := Decode(answers, config)

	if config.DataName != "debug" {
		bins.BasicReRank(answers, config)
	}

	writeAnswers(answers, config)

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

func doPIRSearch(PIRImplimented PIRImpliment, qids []string, k int) map[string]globals.Decodable {

	numQueries := len(qids)
	//numQueries := 300

	decodables := make(map[string]globals.Decodable)
	maintainenceTime := time.Duration(0)
	PIR := PIRImplimented.GetBatchPIRInfo()

	//start := time.Now()

	// TODO REMOVE THIS (?)
	bar := progressbar.Default(int64(numQueries), fmt.Sprintf("Answering Queries"))
	for i := 0; i < numQueries; i++ {

		err := bar.Add(1)
		if err != nil {
			log.Fatal(err)
		}
		q := qids[i]

		// Results should be a 2d array, each item in the first dimension should be a single result and then the lower
		//dimension is an item in the DB
		results, err := PIRImplimented.DoSearch(q, k)

		if err != nil {
			logrus.Errorf("Error querying PIR: %v", err)
			continue
		}

		decodables[q] = results

		if PIR.FinishedBatchNum >= PIR.SupportBatchNum {
			// in this case we need to re-run the preprocessing
			start := time.Now()
			PIR.Preprocessing()
			end := time.Now()
			maintainenceTime += end.Sub(start)
		}
	}

	logrus.Infof("Total maintainence time: %s seconds", maintainenceTime)

	err := bar.Finish()
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
