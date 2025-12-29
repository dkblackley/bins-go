package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
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
	DoSearch(QID string, k int) ([][]uint64, error)
	Preprocess()
}

func main() {

	DBSize := flag.Uint("n", 8841823, "Number of items/vectors in DB")
	searchType := flag.String("t", "bins", "Search type, current options are 'bins'|'Pacmann'")
	dbFileName := flag.String("filename", "msmarco", "Identifier for the dataset to be loaded")
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

	config := globals.Args{
		DatasetsDirectory: *datasetsDirectory,
		K:                 *topK,
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

	flag.Parse()

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
	answers := doPIRSearch(PIRImplemented, qids, int(config.K))
	end = time.Now()
	logrus.Infof("Answers finished in %s seconds", end.Sub(start))

	decodedAnswers := Decode(answers, config)

	// sortedAnswers := bins.BasicReRank(decodedAnswers, config)

	writeAnswers(decodedAnswers, config)

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

	meta := bins.GetDatasets(config.DatasetsDirectory, config.DataName)
	queries, _ := bins.LoadQueries(meta.Queries)

	ids := make([]string, len(queries))
	for i, q := range queries {
		ids[i] = q.ID
	}
	return ids

}

func doPIRSearch(PIRImplimented PIRImpliment, qids []string, k int) map[string][][]uint64 {

	numQueries := len(qids)
	//numQueries := 300

	answers := make(map[string][][]uint64, numQueries)
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

		answers[q] = results

		if PIR.FinishedBatchNum >= PIR.SupportBatchNum {
			// in this case we need to re-run the preprocessing
			start := time.Now()
			PIR.Preprocessing()
			end := time.Now()
			maintainenceTime += end.Sub(start)
		}
	}
	err := bar.Finish()
	if err != nil {
		log.Fatal(err)
	}

	return answers
}

func Decode(answers map[string][][]uint64, config globals.Args) map[string][]string {
	// Each input here is going to be a map of QID to a 2d array of uint64s. We want to produce a map of QID to top-k
	// (larger than k in our case) docIDs.

	metaData := bins.GetDatasets(config.DatasetsDirectory, config.DataName)

	IDLookup := make(map[string]int)
	// TODO: THE BELOW LINE MAY NOT WORK IF USING ANN/PACMANN!!
	vectors, err := bins.LoadFloat32MatrixFromNpy(metaData.Vectors, int(config.DBSize), int(config.Dimensions))
	bins.Must(err)
	for i := 0; i < len(vectors); i++ {
		ID := HashFloat32s(vectors[i])
		IDLookup[ID] = i
	}

	docIDs := make(map[string][]string)
	empty := 0

	logrus.Debugf("Decoding answers with %d items", len(answers))

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
			multipleVectors, err := DecodeEntryToVectors(singleResult, int(config.Dimensions))
			bins.Must(err)

			// TODO: Remove this when not debug
			//if len(multipleVectors) > 0 {
			//	// Check if the first vector is all zeros
			//	isZero := true
			//	for _, val := range multipleVectors[0] {
			//		if val != 0 {
			//			isZero = false
			//			break
			//		}
			//	}
			//	if isZero {
			//		logrus.Warnf("WARNING: Decoded vector is ALL ZEROS for QID %s", qid)
			//	}
			//}

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

func DecodeEntryToVectors(entry []uint64, Dim int) ([][]float32, error) {
	if Dim <= 0 {
		return nil, errors.New("DecodeEntryToVectors: Dim must be > 0")
	}
	if len(entry) == 0 {
		return nil, errors.New("DecodeEntryToVectors: empty entry")
	}

	wordsPerVec := (Dim + 1) / 2 // 2 float32 per uint64
	if len(entry)%wordsPerVec != 0 {
		return nil, fmt.Errorf(
			"decodeEntryToVectors: len(entry)=%d not divisible by wordsPerVec=%d (Dim=%d). "+
				"Wrong Dim or PIR entry sizing mismatch",
			len(entry), wordsPerVec, Dim,
		)
	}

	maxRowSize := len(entry) / wordsPerVec

	// Trim trailing *all-zero vectors* (not trailing zero words)
	actualRows := maxRowSize
	for actualRows > 0 {
		start := (actualRows - 1) * wordsPerVec
		end := start + wordsPerVec

		allZero := true
		for _, w := range entry[start:end] {
			if w != 0 {
				allZero = false
				break
			}
		}
		if !allZero {
			break
		}
		actualRows--
	}

	// Decode only the non-padding vectors
	out := make([][]float32, actualRows)
	pos := 0
	for r := 0; r < actualRows; r++ {
		row := make([]float32, Dim)
		d := 0
		for d < Dim {
			w := entry[pos]
			pos++

			row[d] = math.Float32frombits(uint32(w))
			d++
			if d < Dim {
				row[d] = math.Float32frombits(uint32(w >> 32))
				d++
			}
		}
		out[r] = row
	}

	return out, nil
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

// For degugging, return the first n elements.
//func FirstN[T any](xs []T, n int) []T {
//	if len(xs) <= n {
//		return xs
//	}
//	return xs[:n]
//}
