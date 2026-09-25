package testpir

// testpir.go
// Timing baseline: one PIR database where entry i is the text of doc i, padded to the longest doc.
// No retrieval scheme at all. Each query fetches k uniformly random docs in a single batched PIR call.
//
// Same shape as VecBinsOld: MakeTextDb builds rawDB and wires up one SimpleBatchPianoPIR, and
// DBentryText.Decode unpacks the text and returns the doc IDs.
//
// Run it with -t test (see the main.go note at the bottom of this file).

import (
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"github.com/dkblackley/bins-go/bins"
	"github.com/dkblackley/bins-go/globals"
	"github.com/dkblackley/bins-go/pianopir"
	"github.com/sirupsen/logrus"
)

// Each entry is stored as "<docID>\x00<title> <text>" so Decode can recover the doc ID from the text itself.
const idSep = "\x00"

type TextDB struct {
	N           int    // Number of docs in the DB
	K           int    // Docs retrieved per query
	MaxDocBytes int    // Length of the longest stored string (everything is padded to this)
	DBEntrySize uint64 // Number of bytes in an entry (after padding)
	DBTotalSize uint64 // in bytes
	PIR         *pianopir.SimpleBatchPianoPIR

	rng    *rand.Rand
	config *globals.Args
}

func (t TextDB) PIRPreprocess() time.Duration {
	return t.PIR.Preprocessing()
}

func (t TextDB) Preprocess() {
	t.PIR.Preprocessing()
}

// GetBatchNums counts in per-partition sub-queries rather than batches: k random indices land unevenly across the
// K/2 partitions, so one query can cost up to K sub-queries in a partition (not 2). Reporting it this way means main's
// finished+needed >= supported check re-preprocesses before SimpleBatchPianoPIR.Query hits its own internal fallback.
func (t TextDB) GetBatchNums() (uint64, uint64, uint64) {
	pir := t.PIR
	maxQ := pir.SupportBatchNum * pianopir.QueryPerPartition // <= subPIR MaxQueryNum
	if maxQ <= 2 {
		return pir.FinishedBatchNum, pir.Config().BatchNumNeeded, pir.SupportBatchNum // not preprocessed yet
	}
	return pir.QueriesMadeInPartition, uint64(t.K), maxQ - 2 // Query() redoes preprocessing at MaxQueryNum-2
}

func (t TextDB) GetMetaData() map[string]string {
	meta := t.PIR.PrintInfo()
	meta["TextMaxDocBytes"] = strconv.Itoa(t.MaxDocBytes)
	meta["TextEntryBytes"] = strconv.FormatUint(t.DBEntrySize, 10)
	return meta
}

type DBentryText struct {
	entry [][]uint64
}

// Decode unpacks each PIR response back into its string (same layout as StringsToUint64Grid) and returns the doc ID
// stored in front of the text.
func (d DBentryText) Decode(_ *globals.Args) []string {
	docIDs := make([]string, 0, len(d.entry))
	empty := 0

	for _, row := range d.entry {
		if len(row) <= 1 { // 1-word response means the lookup was dropped
			logrus.Warnf("Got an empty result: %v - Possibly missed an entry", row)
			empty++
			continue
		}

		strs, err := bins.Uint64GridToStrings([][]uint64{row})
		if err != nil {
			logrus.Warnf("Could not decode entry: %v", err)
			continue
		}

		docID, _, ok := strings.Cut(strs[0], idSep)
		if !ok || docID == "" {
			logrus.Warnf("Decoded entry has no doc ID (len %d)", len(strs[0]))
			continue
		}
		docIDs = append(docIDs, docID)
	}

	if empty > 0 && empty == len(d.entry) {
		logrus.Errorf("All results were empty!!!!")
	}
	return docIDs
}

// DoSearch ignores the query entirely: k random docs, one batched PIR call.
func (t TextDB) DoSearch(_ string, k int) (globals.Decodable, error) {
	indices := t.MakeIndices(k)
	results, err := t.PIR.Query(indices)

	return DBentryText{
		results,
	}, err
}

// MakeIndices samples k distinct doc indices uniformly at random.
func (t TextDB) MakeIndices(k int) []uint64 {
	if k > t.N {
		k = t.N
	}

	seen := make(map[uint64]struct{}, k)
	indices := make([]uint64, 0, k)
	for len(indices) < k {
		idx := uint64(t.rng.Intn(t.N))
		if _, dup := seen[idx]; dup {
			continue
		}
		seen[idx] = struct{}{}
		indices = append(indices, idx)
	}
	return indices
}

// MakeTextDb loads the corpus, packs every doc's text into a padded []uint64 entry and builds a single PIR over them.
func MakeTextDb(config *globals.Args) TextDB {

	metaData := config.DatasetMeta

	logrus.Debugf("Loading corpus text from: %s", metaData.OriginalDir)
	docs, err := bins.LoadCorpus(metaData.OriginalDir)
	bins.Must(err)

	N := len(docs)
	if config.DBSize > 0 && int(config.DBSize) < N {
		N = int(config.DBSize)
	}

	texts := make([]string, N)
	maxDocBytes := 0
	for i := 0; i < N; i++ {
		texts[i] = docs[i].ID + idSep + docs[i].Title + " " + docs[i].Text
		if len(texts[i]) > maxDocBytes {
			maxDocBytes = len(texts[i])
		}
	}
	docs = nil

	// Pads every row to the longest string: [ length | packed bytes ... | zero padding ... ], rounded to 4 words.
	rawDB, wordsPerEntry, err := bins.StringsToUint64Grid(texts)
	bins.Must(err)
	texts = nil

	DBEntrySize := uint64(wordsPerEntry) * 8

	// BatchSize = K means one query == one batch, so FinishedBatchNum/SupportBatchNum line up with doPIRSearch.
	K := int(config.K)
	BatchSize := max(K, pianopir.RealQueryPerPartition)

	// Shrink the batch until each partition is big enough for NewSimpleBatchPianoPIR (it log.Fatals otherwise).
	for BatchSize > pianopir.RealQueryPerPartition {
		PartitionNum := BatchSize / pianopir.RealQueryPerPartition
		PartitionSize := (N + PartitionNum - 1) / PartitionNum
		if math.Sqrt(float64(PartitionSize))*math.Log(float64(PartitionSize)) >= 4*pianopir.QueryPerPartition {
			break
		}
		BatchSize--
	}
	if BatchSize != K {
		logrus.Warnf("BatchSize set to %d (K=%d)", BatchSize, K)
	}

	pir := pianopir.NewSimpleBatchPianoPIR(
		uint64(N),
		uint64(wordsPerEntry),
		DBEntrySize,
		uint64(BatchSize),
		rawDB,
		20,
		1,
	)

	config.Metadata["TextDBSize"] = strconv.Itoa(N)

	ret := TextDB{
		N:           N,
		K:           K,
		MaxDocBytes: maxDocBytes,
		DBEntrySize: DBEntrySize,
		DBTotalSize: uint64(N) * DBEntrySize,
		PIR:         pir,
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
		config:      config,
	}

	logrus.Infof("Text DB: %d docs, longest %d B, %d words/entry (%d B), padded DB %.2f GiB",
		N, maxDocBytes, wordsPerEntry, DBEntrySize, float64(ret.DBTotalSize)/(1<<30))
	logrus.Info("PIR Ready for preprocessing")

	return ret
}
