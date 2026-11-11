package run_tree

// Wrapper around the collaborator's updated tree method (bm25-tree main-updated).
// It uses the vendored PIR in ../../pir-old (NOT the new pianopir package), and
// plugs into main.go through the PIRImplement interface.
//
// The logic below is lifted from bm25-tree-main-updated/cmd/main.go with as few
// changes as possible: the per-query body of processAnalyzedQueries became DoSearch.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/dkblackley/bins-go/bins"
	"github.com/dkblackley/bins-go/globals"
	pianopir "github.com/dkblackley/bins-go/pir-old"
	"github.com/dkblackley/bins-go/tree/internal/bm25"
	"github.com/sirupsen/logrus"
)

type PIRTree struct {
	config globals.Args

	stage1DB   *bm25.Stage1PIRDB
	stage2DB   *bm25.Stage2Precomputed
	layout     *bm25.LayoutWithStats
	luceneIdx  *bm25.LuceneIndex
	stage3     *bm25.Stage3Reranker
	qrelsMap   map[string]map[string]bool
	loadedHits map[string][]bm25.HitSubBlock
	savedHits  map[string][]bm25.HitSubBlock

	analyzedQueries []AnalyzedQuery
	queryMap        map[string]AnalyzedQuery
	queryEmbIdx     map[string]int // qid -> row in the Stage3 query embedding file

	enablePIR bool
	batchSize uint64

	queryCount     int
	round1Total    time.Duration
	round2Total    time.Duration
	round3Total    time.Duration
	retrievalTotal time.Duration
}

type AnalyzedQuery struct {
	ID             string   `json:"id"`
	Text           string   `json:"text"`
	Terms          []string `json:"terms"`
	OriginalLength int      // Number of real terms before padding
}

// The collaborator's PIR does its own re-preprocessing inside Query() (it is
// counted in MaintainenceTime), so main.go never needs to trigger it.
func (P *PIRTree) GetBatchNums() (uint64, uint64, uint64) { return 0, 0, 1 }

// stage3BatchSize is -stage3-batch-size, or the original 32 when unset.
func (P *PIRTree) stage3BatchSize() uint64 {
	if P.config.BinsConf.Stage3BatchSize > 0 {
		return uint64(P.config.BinsConf.Stage3BatchSize)
	}
	return 32
}
func (P *PIRTree) PIRPreprocess() time.Duration { return 0 }

// Preprocess builds the Cuckoo tables and runs the PIR offline phase for all
// three stages, so main.go's PreprocessingTime covers it.
func (P *PIRTree) Preprocess() {
	bc := P.config.BinsConf
	cuckooConfig := bm25.CuckooBuildConfig{
		LoadFactor:    bc.CuckooLoadFactor,
		Hashes:        bc.CuckooHashes,
		KeyBytes:      bc.CuckooKeyBytes,
		ValueBits:     bc.CuckooUBBits,
		MaxKicks:      bc.CuckooMaxKicks,
		BuildAttempts: bc.CuckooBuildAttempts,
		CacheDir:      bc.CuckooCacheDir,
	}

	var err error
	P.stage1DB, err = bm25.NewStage1PIRDBWithCuckoo(bc.Stage1DataBin, bc.Stage1IdmapBin, bc.Vocab, bc.R, P.enablePIR, P.batchSize, cuckooConfig)
	if err != nil {
		logrus.Fatalf("Error loading Stage1 database: %v", err)
	}

	if !bc.SkipStage2 && bc.LoadStage2Hits == "" {
		P.stage2DB, err = bm25.NewStage2PrecomputedWithCuckoo(bc.Stage2DataBin, bc.Stage2IdmapBin, bc.Vocab, bc.B, bc.S, bc.Scale, P.enablePIR, P.batchSize, cuckooConfig)
		if err != nil {
			logrus.Fatalf("Error loading Stage2 database: %v", err)
		}
	}

	if bc.DocEmbed != "" && bc.DocIDMap != "" && bc.QueryEmbed != "" {
		sr, err := bm25.NewStage3Reranker(bc.DocEmbed, bc.DocIDMap, bc.QueryEmbed, bc.BlockPerm, bc.EmbedDim, P.enablePIR, P.stage3BatchSize())
		if err != nil {
			logrus.Warnf("failed to initialize Stage3 reranker: %v", err)
		} else {
			P.stage3 = sr
			P.buildQueryEmbIdx()
		}
	} else {
		logrus.Warnf("Stage3 disabled (need doc-embed, doc-id-map and query-embed); returning raw Stage2 sub-blocks")
	}
}

// Stage3 indexes query embeddings by row. If the embedding file has one row per
// analyzed query (the collaborator's stage3_*_query_embeddings.npy) use that
// order, otherwise assume it is the dataset's query file order (e.g. scifact_query_embed_f32.npy).
func (P *PIRTree) buildQueryEmbIdx() {
	P.queryEmbIdx = make(map[string]int)
	if len(P.stage3.QueryEmbeddings) == len(P.analyzedQueries) {
		for i, q := range P.analyzedQueries {
			P.queryEmbIdx[q.ID] = i
		}
		return
	}
	qs, err := bins.LoadQueries(P.config.DatasetMeta.Queries)
	if err != nil {
		logrus.Fatalf("Stage3: %d query embeddings vs %d analyzed queries and could not load %s: %v",
			len(P.stage3.QueryEmbeddings), len(P.analyzedQueries), P.config.DatasetMeta.Queries, err)
	}
	logrus.Infof("Stage3: %d query embeddings != %d analyzed queries, indexing embeddings by %s order",
		len(P.stage3.QueryEmbeddings), len(P.analyzedQueries), P.config.DatasetMeta.Queries)
	for i, q := range qs {
		P.queryEmbIdx[q.ID] = i
	}
}

// FilterQIDs keeps only queries that exist in the analyzed queries file (in the
// file's order) and caps them at -max-queries, like the original binary did.
func (P *PIRTree) FilterQIDs(qids []string) []string {
	want := make(map[string]bool, len(qids))
	for _, q := range qids {
		want[q] = true
	}
	maxQ := P.config.BinsConf.MaxQueries
	out := make([]string, 0)
	for _, q := range P.analyzedQueries {
		if maxQ > 0 && len(out) >= maxQ {
			break
		}
		if want[q.ID] {
			out = append(out, q.ID)
		}
	}
	return out
}

func (P *PIRTree) GetMetaData() map[string]string {
	meta := make(map[string]string)
	if P.stage1DB == nil {
		return meta
	}

	stages := []struct {
		name    string
		metrics map[string]float64
	}{
		{"Stage1", P.stage1DB.GetMetrics()},
	}
	if P.stage2DB != nil {
		stages = append(stages, struct {
			name    string
			metrics map[string]float64
		}{"Stage2", P.stage2DB.GetMetrics()})
	}
	if P.stage3 != nil {
		stages = append(stages, struct {
			name    string
			metrics map[string]float64
		}{"Stage3", P.stage3.GetMetrics()})
	}

	// Per-stage values (same names the original binary wrote) plus totals
	// under the bins/pacmann key names so the plotting scripts can read them.
	totals := make(map[string]float64)
	for _, st := range stages {
		for k, v := range st.metrics {
			meta[st.name+k] = ff(v)
			totals[k] += v
		}
	}
	for _, k := range []string{"TotalByteSent", "TotalWANTime", "TotalLANTime", "ClientStorageMB", "DBSizeInBytesMB", "CommCostPerBatchKB", "Rounds", "TotalPreProc"} {
		if v, ok := totals[k]; ok {
			meta[k] = ff(v)
		}
	}
	meta["PIRPreprocessingTime"] = ff(totals["PreprocessingTime"])
	meta["PreprocessingTime"] = ff(totals["PreprocessingTime"]) // main.go overwrites this with wall-clock after the search, the Pre copy keeps it
	meta["PIRMaintainenceTime"] = ff(totals["MaintainenceTime"])
	// main.go's own MaintainenceTime is always 0 for tree (the PIR re-preprocesses inside Query), so report the PIR's
	meta["MaintainenceTime"] = ff(totals["MaintainenceTime"])

	bc := P.config.BinsConf
	meta["FailureProbLog2"] = "20"
	meta["B"] = strconv.Itoa(bc.B)
	meta["R"] = strconv.Itoa(bc.R)
	meta["L"] = strconv.Itoa(bc.L)
	meta["S"] = strconv.Itoa(bc.S)
	meta["K"] = strconv.Itoa(int(P.config.K))
	meta["KCandidates"] = strconv.Itoa(bc.KCandidates)
	meta["PIRBatchSize"] = strconv.FormatUint(P.batchSize, 10)
	meta["PIREnabled"] = strconv.FormatBool(P.enablePIR)
	meta["query_words"] = strconv.Itoa(bc.QueryWords)
	meta["strict_pir"] = strconv.FormatBool(bm25.StrictPIR)
	meta["pir_constant_work"] = strconv.FormatBool(bm25.PIRConstantWork)
	meta["pir_threads"] = strconv.Itoa(bm25.PIRThreads)
	meta["pir_download_small"] = strconv.FormatBool(pianopir.DownloadIfCheaper)
	meta["pir_chunk_factor"] = ff(pianopir.ChunkSizeFactor)
	meta["pir_query_budget"] = ff(pianopir.QueryBudgetScale)
	meta["Stage3BatchSize"] = strconv.FormatUint(P.stage3BatchSize(), 10)
	meta["Stage3SplitQueries"] = strconv.FormatInt(bm25.Stage3SplitQueries(), 10) // queries whose Stage3 request needed >1 sub-batch (small corpora)
	meta["stage1_level_databases"] = strconv.Itoa(len(P.stage1DB.Levels))
	meta["physical_cuckoo"] = strconv.FormatBool(len(P.stage1DB.Levels) > 0)
	meta["cuckoo_hash_functions"] = "2"
	if len(P.stage1DB.Levels) > 0 && P.stage1DB.Levels[0] != nil {
		t := P.stage1DB.Levels[0].Table
		meta["cuckoo_hash_functions"] = strconv.Itoa(max(t.Hashes, 2))
		meta["cuckoo_key_bytes"] = strconv.Itoa(t.KeyBytes)
		meta["ub_bits"] = strconv.Itoa(t.ValueBits)
		meta["stage1_entry_bytes"] = strconv.FormatUint(t.EntryWords*8, 10)
		meta["cuckoo_load_factor"] = ff(float64(t.ItemCount) / float64(t.SlotCount))
	}

	if P.queryCount > 0 {
		n := float64(P.queryCount)
		meta["round1_total_seconds"] = ff(P.round1Total.Seconds())
		meta["round2_total_seconds"] = ff(P.round2Total.Seconds())
		meta["round3_total_seconds"] = ff(P.round3Total.Seconds())
		meta["retrieval_total_seconds"] = ff(P.retrievalTotal.Seconds())
		meta["round1_avg_seconds"] = ff(P.round1Total.Seconds() / n)
		meta["round2_avg_seconds"] = ff(P.round2Total.Seconds() / n)
		meta["round3_avg_seconds"] = ff(P.round3Total.Seconds() / n)
		meta["retrieval_avg_seconds"] = ff(P.retrievalTotal.Seconds() / n)
	}

	// Save Stage2 hits once queries are done (GetMetaData is called after the search loop)
	if P.savedHits != nil && len(P.savedHits) > 0 && P.config.BinsConf.SaveStage2Hits != "" {
		if data, err := json.MarshalIndent(P.savedHits, "", "  "); err == nil {
			if err := os.WriteFile(P.config.BinsConf.SaveStage2Hits, data, 0644); err != nil {
				logrus.Warnf("failed to write Stage2 hits: %v", err)
			}
		}
	}

	return meta
}

// ff formats floats without e-notation (bins/pacmann metadata uses plain numbers)
func ff(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }

type results struct {
	docIDs []string
}

func (r results) Decode(_ *globals.Args) []string { return r.docIDs }

func (P *PIRTree) DoSearch(QID string, k int) (globals.Decodable, error) {
	query, ok := P.queryMap[QID]
	if !ok {
		return nil, fmt.Errorf("query %s not in analyzed queries file", QID)
	}
	bc := P.config.BinsConf

	query.Terms, query.OriginalLength = normalizeQueryTerms(query.Terms, bc.QueryWords)
	logrus.Debugf("Query %s terms (%d real): %v", QID, query.OriginalLength, query.Terms)

	retrievalStarted := time.Now()

	start := time.Now()
	leafNodes := bm25.Round1BeamPrecomputed(P.layout, query.Terms, bc.L, P.stage1DB)
	P.round1Total += time.Since(start)

	var hitSubs []bm25.HitSubBlock
	if P.loadedHits != nil {
		hitSubs = P.loadedHits[query.ID]
	} else if P.stage2DB != nil {
		start = time.Now()
		hitSubs = bm25.Round2BMWPrecomputed(P.layout, leafNodes, query.Terms, bc.S, bc.KCandidates, P.stage2DB)
		P.round2Total += time.Since(start)
		if P.savedHits != nil {
			P.savedHits[query.ID] = hitSubs
		}
	}

	var ranked []string
	if P.stage3 != nil {
		qIdx, ok := P.queryEmbIdx[query.ID]
		if !ok {
			qIdx = -1
		}
		var gold map[string]bool
		if P.qrelsMap != nil {
			gold = P.qrelsMap[query.ID]
		}
		start = time.Now()
		docIDs, _, mrr := P.stage3.Rerank(hitSubs, qIdx, gold, P.luceneIdx)
		P.round3Total += time.Since(start)
		logrus.Debugf("Stage3 Rerank: Query %s MRR@10=%.4f, candidates=%d", query.ID, mrr, len(docIDs))
		ranked = docIDs[:min(k, len(docIDs))]
	} else {
		for _, hit := range hitSubs {
			for docID := hit.Start; docID < hit.End; docID++ {
				if P.luceneIdx != nil {
					if extID, err := P.luceneIdx.ConvertInternalToExternalID(docID); err == nil && extID != 0 {
						ranked = append(ranked, fmt.Sprintf("%d", extID))
					}
				} else {
					ranked = append(ranked, fmt.Sprintf("%d", docID))
				}
			}
		}
	}

	P.retrievalTotal += time.Since(retrievalStarted)
	P.queryCount++
	return results{docIDs: ranked}, nil
}

// Runtree loads the light-weight stuff (layout, docmap, queries, qrels). The
// heavy PIR/Cuckoo setup happens in Preprocess().
func Runtree(config *globals.Args) *PIRTree {
	bc := config.BinsConf

	bm25.Debug = bc.Debug >= 1
	bm25.StrictPIR = bc.StrictPIR
	bm25.PIRConstantWork = bc.PIRConstantWork
	if bc.PIRThreads < 1 {
		logrus.Fatalf("-pir-threads must be at least 1")
	}
	bm25.PIRThreads = bc.PIRThreads

	// client-storage knobs, all package-level in pir-old (zero values from callers
	// that predate the flags keep the original parameters)
	pianopir.DownloadIfCheaper = bc.PIRDownloadSmall
	if bc.PIRChunkFactor > 0 {
		pianopir.ChunkSizeFactor = bc.PIRChunkFactor
	}
	if bc.PIRQueryBudget > 0 {
		pianopir.QueryBudgetScale = bc.PIRQueryBudget
	}

	if bc.Index == "" || bc.Queries == "" {
		logrus.Fatalf("tree needs -index and -queries")
	}

	P := &PIRTree{
		config:    *config,
		enablePIR: bc.Precompute && !bc.DisablePIR,
		batchSize: uint64(bc.PIRBatchSize),
	}

	var err error
	if bc.Precompute {
		P.layout, err = loadLayout(bc.LayoutPath)
		if err != nil {
			logrus.Fatalf("Error loading layout: %v", err)
		}
	} else {
		P.layout = &bm25.LayoutWithStats{N: int(config.DBSize), B: bc.B, R: bc.R}
	}

	docmapPath := filepath.Join(bc.Index, "docmap.bin")
	if _, err := os.Stat(docmapPath); err == nil {
		P.luceneIdx, err = bm25.NewLuceneIndex(docmapPath)
		if err != nil {
			logrus.Warnf("failed to open docmap at %s: %v", docmapPath, err)
			P.luceneIdx = nil
		}
	} else {
		logrus.Warnf("No docmap found at %s; output will use internal IDs", docmapPath)
	}

	analyzedQueriesPath := strings.Replace(bc.Queries, ".tsv", ".analyzed.json", 1)
	data, err := os.ReadFile(analyzedQueriesPath)
	if err != nil {
		logrus.Fatalf("Error reading analyzed queries file %s: %v", analyzedQueriesPath, err)
	}
	if err := json.Unmarshal(data, &P.analyzedQueries); err != nil {
		logrus.Fatalf("Error parsing analyzed queries JSON: %v", err)
	}
	P.queryMap = make(map[string]AnalyzedQuery, len(P.analyzedQueries))
	for _, q := range P.analyzedQueries {
		P.queryMap[q.ID] = q
	}

	if bc.Qrels != "" {
		P.qrelsMap = loadQrels(bc.Qrels)
	}

	if bc.LoadStage2Hits != "" {
		raw, err := os.ReadFile(bc.LoadStage2Hits)
		if err != nil {
			logrus.Warnf("failed to read load-stage2-hits %s: %v", bc.LoadStage2Hits, err)
		} else if err := json.Unmarshal(raw, &P.loadedHits); err != nil {
			logrus.Warnf("failed to parse load-stage2-hits JSON: %v", err)
			P.loadedHits = nil
		}
	}
	if bc.SaveStage2Hits != "" {
		P.savedHits = make(map[string][]bm25.HitSubBlock)
	}

	return P
}

// normalizeQueryTerms pads or truncates the query terms to a fixed length.
// Padding uses OOV tokens: they still cost the same Cuckoo probes but score zero.
func normalizeQueryTerms(terms []string, targetLength int) ([]string, int) {
	originalLen := len(terms)
	if targetLength <= 0 {
		return terms, originalLen
	}
	normalized := make([]string, targetLength)
	if len(terms) >= targetLength {
		copy(normalized, terms[:targetLength])
		return normalized, targetLength
	}
	copy(normalized, terms)
	for i := len(terms); i < targetLength; i++ {
		normalized[i] = fmt.Sprintf("__pir_padding_%d__", i)
	}
	return normalized, originalLen
}

// loadQrels loads qrels file (columns: qid _ docid relevance, i.e. TREC format) into map[qid]set(docid)
func loadQrels(path string) map[string]map[string]bool {
	m := make(map[string]map[string]bool)
	f, err := os.Open(path)
	if err != nil {
		logrus.Warnf("failed to open qrels %s: %v", path, err)
		return m
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		parts := strings.Fields(scanner.Text())
		if len(parts) < 3 {
			continue
		}
		qid, docid := parts[0], parts[2]
		if _, ok := m[qid]; !ok {
			m[qid] = make(map[string]bool)
		}
		m[qid][docid] = true
	}
	return m
}

func loadLayout(path string) (*bm25.LayoutWithStats, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("error reading layout file: %v", err)
	}
	var raw struct {
		N        int                `json:"N"`
		B        int                `json:"B"`
		R        int                `json:"r"`
		MinDLMap map[string]float64 `json:"min_dl_map"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("error parsing layout JSON: %v", err)
	}
	l := &bm25.LayoutWithStats{N: raw.N, B: raw.B, R: raw.R, MinDLMap: make(map[int]int)}
	for k, v := range raw.MinDLMap {
		var ik int
		fmt.Sscanf(k, "%d", &ik)
		l.MinDLMap[ik] = int(v)
	}
	return l, nil
}
