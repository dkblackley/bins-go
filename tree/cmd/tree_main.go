package run_tree

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/dkblackley/bins-go/pianopir"
	"github.com/dkblackley/bins-go/tree/internal/bm25"
	"github.com/sirupsen/logrus"

	"github.com/dkblackley/bins-go/globals"
)

type PIRTree struct {
	// Dimensions           int                      // Dimension of vectors
	// DBTotalSize          uint64                   // in bytes
	//Queries              map[string]globals.Query // A mapping from QID to query

	stage1DB   *bm25.Stage1PIRDB
	stage2DB   *bm25.Stage2Precomputed
	layout     *bm25.LayoutWithStats
	luceneIdx  *bm25.LuceneIndex
	stage3     *bm25.Stage3Reranker
	qrelsMap   map[string]map[string]bool
	loadedHits map[string][]bm25.HitSubBlock
	queryMap   map[string]AnalyzedQuery
	queryCount int
	queryWords int

	config globals.Args
}

func (P PIRTree) GetBatchNums() (uint64, uint64, uint64) {

	pirs := []struct {
		FinishedBatchNum uint64
		SupportBatchNum  uint64
		BatchNumNeeded   uint64 // Assuming Config returns a struct with BatchNumNeeded
	}{
		{P.stage1DB.Pir.FinishedBatchNum, P.stage1DB.Pir.SupportBatchNum, P.stage1DB.Pir.Config().BatchNumNeeded},
		{P.stage2DB.Pir.FinishedBatchNum, P.stage2DB.Pir.SupportBatchNum, P.stage1DB.Pir.Config().BatchNumNeeded},
		{P.stage3.Pir.FinishedBatchNum, P.stage3.Pir.SupportBatchNum, P.stage1DB.Pir.Config().BatchNumNeeded},
	}

	for _, pir := range pirs {
		f := pir.FinishedBatchNum
		n := pir.BatchNumNeeded
		s := pir.SupportBatchNum

		if f+n >= s {
			return f, n, s
		}
	}

	lastPir := pirs[len(pirs)-1]
	return lastPir.FinishedBatchNum, lastPir.BatchNumNeeded, lastPir.SupportBatchNum
}

func (P *PIRTree) PIRPreprocess() time.Duration {
	var total time.Duration

	pirs := []*pianopir.SimpleBatchPianoPIR{
		P.stage1DB.Pir,
		P.stage2DB.Pir,
		P.stage3.Pir,
	}

	for _, pir := range pirs {
		f := pir.FinishedBatchNum
		n := pir.Config().BatchNumNeeded
		s := pir.SupportBatchNum

		if f+n >= s {
			total += pir.Preprocessing()
		}
	}

	return total
}

func (P *PIRTree) GetMetaData() map[string]string {
	// Initialize the combined map
	meta := make(map[string]string)

	// Define the stages and their corresponding suffixes
	stages := []struct {
		pir    *pianopir.SimpleBatchPianoPIR
		suffix string
	}{
		{P.stage1DB.Pir, "_stage1"},
		{P.stage2DB.Pir, "_stage2"},
		{P.stage3.Pir, "_stage3"},
	}

	// Loop through each stage, retrieve info, append suffix, and merge
	for _, stage := range stages {
		info := stage.pir.PrintInfo()
		for k, v := range info {
			meta[k+stage.suffix] = v
		}
	}

	return meta
}

func (P *PIRTree) DoSearch(QID string, k int) (globals.Decodable, error) {

	query := P.queryMap[QID]

	temp := processAnalyzedQuery(P.layout, P.stage1DB, P.stage2DB, P.luceneIdx, P.config.BinsConf.L, P.config.BinsConf.S, P.config.BinsConf.KCandidates, P.config.BinsConf.K, P.stage3, P.config.BinsConf.SkipStage2, P.queryWords, query, P.loadedHits, P.queryCount, P.qrelsMap)
	P.queryCount++

	return temp, nil
}

func (P PIRTree) Preprocess() {

	pirs := []*pianopir.SimpleBatchPianoPIR{

		P.stage1DB.Pir,
		P.stage2DB.Pir,
		P.stage3.Pir,
	}

	for _, pir := range pirs {

		pir.Preprocessing()
	}

}

func Runtree(config globals.Args) *PIRTree {

	binsconf := config.BinsConf

	index := binsconf.Index
	queries := binsconf.Queries
	B := binsconf.B
	r := binsconf.R
	// L := binsconf.L
	s := binsconf.S
	// k := binsconf.K
	// kCandidates := binsconf.KCandidates
	// maxQueries := binsconf.MaxQueries
	k1 := binsconf.K1
	bParam := binsconf.BParam
	qrels := binsconf.Qrels
	precompute := binsconf.Precompute
	stage1DataBin := binsconf.Stage1DataBin
	stage1IdmapBin := binsconf.Stage1IdmapBin
	vocab := binsconf.Vocab
	scale := binsconf.Scale
	pirBatchSize := binsconf.PIRBatchSize
	docEmbed := binsconf.DocEmbed
	docIDMap := binsconf.DocIDMap
	queryEmbed := binsconf.QueryEmbed
	embedDim := binsconf.EmbedDim
	debug := binsconf.Debug
	// queryWords := binsconf.QueryWords
	loadStage2Hits := binsconf.LoadStage2Hits
	saveStage2Hits := binsconf.SaveStage2Hits
	skipStage2 := binsconf.SkipStage2

	dataRoot := config.DatasetsDirectory

	// TODO: Change all print to logrus statements

	logrus.Debugln(stage1DataBin, stage1IdmapBin, scale)

	if debug >= 1 {
		bm25.Debug = true
	}
	// Validate required flags (qrels optional for Stage3-only runs)
	if index == "" || queries == "" {
		logrus.Errorln("Required flags: --index, --queries")
		flag.Usage()
		os.Exit(1)
	}
	if qrels == "" {
		fmt.Println("Warning: --qrels not provided; evaluation (MRR) will be disabled")
	}

	// Create BM25, parameters
	params := &bm25.BM25Params{
		K1: k1,
		B:  bParam,
	}
	fmt.Printf("BM25, Parameters: K1=%.2f, B=%.2f\n", params.K1, params.B)

	// Load Stage1, database
	// Enable PIR if using precomputation
	enablePIR := precompute
	batchSize := uint64(pirBatchSize)

	var err error

	// Load Stage1, database with optional PIR support
	stage1DB, err := bm25.NewStage1PIRDBWithPIR(stage1DataBin, stage1IdmapBin, vocab, r, enablePIR, batchSize)
	if err != nil {
		fmt.Printf("Error loading Stage1, database: %v\n", err)
		os.Exit(1)
	}
	defer stage1DB.Close()

	// Load Stage2, database (skip if requested or if loading saved hitSubs)
	var stage2DB *bm25.Stage2Precomputed = nil
	if !skipStage2 && loadStage2Hits == "" {
		stage2DB, err = bm25.NewStage2PrecomputedWithPIR(
			dataRoot+"/tree/stage2/data.bin",
			dataRoot+"/tree/stage2/idmap.bin",
			vocab,
			B,
			s,
			scale,
			enablePIR, // Use same enablePIR flag as Stage1
			batchSize,
		)
		if err != nil {
			fmt.Printf("Error loading Stage2, database: %v\n", err)
			os.Exit(1)
		}
		defer stage2DB.Close()
	} else if loadStage2Hits != "" {
		fmt.Printf("Loading Stage2 hits from %s; skipping Stage2 DB initialization\n", loadStage2Hits)
	} else if skipStage2 {
		fmt.Printf("Skipping Stage2 DB initialization as requested (--skip-stage2)\n")
	}

	if enablePIR {
		fmt.Printf("PIR enabled with batch size %d\n", batchSize)
	}

	// Load or create layout
	var layout *bm25.LayoutWithStats
	if precompute {
		// Load from file
		layout, err = loadLayout(dataRoot + "/tree/layout_with_stats.json")
		if err != nil {
			fmt.Printf("Error loading layout: %v\n", err)
			os.Exit(1)
		}
	} else {
		// Create new layout
		layout = &bm25.LayoutWithStats{
			N: getNumDocs(index),
			B: B,
			R: r,
		}
		// Precompute stats would go here if we had Lucene adapter
	}

	// If a docmap exists in the index directory, load it so we can write external IDs
	var luceneIdx *bm25.LuceneIndex
	docmapPath := filepath.Join(index, "docmap.bin")
	if _, err := os.Stat(docmapPath); err == nil {
		li, err := bm25.NewLuceneIndex(docmapPath)
		if err != nil {
			fmt.Printf("Warning: failed to open docmap at %s: %v\n", docmapPath, err)
		} else {
			luceneIdx = li
			defer luceneIdx.Close()
		}
	} else {
		fmt.Printf("No docmap found at %s; output will use internal IDs unless you provide a docmap.bin\n", docmapPath)
	}

	// Print some vocab information for debugging (only if Stage2 DB initialized)
	if bm25.Debug {
		if stage2DB != nil {
			fmt.Printf("\nStage 2 Vocab Info:\n")
			fmt.Printf("Total terms: %d\n", len(stage2DB.Vocab))
			for term, id := range stage2DB.Vocab {
				if term == "androgen" || term == "receptor" || term == "define" {
					fmt.Printf("Found term '%s' with id %d\n", term, id)
				}
			}
		} else {
			fmt.Printf("\nStage 2 DB not initialized (using loaded Stage2 hits or --skip-stage2)\n")
		}
	}

	analyzedQueriesPath := strings.Replace(queries, ".tsv", ".analyzed.json", 1)
	// processAnalyzedQueries(layout, stage1DB, stage2DB, luceneIdx, analyzedQueriesPath, maxQueries, L, s, kCandidates, k, stage3, qrels, loadStage2Hits, saveStage2Hits, skipStage2, queryWords)

	// Read analyzed queries JSON file
	data, err := os.ReadFile(analyzedQueriesPath)
	if err != nil {
		logrus.Errorf("Error reading analyzed queries file: %v\n", err)
		os.Exit(1)
	}

	var analyzedQueries []AnalyzedQuery
	if err := json.Unmarshal(data, &analyzedQueries); err != nil {
		logrus.Errorf("Error parsing analyzed queries JSON: %v\n", err)
		os.Exit(1)
	}
	var stage3Idx = make([]string, len(analyzedQueries))

	queryMap := make(map[string]AnalyzedQuery)
	for i, query := range analyzedQueries {
		queryMap[query.ID] = query
		stage3Idx[i] = query.ID
	}

	// var stage1DB *bm25.Stage1PIRDB = nil
	// var stage2DB *bm25.Stage2Precomputed = nil
	// Initialize Stage3 reranker if embeddings provided
	var stage3 *bm25.Stage3Reranker = nil
	if docEmbed != "" && docIDMap != "" && queryEmbed != "" {
		sr, err := bm25.NewStage3Reranker(docEmbed, docIDMap, queryEmbed, dataRoot+"/tree/block_permutations.json", embedDim, enablePIR, uint64(32), stage3Idx)
		// I think there is a difference in the files that I and Son use! My ones seem to be aligned and his are not?
		// sr, err := bm25.NewStage3Reranker(docEmbed, docIDMap, queryEmbed, "", embedDim, enablePIR, uint64(32), stage3Idx)
		if err != nil {
			fmt.Printf("Warning: failed to initialize Stage3 reranker: %v\n", err)
		} else {
			stage3 = sr
			defer stage3.Close()
		}
	}

	// Load qrels (optional) for evaluation and MRR calculation
	var qrelsMap map[string]map[string]bool
	if qrels != "" {
		qrelsMap = loadQrels(qrels)
		if bm25.Debug {
			fmt.Printf("DEBUG: loaded %d qids from qrels (%s)\n", len(qrelsMap), qrels)
			if v, ok := qrelsMap["1049085"]; ok {
				fmt.Printf("DEBUG: qrels[1049085] has %d entries\n", len(v))
			} else {
				i := 0
				for k := range qrelsMap {
					if i == 0 {
						fmt.Printf("DEBUG: sample qid from qrels: %s\n", k)
					} else if i < 3 {
						fmt.Printf("DEBUG: another qid: %s\n", k)
					} else {
						break
					}
					i++
				}
			}
		}
	}

	// If requested, load saved Stage2 hits (map qid -> []HitSubBlock)
	var loadedHits map[string][]bm25.HitSubBlock = nil
	if loadStage2Hits != "" {
		raw, err := os.ReadFile(saveStage2Hits)
		if err != nil {
			fmt.Printf("Warning: failed to read load-stage2-hits %s: %v\n", loadStage2Hits, err)
		} else {
			if err := json.Unmarshal(raw, &loadedHits); err != nil {
				fmt.Printf("Warning: failed to parse load-stage2-hits JSON: %v\n", err)
				loadedHits = nil
			} else {
				fmt.Printf("Loaded saved Stage2 hits for %d queries\n", len(loadedHits))
			}
		}
	}

	// If requested, prepare map to collect hits to save
	//var savedHits map[string][]bm25.HitSubBlock = nil
	//if saveStage2Hits != "" {
	//	savedHits = make(map[string][]bm25.HitSubBlock)
	//}

	return &PIRTree{
		stage1DB:   stage1DB,
		stage2DB:   stage2DB,
		layout:     layout,
		luceneIdx:  luceneIdx,
		stage3:     stage3,
		config:     config,
		qrelsMap:   qrelsMap,
		loadedHits: loadedHits,
		queryMap:   queryMap,
		queryCount: 0,
	}

}

type AnalyzedQuery struct {
	ID             string   `json:"id"`
	Text           string   `json:"text"`
	Terms          []string `json:"terms"`
	OriginalLength int      // Number of real terms before padding
}

// normalizeQueryTerms pads or truncates the query terms to a fixed length
// Returns: (paddedTerms, originalLength)
// Note: Padding words should NOT be used for scoring, only for PIR privacy
func normalizeQueryTerms(terms []string, targetLength int) ([]string, int) {
	originalLen := len(terms)
	if targetLength <= 0 {
		return terms, originalLen
	}

	normalized := make([]string, targetLength)

	if len(terms) >= targetLength {
		// Truncate if too many terms
		copy(normalized, terms[:targetLength])
		return normalized, targetLength
	} else {
		// Copy existing terms
		copy(normalized, terms)
		// Pad with a word from vocabulary (for PIR privacy)
		// Note: "what" has low IDF but will still affect scores!
		// TODO: Use a special padding token with no postings
		for i := len(terms); i < targetLength; i++ {
			normalized[i] = "what"
		}
		return normalized, originalLen
	}
}

type results struct {
	docIDs    []string
	hits      []bm25.HitSubBlock
	luceneIdx *bm25.LuceneIndex
}

func (r results) Decode(_ globals.Args) []string {

	if r.hits == nil {
		return r.docIDs
	}

	var stringIDs []string
	docIDs := make(map[int]bool)
	for _, hit := range r.hits {
		for docID := hit.Start; docID < hit.End; docID++ {
			docIDs[docID] = true
		}
	}

	for docID := range docIDs {
		if r.luceneIdx != nil {
			extID, err := r.luceneIdx.ConvertInternalToExternalID(docID)
			if err != nil || extID == 0 {
				// skip unmapped IDs
				logrus.Warnf("Warning: failed to convert internal docID %d to external ID, skipping\n", docID)
				continue
			}
			stringIDs = append(stringIDs, fmt.Sprintf("%d", docID))
		} else {
			stringIDs = append(stringIDs, fmt.Sprintf("%d", docID))
		}
	}

	return stringIDs

}

func processAnalyzedQuery(
	layout *bm25.LayoutWithStats,
	stage1DB *bm25.Stage1PIRDB,
	stage2DB *bm25.Stage2Precomputed,
	luceneIdx *bm25.LuceneIndex,
	L int,
	s int,
	kCandidates int,
	k int,
	stage3 *bm25.Stage3Reranker,
	skipStage2 bool,
	queryWords int,
	query AnalyzedQuery,
	loadedHits map[string][]bm25.HitSubBlock,
	queryCount int,
	qrelsMap map[string]map[string]bool,
) results {

	// Normalize query terms to fixed length (for PIR privacy) - This should no longer be an issue? TODO: Don't pad
	// But track original length so padding doesn't affect scores
	// query.Terms, query.OriginalLength = normalizeQueryTerms(query.Terms, queryWords)
	//realTerms := query.Terms[:query.OriginalLength] // Terms to use for actual scoring
	realTerms := query.Terms
	query.OriginalLength = len(realTerms)

	if bm25.Debug || true {
		fmt.Printf("  Original terms (%d): %v\n", query.OriginalLength, realTerms)
		if len(query.Terms) > query.OriginalLength {
			fmt.Printf("  Padded terms  (%d): %v\n", len(query.Terms), query.Terms)
		}
	}

	if bm25.Debug && stage2DB != nil {
		// Debug: check how many query terms exist in the Stage-2 vocab (precomputed)
		matched := 0
		fmt.Printf("DEBUG Query %s terms: %v\n", query.ID, query.Terms)
		for _, t := range query.Terms {
			if _, ok := stage2DB.Vocab[t]; ok {
				matched++
				fmt.Printf("  Term '%s' found in vocab\n", t)
			} else {
				fmt.Printf("  Term '%s' NOT found in vocab\n", t)
			}
		}
		fmt.Printf("  Total matched terms: %d/%d\n", matched, len(query.Terms))
	}

	// Round 1: Beam search (use only real terms, not padding)
	// time the runtime
	start := time.Now()
	// TODO: EXTRACT PIR INFO FROM HERE
	leafNodes := bm25.Round1BeamPrecomputed(layout, realTerms, L, stage1DB)
	elapsed := time.Since(start)
	fmt.Printf("Round 1, took %s\n", elapsed)

	// fmt.Println("Round 1, completed. Result:", leafNodes) // Debug output removed
	// Round 2: Sub-block selection
	// leafNodes := []int{2193474, 2189759, 2121932, 2174210, 2315792, 2291417, 2166755, 2315801, 2124788, 2122040, 2278572, 2230491, 2139274, 2180257, 2121978, 2144827, 2189796, 2247590, 2166840, 2157385, 2114415, 2269328, 2366289, 2121935, 2311483, 2247561, 2151012, 2278378, 2163791, 2237401, 2121720, 2234741, 2315746, 2161818, 2267755, 2121946, 2176059, 2128599, 2263975, 2132566, 2234833, 2352127, 2237396, 2127324, 2352117, 2128587, 2311967, 2288806, 2315760, 2289406, 2187854, 2366165, 2147264, 2366304, 2132352, 2119908, 2363843, 2272120, 2286738, 2121890, 2116528, 2251177, 2166632, 2318993, 2114029, 2315321, 2142479, 2114395, 2243199, 2259383, 2366256, 2246155, 2281334, 2352118, 2352116, 2266854, 2165070, 2237390, 2298530, 2266850, 2166976, 2187763, 2210312, 2164317, 2166754, 2210216, 2174761, 2190295, 2304150, 2204643, 2144795, 2146004, 2285221, 2351073, 2114211, 2197911, 2210331, 2210171, 2234869, 2227354, 2164445, 2164473, 2116617, 2315781, 2117996, 2168640, 2126629, 2287422, 2125017, 2146044, 2203212, 2136259, 2210355, 2193462, 2210145, 2210207, 2144745, 2227402, 2166843, 2166812, 2203267, 2187869, 2204451, 2166899, 2210189, 2166686, 2164397, 2251449, 2126713, 2114276, 2286775, 2119964, 2315275, 2166794, 2266856, 2189219, 2099952, 2153098, 2345976, 2266771, 2267787, 2228613, 2244799, 2247519, 2278443, 2183363, 2164403, 2204874, 2164304, 2187868, 2164524, 2136302, 2344588, 2278418, 2164296, 2207829, 2164324, 2125018, 2164368, 2124661, 2279515, 2144831, 2196621, 2285260, 2285361, 2285203, 2124677, 2124756, 2141663, 2187231, 2317965, 2116570, 2210154, 2180657, 2136276, 2193421, 2174214, 2203320, 2315780, 2193468, 2193380, 2147542, 2147499, 2251777, 2211971, 2187725, 2107209, 2114106, 2114097, 2111459, 2187247, 2206143, 2190322, 2124616, 2203321, 2164341, 2234718, 2203237, 2137579, 2315794}
	var hitSubs []bm25.HitSubBlock
	// If loaded hits are available, use them
	if loadedHits != nil {
		if hs, ok := loadedHits[query.ID]; ok {
			hitSubs = hs
		} else {
			fmt.Printf("Warning: no loaded Stage2 hits for query %s\n", query.ID)
			hitSubs = nil
		}
	} else if skipStage2 {
		// Skipping Stage2: no hits
		hitSubs = nil
	} else {
		// record runtime (use only real terms, not padding)
		timeStart := time.Now()
		// TODO: EXTRACT PIR INFO FROM HERE
		hitSubs = bm25.Round2BMWPrecomputed(layout, leafNodes, realTerms, s, kCandidates, stage2DB)
		timeElapsed := time.Since(timeStart)
		fmt.Printf("Round 2, took %s\n", timeElapsed)
	}

	// Debug: log counts to diagnose recall differences
	if bm25.Debug {
		fmt.Printf("Query %s: leafNodes=%d, hitSubBlocks=%d\n", query.ID, len(leafNodes), len(hitSubs))

		if len(hitSubs) > 0 {
			// Print first few hit blocks to check ranges
			fmt.Printf("First few hit blocks for query %s:\n", query.ID)
			for i := 0; i < min(5, len(hitSubs)); i++ {
				fmt.Printf("  Block %d: start=%d, end=%d, score=%f\n",
					i, hitSubs[i].Start, hitSubs[i].End, hitSubs[i].ScoreUB)
			}
		}

		// Count total docs in hitSubs for debugging
		totalDocs := 0
		for _, hit := range hitSubs {
			totalDocs += hit.End - hit.Start
		}
		fmt.Printf("DEBUG: Query %s has %d hitSubs containing %d total docs\n", query.ID, len(hitSubs), totalDocs)
	}

	// If stage3 reranker is available, rerank and write reranked results
	if stage3 != nil {
		// Assume query embeddings are in same order as analyzedQueries; use queryCount as index
		// qIdx := queryCount
		// Prepare gold set for Stage3 (if available) so Stage3 can include debug and MRR
		var goldSetForStage3 map[string]bool = nil
		if qrelsMap != nil {
			goldSetForStage3 = qrelsMap[query.ID]
		}

		// Check whether any gold doc exists in the Stage2 candidate pool (hitSubs)
		if goldSetForStage3 != nil && len(goldSetForStage3) > 0 {
			found := false
			// create a small helper to test a single internal docID
			for _, hs := range hitSubs {
				for docID := hs.Start; docID < hs.End; docID++ {
					// Try luceneIdx conversion first (internal -> external)
					if luceneIdx != nil {
						if ext, err := luceneIdx.ConvertInternalToExternalID(docID); err == nil && ext != 0 {
							if goldSetForStage3[fmt.Sprintf("%d", ext)] {
								found = true
								break
							}
						}
					}
					// If stage3 has a DocIDMap, try that mapping
					if !found && stage3 != nil && len(stage3.DocIDMap) > docID {
						if goldSetForStage3[stage3.DocIDMap[docID]] {
							found = true
							break
						}
					}
					// As a fallback, compare internal numeric id string
					if !found {
						if goldSetForStage3[fmt.Sprintf("%d", docID)] {
							found = true
							break
						}
					}
				}
				if found {
					break
				}
			}
			if bm25.Debug {
				if found {
					fmt.Printf("DEBUG: gold doc FOUND in Stage2 candidates for query %s\n", query.ID)
				} else {
					fmt.Printf("DEBUG: gold doc NOT FOUND in Stage2 candidates for query %s\n", query.ID)
				}
			}
		}
		// Get reranked IDs/scores from Stage3

		start = time.Now()
		// TODO: MAKE THIS INTO POSTPROC FOR ALL QUERIES(?)
		docIDs, _, mrrFromStage3 := stage3.Rerank(hitSubs, query.ID, goldSetForStage3, luceneIdx)
		elapsed = time.Since(start)
		fmt.Printf("Round 3, took %s\n", elapsed)

		// Stage3 reranker already returns external doc IDs via DocIDMap, so no conversion needed
		fmt.Printf("Stage3 Rerank: Query %s MRR@10=%.4f, candidates=%d\n", query.ID, mrrFromStage3, len(docIDs))
		return results{docIDs: docIDs, luceneIdx: nil, hits: nil}
		// write top-k reranked results (Stage3 already returns external IDs)
		//if len(docIDs) > 0 {
		//	writeResultsReranked(query.ID, docIDs, scores, k)
		//}
	} else {

		return results{docIDs: nil, luceneIdx: luceneIdx, hits: hitSubs}

		// Write original hitSubs results
		//writeResults(query.ID, hitSubs, k, luceneIdx)
	}

	// If we collected Stage2 hits to save, write them out as JSON
	//if savedHits != nil && saveStage2HitsPath != "" {
	//	data, err := json.MarshalIndent(savedHits, "", "  ")
	//	if err != nil {
	//		fmt.Printf("Warning: failed to marshal saved Stage2 hits: %v\n", err)
	//	} else {
	//		if err := os.WriteFile(saveStage2HitsPath, data, 0644); err != nil {
	//			fmt.Printf("Warning: failed to write saved Stage2 hits to %s: %v\n", saveStage2HitsPath, err)
	//		} else {
	//			fmt.Printf("Saved Stage2 hits to %s\n", saveStage2HitsPath)
	//		}
	//	}
	//}
}

// loadQrels loads qrels file (tsv with columns: qid docid relevance) into a map[qid]set(docid)
func loadQrels(path string) map[string]map[string]bool {
	m := make(map[string]map[string]bool)
	f, err := os.Open(path)
	if err != nil {
		fmt.Printf("Warning: failed to open qrels %s: %v\n", path, err)
		return m
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)
		if len(parts) < 3 {
			continue
		}
		qid := parts[0]
		docid := parts[2]
		if _, ok := m[qid]; !ok {
			m[qid] = make(map[string]bool)
		}
		m[qid][docid] = true
	}
	if err := scanner.Err(); err != nil {
		fmt.Printf("Warning: error reading qrels %s: %v\n", path, err)
	}
	return m
}

// writeResultsReranked writes reranked results to a file `approx_precompute_reranked_<k>.tsv`
func writeResultsReranked(qid string, docIDs []string, scores []float32, k int) {
	fname := fmt.Sprintf("approx_precompute_reranked_%d.tsv", k)
	f, err := os.OpenFile(fname, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("Error opening reranked results file: %v\n", err)
		return
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	limit := k
	if len(docIDs) < limit {
		limit = len(docIDs)
	}
	for i := 0; i < limit; i++ {
		fmt.Fprintf(w, "%s\t%s\t%f\n", qid, docIDs[i], scores[i])
	}
	w.Flush()
}

func loadLayout(path string) (*bm25.LayoutWithStats, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("error reading layout file: %v", err)
	}

	// expected JSON fields: N, B, r, min_dl_map
	var raw struct {
		N        int                `json:"N"`
		B        int                `json:"B"`
		R        int                `json:"r"`
		MinDLMap map[string]float64 `json:"min_dl_map"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("error parsing layout JSON: %v", err)
	}

	l := &bm25.LayoutWithStats{
		N:        raw.N,
		B:        raw.B,
		R:        raw.R,
		MinDLMap: make(map[int]int),
	}
	for k, v := range raw.MinDLMap {
		var ik int
		fmt.Sscanf(k, "%d", &ik)
		l.MinDLMap[ik] = int(v)
	}
	return l, nil
}

func getNumDocs(indexPath string) int {
	// This is a placeholder - implement actual index stats retrieval
	return 8841823
}

// Removed unused function processQueries

func writeResults(qid string, hits []bm25.HitSubBlock, k int, luceneIdx *bm25.LuceneIndex) int {
	// Open results file in append mode
	f, err := os.OpenFile(fmt.Sprintf("approx_precompute_%d.tsv", k), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("Error opening results file: %v\n", err)
		return 0
	}
	defer f.Close()

	// First collect all unique docIDs from the hits
	docIDs := make(map[int]bool)
	for _, hit := range hits {
		for docID := hit.Start; docID < hit.End; docID++ {
			docIDs[docID] = true
		}
	}

	// Write results for all collected docIDs
	writer := bufio.NewWriter(f)
	written := 0
	for docID := range docIDs {
		if luceneIdx != nil {
			extID, err := luceneIdx.ConvertInternalToExternalID(docID)
			if err != nil || extID == 0 {
				// skip unmapped IDs
				logrus.Warnf("Unable to convert internal docID %d to external ID when writing results, skipping")
				continue
			}
			fmt.Fprintf(writer, "%s\t%d\t1.0\n", qid, extID)
		} else {
			fmt.Fprintf(writer, "%s\t%d\t1.0\n", qid, docID)
		}
		written++
	}
	writer.Flush()
	return written
}
