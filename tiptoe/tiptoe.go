//go:build tiptoe

// Package tiptoe runs the real Tiptoe protocol (Henzinger et al., SOSP'23, github.com/ahenzinger/tiptoe) behind the
// same PIRImplement interface as bins/tree/pacmann, so it is timed, scored and written to metadata.json in exactly
// the same way.
//
// Nothing cryptographic is re-implemented here: corpus loading, database layout, the SimplePIR-based linearly
// homomorphic ranking service, the URL PIR service and the "underhood" hint-compression (the per-query token that
// removes the client-side SimplePIR hint) are all upstream code. This file only does what Tiptoe's coordinator and
// client binaries do, in-process, with a single embedding server and a single URL server:
//
//	offline (per query): client builds a token query -> server applies the hint under RLWE -> client recovers it
//	round 1:             client picks the nearest centroid, sends an LHE-encrypted query embedding for that
//	                     cluster -> server returns encrypted inner products for every doc -> client decrypts the
//	                     scores of the docs in its cluster
//	round 2:             client PIR-fetches the URL bundle holding its best doc and returns the docs of that bundle,
//	                     ordered by score (exactly what search/protocol/client.go runRound prints as its top results)
//
// The data is produced by baselines/tiptoe/preprocess_tiptoe.py from the same embeddings PILLAR uses.
//
// Build with: go build -tags tiptoe (needs Microsoft SEAL 4.1, see baselines/README.md).
package tiptoe

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"

	"github.com/ahenzinger/tiptoe/search/config"
	"github.com/ahenzinger/tiptoe/search/corpus"
	"github.com/ahenzinger/tiptoe/search/embeddings"
	"github.com/ahenzinger/tiptoe/search/protocol"
	tutils "github.com/ahenzinger/tiptoe/search/utils"
	"github.com/ahenzinger/underhood/underhood"
	"github.com/henrycg/simplepir/matrix"
	"github.com/henrycg/simplepir/pir"
	"github.com/kshedden/gonpy"
	"github.com/sirupsen/logrus"

	"github.com/dkblackley/bins-go/bins"
	"github.com/dkblackley/bins-go/globals"
)

// Same network model as pianopir.netTime, so LAN/WAN numbers are comparable across methods.
func netTime(bytes, rounds uint64, mbps, rttSec float64) float64 {
	return float64(bytes)*8/(mbps*1e6) + float64(rounds)*rttSec
}

type preprocMeta struct {
	NumDocs     int     `json:"num_docs"`
	Dim         int     `json:"dim"`
	NumClusters int     `json:"num_clusters"`
	SlotBits    int     `json:"slot_bits"`
	Scale       float64 `json:"scale"`
	MultiAssign int     `json:"multi_assign"`
	BundleSize  int     `json:"bundle_size"`
	NumBundles  int     `json:"num_bundles"`
	MaxCluster  int     `json:"max_cluster_size"`
}

type TiptoeSearch struct {
	maxQueries int
	meta       preprocMeta
	dir        string
	conf       *config.Config
	centroids  [][]float32
	queryEmb   map[string][]float32
	docIDs     map[int]string

	embServer *protocol.Server
	urlServer *protocol.Server
	hint      *protocol.TiptoeHint
	embHint   *underhood.Server[matrix.Elem64]
	urlHint   *underhood.Server[matrix.Elem32]
	client    *protocol.Client

	tokenReady bool

	// counters, summed over the run
	offlineUp, offlineDown uint64
	onlineUp, onlineDown   uint64
	offlineRounds          uint64
	onlineRounds           uint64
	offlineTime            time.Duration
	round1Time, round2Time time.Duration
	clientTime             time.Duration
	numSearches            uint64
	numTokens              uint64
	clientStateBytes       uint64
	tokenBytesLast         uint64
	dbBytes                uint64
}

// DBentryIDs is already decoded: Tiptoe returns URLs (here: corpus row indices) which are mapped to doc IDs.
type DBentryIDs struct{ ids []string }

func (d DBentryIDs) Decode(_ *globals.Args) []string { return d.ids }

func loadNpy(path string) ([][]float32, error) {
	r, err := gonpy.NewFileReader(path)
	if err != nil {
		return nil, err
	}
	if len(r.Shape) != 2 {
		return nil, fmt.Errorf("%s: expected a 2-D array, got shape %v", path, r.Shape)
	}
	flat, err := r.GetFloat32()
	if err != nil {
		return nil, err
	}
	n, dim := r.Shape[0], r.Shape[1]
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		out[i] = flat[i*dim : (i+1)*dim]
	}
	return out, nil
}

// MakeTiptoe loads the preprocessed Tiptoe corpus from config.BinsConf.TiptoeDir and builds both servers + the client.
func MakeTiptoe(cfg *globals.Args) *TiptoeSearch {
	dir := cfg.BinsConf.TiptoeDir
	if dir == "" {
		logrus.Fatalf("-tiptoe-dir is required (output of baselines/tiptoe/preprocess_tiptoe.py)")
	}
	t := &TiptoeSearch{dir: dir, maxQueries: cfg.BinsConf.TiptoeMaxQueries}

	raw, err := os.ReadFile(filepath.Join(dir, "tiptoe_meta.json"))
	bins.Must(err)
	bins.Must(json.Unmarshal(raw, &t.meta))
	if uint64(t.meta.SlotBits) != config.SLOT_BITS() {
		logrus.Fatalf("slot bits %d != Tiptoe's SLOT_BITS %d", t.meta.SlotBits, config.SLOT_BITS())
	}

	t.centroids, err = loadNpy(filepath.Join(dir, "centroids.npy"))
	bins.Must(err)

	t.conf = config.MakeConfig(dir, false /* text search */)
	if uint64(t.meta.Dim) != t.conf.EMBEDDINGS_DIM() {
		logrus.Fatalf("embedding dim %d != Tiptoe text dim %d", t.meta.Dim, t.conf.EMBEDDINGS_DIM())
	}
	if t.meta.NumClusters > t.conf.TOTAL_NUM_CLUSTERS() {
		logrus.Fatalf("%d clusters > Tiptoe's TOTAL_NUM_CLUSTERS %d", t.meta.NumClusters, t.conf.TOTAL_NUM_CLUSTERS())
	}

	// Doc IDs (corpus row -> BEIR id) and query embeddings, in the order main.go uses.
	t.docIDs, _ = bins.MakeDocIDAndQueryIDMap(cfg.DatasetMeta)
	queries, err := bins.LoadQueries(cfg.DatasetMeta.Queries)
	bins.Must(err)
	qemb, err := globals.LoadFloat32MatrixFromNpy(cfg.DatasetMeta.Vectors.QueryVec, len(queries), t.meta.Dim)
	bins.Must(err)
	t.queryEmb = make(map[string][]float32, len(queries))
	for i, q := range queries {
		t.queryEmb[q.ID] = qemb[i]
	}

	// ---- server side (what NewEmbeddingServers / NewUrlServers do, for one server each) ----
	logrus.Infof("Tiptoe: reading %d clusters from %s", t.meta.NumClusters, dir)
	embCorpus := corpus.ReadEmbeddingsTxt(0, t.meta.NumClusters, t.conf)
	urlCorpus := corpus.ReadUrlsTxt(0, t.meta.NumClusters, t.conf)

	t.embServer = new(protocol.Server)
	t.embServer.PreprocessEmbeddingsFromCorpus(embCorpus, uint64(cfg.BinsConf.TiptoeEmbHintMB), t.conf)
	t.urlServer = new(protocol.Server)
	t.urlServer.PreprocessUrlsFromCorpus(urlCorpus, uint64(cfg.BinsConf.TiptoeURLHintMB))

	// Server DB size: embeddings are SlotBits-bit values, URLs are compressed bundles.
	t.dbBytes = embCorpus.GetNumDocs()*embCorpus.GetEmbeddingSlots()*embCorpus.GetSlotBits()/8 +
		uint64(urlCorpus.NumSubclusters())*urlCorpus.GetUrlBytes()

	// ---- coordinator: merge both hints, keep the hint matrices server-side only ----
	var he, hu protocol.TiptoeHint
	bins.Must(t.embServer.GetHint(true, &he))
	bins.Must(t.urlServer.GetHint(true, &hu))
	h := he // embeddingsInitInfo
	h.ServeUrls = true
	h.CParams.UrlBytes = hu.CParams.UrlBytes // urlsInitInfo
	h.CParams.CompressUrl = hu.CParams.CompressUrl
	h.UrlsHint = hu.UrlsHint
	h.UrlsIndexMap = hu.UrlsIndexMap

	t.embHint = underhood.NewServerHintOnly(&h.EmbeddingsHint.Hint) // preprocessEmbHint
	t.urlHint = underhood.NewServerHintOnly(&h.UrlsHint.Hint)       // preprocessUrlHint
	// Drop the hint matrices from what the client gets (h holds its own Matrix headers, the servers are untouched).
	h.EmbeddingsHint.Hint.DropLastrows(h.EmbeddingsHint.Hint.Rows())
	h.UrlsHint.Hint.DropLastrows(h.UrlsHint.Hint.Rows())
	t.hint = &h

	// ---- client ----
	t.client = protocol.NewClient(false)
	t.client.Setup(t.hint)

	// Client state = what it downloads once and keeps: the (row-dropped) hint metadata + index maps,
	// and the cluster centroids it needs to pick a cluster.
	t.clientStateBytes = tutils.MessageSizeBytes(*t.hint) + uint64(len(t.centroids)*t.meta.Dim*4)

	logrus.Infof("Tiptoe ready: %d docs, %d clusters, client state %.2f MB, server DB %.2f MB",
		embCorpus.GetNumDocs(), t.meta.NumClusters, float64(t.clientStateBytes)/(1<<20), float64(t.dbBytes)/(1<<20))
	return t
}

// applyHint is Coordinator.ApplyHint: the server answers the token query for both services.
func (t *TiptoeSearch) applyHint(ct *underhood.HintQuery) *protocol.UnderhoodAnswer {
	out := new(protocol.UnderhoodAnswer)
	out.EmbAnswer = *t.embHint.HintAnswer(ct)
	toDrop := int(t.hint.EmbeddingsHint.Info.Params.N - t.hint.UrlsHint.Info.Params.N)
	trimmed := (*ct)[:len(*ct)-toDrop]
	out.UrlAnswer = *t.urlHint.HintAnswer(&trimmed)
	return out
}

// PIRPreprocess fetches one per-query token (Tiptoe's offline phase). main calls it before a query whenever
// GetBatchNums says no token is left, so its cost lands in MaintainenceTime like PIR preprocessing does for the others.
func (t *TiptoeSearch) PIRPreprocess() time.Duration {
	start := time.Now()
	ct := t.client.PreprocessQuery()
	ans := t.applyHint(ct)
	up := tutils.MessageSizeBytes(*ct)
	down := tutils.MessageSizeBytes(ans.EmbAnswer) + tutils.MessageSizeBytes(ans.UrlAnswer)
	t.client.ProcessHintApply(ans)
	t.offlineUp += up
	t.offlineDown += down
	t.tokenBytesLast = up + down
	t.offlineRounds++
	t.numTokens++
	t.tokenReady = true
	d := time.Since(start)
	t.offlineTime += d
	return d
}

// FilterQIDs lets main answer only the first -tiptoe-max-queries queries (the per-query token makes full MS MARCO slow).
func (t *TiptoeSearch) FilterQIDs(qids []string) []string {
	if t.maxQueries > 0 && len(qids) > t.maxQueries {
		return qids[:t.maxQueries]
	}
	return qids
}

func (t *TiptoeSearch) Preprocess() {
	// Server-side preprocessing already happened in MakeTiptoe; the first token is fetched lazily by main.
}

// GetBatchNums: one token per query. (finished+needed >= supported) triggers PIRPreprocess.
func (t *TiptoeSearch) GetBatchNums() (uint64, uint64, uint64) {
	if t.tokenReady {
		return 0, 0, 1
	}
	return 0, 1, 1
}

func (t *TiptoeSearch) quantise(v []float32) []int8 {
	lo, hi := -(1 << (t.meta.SlotBits - 1)), (1<<(t.meta.SlotBits-1))-1
	out := make([]int8, len(v))
	for i, x := range v {
		q := int(math.Round(float64(x) * t.meta.Scale))
		if q < lo {
			q = lo
		} else if q > hi {
			q = hi
		}
		out[i] = int8(q)
	}
	return out
}

func (t *TiptoeSearch) nearestCluster(v []float32) uint64 {
	best, bestD := 0, math.Inf(1)
	for c, cen := range t.centroids {
		d := 0.0
		for j := range cen {
			x := float64(v[j] - cen[j])
			d += x * x
		}
		if d < bestD {
			best, bestD = c, d
		}
	}
	return uint64(best)
}

func (t *TiptoeSearch) DoSearch(qid string, k int) (globals.Decodable, error) {
	if !t.tokenReady {
		t.PIRPreprocess()
	}
	t.tokenReady = false
	v, ok := t.queryEmb[qid]
	if !ok {
		return nil, fmt.Errorf("no embedding for query %s", qid)
	}

	cStart := time.Now()
	cluster := t.nearestCluster(v)
	emb := t.quantise(v)
	embQuery := t.client.QueryEmbeddings(emb, cluster)
	clientTime := time.Since(cStart)

	// Round 1: encrypted inner products
	r1 := time.Now()
	var embAns pir.Answer[matrix.Elem64]
	if err := t.embServer.GetEmbeddingsAnswer(embQuery, &embAns); err != nil {
		return nil, err
	}
	t.round1Time += time.Since(r1)
	t.onlineUp += tutils.MessageSizeBytes(*embQuery)
	t.onlineDown += tutils.MessageSizeBytes(embAns)

	cStart = time.Now()
	dec := t.client.ReconstructEmbeddingsWithinCluster(&embAns, cluster)
	scores := embeddings.SmoothResults(dec, t.hint.EmbeddingsHint.Info.P())
	// Rows past the cluster's docs are DB padding (score 0); drop them. Signed scores mean we can't use
	// runRound's "stop at score 0" shortcut.
	clusterSize := uint64(0)
	for _, sc := range t.hint.UrlsIndexMap[uint(cluster)] {
		clusterSize += sc.Size()
	}
	if uint64(len(scores)) > clusterSize {
		scores = scores[:clusterSize]
	}
	// SortByScores sorts its argument in place, so hand it a copy.
	order := tutils.SortByScores(append([]int(nil), scores...))
	if len(order) == 0 {
		return DBentryIDs{}, fmt.Errorf("empty cluster %d", cluster)
	}
	best := order[0]
	urlQuery, chunk := t.client.QueryUrls(cluster, best)
	clientTime += time.Since(cStart)

	// Round 2: PIR fetch of the URL bundle that holds the best doc
	r2 := time.Now()
	var urlAns pir.Answer[matrix.Elem32]
	if err := t.urlServer.GetUrlsAnswer(urlQuery, &urlAns); err != nil {
		return nil, err
	}
	t.round2Time += time.Since(r2)
	t.onlineUp += tutils.MessageSizeBytes(*urlQuery)
	t.onlineDown += tutils.MessageSizeBytes(urlAns)

	cStart = time.Now()
	urls := t.client.ReconstructUrls(&urlAns, cluster, best)
	// Same selection as runRound: docs of the fetched bundle, in score order.
	type scored struct {
		id    string
		score int
	}
	var res []scored
	seen := map[string]bool{}
	for _, doc := range order {
		_, c, idx := t.hint.UrlsIndexMap.SubclusterToIndex(cluster, doc)
		if c != chunk {
			continue
		}
		row, err := strconv.Atoi(corpus.GetIthUrl(urls, idx))
		if err != nil {
			continue
		}
		id := t.docIDs[row]
		if seen[id] {
			continue
		}
		seen[id] = true
		res = append(res, scored{id, scores[doc]})
	}
	sort.SliceStable(res, func(i, j int) bool { return res[i].score > res[j].score })
	if len(res) > k {
		res = res[:k]
	}
	ids := make([]string, len(res))
	for i, r := range res {
		ids[i] = r.id
	}
	t.clientTime += clientTime + time.Since(cStart)
	t.onlineRounds += 2
	t.numSearches++
	return DBentryIDs{ids}, nil
}

func mb(b uint64) string { return strconv.FormatFloat(float64(b)/(1<<20), 'f', 4, 64) }
func kb(b uint64) string { return strconv.FormatFloat(float64(b)/1024, 'f', 4, 64) }

// GetMetaData uses the same keys as pianopir's PrintInfo so plotting/load_results.py reads it unchanged.
func (t *TiptoeSearch) GetMetaData() map[string]string {
	m := map[string]string{}
	totalBytes := t.offlineUp + t.offlineDown + t.onlineUp + t.onlineDown
	// Network time: online rounds only; the per-query token is offline (like PIR preprocessing for the others),
	// and its bytes are still counted in TotalByteSent.
	onlineBytes := t.onlineUp + t.onlineDown
	m["ClientStorageMB"] = mb(t.clientStateBytes)
	m["DBSizeInBytesMB"] = mb(t.dbBytes)
	m["TotalByteSent"] = strconv.FormatUint(totalBytes, 10)
	m["TotalLANTime"] = strconv.FormatFloat(netTime(onlineBytes, t.onlineRounds, 1000, 0.005), 'f', 6, 64)
	m["TotalWANTime"] = strconv.FormatFloat(netTime(onlineBytes, t.onlineRounds, 400, 0.05), 'f', 6, 64)
	m["TiptoeOfflineLANTime"] = strconv.FormatFloat(netTime(t.offlineUp+t.offlineDown, t.offlineRounds, 1000, 0.005), 'f', 6, 64)
	m["TiptoeOfflineWANTime"] = strconv.FormatFloat(netTime(t.offlineUp+t.offlineDown, t.offlineRounds, 400, 0.05), 'f', 6, 64)
	if t.numSearches > 0 {
		m["CommCostPerBatchKB"] = kb(onlineBytes / t.numSearches)
		m["TiptoeTokenKBPerQuery"] = kb((t.offlineUp + t.offlineDown) / t.numSearches)
	}
	m["TiptoeOfflineUpBytes"] = strconv.FormatUint(t.offlineUp, 10)
	m["TiptoeOfflineDownBytes"] = strconv.FormatUint(t.offlineDown, 10)
	m["TiptoeOnlineUpBytes"] = strconv.FormatUint(t.onlineUp, 10)
	m["TiptoeOnlineDownBytes"] = strconv.FormatUint(t.onlineDown, 10)
	m["TiptoeOfflineTime"] = t.offlineTime.String()
	m["TiptoeRound1Time"] = t.round1Time.String()
	m["TiptoeRound2Time"] = t.round2Time.String()
	m["TiptoeClientTime"] = t.clientTime.String()
	m["TiptoeRounds"] = strconv.FormatUint(t.onlineRounds, 10)
	m["TiptoeNumClusters"] = strconv.Itoa(t.meta.NumClusters)
	m["TiptoeMaxClusterSize"] = strconv.Itoa(t.meta.MaxCluster)
	m["TiptoeMultiAssign"] = strconv.Itoa(t.meta.MultiAssign)
	m["TiptoeBundleSize"] = strconv.Itoa(t.meta.BundleSize)
	m["TiptoeNumBundles"] = strconv.Itoa(t.meta.NumBundles)
	m["TiptoeEmbM"] = strconv.FormatUint(t.hint.EmbeddingsHint.Info.M, 10)
	m["TiptoeEmbL"] = strconv.FormatUint(t.hint.EmbeddingsHint.Info.L, 10)
	m["TiptoeURLM"] = strconv.FormatUint(t.hint.UrlsHint.Info.M, 10)
	m["TiptoeURLL"] = strconv.FormatUint(t.hint.UrlsHint.Info.L, 10)
	m["TiptoeDir"] = t.dir
	return m
}
