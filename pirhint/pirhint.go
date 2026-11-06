// Package pirhint computes the client storage (the Piano "hint") that a
// SimpleBatchPianoPIR allocates, without building one, so DB layouts and PIR
// parameters can be compared at full MS MARCO scale in milliseconds.
//
// The formulas mirror the two PIR copies in this repo exactly:
//
//	FlavorOld  pir-old  (tree method): PartitionNum = Batch/2, last partition
//	           padded, hint counts rounded to multiples of 8.
//	FlavorNew  pianopir (bins method): partitions past the DB are dropped,
//	           hint counts rounded to multiples of pianopir.ThreadNum (30).
//
// pirhint_test.go checks both against the real allocations.
//
// Per partition of P rows and E-byte entries (FailureProbLog2 = 20):
//
//	ChunkSize C  = next pow2 >= ChunkFactor*sqrt(P)         (ChunkFactor 2 originally)
//	SetSize   S  = ceil(P/C) rounded up to a multiple of 4
//	primary      = ceil(ln2*21) * C = 15*C hints            (each E bytes + 16 B of tags)
//	Q            = QueryBudget * sqrt(P)*ln(P)              (queries between refreshes)
//	backup       = S * 3*floor(Q/S) hints                   (each 2E bytes + 16 B)
//
// so the client holds ~(15*C + 6*sqrt(P)*ln P)*E per partition, about
// 40*sqrt(P)+6*sqrt(P)*ln(P) entries: MORE than the partition itself whenever
// P is below roughly 10^4 rows. Over k partitions of an N-row DB the total grows
// as sqrt(k*N)*E, so splitting one DB into k DBs costs sqrt(k) more, not less.
package pirhint

import "math"

type Flavor int

const (
	FlavorOld Flavor = iota // pir-old, used by the tree
	FlavorNew               // pianopir, used by the bins
)

func (f Flavor) String() string {
	if f == FlavorNew {
		return "pianopir"
	}
	return "pir-old"
}

// Knobs are the client-storage parameters that pir-old exposes as flags.
type Knobs struct {
	ChunkFactor float64 // ChunkSize = next pow2 >= ChunkFactor*sqrt(P); 2 originally
	QueryBudget float64 // scale on sqrt(P)ln(P) queries per refresh; 1 originally
	FailBits    uint64  // FailureProbLog2; 20 in this repo
}

// Default is the parameter set every existing result was produced with.
var Default = Knobs{ChunkFactor: 2, QueryBudget: 1, FailBits: 20}

// DB describes one PIR database.
type DB struct {
	Name       string
	Rows       uint64 // physical rows (Cuckoo slots, bins, blocks...)
	EntryBytes uint64 // bytes per row as the PIR sees it (a multiple of 8)
	Batch      uint64 // BatchSize handed to NewSimpleBatchPianoPIR
	Copies     uint64 // identical DBs of this shape (0 = 1), e.g. one per leaf block
	Probes     uint64 // PIR lookups one search makes in each copy (for the online estimate)
}

func (db DB) copies() float64 { return float64(max(db.Copies, 1)) }

// Result is the client cost of one DB.
type Result struct {
	DB
	Flavor        Flavor
	Partitions    uint64
	PartitionSize uint64
	ChunkSize     uint64
	SetSize       uint64
	MaxQueries    uint64 // per partition, between refreshes
	PrimaryHints  uint64 // per partition
	BackupHints   uint64 // per partition
	HintBytes     float64
	DBBytes       float64
	QueryBytes    float64 // upload + download of one Piano query (SetSize*4 + E)
	OnlineBytes   float64 // estimated bytes one search sends to this DB (all copies)
	// Piano re-streams the whole DB when a partition runs out of queries: searches
	// between refreshes, and the offline download that adds per search
	SearchesPerRefresh float64
	OfflineBytes       float64
	Downloaded         bool // the plain DB was cheaper than the hints and was taken instead
}

// ClientBytes is what the client stores for this DB.
func (r Result) ClientBytes() float64 {
	if r.Downloaded {
		return r.DBBytes
	}
	return r.HintBytes
}

// HintOverDB is how many times larger the hints are than the DB itself.
func (r Result) HintOverDB() float64 { return r.HintBytes / r.DBBytes }

func roundUp(x, m uint64) uint64 { return (x + m - 1) / m * m }

// Compute returns the client cost of db. With download set, a DB whose hints
// would be at least as large as the DB is downloaded instead (pir-old's
// DownloadIfCheaper).
func Compute(db DB, flavor Flavor, k Knobs, download bool) Result {
	r := Result{DB: db, Flavor: flavor}
	r.DBBytes = float64(db.Rows) * float64(db.EntryBytes) * db.copies()
	if db.Rows == 0 {
		return r
	}
	r.Partitions = max(db.Batch/2, 1)
	r.PartitionSize = (db.Rows + r.Partitions - 1) / r.Partitions
	if flavor == FlavorNew {
		r.Partitions = (db.Rows + r.PartitionSize - 1) / r.PartitionSize
	}
	threadRound := uint64(8)
	if flavor == FlavorNew {
		threadRound = 30
	}

	p := r.PartitionSize
	target := uint64(k.ChunkFactor * math.Sqrt(float64(p)))
	r.ChunkSize = 1
	for r.ChunkSize < target {
		r.ChunkSize *= 2
	}
	r.SetSize = roundUp(uint64(math.Ceil(float64(p)/float64(r.ChunkSize))), 4)

	r.MaxQueries = uint64(math.Sqrt(float64(p)) * math.Log(float64(p)))
	if flavor == FlavorOld && k.QueryBudget != 1 {
		r.MaxQueries = max(uint64(k.QueryBudget*float64(r.MaxQueries)), r.SetSize)
	}
	r.PrimaryHints = roundUp(uint64(math.Ceil(math.Log(2)*float64(k.FailBits+1)))*r.ChunkSize, threadRound)
	perChunk := roundUp(3*uint64(float64(r.MaxQueries)/float64(r.SetSize)), threadRound)
	r.BackupHints = r.SetSize * perChunk

	e := float64(db.EntryBytes)
	perPartition := float64(r.PrimaryHints)*(16+e) + float64(r.BackupHints)*(16+2*e)
	r.HintBytes = float64(r.Partitions) * perPartition * db.copies()
	r.QueryBytes = float64(r.SetSize*4) + e
	r.Downloaded = download && r.HintBytes >= r.DBBytes
	if !r.Downloaded && db.Probes > 0 {
		// every partition answers as many queries as the fullest one gets
		mean := float64(db.Probes) / float64(r.Partitions)
		occupancy := math.Ceil(mean + math.Sqrt(2*mean*math.Log(float64(r.Partitions)+1)))
		r.OnlineBytes = float64(r.Partitions) * occupancy * r.QueryBytes * db.copies()
		r.SearchesPerRefresh = math.Max(float64(r.MaxQueries)-2, 1) / occupancy
		r.OfflineBytes = r.DBBytes / math.Max(r.SearchesPerRefresh, 1)
	}
	return r
}

// Sum adds up the client bytes, hint bytes and DB bytes of several DBs.
func Sum(rs []Result) (client, hints, db float64) {
	for _, r := range rs {
		client += r.ClientBytes()
		hints += r.HintBytes
		db += r.DBBytes
	}
	return
}

// CuckooRows is the slot count of a Cuckoo table (bm25.cuckooSlotCount).
func CuckooRows(items uint64, loadFactor float64) uint64 {
	if items == 0 {
		return 1
	}
	return uint64(math.Ceil(float64(items) / loadFactor))
}

// CuckooEntryBytes is the PIR entry of a Cuckoo slot: an 8-byte key plus the
// value, rounded up to whole words (bm25.cuckooEntryWords).
func CuckooEntryBytes(valueBytes uint64) uint64 { return roundUp(8+valueBytes, 8) }

// Words rounds a byte count up to whole uint64 words.
func Words(bytes uint64) uint64 { return roundUp(bytes, 8) }
