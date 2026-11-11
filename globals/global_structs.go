package globals

import (
	"fmt"

	"github.com/dkblackley/bins-go/diskstore"
	"github.com/kshedden/gonpy"
)

type Args struct {
	DatasetsDirectory string
	K                 uint
	SearchType        string
	DataName          string
	Vectors           bool
	Dimensions        uint
	DBSize            uint
	BinSize           uint
	DocsPerBin        uint
	IDBatch           uint // bins vec0: batch size of the ID DB (0 = 24)
	VecBatch          uint // bins vec0: batch size of each embedding-DB round (0 = automatic)
	VecRounds         uint // bins vec0: embedding-DB rounds per query (0 = automatic)
	VecOrder          string // bins vec0: candidate order for the embedding DB: "bin" (term by term) or "rr" (round robin)
	VecPrune          bool // bins vec0: leave docs that are in no bin out of the embedding DB
	Threshold         uint
	DChoice           uint
	Save              bool
	Load              bool
	DebugLevel        int
	CheckPointFolder  string
	// RTT               uint
	OutFile         string
	OutDir          string
	QueryNum        uint
	DatasetMeta     DatasetMetadata
	BinsConf        BinsConf
	IDLookup        map[[32]byte]string
	Metadata        map[string]string
	NeighbhourNum   uint
	StepN           uint
	DocIDMapPacmann map[int]string
	Text            TextConf
}

// TextConf is the bins text-retrieval mode (-text): bins hold compressed document text instead of embeddings and
// the client re-ranks with BM25.
type TextConf struct {
	Enabled   bool
	Format    string // "bow" (sorted stems + tf) or "seq" (stems in order)
	Order     int    // context length of the character model
	IDFBits   uint   // log2 cells per row of the client's IDF sketch (0 = no idf, plain tf/length)
	TrainDocs int    // documents used to train the model
	K1, B     float64
}

type BinsConf struct {
	// Stage0 / I/O
	Index   string
	Queries string
	Qrels   string
	Vocab   string

	// Stage1/2 params
	B           int
	R           int
	L           int
	S           int
	K           int
	KCandidates int
	MaxQueries  int
	Scale       float64

	// BM25 params
	K1     float64
	BParam float64

	// Precompute / bins
	Precompute     bool
	Stage1DataBin  string
	Stage1IdmapBin string

	// PIR
	PIRBatchSize int64

	// Stage3 (dense rerank)
	DocEmbed   string
	DocIDMap   string
	QueryEmbed string
	EmbedDim   int

	// Debug / query shaping
	Debug      int
	QueryWords int

	// Stage2 hit load/save for Stage3-only runs
	LoadStage2Hits string
	SaveStage2Hits string
	SkipStage2     bool

	SearchType string

	// Extra flags for the updated tree method (uses pir-old + Cuckoo tables)
	Stage2DataBin       string
	Stage2IdmapBin      string
	LayoutPath          string
	BlockPerm           string
	DisablePIR          bool
	StrictPIR           bool
	PIRConstantWork     bool
	PIRThreads          int
	CuckooLoadFactor    float64
	CuckooHashes        int     // tree: 2 or 3 hash locations per Cuckoo key (2 = original)
	CuckooKeyBytes      int     // tree: bytes of the (term,node) key kept per row (8 = original)
	CuckooUBBits        int     // tree: bits per upper bound in Stage1/2 rows (8 = original)
	PIRDownloadSmall    bool    // pir-old: download a DB whose hints would be larger than it
	PIRChunkFactor      float64 // pir-old: ChunkSize factor (2 = original)
	PIRQueryBudget      float64 // pir-old: queries between refreshes, as a fraction of sqrt(n)ln(n)
	Stage3BatchSize     int64   // tree: Stage3 BatchPIR batch size
	CuckooMaxKicks      int
	CuckooBuildAttempts int
	CuckooCacheDir      string

	// Tiptoe baseline (-t tiptoe, build with -tags tiptoe)
	TiptoeDir        string // output of baselines/tiptoe/preprocess_tiptoe.py
	TiptoeEmbHintMB  int    // SimplePIR hint size target for the ranking service (Tiptoe default: 500)
	TiptoeURLHintMB  int    // same for the URL service (Tiptoe default: 100)
	TiptoeMaxQueries int    // answer only the first N queries (0 = all)
}

// strconv.Itoa(docID)

type Vectors struct {
	CorpusVec string
	//CorpusVec64 string
	QueryVec string
	//QueryVec64  string
	Graph string
}

type DatasetMetadata struct {
	Name        string
	IndexDir    string
	OriginalDir string
	Queries     string
	Qrels       string
	Vectors     Vectors
}

// Hacky interface
type Decodable interface {
	Decode(config *Args) []string
}

type Query struct {
	ID      string `json:"_id"`
	AltID   string `json:"id"`       // Add this to catch "id"
	QueryID string `json:"query_id"` // Add this to catch "query_id"
	Text    string `json:"text"`
	// Metadata string `json:"metadata"`
}

// Taken from graphann package. I think dim should be 192 and n should be 8841823 (ms marco size)
func LoadFloat32MatrixFromNpy(filename string, n int, dim int) ([][]float32, error) {
	if diskstore.Enabled { // -noram: map the file instead of reading it onto the heap
		return diskstore.MmapNpyFloat32(filename, n, dim)
	}
	r, err := gonpy.NewFileReader(filename)
	if err != nil {
		fmt.Println(err)
		return nil, err
	}

	shape := r.Shape

	// check the shape
	if len(shape) != 2 || shape[0] < n || shape[1] != dim {
		fmt.Printf("Invalid shape: %v\n", shape)
		fmt.Printf("Expected shape: (%d, %d)\n", n, dim)
		return nil, fmt.Errorf("invalid shape: %v", shape)
	}

	data, err := r.GetFloat32()

	// data, err := r.GetFloat64()
	if err != nil {
		fmt.Println(err)
		return nil, err
	}

	//bar := progressbar.Default(int64(n), "Loading BM25 vectors")

	// Convert to a 2D slice by re-slicing the flat buffer instead of copying it. The old version copied every
	// row, so for a 100M+ corpus (~77 GB at 192 dims) the vectors briefly lived in memory twice. Each row is
	// capped (full slice expression) so an append on one row can never spill into the next.
	ret := make([][]float32, n)
	for i := 0; i < n; i++ {
		ret[i] = data[i*dim : (i+1)*dim : (i+1)*dim]
	}

	//bar.Finish()

	return ret, nil
}
