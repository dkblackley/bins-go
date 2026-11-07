package globals

import (
	"fmt"

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
	CuckooMaxKicks      int
	CuckooBuildAttempts int
	CuckooCacheDir      string
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
