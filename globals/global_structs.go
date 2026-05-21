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
	BinSize           float64
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
	Decode(config Args) []string
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

	// we now convert the data to a 2D slice
	ret := make([][]float32, n)
	for i := 0; i < n; i++ {
		ret[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			ret[i][j] = float32(data[i*dim+j])
		}
		//bar.Add64(int64(1))
	}

	//bar.Finish()

	return ret, nil
}
