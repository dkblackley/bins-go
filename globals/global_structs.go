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
	Threshold         uint
	DChoice           uint
	Save              bool
	Load              bool
	DebugLevel        int
	CheckPointFolder  string
	RTT               uint
	OutFile           string
	QueryNum          uint
	DatasetMeta       DatasetMetadata
	IDLookup          map[[32]byte]string
	Metadata          map[string]string
}

// strconv.Itoa(docID)

type Vectors struct {
	CorpusVec   string
	CorpusVec64 string
	QueryVec    string
	QueryVec64  string
	Graph       string
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
