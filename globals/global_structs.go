package globals

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
	IDLookup          map[string]int
}

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
