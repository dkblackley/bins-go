package globals

type Args struct {
	DatasetsDirectory string
	K                 uint
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
}
