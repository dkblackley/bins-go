module github.com/dkblackley/bins-go

go 1.24.0

require (
	github.com/ahenzinger/tiptoe v0.0.0-20231001172502-ba7cae43ab86
	github.com/ahenzinger/underhood v0.0.0-20230921190418-f053a81c6385
	github.com/blugelabs/bluge v0.2.2
	github.com/evan176/hnswgo v0.0.0-20220622031020-39253a76f9e4
	github.com/henrycg/simplepir v0.0.0-20230920020624-026ee7bd6783
	github.com/kshard/fvecs v0.0.1
	github.com/kshedden/gonpy v0.0.0-20210519231815-fa3c8dd8e59b
	github.com/schollz/progressbar/v3 v3.18.0
	github.com/sirupsen/logrus v1.9.3
	github.com/yahoojapan/gongt v0.0.0-20190517050727-966dcc7aa5e8
)

require (
	github.com/RoaringBitmap/roaring v1.9.4 // indirect
	github.com/axiomhq/hyperloglog v0.2.5 // indirect
	github.com/bits-and-blooms/bitset v1.24.4 // indirect
	github.com/blevesearch/go-porterstemmer v1.0.3 // indirect
	github.com/blevesearch/mmap-go v1.0.4 // indirect
	github.com/blevesearch/segment v0.9.1 // indirect
	github.com/blevesearch/snowballstem v0.9.0 // indirect
	github.com/blevesearch/vellum v1.1.0 // indirect
	github.com/blugelabs/bluge_segment_api v0.2.0 // indirect
	github.com/blugelabs/ice v1.0.0 // indirect
	github.com/blugelabs/ice/v2 v2.0.1 // indirect
	github.com/caio/go-tdigest v3.1.0+incompatible // indirect
	github.com/dgryski/go-metro v0.0.0-20250106013310-edb8663e5e33 // indirect
	github.com/fatih/color v1.15.0 // indirect
	github.com/golang/snappy v1.0.0 // indirect
	github.com/kamstrup/intmap v0.5.1 // indirect
	github.com/klauspost/compress v1.18.2 // indirect
	github.com/mattn/go-colorable v0.1.13 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	github.com/mitchellh/colorstring v0.0.0-20190213212951-d06e56a500db // indirect
	github.com/mschoch/smat v0.2.0 // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	golang.org/x/sys v0.38.0 // indirect
	golang.org/x/term v0.37.0 // indirect
	golang.org/x/text v0.3.0 // indirect
)

// Tiptoe pins underhood at a pseudo-version whose timestamp does not match the commit; point at the same commit (f053a81c6385) with the correct timestamp.
replace github.com/ahenzinger/underhood v0.0.0-20230921190418-f053a81c6385 => github.com/ahenzinger/underhood v0.0.0-20230922182337-f053a81c6385

// Patched copy: Go >= 1.24 rejects methods on cgo types, which upstream simplepir defines (see third_party/README.md).
replace github.com/henrycg/simplepir => ./third_party/simplepir
