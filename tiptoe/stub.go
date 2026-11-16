//go:build !tiptoe

// Default build: the Tiptoe baseline links Microsoft SEAL through cgo (github.com/ahenzinger/underhood), so it is only
// compiled with -tags tiptoe. This stub keeps `go build` working everywhere else.
package tiptoe

import (
	"time"

	"github.com/dkblackley/bins-go/globals"
	"github.com/sirupsen/logrus"
)

type TiptoeSearch struct{}

func MakeTiptoe(_ *globals.Args) *TiptoeSearch {
	logrus.Fatalf("this binary was built without Tiptoe; rebuild with `go build -tags tiptoe` (see baselines/README.md)")
	return nil
}

func (t *TiptoeSearch) GetMetaData() map[string]string                  { return nil }
func (t *TiptoeSearch) DoSearch(string, int) (globals.Decodable, error) { return nil, nil }
func (t *TiptoeSearch) Preprocess()                                     {}
func (t *TiptoeSearch) PIRPreprocess() time.Duration                    { return 0 }
func (t *TiptoeSearch) GetBatchNums() (uint64, uint64, uint64)          { return 0, 0, 1 }
