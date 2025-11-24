package bins_go

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"time"

	"github.com/blugelabs/bluge"
	"github.com/blugelabs/bluge/analysis"
	"github.com/blugelabs/bluge/analysis/char"
	"github.com/blugelabs/bluge/analysis/lang/en"
	"github.com/blugelabs/bluge/analysis/token"
	"github.com/blugelabs/bluge/analysis/tokenizer"
	"github.com/dkblackley/bins-go/bins"
	"github.com/dkblackley/bins-go/pianopir"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

const MAX_UINT32 = ^uint32(0)
const MARCO_SIZE = 8841823

// const MARCO_SIZE = 1105 //Debug size
const DIM = 192
const RTT = 50

type PIRImpliment interface {
	intoDb() []uint64
	getPIRInfo() *pianopir.SimpleBatchPianoPIR
}
