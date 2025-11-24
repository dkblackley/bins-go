package main

import (
	"github.com/dkblackley/bins-go/pianopir"
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

func main() {
	logrus.SetLevel(logrus.InfoLevel)
}
