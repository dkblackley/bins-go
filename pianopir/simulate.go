package pianopir

import (
	"fmt"
	"math/rand"
	"slices"
	"time"
)

// Simulate turns on simulated PIR. A simulated client never builds its hint
// tables and a simulated server never XORs DB rows: a lookup returns the row
// itself, which is exactly what a real Piano query decodes to. Everything the
// results depend on still runs as in a real run: the batch partitioning and
// padding, dummy queries, the client's local cache, the per-chunk query budget
// (so re-preprocessing happens at the same points), the round count and the
// byte accounting. Only the timings are meaningless, so main.go writes -1 for
// them. Off by default; set it before building any SimpleBatchPianoPIR.
//
// What is exact and what is not:
//   - the answers, so MRR/Recall before and after the re-rank: identical
//   - rounds, re-preprocessing points, TotalByteSentPadded, ClientStoragePaddedMB,
//     CommCostPerBatchKB, DBSizeInBytesMB: identical (they do not depend on the hints)
//   - TotalByteSent and the LAN/WAN times: each response is charged the longest
//     row among SetSize rows picked uniformly at random, one per chunk. A real
//     query's rows come from the PRF (also uniform), so this is the same
//     distribution, but not the same draw. Real runs are not reproducible either
//     (the PRF keys are seeded from the clock). For equal-size rows it is exact.
//   - ClientStorageMB: the expected trimmed hint size (see expectedTrimmedStorage).
//     For equal-size rows it is exact up to the random replacement picks that
//     land on padding in a partly padded last chunk (a few entries out of millions).
var Simulate = false

// simClient is the state a simulated client keeps in place of the hint tables.
type simClient struct {
	rng            *rand.Rand
	trimmedStorage float64 // cached expectedTrimmedStorage, -1 until computed
}

func newSimClient() *simClient {
	return &simClient{rng: rand.New(rand.NewSource(time.Now().UnixNano())), trimmedStorage: -1}
}

// simInitialization is Initialization without the hint tables: a fresh query
// budget and an empty local cache.
func (c *PianoPIRClient) simInitialization() {
	c.FinishedQueryNum = 0
	c.QueryHistogram = resetUint64s(c.QueryHistogram, c.config.SetSize)
	c.localCache = make(map[uint64][][]uint64)
}

// simQuery follows Query step for step (same cache, same budget checks, same
// errors) but sends uniformly random offsets and returns the row directly.
func (c *PianoPIRClient) simQuery(idx uint64, server *PianoPIRServer, realQuery bool) ([]uint64, error) {
	ret := make([]uint64, c.config.MaxDBEntrySize)
	offsets := make([]uint32, c.config.SetSize)
	for i := range offsets {
		offsets[i] = uint32(c.sim.rng.Uint64() & (c.config.ChunkSize - 1))
	}

	if !realQuery {
		server.simPrivateQuery(offsets)
		return ret, nil
	}

	if idx >= c.config.DBSize {
		return ret, fmt.Errorf("idx %v is out of range", idx)
	}
	if v, ok := c.localCache[idx]; ok {
		return v[0], nil
	}
	if c.FinishedQueryNum >= c.MaxQueryNum {
		return ret, fmt.Errorf("exceed the maximum number of queries")
	}
	chunkId := idx / c.config.ChunkSize
	if c.QueryHistogram[chunkId] >= c.maxQueryPerChunk {
		return ret, fmt.Errorf("too many queries in chunk %v (limit %d)", chunkId, c.maxQueryPerChunk)
	}

	server.simPrivateQuery(offsets)
	if idx < uint64(len(server.rawDB)) {
		copy(ret, server.rawDB[idx])
	}

	c.FinishedQueryNum += 1
	c.QueryHistogram[chunkId] += 1
	c.localCache[idx] = [][]uint64{ret}
	return ret, nil
}

// simPrivateQuery charges the bytes PrivateQuery would, without XORing.
func (s *PianoPIRServer) simPrivateQuery(offsets []uint32) {
	maxLen := 0
	for i := uint64(0); i < s.config.SetSize; i++ {
		idx := uint64(offsets[i]) + i*s.config.ChunkSize
		if idx >= s.config.DBSize || idx >= uint64(len(s.rawDB)) {
			continue
		}
		if n := len(s.rawDB[idx]); n > maxLen {
			maxLen = n
		}
	}
	s.chargeQuery(len(offsets), maxLen)
}

// expectedTrimmedStorage is the expected value of LocalStorageSizeTrimmed for
// a client preprocessed over rawDB, computed from the row lengths alone.
//
// A primary parity is the XOR of one uniformly random row from every chunk, so
// its trimmed length is the longest of those rows: with F_c(x) the fraction of
// chunk c's slots whose trimmed length is <= x,
//
//	E[max] = sum over L >= 1 of (1 - prod_c F_c(L-1)).
//
// A backup parity of group i is the same over every chunk except i, and a
// replacement value is one uniformly random row of chunk i. Rows past the end
// of the partition are padding of length 0, as in Preprocessing.
func (c *PianoPIRClient) expectedTrimmedStorage(rawDB [][]uint64) float64 {
	setSize := int(c.config.SetSize)
	chunkSize := int(c.config.ChunkSize)

	// sorted trimmed lengths of each chunk, and every distinct length
	chunks := make([][]int, setSize)
	distinct := map[int]struct{}{0: {}}
	for ch := 0; ch < setSize; ch++ {
		lens := make([]int, chunkSize)
		for o := 0; o < chunkSize; o++ {
			if r := ch*chunkSize + o; r < len(rawDB) {
				lens[o] = trimmedWords(rawDB[r])
			}
			distinct[lens[o]] = struct{}{}
		}
		slices.Sort(lens)
		chunks[ch] = lens
	}
	u := make([]int, 0, len(distinct))
	for v := range distinct {
		u = append(u, v)
	}
	slices.Sort(u)

	primary := 0.0
	backup := make([]float64, setSize)
	pos := make([]int, setSize)   // slots of each chunk with length <= current u
	F := make([]float64, setSize) // pos / chunkSize
	pre := make([]float64, setSize+1)
	suf := make([]float64, setSize+1)
	for k := 1; k < len(u); k++ {
		prev, width := u[k-1], float64(u[k]-u[k-1])
		for ch := 0; ch < setSize; ch++ {
			for pos[ch] < chunkSize && chunks[ch][pos[ch]] <= prev {
				pos[ch]++
			}
			F[ch] = float64(pos[ch]) / float64(chunkSize)
		}
		pre[0], suf[setSize] = 1, 1
		for ch := 0; ch < setSize; ch++ {
			pre[ch+1] = pre[ch] * F[ch]
		}
		for ch := setSize - 1; ch >= 0; ch-- {
			suf[ch] = suf[ch+1] * F[ch]
		}
		primary += width * (1 - pre[setSize])
		for ch := 0; ch < setSize; ch++ {
			backup[ch] += width * (1 - pre[ch]*suf[ch+1])
		}
	}

	words := float64(c.primaryHintNum) * primary
	for ch := 0; ch < setSize; ch++ {
		sum := 0
		for _, l := range chunks[ch] {
			sum += l
		}
		mean := float64(sum) / float64(chunkSize)
		words += float64(c.maxQueryPerChunk) * (backup[ch] + mean)
	}

	totalBackupHintNum := c.config.SetSize * c.maxQueryPerChunk
	tags := 2*c.primaryHintNum*8 + 2*totalBackupHintNum*8
	stored := c.primaryHintNum + 2*totalBackupHintNum
	return float64(tags) + words*8 + float64(stored*ResponseLengthHeaderBytes)
}
