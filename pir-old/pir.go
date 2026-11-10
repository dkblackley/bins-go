package pianopir

import (
	"fmt"
	//"encoding/binary"

	"log"
	"math"
	"math/rand"
	"time"
)

const (
	//FailureProbLog2     = 40
	DefaultProgramPoint = 0x7fffffff
)

type PianoPIRConfig struct {
	DBEntryByteNum  uint64 // the number of bytes in a DB entry
	DBEntrySize     uint64 // the number of uint64 in a DB entry
	DBSize          uint64
	ChunkSize       uint64
	SetSize         uint64
	ThreadNum       uint64
	FailureProbLog2 uint64
}

type PianoPIRServer struct {
	config            *PianoPIRConfig
	rawDB             []uint64
	privateQueryCount uint64
	RetrievalCount    uint64
}

// an initialization function for the server
func NewPianoPIRServer(config *PianoPIRConfig, rawDB []uint64) *PianoPIRServer {
	return &PianoPIRServer{
		config: config,
		rawDB:  rawDB,
	}
}

func (s *PianoPIRServer) NonePrivateQuery(idx uint64) ([]uint64, error) {
	ret := make([]uint64, s.config.DBEntrySize)
	// initialize ret to be all zeros
	for i := uint64(0); i < s.config.DBEntrySize; i++ {
		ret[i] = 0
	}

	if idx >= s.config.DBSize {
		//log.Fatalf("idx %v is out of range", idx)
		if idx < s.config.ChunkSize*s.config.SetSize {
			// caused by the padding
			return ret, nil
		} else {
			// return an empty entry and an error
			return ret, fmt.Errorf("idx %v is out of range", idx)
		}
	}

	// copy the idx*DBEntrySize-th to (idx+1)*DBEntrySize-th elements to ret
	copy(ret, s.rawDB[idx*s.config.DBEntrySize:(idx+1)*s.config.DBEntrySize])
	return ret, nil
}

// the private query just computes the xor sum of the elements in the idxs list
func (s *PianoPIRServer) PrivateQuery(offsets []uint32) ([]uint64, error) {
	s.privateQueryCount++
	s.RetrievalCount += uint64(len(offsets))*4 + s.config.DBEntrySize*8
	ret := make([]uint64, s.config.DBEntrySize)
	// initialize ret to be all zeros
	for i := uint64(0); i < s.config.DBEntrySize; i++ {
		ret[i] = 0
	}

	for i := uint64(0); i < s.config.SetSize; i++ {
		idx := uint64(offsets[i]) + i*s.config.ChunkSize

		if idx >= s.config.DBSize {
			continue
		}

		// xor the idx*DBEntrySize-th to (idx+1)*DBEntrySize-th elements to ret
		EntryXor(ret, s.rawDB[idx*s.config.DBEntrySize:(idx+1)*s.config.DBEntrySize], s.config.DBEntrySize)

		//for i := uint64(0); i < s.config.DBEntrySize; i++ {
		//	ret[i] ^= s.rawDB[idx*s.config.DBEntrySize+i]
		//	}
	}

	return ret, nil
}

// PianoPIRClient is the stateful client for PianoPIR
type PianoPIRClient struct {
	config   *PianoPIRConfig
	skipPrep bool

	// the master keys for the client
	//rng       *rand.Rand
	masterKey PrfKey
	longKey   []uint32

	MaxQueryNum      uint64
	FinishedQueryNum uint64

	// an upper bound of the number of queries in each chunk
	maxQueryPerChunk uint64
	QueryHistogram   []uint64

	// primary hint table
	primaryHintNum      uint64
	primaryShortTag     []uint64 // the prf short tag
	primaryParity       []uint64 // notice that we group DBEntrySize uint64 into one entry
	primaryProgramPoint []uint64 // the point that the set is programmed

	replacementIdx [][]uint64 // the replacement indices. We have one array for each chunk
	replacementVal [][]uint64 // the replacement values. We have one array for each chunk

	// backup hint table
	backupShortTag [][]uint64 // the prf short tag
	backupParity   [][]uint64 // notice that we group DBEntrySize uint64 into one entry

	// local cache
	localCache map[uint64][]uint64

	constantWork  bool
	dummyScanSink uint64

	sim *simClient // non-nil when Simulate was on at creation: no hint tables, see simulate.go
}

func primaryNumParam(Q float64, ChunkSize float64, target uint64) uint64 {
	k := math.Ceil(math.Log(2) * (float64(target)))
	return uint64(k) * uint64(ChunkSize)
}

// Client-storage knobs. The defaults reproduce the original Piano parameters,
// so existing results do not change unless a flag sets them.
var (
	// ChunkSizeFactor sets ChunkSize to the next power of two >= factor*sqrt(DBSize).
	// The primary hint table has ~ln(2)*(FailureProbLog2+1)*ChunkSize entries, so it
	// shrinks linearly with the factor, while SetSize (upload size and server work per
	// query) grows as 1/factor. 2.0 is the original setting.
	ChunkSizeFactor = 2.0

	// QueryBudgetScale scales the number of queries a hint table supports before it is
	// refreshed (sqrt(n)*ln(n) originally). The backup hint tables are proportional to
	// it, so a smaller value trades client storage for more frequent re-preprocessing.
	QueryBudgetScale = 1.0
)

// pianoChunkLayout returns the (ChunkSize, SetSize) NewPianoPIR uses for a DB of n rows.
func pianoChunkLayout(n uint64) (uint64, uint64) {
	targetChunkSize := uint64(ChunkSizeFactor * math.Sqrt(float64(n)))
	ChunkSize := uint64(1)
	for ChunkSize < targetChunkSize {
		ChunkSize *= 2
	}
	SetSize := uint64(math.Ceil(float64(n) / float64(ChunkSize)))
	// round up to the next mulitple of 4
	SetSize = (SetSize + 3) / 4 * 4
	return ChunkSize, SetSize
}

// clientHintCounts returns how many queries the client supports between
// preprocessings, how many primary hints it keeps, and how many backup hints it
// keeps per chunk. LocalStorageSize and the hint allocation both follow it.
func clientHintCounts(config *PianoPIRConfig) (maxQueryNum, primaryHintNum, maxQueryPerChunk uint64) {
	maxQueryNum = uint64(math.Sqrt(float64(config.DBSize)) * math.Log(float64(config.DBSize)))
	if QueryBudgetScale != 1.0 {
		// keep at least one backup per chunk, or the first query in a chunk exhausts it
		maxQueryNum = max(uint64(QueryBudgetScale*float64(maxQueryNum)), config.SetSize)
	}
	primaryHintNum = primaryNumParam(float64(maxQueryNum), float64(config.ChunkSize), config.FailureProbLog2+1) // fail prob 2^(-41)
	primaryHintNum = (primaryHintNum + config.ThreadNum - 1) / config.ThreadNum * config.ThreadNum
	maxQueryPerChunk = 3 * uint64(float64(maxQueryNum)/float64(config.SetSize))
	maxQueryPerChunk = (maxQueryPerChunk + config.ThreadNum - 1) / config.ThreadNum * config.ThreadNum
	return maxQueryNum, primaryHintNum, maxQueryPerChunk
}

// hintStorageBytes is LocalStorageSize computed from the config alone, before
// any client is allocated.
func hintStorageBytes(config *PianoPIRConfig) float64 {
	_, primaryHintNum, maxQueryPerChunk := clientHintCounts(config)
	totalBackupHintNum := float64(config.SetSize) * float64(maxQueryPerChunk)
	return float64(primaryHintNum)*(16+float64(config.DBEntryByteNum)) +
		totalBackupHintNum*(16+2*float64(config.DBEntryByteNum))
}

// NewPianoPIRClient is an initialization function for the client
func NewPianoPIRClient(config *PianoPIRConfig) *PianoPIRClient {

	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	masterKey := RandKey(rng)
	longKey := GetLongKey((*PrfKey128)(&masterKey))
	//seed := int64(1678852332934430000)

	maxQueryNum, primaryHintNum, maxQueryPerChunk := clientHintCounts(config)

	if Simulate {
		return &PianoPIRClient{
			config:           config,
			MaxQueryNum:      maxQueryNum,
			QueryHistogram:   make([]uint64, config.SetSize),
			primaryHintNum:   primaryHintNum,
			maxQueryPerChunk: maxQueryPerChunk,
			localCache:       make(map[uint64][]uint64),
			sim:              &simClient{},
		}
	}

	//fmt.Printf("maxQueryNum = %v\n", maxQueryNum)
	//fmt.Printf("primaryHintNum = %v\n", primaryHintNum)
	//fmt.Printf("maxQueryPerChunk = %v\n", maxQueryPerChunk)

	masterKey = RandKey(rng)
	return &PianoPIRClient{
		config:   config,
		skipPrep: false, // default to false

		//rng:       rng,
		masterKey: masterKey,
		longKey:   longKey,

		MaxQueryNum:      maxQueryNum,
		FinishedQueryNum: 0,
		QueryHistogram:   make([]uint64, config.SetSize),

		primaryHintNum:      primaryHintNum,
		primaryShortTag:     make([]uint64, primaryHintNum),
		primaryParity:       make([]uint64, primaryHintNum*config.DBEntrySize),
		primaryProgramPoint: make([]uint64, primaryHintNum),

		maxQueryPerChunk: maxQueryPerChunk,
		replacementIdx:   make([][]uint64, config.SetSize),
		replacementVal:   make([][]uint64, config.SetSize),

		backupShortTag: make([][]uint64, config.SetSize),
		backupParity:   make([][]uint64, config.SetSize),

		localCache: make(map[uint64][]uint64),
	}
}

// return the local storage in bytes
func (c *PianoPIRClient) LocalStorageSize() float64 {
	localStorageSize := float64(0)
	localStorageSize = localStorageSize + float64(c.primaryHintNum)*8                                // the primary hint short tag
	localStorageSize = localStorageSize + float64(c.primaryHintNum)*float64(c.config.DBEntryByteNum) // the primary parity
	localStorageSize = localStorageSize + float64(c.primaryHintNum)*8                                // the primary program point
	totalBackupHintNum := float64(c.config.SetSize) * float64(c.maxQueryPerChunk)
	localStorageSize = localStorageSize + float64(totalBackupHintNum)*8                                // the replacement indices
	localStorageSize = localStorageSize + float64(totalBackupHintNum)*float64(c.config.DBEntryByteNum) // the replacement values
	localStorageSize = localStorageSize + float64(totalBackupHintNum)*8                                // the backup short tag
	localStorageSize = localStorageSize + float64(totalBackupHintNum)*float64(c.config.DBEntryByteNum) // the backup parities

	return localStorageSize
}

func (c *PianoPIRClient) PrintStorageBreakdown() {
	fmt.Printf("primary hint short tag = %v\n", c.primaryHintNum*4)
	fmt.Printf("primary parity = %v\n", c.primaryHintNum*c.config.DBEntryByteNum)
	fmt.Printf("primary program point = %v\n", c.primaryHintNum*4)
	totalBackupHintNum := c.config.SetSize * c.maxQueryPerChunk
	fmt.Printf("replacement indices = %v\n", totalBackupHintNum*4)
	fmt.Printf("replacement values = %v\n", totalBackupHintNum*c.config.DBEntryByteNum)
	fmt.Printf("backup short tag = %v\n", totalBackupHintNum*4)
	fmt.Printf("backup parities = %v\n", totalBackupHintNum*c.config.DBEntryByteNum)
}

func (c *PianoPIRClient) Initialization() {
	if c.sim != nil {
		c.simInitialization()
		return
	}
	//TODO: implemente the preprocessing logic
	c.FinishedQueryNum = 0

	// resample the key
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	c.masterKey = RandKey(rng)
	c.longKey = GetLongKey((*PrfKey128)(&c.masterKey))

	c.QueryHistogram = make([]uint64, c.config.SetSize)
	for i := 0; i < int(c.config.SetSize); i++ {
		c.QueryHistogram[i] = 0
	}

	// first initialize everything to be zero

	shortTagCount := uint64(0)

	c.primaryShortTag = make([]uint64, c.primaryHintNum)
	c.primaryParity = make([]uint64, c.primaryHintNum*c.config.DBEntrySize)
	c.primaryProgramPoint = make([]uint64, c.primaryHintNum)

	for i := 0; i < int(c.primaryHintNum); i++ {
		c.primaryShortTag[i] = shortTagCount
		c.primaryParity[i] = 0
		c.primaryProgramPoint[i] = DefaultProgramPoint
		shortTagCount += 1
	}

	c.replacementIdx = make([][]uint64, c.config.SetSize)
	c.replacementVal = make([][]uint64, c.config.SetSize)
	c.backupShortTag = make([][]uint64, c.config.SetSize)
	c.backupParity = make([][]uint64, c.config.SetSize)

	for i := 0; i < int(c.config.SetSize); i++ {
		c.replacementIdx[i] = make([]uint64, c.maxQueryPerChunk)
		c.replacementVal[i] = make([]uint64, c.maxQueryPerChunk*c.config.DBEntrySize)
		c.backupShortTag[i] = make([]uint64, c.maxQueryPerChunk)
		c.backupParity[i] = make([]uint64, c.maxQueryPerChunk*c.config.DBEntrySize)

		for j := 0; j < int(c.maxQueryPerChunk); j++ {
			c.replacementIdx[i][j] = DefaultProgramPoint
			c.replacementVal[i][j] = 0
			c.backupShortTag[i][j] = shortTagCount
			c.backupParity[i][j] = 0
			shortTagCount += 1
		}
	}

	// clean the cache
	c.localCache = make(map[uint64][]uint64)
}

// EntryXor uses the vectorized kernel for complete four-word groups and a
// scalar tail for short/non-aligned entries. Stage 2 stores four uint8 scores
// in one uint64, so silently requiring a four-word entry corrupts its result.
func EntryXor(a []uint64, b []uint64, entrySize uint64) {
	vectorWords := entrySize - entrySize%4
	if vectorWords > 0 {
		xorSlices(a[:vectorWords], b[:vectorWords], int(vectorWords))
	}
	for i := vectorWords; i < entrySize; i++ {
		a[i] ^= b[i]
	}
}

func (c *PianoPIRClient) Preprocessing(rawDB []uint64) {
	c.Initialization() // first clean everything
	if c.skipPrep || c.sim != nil {
		// only for debugging and benchmarking, or simulated (no hints to build)
		return
	}

	//log.Printf("len(rawDB) %v\n", len(rawDB))
	//if len(rawDB) < int(c.config.ChunkSize*c.config.SetSize*c.config.DBEntrySize) {
	//append with zeros
	//rawDB = append(rawDB, make([]uint64, int(c.config.ChunkSize*c.config.SetSize*c.config.DBEntrySize)-len(rawDB))...)
	//}

	//TODO: using multiple threads
	for i := uint64(0); i < c.config.SetSize; i++ {
		start := i * c.config.ChunkSize
		//end := min((i+1)*c.config.ChunkSize, c.config.DBSize)
		end := (i + 1) * c.config.ChunkSize
		if end*c.config.DBEntrySize > uint64(len(rawDB)) {
			// in this case, we
			tmpChunk := make([]uint64, c.config.ChunkSize*c.config.DBEntrySize)
			for j := start * c.config.DBEntrySize; j < end*c.config.DBEntrySize; j++ {
				if j >= uint64(len(rawDB)) {
					tmpChunk[j-start*c.config.DBEntrySize] = 0
				} else {
					tmpChunk[j-start*c.config.DBEntrySize] = rawDB[j]
				}
			}
			c.UpdatePreprocessing(i, tmpChunk)
		} else {
			//fmt.Println("preprocessing chunk ", i, "start ", start, "end ", end)
			c.UpdatePreprocessing(i, rawDB[start*c.config.DBEntrySize:end*c.config.DBEntrySize])
		}
	}
}

func (c *PianoPIRClient) UpdatePreprocessing(chunkId uint64, chunk []uint64) {

	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))

	if len(chunk) < int(c.config.ChunkSize*c.config.DBEntrySize) {
		fmt.Println("not enough chunk size")
		//chunk = append(chunk, make([]uint64, int(c.config.ChunkSize*c.config.DBEntrySize)-len(chunk))...)
	}

	//fmt.Printf("primary hint num = %v\n", c.primaryHintNum)

	// first enumerate all primar hints
	for i := uint64(0); i < c.primaryHintNum; i++ {
		//fmt.Println("i = ", i)
		offset := PRFEvalWithLongKeyAndTag(c.longKey, c.primaryShortTag[i], uint64(chunkId)) & (c.config.ChunkSize - 1)
		//fmt.Printf("i = %v, offset = %v\n", i, offset)
		if (i+1)*c.config.DBEntrySize > uint64(len(c.primaryParity)) {
			//fmt.Errorf("i = %v, i*c.config.DBEntrySize = %v, len(c.primaryParity) = %v", i, i*c.config.DBEntrySize, len(c.primaryParity)
			log.Fatalf("i = %v, i*c.config.DBEntrySize = %v, len(c.primaryParity) = %v", i, i*c.config.DBEntrySize, len(c.primaryParity))
		}
		EntryXor(c.primaryParity[i*c.config.DBEntrySize:(i+1)*c.config.DBEntrySize], chunk[offset*c.config.DBEntrySize:(offset+1)*c.config.DBEntrySize], c.config.DBEntrySize)
	}

	//fmt.Println("finished primary hints")

	// second enumerate all backup hints
	for i := uint64(0); i < c.config.SetSize; i++ {
		// ignore if i == chunkId
		if i == chunkId {
			continue
		}
		for j := uint64(0); j < c.maxQueryPerChunk; j++ {
			offset := PRFEvalWithLongKeyAndTag(c.longKey, c.backupShortTag[i][j], uint64(chunkId)) & (c.config.ChunkSize - 1)
			EntryXor(c.backupParity[i][j*c.config.DBEntrySize:(j+1)*c.config.DBEntrySize], chunk[offset*c.config.DBEntrySize:(offset+1)*c.config.DBEntrySize], c.config.DBEntrySize)
		}
	}

	//fmt.Println("finished backup hints")

	// finally store the replacement

	for j := uint64(0); j < c.maxQueryPerChunk; j++ {
		offset := rng.Uint64() & (c.config.ChunkSize - 1)
		c.replacementIdx[chunkId][j] = offset + chunkId*c.config.ChunkSize
		copy(c.replacementVal[chunkId][j*c.config.DBEntrySize:(j+1)*c.config.DBEntrySize], chunk[offset*c.config.DBEntrySize:(offset+1)*c.config.DBEntrySize])
	}

	//fmt.Println("finished replacement")
}

func (c *PianoPIRClient) Query(idx uint64, server *PianoPIRServer, realQuery bool) ([]uint64, error) {
	if c.sim != nil {
		return c.simQuery(idx, server, realQuery)
	}

	ret := make([]uint64, c.config.DBEntrySize)
	// initialize ret to be all zeros
	for i := uint64(0); i < c.config.DBEntrySize; i++ {
		ret[i] = 0
	}

	//// if it's a dummy query, then just generate c.config.SetSize random numbers between 0...c.config.ChunkSize
	//if !realQuery {
	//	return ret, c.constantWorkDummyQuery(idx, server)
	//}

	// if it's a dummy query, then just generate c.config.SetSize random numbers between 0...c.config.ChunkSize
	if !realQuery {
		offsets := make([]uint32, c.config.SetSize)
		for i := uint64(0); i < c.config.SetSize; i++ {
			offsets[i] = uint32(rand.Uint64() & (c.config.ChunkSize - 1))
		}
		// still simulate/count the time and bits for the query
		_, err := server.PrivateQuery(offsets)

		return ret, err
	}

	if idx >= c.config.DBSize {
		log.Fatalf("idx %v is out of range", idx)

		// return an empty entry and an error
		return ret, fmt.Errorf("idx %v is out of range", idx)
	}

	//// if the idx is in the local cache, then return the result from the local cache
	//if v, ok := c.localCache[idx]; ok {
	//	if c.constantWork {
	//		if err := c.constantWorkDummyQuery(idx, server); err != nil {
	//			return ret, err
	//		}
	//	}
	//	return v, nil
	//}

	if v, ok := c.localCache[idx]; ok {
		return v, nil
	}

	// now we need to make a real query
	if c.FinishedQueryNum >= c.MaxQueryNum {
		log.Printf("fnished query = %v", c.FinishedQueryNum)
		log.Printf("max query num = %v", c.MaxQueryNum)
		log.Printf("exceed the maximum number of queries")
		return ret, fmt.Errorf("exceed the maximum number of queries")
	}

	chunkId := idx / c.config.ChunkSize
	offset := idx % c.config.ChunkSize

	if c.QueryHistogram[chunkId] >= c.maxQueryPerChunk {
		log.Printf("Too many queries in chunk %v", chunkId)
		log.Printf("Max query per chunk = %v", c.maxQueryPerChunk)
		return ret, fmt.Errorf("too many queries in chunk %v", chunkId)
	}

	// now we find the hit hint in the primary hint table

	hitId := uint64(DefaultProgramPoint)
	for i := uint64(0); i < c.primaryHintNum; i++ {
		hintOffset := PRFEvalWithLongKeyAndTag(c.longKey, c.primaryShortTag[i], uint64(chunkId)) & (c.config.ChunkSize - 1)
		if hintOffset == offset {
			// if this chunk has been programmed in this chunk before, then it shouldn't count
			if hitId == DefaultProgramPoint && (c.primaryProgramPoint[i] == DefaultProgramPoint || (c.primaryProgramPoint[i]/c.config.ChunkSize != chunkId)) {
				hitId = i
				if !c.constantWork {
					break
				}
			}
		}
	}

	if hitId == DefaultProgramPoint {
		//log.Printf("No hit hint in the primary hint table, current idx = %v", idx)
		return ret, fmt.Errorf("no hit hint in the primary hint table")
	}

	// now we expand this hit hint to a full set
	querySet := make([]uint64, c.config.SetSize)

	for i := uint64(0); i < c.config.SetSize; i++ {
		hintOffset := PRFEvalWithLongKeyAndTag(c.longKey, c.primaryShortTag[hitId], uint64(i)) & (c.config.ChunkSize - 1)
		querySet[i] = i*c.config.ChunkSize + hintOffset
	}

	// if it's programmed, we need to enforce it
	if c.primaryProgramPoint[hitId] != DefaultProgramPoint {
		//log.Printf("hitId = %v, c.primaryProgramPoint[hitId] = %v", hitId, c.primaryProgramPoint[hitId])
		querySet[c.primaryProgramPoint[hitId]/c.config.ChunkSize] = c.primaryProgramPoint[hitId]
	}

	// now we find the first unconsumed replacement idx and val in the chunkId-th group
	inGroupIdx := uint64(c.QueryHistogram[chunkId])
	replIdx := c.replacementIdx[chunkId][inGroupIdx]
	replVal := c.replacementVal[chunkId][inGroupIdx*c.config.DBEntrySize : (inGroupIdx+1)*c.config.DBEntrySize]
	querySet[chunkId] = replIdx

	// now we make a private query
	// we only send the offset to the server, so that we can save some bandwidth
	querySetOffset := make([]uint32, c.config.SetSize)
	for i := uint64(0); i < c.config.SetSize; i++ {
		querySetOffset[i] = uint32(querySet[i] & (c.config.ChunkSize - 1))
	}

	response, err := server.PrivateQuery(querySetOffset)

	// we revert the influence of the replacement
	EntryXor(response, replVal, c.config.DBEntrySize)
	// we also xor the original parity
	EntryXor(response, c.primaryParity[hitId*c.config.DBEntrySize:(hitId+1)*c.config.DBEntrySize], c.config.DBEntrySize)
	// now response is the answer.

	// for now just do non private query
	//response, err := server.NonePrivateQuery(idx)

	// now we need to refresh
	c.primaryShortTag[hitId] = c.backupShortTag[chunkId][inGroupIdx]
	copy(c.primaryParity[hitId*c.config.DBEntrySize:(hitId+1)*c.config.DBEntrySize], c.backupParity[chunkId][inGroupIdx*c.config.DBEntrySize:(inGroupIdx+1)*c.config.DBEntrySize])
	c.primaryProgramPoint[hitId] = idx                                                                                   // program the original index
	EntryXor(c.primaryParity[hitId*c.config.DBEntrySize:(hitId+1)*c.config.DBEntrySize], response, c.config.DBEntrySize) // also need to add the current response to the parity

	//finally we need to update the history information
	c.FinishedQueryNum += 1
	c.QueryHistogram[chunkId] += 1
	c.localCache[idx] = response

	return response, err
}

// constantWorkDummyQuery performs the same server-sized request as a real
// query. In constant-work mode it also scans every primary hint so a local
// cache hit does not create the large client-side timing shortcut.
func (c *PianoPIRClient) constantWorkDummyQuery(idx uint64, server *PianoPIRServer) error {
	if c.constantWork {
		chunkID := uint64(0)
		offset := uint64(0)
		if c.config.DBSize > 0 {
			idx %= c.config.DBSize
			chunkID = idx / c.config.ChunkSize
			offset = idx % c.config.ChunkSize
		}
		matched := uint64(0)
		for i := uint64(0); i < c.primaryHintNum; i++ {
			hintOffset := PRFEvalWithLongKeyAndTag(c.longKey, c.primaryShortTag[i], chunkID) & (c.config.ChunkSize - 1)
			if hintOffset == offset {
				matched ^= i + 1
			}
		}
		c.dummyScanSink = matched
	}
	offsets := make([]uint32, c.config.SetSize)
	for i := uint64(0); i < c.config.SetSize; i++ {
		offsets[i] = uint32(rand.Uint64() & (c.config.ChunkSize - 1))
	}
	_, err := server.PrivateQuery(offsets)
	return err
}

type PianoPIR struct {
	config *PianoPIRConfig
	client *PianoPIRClient
	server *PianoPIRServer
}

func (p *PianoPIR) SetConstantWork(enabled bool) {
	p.client.constantWork = enabled
}

func newPianoPIRConfig(DBSize, DBEntryByteNum, FailureProbLog2 uint64) *PianoPIRConfig {
	ChunkSize, SetSize := pianoChunkLayout(DBSize)
	return &PianoPIRConfig{
		DBEntryByteNum:  DBEntryByteNum,
		DBEntrySize:     DBEntryByteNum / 8,
		DBSize:          DBSize,
		ChunkSize:       ChunkSize,
		SetSize:         SetSize,
		ThreadNum:       8,
		FailureProbLog2: FailureProbLog2,
	}
}

func NewPianoPIR(DBSize uint64, DBEntryByteNum uint64, rawDB []uint64, FailureProbLog2 uint64) *PianoPIR {
	DBEntrySize := DBEntryByteNum / 8

	// assert that the rawDB is of the correct size
	if uint64(len(rawDB)) != DBSize*DBEntrySize {
		log.Fatalf("Piano PIR len(rawDB) = %v; want %v", len(rawDB), DBSize*DBEntrySize)
	}

	config := newPianoPIRConfig(DBSize, DBEntryByteNum, FailureProbLog2)

	client := NewPianoPIRClient(config)
	server := NewPianoPIRServer(config, rawDB)

	return &PianoPIR{
		config: config,
		client: client,
		server: server,
	}
}

func (p *PianoPIR) Preprocessing() {
	p.client.Preprocessing(p.server.rawDB)
}

func (p *PianoPIR) DummyPreprocessing() {
	p.client.Initialization()
	p.client.skipPrep = true
}

func (p *PianoPIR) Query(idx uint64, realQuery bool) ([]uint64, error) {

	if p.client.FinishedQueryNum == p.client.MaxQueryNum {
		fmt.Printf("exceed the maximum number of queries %v and redo preprocessing\n", p.client.MaxQueryNum)
		p.client.Preprocessing(p.server.rawDB)
	}

	return p.client.Query(idx, p.server, realQuery)
}

func (p *PianoPIR) LocalStorageSize() float64 {
	return p.client.LocalStorageSize()
}

func (p *PianoPIR) CommCostPerQuery() float64 {

	// upload contains p.config.SetSize 32-bit integers
	// download contains p.config.DBEntrySize 64-bit integers
	return float64(p.config.SetSize*4 + p.config.DBEntrySize*8)
}

func (p *PianoPIR) Config() *PianoPIRConfig {
	return p.config
}
