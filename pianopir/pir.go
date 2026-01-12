package pianopir

import (
	"fmt"
	//"encoding/binary"

	"log"
	"math"
	"math/rand"
	"time"

	"github.com/sirupsen/logrus"
)

const (
	//FailureProbLog2     = 40
	DefaultProgramPoint = 0x7fffffff
)

type PianoPIRConfig struct {
	DBEntryByteNum  uint64 // the (average) number of bytes in a DB entry - just used for debugging/printing
	MaxDBEntrySize  uint64 // The maximum number of uint64 in a DB entry (this is actually just the size of the largest entry)
	DBSize          uint64
	ChunkSize       uint64 // chunksize is 2 times sqrt of DBSize and then rounded up to power of 2.
	SetSize         uint64 // Set size is the number of chunks we have (rounde dup to multiple of 4)
	ThreadNum       uint64
	FailureProbLog2 uint64
}

type PianoPIRServer struct {
	config *PianoPIRConfig
	// Different from the original implementation, we store the DB as a two dimensional array, the first
	// Dimentions is the index and then the second is all the items stored at that index
	rawDB [][]uint64
}

// an initialization function for the server
func NewPianoPIRServer(config *PianoPIRConfig, rawDB [][]uint64) *PianoPIRServer {
	return &PianoPIRServer{
		config: config,
		rawDB:  rawDB,
	}
}

func (s *PianoPIRServer) NonePrivateQuery(idx uint64) ([]uint64, error) {
	ret := make([]uint64, s.config.MaxDBEntrySize)
	// initialize ret to be all zeros
	for i := uint64(0); i < s.config.MaxDBEntrySize; i++ {
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
	// copy(ret, s.rawDB[idx*s.config.DBEntrySize:(idx+1)*s.config.DBEntrySize])
	// Don't need to do the fancy offset as above
	copy(ret, s.rawDB[idx])
	return ret, nil
}

// the private query just computes the xor sum of the elements in the idxs list. The return should be a list of size
// 1 but with all the offsets XOR'd into it. It's easier to return it as a 2d array to get it to work nice with everything else
func (s *PianoPIRServer) PrivateQuery(offsets []uint32) ([][]uint64, error) {
	tempRet := make([][]uint64, 1)
	tempRet[0] = make([]uint64, s.config.MaxDBEntrySize)
	// initialize temp_ret[0] to be all zeros
	for i := uint64(0); i < s.config.MaxDBEntrySize; i++ {
		tempRet[0][i] = 0
	}

	for i := uint64(0); i < s.config.SetSize; i++ {
		idx := uint64(offsets[i]) + i*s.config.ChunkSize

		if idx >= s.config.DBSize {
			continue
		}

		// xor the idx*DBEntrySize-th to (idx+1)*DBEntrySize-th elements to temp_ret[0]
		// EntryXor(temp_ret[0], s.rawDB[idx*s.config.DBEntrySize:(idx+1)*s.config.DBEntrySize], s.config.DBEntrySize)

		// This is slightly more complicated. Before all entries where the same size, now we have to
		// handle the case where one entry is shorter. to do this we shorten the larger array to be the
		// same size as the shorter one

		// This is probably going to cause issues as well, so I will leave a reminder: TODO Allo entries
		// should be a multiple of 4!

		entry := s.rawDB[idx : idx+1]
		//TODO: this might be causing issues, uncomment?
		//n := len(entry[0]) // uint64 elements
		//
		//// Debug check for size:
		//if n%4 != 0 {
		//	logrus.Fatalf("idx %v is out of range", idx)
		//	panic("DB entry size mismatch - not a multiple of 4!!!!")
		//}

		// I think this is an update 'in place', so I shouldn't need to copy temp_ret[0]/do any fancy memory allocs
		EntryXor(tempRet, entry)

	}

	return tempRet, nil
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

	// an upper bound of the number of queries in each chunk (A chunk is just one of the sqrt(n) sets we split the DB into)
	maxQueryPerChunk uint64
	QueryHistogram   []uint64

	// primary hint table - From the original PIR paper: The primary hint table contains sqrt(n) indices chosen (One per chunk)
	//according to the PRF keys and the partiy of the entire DB.
	primaryHintNum      uint64     // the number of hints in the primary hint table
	primaryShortTag     []uint64   // the prf short tag - used to find the indexes the hint has
	primaryParity       [][]uint64 // Made 2 dimensional: 1st dimension is the chunk for which we have the parity (second is the parity of entries)
	primaryProgramPoint []uint64   // the point that the set is programmed

	// Replacement indices - this is the actual value in the DB. When we make a query, we sent out sprt(n) hints but we
	// replace one value in S, specifically one of which we have the a replacement value for. When the we replace that
	// value we replace it with the truly wanted value. Now we can XOR our hint chunk with the response and get the
	// actual value XOR the replacement value.
	replacementIdx [][]uint64   // the replacement indices. We have one array for each chunk
	replacementVal [][][]uint64 // the replacement values. We have one array for each chunk

	// backup hint table - Holds a few more Sets and hints so that we can replace the queries we lose. Note that we
	//have to discard the entire hint when using the replacement stuff, this is because it skews the random
	//distribution. For a backup we look over all the backup sets until we find one that just happens to have the index
	//we used (so it retains ieparture from just a year ago when she sought to distance herself from a scandal that centred on the highly cts secure random properties)
	backupShortTag [][]uint64   // the prf short tag
	backupParity   [][][]uint64 // notice that we group DBEntrySize uint64 into one entry

	// local cache
	localCache map[uint64][][]uint64
}

func primaryNumParam(Q float64, ChunkSize float64, target uint64) uint64 {
	k := math.Ceil(math.Log(2) * (float64(target)))
	return uint64(k) * uint64(ChunkSize)
}

// NewPianoPIRClient is an initialization function for the client
func NewPianoPIRClient(config *PianoPIRConfig) *PianoPIRClient {

	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	masterKey := RandKey(rng)
	longKey := GetLongKey((*PrfKey128)(&masterKey))
	//seed := int64(1678852332934430000)

	maxQueryNum := uint64(math.Sqrt(float64(config.DBSize)) * math.Log(float64(config.DBSize)))
	primaryHintNum := primaryNumParam(float64(maxQueryNum), float64(config.ChunkSize), config.FailureProbLog2+1) // fail prob 2^(-41)
	primaryHintNum = (primaryHintNum + config.ThreadNum - 1) / config.ThreadNum * config.ThreadNum
	maxQueryPerChunk := 3 * uint64(float64(maxQueryNum)/float64(config.SetSize))
	maxQueryPerChunk = (maxQueryPerChunk + config.ThreadNum - 1) / config.ThreadNum * config.ThreadNum

	primaryParity := make([][]uint64, primaryHintNum)

	for i := 0; i < int(primaryHintNum); i++ {
		primaryParity[i] = make([]uint64, config.MaxDBEntrySize)
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

		primaryHintNum:  primaryHintNum,
		primaryShortTag: make([]uint64, primaryHintNum),
		// TODO: Check this is still fine. I think it should be as it'll just be padded with 0s (which do nothing for XOR)?
		primaryParity:       primaryParity,
		primaryProgramPoint: make([]uint64, primaryHintNum),

		maxQueryPerChunk: maxQueryPerChunk,
		replacementIdx:   make([][]uint64, config.SetSize),
		replacementVal:   make([][][]uint64, config.SetSize),

		backupShortTag: make([][]uint64, config.SetSize),
		// Outer dim should be config.SetSize, 2nd is maxQueryPerChunk, 3rd is MaxDBEntrySize
		backupParity: make([][][]uint64, config.SetSize),

		localCache: make(map[uint64][][]uint64),
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
	// TODO: Check this is still fine. I think it should be as it'll just be padded with 0s?
	c.primaryParity = make([][]uint64, c.primaryHintNum)
	c.primaryProgramPoint = make([]uint64, c.primaryHintNum)

	for i := 0; i < int(c.primaryHintNum); i++ {
		c.primaryShortTag[i] = shortTagCount
		c.primaryParity[i] = make([]uint64, c.config.MaxDBEntrySize)
		c.primaryParity[i][0] = 0
		c.primaryProgramPoint[i] = DefaultProgramPoint
		shortTagCount += 1
	}

	// TODO: BE CAREFUL WITH REPLACEMENT/BACKUP VALS!! I might accidentally writing 0s over good data I think the parity
	// will be fine to pad out, even for running time as it should just be done in pre-processing. I could perhaps do it
	// dynamically instead of what I'm doing below for faster speed.
	c.replacementIdx = make([][]uint64, c.config.SetSize)
	c.replacementVal = make([][][]uint64, c.config.SetSize)
	c.backupShortTag = make([][]uint64, c.config.SetSize)
	c.backupParity = make([][][]uint64, c.config.SetSize)

	for i := 0; i < int(c.config.SetSize); i++ {
		c.replacementIdx[i] = make([]uint64, c.maxQueryPerChunk)
		c.backupShortTag[i] = make([]uint64, c.maxQueryPerChunk)

		c.replacementVal[i] = make([][]uint64, c.maxQueryPerChunk)
		c.backupParity[i] = make([][]uint64, c.maxQueryPerChunk)

		for j := 0; j < int(c.maxQueryPerChunk); j++ {

			c.backupParity[i][j] = make([]uint64, c.config.MaxDBEntrySize)
			c.replacementVal[i][j] = make([]uint64, c.config.MaxDBEntrySize)

			c.replacementIdx[i][j] = DefaultProgramPoint
			c.backupShortTag[i][j] = shortTagCount
			shortTagCount += 1

			//TODO: this dynamically?
			for k := 0; uint64(k) < c.config.MaxDBEntrySize; k++ {

				c.backupParity[i][j][k] = 0
				c.replacementVal[i][j][k] = 0

			}
		}
	}

	// clean the cache
	c.localCache = make(map[uint64][][]uint64)
}

// Use native XOR instead of ASM. I think the ASM version has hardware acceleration though, so I'll leave the other
// below and commented out for now.
func EntryXor(a, b [][]uint64) {

	for i := 0; i < len(b); i++ {
		ai := a[i]
		bi := b[i]

		n := len(ai)
		if lb := len(bi); lb < n {
			n = lb
		}
		if n == 0 {
			continue
		}

		j := 0
		n8 := n &^ 7 // largest multiple of 8 <= n
		if n8 > 0 {
			// Help the compiler prove bounds safety for the unrolled loop.
			_ = ai[n8-1]
			_ = bi[n8-1]

			for ; j < n8; j += 8 {
				ai[j+0] ^= bi[j+0]
				ai[j+1] ^= bi[j+1]
				ai[j+2] ^= bi[j+2]
				ai[j+3] ^= bi[j+3]
				ai[j+4] ^= bi[j+4]
				ai[j+5] ^= bi[j+5]
				ai[j+6] ^= bi[j+6]
				ai[j+7] ^= bi[j+7]
			}
		}

		for ; j < n; j++ {
			ai[j] ^= bi[j]
		}
	}
}

//func EntryXor(a, b [][]uint64) {
//	limit := len(a)
//	if lb := len(b); lb < limit {
//		limit = lb
//	}
//
//	for i := 0; i < limit; i++ {
//		ai := a[i]
//		bi := b[i]
//
//		n := len(ai)
//		if lb := len(bi); lb < n {
//			n = lb
//		}
//		if n == 0 {
//			continue
//		}
//
//		n4 := n &^ 3
//		if n4 != 0 {
//			// Option A: xorSlices(ai[:n4], bi[:n4])
//			// Option B: xorSlices(ai, bi, n4)
//			xorSlices(ai[:n4], bi[:n4])
//		}
//		for j := n4; j < n; j++ {
//			ai[j] ^= bi[j]
//		}
//	}
//
//}

func (c *PianoPIRClient) Preprocessing(rawDB [][]uint64) [][]uint64 {
	c.Initialization() // first clean everything

	if len(rawDB) < int(c.config.ChunkSize*c.config.SetSize) {
		// append with zeros
		prev_len := len(rawDB)
		rawDB = append(rawDB, make([][]uint64, int(c.config.ChunkSize*c.config.SetSize)-len(rawDB))...)
		for j := 1; j < int(c.config.ChunkSize*c.config.SetSize)-prev_len; j++ {
			rawDB[j+prev_len] = make([]uint64, c.config.MaxDBEntrySize)
			for k := 0; k < len(rawDB[j+prev_len]); k++ {
				rawDB[j+prev_len][k] = 0
			}
		}
	}

	if c.skipPrep {
		// only for debugging and benchmarking
		return rawDB
	}

	//log.Printf("len(rawDB) %v\n", len(rawDB))
	//if len(rawDB) < int(c.config.ChunkSize*c.config.SetSize*c.config.DBEntrySize) {
	//append with zeros
	//rawDB = append(rawDB, make([]uint64, int(c.config.ChunkSize*c.config.SetSize*c.config.DBEntrySize)-len(rawDB))...)
	//}

	//TODO: using multiple threads
	// Remember setsize is the number of chunks and chunksize is the number of entries per chunk
	for i := uint64(0); i < c.config.SetSize; i++ {
		start := i * c.config.ChunkSize
		//end := min((i+1)*c.config.ChunkSize, c.config.DBSize)
		end := (i + 1) * c.config.ChunkSize
		// If our chunk would go outside of the database, we need to append zeros
		// I shouldn't need this in my new fancy 2d rawDb array(?)
		//if end*c.config.DBEntrySize > uint64(len(rawDB)) {
		//	// in this case, we
		//	tmpChunk := make([]uint64, c.config.ChunkSize*c.config.DBEntrySize)
		//	for j := start * c.config.DBEntrySize; j < end*c.config.DBEntrySize; j++ {
		//		if j >= uint64(len(rawDB)) {
		//			tmpChunk[j-start*c.config.DBEntrySize] = 0
		//		} else {
		//			tmpChunk[j-start*c.config.DBEntrySize] = rawDB[j]
		//		}
		//	}
		//	c.UpdatePreprocessing(i, tmpChunk)
		//} else {
		//	//fmt.Println("preprocessing chunk ", i, "start ", start, "end ", end)
		//	c.UpdatePreprocessing(i, rawDB[start*c.config.DBEntrySize:end*c.config.DBEntrySize])
		//}
		// Turns out we do still need to append empty items to our rawDB...

		c.UpdatePreprocessing(i, rawDB[start:end])
	}

	return rawDB

}

func (c *PianoPIRClient) UpdatePreprocessing(chunkId uint64, chunk [][]uint64) {

	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))

	// if len(chunk) < int(c.config.ChunkSize*c.config.DBEntrySize) {
	if len(chunk) < int(c.config.ChunkSize) {
		fmt.Println("not enough chunk size")
		//chunk = append(chunk, make([]uint64, int(c.config.ChunkSize*c.config.DBEntrySize)-len(chunk))...)
	}

	//fmt.Printf("primary hint num = %v\n", c.primaryHintNum)

	// first enumerate all primar hints
	for i := uint64(0); i < c.primaryHintNum; i++ {
		//fmt.Println("i = ", i)
		// offset is the offset of the entry in the chunk
		offset := PRFEvalWithLongKeyAndTag(c.longKey, c.primaryShortTag[i], uint64(chunkId)) & (c.config.ChunkSize - 1)
		//fmt.Printf("i = %v, offset = %v\n", i, offset)
		//if (i+1)*c.config.DBEntrySize > uint64(len(c.primaryParity)) {
		//	//fmt.Errorf("i = %v, i*c.config.DBEntrySize = %v, len(c.primaryParity) = %v", i, i*c.config.DBEntrySize, len(c.primaryParity)
		//	log.Fatalf("i = %v, i*c.config.DBEntrySize = %v, len(c.primaryParity) = %v", i, i*c.config.DBEntrySize, len(c.primaryParity))
		//}
		// TODO: I think the primary parity has to be made 2D as well? 1st dim would be the hint and 2nd dim would be the XOR of the entries
		// (Probably of size maxentrysize)
		EntryXor(c.primaryParity[i:(i+1)], chunk[offset:(offset+1)])
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

			EntryXor(c.backupParity[i][j:(j+1)], chunk[offset:(offset+1)])
		}
	}

	//fmt.Println("finished backup hints")

	// finally store the replacement

	for j := uint64(0); j < c.maxQueryPerChunk; j++ {
		offset := rng.Uint64() & (c.config.ChunkSize - 1)
		c.replacementIdx[chunkId][j] = offset + chunkId*c.config.ChunkSize
		copy(c.replacementVal[chunkId][j:(j+1)], chunk[offset:(offset+1)])
	}

	//fmt.Println("finished replacement")
}

func (c *PianoPIRClient) Query(idx uint64, server *PianoPIRServer, realQuery bool) ([]uint64, error) {

	// TODO: Should just make red dynamically sized?
	ret := make([]uint64, c.config.MaxDBEntrySize)
	// initialize ret to be all zeros
	for i := uint64(0); i < c.config.MaxDBEntrySize; i++ {
		ret[i] = 0
	}

	// if it's a dummy query, then just generate c.config.SetSize random numbers between 0...c.config.ChunkSize
	if !realQuery {
		offsets := make([]uint32, c.config.SetSize)
		for i := uint64(0); i < c.config.SetSize; i++ {
			offsets[i] = uint32(rand.Uint64() & (c.config.ChunkSize - 1))
		}
		_, err := server.PrivateQuery(offsets)

		return ret, err
	}

	if idx >= c.config.DBSize {
		log.Fatalf("idx %v is out of range", idx)

		// return an empty entry and an error
		return ret, fmt.Errorf("idx %v is out of range", idx)
	}

	// if the idx is in the local cache, then return the result from the local cache
	if v, ok := c.localCache[idx]; ok {
		return v[0], nil
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
			if c.primaryProgramPoint[i] == DefaultProgramPoint || (c.primaryProgramPoint[i]/c.config.ChunkSize != chunkId) {
				hitId = i
				break
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
	// This is really just one value... From inGroupIdx to (inGroupIdx+1)
	replVal := c.replacementVal[chunkId][inGroupIdx:(inGroupIdx + 1)]
	querySet[chunkId] = replIdx

	// now we make a private query
	// we only send the offset to the server, so that we can save some bandwidth
	querySetOffset := make([]uint32, c.config.SetSize)
	for i := uint64(0); i < c.config.SetSize; i++ {
		querySetOffset[i] = uint32(querySet[i] & (c.config.ChunkSize - 1))
	}

	response, err := server.PrivateQuery(querySetOffset)

	// we revert the influence of the replacement
	// (These should both be of len 1)
	EntryXor(response, replVal)
	// we also xor the original parity
	EntryXor(response, c.primaryParity[hitId:(hitId+1)])
	// now response is the answer.

	// for now just do non private query
	//response, err := server.NonePrivateQuery(idx)

	// now we need to refresh
	c.primaryShortTag[hitId] = c.backupShortTag[chunkId][inGroupIdx]
	copy(c.primaryParity[hitId:(hitId+1)], c.backupParity[chunkId][inGroupIdx:(inGroupIdx+1)])
	c.primaryProgramPoint[hitId] = idx                   // program the original index
	EntryXor(c.primaryParity[hitId:(hitId+1)], response) // also need to add the current response to the parity

	//finally we need to update the history information
	c.FinishedQueryNum += 1
	c.QueryHistogram[chunkId] += 1
	c.localCache[idx] = response

	if len(response) > 1 {
		log.Printf("len(response) = %v; want 1", len(response))

		return ret, fmt.Errorf("got too many responses %v", len(response))
	}

	return response[0], err
}

type PianoPIR struct {
	config *PianoPIRConfig
	client *PianoPIRClient
	server *PianoPIRServer
}

func NewPianoPIR(config *PianoPIRConfig, rawDB [][]uint64) *PianoPIR {
	//DBEntrySize := DBEntryByteNum / 8
	//
	//// assert that the rawDB is of the correct size
	//if uint64(len(rawDB)) != DBSize {
	//	log.Fatalf("Piano PIR len(rawDB) = %v; want %v", len(rawDB), DBSize)
	//}
	//
	//targetChunkSize := uint64(2 * math.Sqrt(float64(DBSize)))
	//ChunkSize := uint64(1)
	//for ChunkSize < targetChunkSize {
	//	ChunkSize *= 2
	//}
	//SetSize := uint64(math.Ceil(float64(DBSize) / float64(ChunkSize)))
	//// round up to the next mulitple of 4
	//SetSize = (SetSize + 3) / 4 * 4

	client := NewPianoPIRClient(config)
	server := NewPianoPIRServer(config, rawDB)

	return &PianoPIR{
		config: config,
		client: client,
		server: server,
	}
}

func DeepCopy2DUint64(src [][]uint64) [][]uint64 {
	if src == nil {
		return nil
	}

	dst := make([][]uint64, len(src))
	for i := range src {
		if src[i] == nil {
			// Preserve nil rows distinctly from empty rows.
			dst[i] = nil
			continue
		}
		dst[i] = make([]uint64, len(src[i]))
		copy(dst[i], src[i])
	}
	return dst
}

func (p *PianoPIR) Preprocessing() {
	p.server.rawDB = DeepCopy2DUint64(p.server.rawDB)
	p.server.rawDB = p.client.Preprocessing(p.server.rawDB)
}

func (p *PianoPIR) DummyPreprocessing() {
	p.client.Initialization()
	p.client.skipPrep = true
}

func (p *PianoPIR) Query(idx uint64, realQuery bool) ([]uint64, error) {

	if p.client.FinishedQueryNum == p.client.MaxQueryNum {
		logrus.Warnf("exceed the maximum number of queries %v and redo preprocessing\n", p.client.MaxQueryNum)
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
	// THis is a pretty 'worst-case' estimate.... TODO: AverageDBEntrySize?
	return float64(p.config.SetSize*4 + p.config.MaxDBEntrySize*8)
}

func (p *PianoPIR) Config() *PianoPIRConfig {
	return p.config
}
