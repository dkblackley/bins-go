package pianopir

import (
	"fmt"
	"log"
	"math"
	"sync"
	"time"
)

//"encoding/binary"

const (
	RealQueryPerPartition = 2
	QueryPerPartition     = 2
	DefaultValue          = 0xdeadbeef
	DefaultThreadNum      = 1
)

type SimpleBatchPianoPIRConfig struct {
	DBEntryByteNum  uint64 // the number of bytes in a DB entry
	DBEntrySize     uint64 // the number of uint64 in a DB entry
	DBSize          uint64
	BatchSize       uint64
	PartitionNum    uint64
	PartitionSize   uint64
	ThreadNum       uint64
	FailureProbLog2 uint64
	ConstantWork    bool // use public fixed partition padding and cache-hit dummies
}

const constantWorkOccupancyFailureBits = uint64(16)

type constantWorkCapKey struct {
	queryCount   int
	partitionNum uint64
	failureBits  uint64
}

var constantWorkCapCache sync.Map

// constantWorkPartitionQueries returns a public high-probability upper bound
// on the maximum occupancy when query rows are distributed among partitions.
// It depends only on the public request count and partition count, never on the
// queried locations. The union-bound overflow probability is at most 2^-bits.
func constantWorkPartitionQueries(queryCount int, partitionNum, failureBits uint64) int {
	if queryCount <= 0 {
		return 0
	}
	if partitionNum <= 1 {
		return queryCount
	}
	key := constantWorkCapKey{queryCount, partitionNum, failureBits}
	if cached, ok := constantWorkCapCache.Load(key); ok {
		return cached.(int)
	}

	p := 1.0 / float64(partitionNum)
	probabilities := make([]float64, queryCount+1)
	probabilities[0] = math.Pow(1.0-p, float64(queryCount))
	for occupied := 1; occupied <= queryCount; occupied++ {
		probabilities[occupied] = probabilities[occupied-1] *
			float64(queryCount-occupied+1) / float64(occupied) * p / (1.0 - p)
	}
	targetTail := math.Ldexp(1.0, -int(failureBits)) / float64(partitionNum)
	tail := 0.0
	bound := queryCount
	for occupied := queryCount; occupied >= 0; occupied-- {
		tail += probabilities[occupied]
		if tail > targetTail {
			bound = occupied + 1
			if bound > queryCount {
				bound = queryCount
			}
			break
		}
	}
	constantWorkCapCache.Store(key, bound)
	return bound
}

// it's a simple batch PIR client
// it does not guarantee to output all the queries
// the strategy is simple
// 1. divide the DB into BatchSize / K partitions
// 2. for each partition, create a sub PIR class
// 3a. For each batch of queries, it will first arrange the queries into the partitions.
//     Each partition will have at most K queries (first come first serve).
// 3b. For each partition, it makes at most K queries to the sub PIR class
// 4. It will output the queries in the order of the partitions

type SimpleBatchPianoPIR struct {
	config *SimpleBatchPianoPIRConfig
	subPIR []*PianoPIR

	// the following are stats

	FinishedBatchNum         uint64
	QueriesMadeInPartition   uint64
	SupportBatchNum          uint64
	localStorage             uint64 // bytes
	commCostPerBatchOnline   uint64 // bytes
	commCostPerBatchOffline  uint64 // bytes
	roundsMade               uint64
	InitialPreprocessingTime float64
	MaintainenceTime         float64
	TotalPreProcs            uint

	// download is set when the whole DB is smaller than the Piano client hints for
	// it would be (see DownloadIfCheaper). The client then keeps rawDB and answers
	// every lookup locally: no server interaction, so nothing is revealed.
	download bool
	rawDB    []uint64
}

// DownloadIfCheaper makes NewSimpleBatchPianoPIR hand the client the plain DB
// whenever the Piano hints for it would take at least as much client storage as
// the DB itself. The choice depends only on public sizes, and a local lookup
// sends nothing, so it is trivially private. Off by default.
var DownloadIfCheaper = false

// BatchHintStorageBytes is the client storage NewSimpleBatchPianoPIR would
// allocate for these parameters, without building anything.
func BatchHintStorageBytes(DBSize, DBEntryByteNum, BatchSize, FailureProbLog2 uint64) float64 {
	PartitionNum := BatchSize / RealQueryPerPartition
	PartitionSize := (DBSize + PartitionNum - 1) / PartitionNum
	return float64(PartitionNum) * hintStorageBytes(newPianoPIRConfig(PartitionSize, DBEntryByteNum, FailureProbLog2))
}

func NewSimpleBatchPianoPIR(DBSize uint64, DBEntryByteNum uint64, BatchSize uint64, rawDB []uint64, FailureProbLog2 uint64) *SimpleBatchPianoPIR {
	DBEntrySize := DBEntryByteNum / 8
	// rawDB may stop short of the last entry's end (by less than one entry): the missing words are zeros. That lets
	// a caller pass a read-only mmap of a file whose length is not a whole number of entries (the tree's Stage3
	// embeddings, 8 docs per entry) without copying it to the heap just to pad it; the short last partition is
	// copied and zero-padded below anyway.
	if want := int(DBSize * DBEntrySize); len(rawDB) > want || (DBSize > 0 && len(rawDB) <= want-int(DBEntrySize)) {
		log.Fatalf("BatchPIR: len(rawDB) = %v; want %v (or less than one entry short)", len(rawDB), want)
	}

	// create the sub PIR classes
	PartitionNum := BatchSize / RealQueryPerPartition
	//PartitionSize := DBSize / PartitionNum and round up
	PartitionSize := (DBSize + PartitionNum - 1) / PartitionNum

	config := &SimpleBatchPianoPIRConfig{
		DBEntryByteNum:  DBEntryByteNum,
		DBEntrySize:     DBEntrySize,
		DBSize:          DBSize,
		BatchSize:       BatchSize,
		PartitionNum:    PartitionNum,
		PartitionSize:   PartitionSize,
		ThreadNum:       DefaultThreadNum,
		FailureProbLog2: FailureProbLog2,
	}

	if DownloadIfCheaper {
		hintBytes := BatchHintStorageBytes(DBSize, DBEntryByteNum, BatchSize, FailureProbLog2)
		dbBytes := float64(DBSize) * float64(DBEntryByteNum)
		if hintBytes >= dbBytes {
			fmt.Printf("BatchPIR: hints would take %.2f MB but the DB is only %.2f MB, the client downloads the DB instead\n",
				hintBytes/1024/1024, dbBytes/1024/1024)
			if len(rawDB) < int(DBSize*DBEntrySize) { // small DB: pad a copy
				rawDB = append(append(make([]uint64, 0, DBSize*DBEntrySize), rawDB...),
					make([]uint64, int(DBSize*DBEntrySize)-len(rawDB))...)
			}
			return &SimpleBatchPianoPIR{config: config, download: true, rawDB: rawDB}
		}
	}

	subPIR := make([]*PianoPIR, PartitionNum)

	for i := uint64(0); i < PartitionNum; i++ {
		start := min(i*PartitionSize, DBSize)
		end := min((i+1)*PartitionSize, DBSize)
		partEnd := min(end*DBEntrySize, uint64(len(rawDB)))
		part := rawDB[min(start*DBEntrySize, partEnd):partEnd]
		if end-start < PartitionSize || partEnd < end*DBEntrySize {
			padded := make([]uint64, PartitionSize*DBEntrySize) // copy, cached Cuckoo tables are read-only mmaps
			copy(padded, part)
			part = padded
		}
		subPIR[i] = NewPianoPIR(PartitionSize, DBEntryByteNum, part, FailureProbLog2)
	}

	return &SimpleBatchPianoPIR{
		config:                 config,
		subPIR:                 subPIR,
		FinishedBatchNum:       0,
		QueriesMadeInPartition: 0,
	}
}

// SetConstantWork makes the amount of online work depend only on the public
// request count. Cache hits are replaced with dummy server work by each sub-PIR.
func (p *SimpleBatchPianoPIR) SetConstantWork(enabled bool) {
	p.config.ConstantWork = enabled
	for _, subPIR := range p.subPIR {
		subPIR.SetConstantWork(enabled)
	}
}

// SetThreadNum controls partition-level parallelism for preprocessing and
// online queries. Each partition owns an independent PianoPIR instance, so
// partitions can run concurrently while queries inside one partition remain
// ordered. Values larger than the partition count are clamped.
func (p *SimpleBatchPianoPIR) SetThreadNum(threadNum int) {
	if threadNum < 1 {
		threadNum = 1
	}
	if uint64(threadNum) > p.config.PartitionNum {
		threadNum = int(p.config.PartitionNum)
	}
	p.config.ThreadNum = uint64(threadNum)
}

func (p *SimpleBatchPianoPIR) PrintInfo() {
	if p.download {
		fmt.Printf("BatchPIR (downloaded): DBSize: %v, DBEntryByteNum: %v, client storage = %v MB\n",
			p.config.DBSize, p.config.DBEntryByteNum, p.LocalStorageSize()/1024/1024)
		return
	}
	fmt.Printf("-----------BatchPIR config --------\n")
	DBSizeInBytes := float64(p.config.DBSize) * float64(p.config.DBEntryByteNum)
	fmt.Printf("DB size in MB = %v\n", DBSizeInBytes/1024/1024)
	fmt.Printf("DBSize: %v, DBEntryByteNum: %v, BatchSize: %v, PartitionNum: %v, PartitionSize: %v, ThreadNum: %v, FailureProbLog2: %v, ConstantWork: %v\n", p.config.DBSize, p.config.DBEntryByteNum, p.config.BatchSize, p.config.PartitionNum, p.config.PartitionSize, p.config.ThreadNum, p.config.FailureProbLog2, p.config.ConstantWork)
	maxQuery := p.subPIR[0].client.MaxQueryNum / QueryPerPartition
	fmt.Printf("max query num = %v\n", maxQuery)
	fmt.Printf("max query per chunk = %v\n", p.subPIR[0].client.maxQueryPerChunk)
	fmt.Printf("total storage = %v MB\n", p.LocalStorageSize()/1024/1024)
	fmt.Printf("comm cost per batch = %v KB\n", p.CommCostPerBatchOnline()/1024)
	fmt.Printf("amortized preprocessing comm cost = %v KB\n", float64(DBSizeInBytes)/float64(maxQuery)/1024)
	fmt.Printf("total amortized comm cost = %v KB\n", float64(DBSizeInBytes)/float64(maxQuery)/1024+float64(p.CommCostPerBatchOnline())/1024)
	fmt.Printf("-----------------------------\n")
}

func (p *SimpleBatchPianoPIR) RecordStats() {
	if p.download {
		p.localStorage = uint64(p.LocalStorageSize())
		return
	}
	p.localStorage = uint64(p.LocalStorageSize())                 // bytes
	p.commCostPerBatchOnline = uint64(p.CommCostPerBatchOnline()) // bytes
	p.SupportBatchNum = p.subPIR[0].client.MaxQueryNum / QueryPerPartition
	DBSizeInBytes := float64(p.config.DBSize) * float64(p.config.DBEntryByteNum)
	p.commCostPerBatchOffline = uint64(float64(DBSizeInBytes) / float64(p.SupportBatchNum)) // bytes
}

func (p *SimpleBatchPianoPIR) Preprocessing() {
	p.PrintInfo()
	if p.download {
		// the offline phase is the download itself, which Piano's preprocessing
		// (streaming the whole DB) already costs; there is nothing to compute
		p.FinishedBatchNum, p.QueriesMadeInPartition = 0, 0
		if p.TotalPreProcs == 0 {
			p.TotalPreProcs = 1
		}
		p.RecordStats()
		return
	}

	// now we do the preprocessing
	// we need to clock the time

	// we now use p.config.ThreadNum threads to do the preprocessing
	p.FinishedBatchNum = 0
	p.QueriesMadeInPartition = 0
	startTime := time.Now()

	var wg sync.WaitGroup
	wg.Add(int(p.config.ThreadNum))

	perThreadPartitionNum := (p.config.PartitionNum + p.config.ThreadNum - 1) / p.config.ThreadNum

	for tid := uint64(0); tid < p.config.ThreadNum; tid++ {
		go func(tid uint64) {
			start := tid * perThreadPartitionNum
			end := min((tid+1)*perThreadPartitionNum, p.config.PartitionNum)
			//log.Printf("Thread %v preprocessing partitions [%v, %v)\n", tid, start, end)
			for i := start; i < end; i++ {
				p.subPIR[i].Preprocessing()
			}
			//log.Print("Thread ", tid, " finished preprocessing")
			wg.Done()
		}(tid)
	}

	wg.Wait()

	endTime := time.Now()
	prepTime := endTime.Sub(startTime).Seconds()
	if p.TotalPreProcs >= 1 {
		p.MaintainenceTime += prepTime
		p.TotalPreProcs += 1
	} else {
		p.InitialPreprocessingTime = prepTime
		p.TotalPreProcs = 1
	}

	log.Printf("Preprocessing time = %v\n", endTime.Sub(startTime))

	p.RecordStats()
}

func (p *SimpleBatchPianoPIR) DummyPreprocessing() {
	p.PrintInfo()
	if p.download {
		p.RecordStats()
		return
	}
	// directly initialize all subPIR
	for i := uint64(0); i < p.config.PartitionNum; i++ {
		p.subPIR[i].DummyPreprocessing()
	}

	log.Printf("Skipping Prep")
	p.RecordStats()
}

/// TODO: optimize for multiple batch

// Replace the Query function in batch-pir.go

// func (p *SimpleBatchPianoPIR) Query(idx []uint32) ([][]byte, error) {
//     // 1. USE BATCH SIZE, NOT PARTITION NUM
//     // The server is configured for BatchSize (1024), so we must iterate by that.
//     batchSize := int(p.config.BatchSize)
//     totalQueries := len(idx)

//     // 2. CEILING DIVISION FORMULA
//     // (N + B - 1) / B ensures we calculate enough batches to cover the remainder.
//     numBatches := (totalQueries + batchSize - 1) / batchSize

//     // Pre-allocate results slice
//     allResults := make([][]byte, 0, totalQueries)

//     for i := 0; i < numBatches; i++ {
//         start := i * batchSize
//         end := start + batchSize

//         // Clamp the end index for the last partial batch
//         if end > totalQueries {
//             end = totalQueries
//         }

//         // Get the actual user queries for this iteration
//         currentBatch := idx[start:end]
//         realBatchLen := len(currentBatch)

//         // 3. EXPLICIT PADDING (CRITICAL FIX)
//         // If the batch is smaller than 1024, we must pad it.
//         // Sending partial batches causes checksum mismatches.
//         // Padding with 0s causes hash bucket overflows.
//         if realBatchLen < batchSize {
//             // Allocate a full-sized batch buffer
//             paddedBatch := make([]uint32, batchSize)
//             copy(paddedBatch, currentBatch)

//             // Fill the empty slots with RANDOM indices.
//             // We use random valid indices to distribute load evenly across hash buckets.
//             // Using 0 here would crash the server's Cuckoo hashing.
//             for k := realBatchLen; k < batchSize; k++ {
//                 // Use a random index within the database range
//                 paddedBatch[k] = uint32(rand.Intn(int(p.config.DBSize)))
//             }

//             // Swap the slice to use the padded version
//             currentBatch = paddedBatch
//         }

//         // 4. PERFORM QUERY
//         // This now always sends exactly 1024 items.
//         batchResults, err := p.client.BatchQuery(currentBatch)
//         if err != nil {
//             return nil, fmt.Errorf("batch query failed at batch %d: %v", i, err)
//         }

//         // 5. STRIP PADDING
//         // We only append the results corresponding to the real user queries.
//         if len(batchResults) >= realBatchLen {
//             allResults = append(allResults, batchResults[:realBatchLen]...)
//         } else {
//             // Safety fallback
//             allResults = append(allResults, batchResults...)
//         }
//     }

//     return allResults, nil
// }

func (p *SimpleBatchPianoPIR) Query(idx []uint64) ([][]uint64, error) {
	if len(idx) == 0 {
		return [][]uint64{}, nil
	}
	if p.download {
		return p.localQuery(idx)
	}

	// Arrange the requests first, then use the maximum occupancy. The previous
	// average occupancy silently discarded requests in overloaded partitions.
	partitionQueries := make([][]uint64, p.config.PartitionNum)
	for i := 0; i < len(idx); i++ {
		partitionIdx := idx[i] / p.config.PartitionSize
		if idx[i] >= p.config.DBSize {
			return nil, fmt.Errorf("BatchPIR index %d is outside database of size %d", idx[i], p.config.DBSize)
		}
		partitionQueries[partitionIdx] = append(partitionQueries[partitionIdx], idx[i])
	}
	observedMaxOccupancy := 0
	for i := uint64(0); i < p.config.PartitionNum; i++ {
		if len(partitionQueries[i]) > observedMaxOccupancy {
			observedMaxOccupancy = len(partitionQueries[i])
		}
	}
	queryNumToMake := observedMaxOccupancy
	if p.config.ConstantWork {
		queryNumToMake = constantWorkPartitionQueries(
			len(idx), p.config.PartitionNum, constantWorkOccupancyFailureBits,
		)
		if observedMaxOccupancy > queryNumToMake {
			return nil, fmt.Errorf(
				"BatchPIR constant-work occupancy overflow: observed %d queries in one partition, public bound is %d",
				observedMaxOccupancy, queryNumToMake,
			)
		}
	}
	if uint64(queryNumToMake) >= p.subPIR[0].client.MaxQueryNum-2 {
		return nil, fmt.Errorf(
			"BatchPIR request needs %d queries in one partition; maximum is %d",
			queryNumToMake, p.subPIR[0].client.MaxQueryNum-2,
		)
	}
	if p.QueriesMadeInPartition+uint64(queryNumToMake) >= p.subPIR[0].client.MaxQueryNum-2 {
		fmt.Printf("Redo preprocessing before query (%d queries used, %d requested)\n",
			p.QueriesMadeInPartition, queryNumToMake)
		p.Preprocessing()
	}

	//fmt.Println("partitionQueries: ", partitionQueries)

	// Each partition owns an independent Piano client/server pair. Run
	// partitions concurrently, but preserve request order within a partition
	// because Piano updates state after every real query.
	p.roundsMade++
	partitionResponses := make([]map[uint64][]uint64, p.config.PartitionNum)
	jobs := make(chan uint64)
	var workers sync.WaitGroup
	var firstErr error
	var errOnce sync.Once
	workers.Add(int(p.config.ThreadNum))
	for worker := uint64(0); worker < p.config.ThreadNum; worker++ {
		go func() {
			defer workers.Done()
			for partition := range jobs {
				queries := partitionQueries[partition]
				for len(queries) < queryNumToMake {
					queries = append(queries, DefaultValue)
				}
				localResponses := make(map[uint64][]uint64)
				for queryIndex := 0; queryIndex < queryNumToMake; queryIndex++ {
					row := queries[queryIndex]
					if row == DefaultValue {
						if _, err := p.subPIR[partition].Query(0, false); err != nil {
							errOnce.Do(func() {
								firstErr = fmt.Errorf("partition %d dummy query %d failed: %w", partition, queryIndex, err)
							})
							break
						}
						continue
					}
					response, err := p.subPIR[partition].Query(row-partition*p.config.PartitionSize, true)
					if err != nil {
						// hint miss / chunk saturation: refresh only this partition and retry once
						fmt.Printf("partition %d query failed (%v); redoing its preprocessing\n", partition, err)
						p.subPIR[partition].Preprocessing()
						response, err = p.subPIR[partition].Query(row-partition*p.config.PartitionSize, true)
					}
					if err != nil {
						errOnce.Do(func() {
							firstErr = fmt.Errorf("partition %d query %d failed: %w", partition, queryIndex, err)
						})
						break
					}
					localResponses[row] = response
				}
				partitionResponses[partition] = localResponses
			}
		}()
	}
	for partition := uint64(0); partition < p.config.PartitionNum; partition++ {
		jobs <- partition
	}
	close(jobs)
	workers.Wait()
	if firstErr != nil {
		return nil, firstErr
	}

	responses := make(map[uint64][]uint64, len(idx))
	for _, localResponses := range partitionResponses {
		for row, response := range localResponses {
			responses[row] = response
		}
	}

	// print all the indices in responses
	//for k, v := range responses {
	//	fmt.Printf("responses[%v] = %v\n", k, v[0])
	//	}

	// now we output the responses in the order of the queries
	ret := make([][]uint64, len(idx))
	for i := 0; i < len(idx); i++ {
		if response, ok := responses[idx[i]]; ok {
			ret[i] = response
		} else {
			return nil, fmt.Errorf("BatchPIR produced no response for database row %d", idx[i])
		}
	}

	p.FinishedBatchNum += (uint64(queryNumToMake) + QueryPerPartition - 1) / QueryPerPartition
	p.QueriesMadeInPartition += uint64(queryNumToMake)

	return ret, nil
}

// localQuery answers lookups from the downloaded DB, with no server round.
func (p *SimpleBatchPianoPIR) localQuery(idx []uint64) ([][]uint64, error) {
	w := p.config.DBEntrySize
	ret := make([][]uint64, len(idx))
	for i, row := range idx {
		if row >= p.config.DBSize {
			return nil, fmt.Errorf("BatchPIR index %d is outside database of size %d", row, p.config.DBSize)
		}
		ret[i] = append([]uint64(nil), p.rawDB[row*w:(row+1)*w]...)
	}
	return ret, nil
}

// Downloaded reports whether the client holds the plain DB instead of hints.
func (p *SimpleBatchPianoPIR) Downloaded() bool { return p.download }

func (p *SimpleBatchPianoPIR) LocalStorageSize() float64 {
	if p.download {
		return float64(p.config.DBSize) * float64(p.config.DBEntryByteNum)
	}
	ret := float64(0)
	for i := uint64(0); i < p.config.PartitionNum; i++ {
		ret += p.subPIR[i].LocalStorageSize()
	}
	return ret
}

func (p *SimpleBatchPianoPIR) CommCostPerBatchOnline() uint64 {
	ret := float64(0)
	for i := range p.subPIR { // none when the DB was downloaded
		ret += p.subPIR[i].CommCostPerQuery() * float64(QueryPerPartition)
	}
	return uint64(ret)
}

func (p *SimpleBatchPianoPIR) CommCostPerBatchOffline() uint64 {
	return p.commCostPerBatchOffline
}

func (p *SimpleBatchPianoPIR) Config() *SimpleBatchPianoPIRConfig {
	return p.config
}

// SplitByPartitionLimit splits idx into consecutive sub-batches, each of which Query accepts: at most
// MaxQueryNum-3 lookups in any one partition (Query refuses a request that needs MaxQueryNum-2 or more), and, with
// constant work on, few enough lookups that the public per-partition bound for the sub-batch stays under that limit.
// A request that already fits comes back as the one batch idx. Order is kept within and across sub-batches.
func (p *SimpleBatchPianoPIR) SplitByPartitionLimit(idx []uint64) [][]uint64 {
	if p.download || len(p.subPIR) == 0 || len(idx) == 0 {
		return [][]uint64{idx}
	}
	limit := int(p.subPIR[0].client.MaxQueryNum) - 3
	if limit < 1 {
		limit = 1
	}
	fits := func(b []uint64) bool {
		occ := make(map[uint64]int)
		for _, i := range b {
			occ[i/p.config.PartitionSize]++
			if occ[i/p.config.PartitionSize] > limit {
				return false
			}
		}
		if p.config.ConstantWork &&
			constantWorkPartitionQueries(len(b), p.config.PartitionNum, constantWorkOccupancyFailureBits) > limit {
			return false
		}
		return true
	}
	if fits(idx) {
		return [][]uint64{idx}
	}
	var out [][]uint64
	cur := make([]uint64, 0, len(idx))
	for _, i := range idx {
		if len(cur) > 0 && !fits(append(cur, i)) {
			out = append(out, cur)
			cur = make([]uint64, 0, len(idx))
		}
		cur = append(cur, i)
	}
	return append(out, cur)
}

// GetMetrics returns this DB's metrics under bins-style key names.
// Values are float64 so stages can sum them key-by-key.
func (p *SimpleBatchPianoPIR) GetMetrics() map[string]float64 {
	totalByte := uint64(0)
	for _, sub := range p.subPIR {
		totalByte += sub.server.RetrievalCount
	}
	return map[string]float64{
		"TotalByteSent":      float64(totalByte),
		"TotalWANTime":       netTime(totalByte, p.roundsMade, 400, 0.05),
		"TotalLANTime":       netTime(totalByte, p.roundsMade, 1000, 0.005),
		"Rounds":             float64(p.roundsMade),
		"PreprocessingTime":  p.InitialPreprocessingTime,
		"MaintainenceTime":   p.MaintainenceTime,
		"TotalPreProc":       float64(p.TotalPreProcs),
		"ClientStorageMB":    p.LocalStorageSize() / 1024 / 1024,
		"DBSizeInBytesMB":    float64(p.config.DBSize) * float64(p.config.DBEntryByteNum) / 1024 / 1024,
		"CommCostPerBatchKB": float64(p.CommCostPerBatchOnline()) / 1024,
		"DownloadedDBs":      boolToFloat(p.download),
	}
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func netTime(bytes, rounds uint64, mbps, rttSec float64) float64 {
	return float64(bytes)*8/(mbps*1e6) + float64(rounds)*rttSec
}
