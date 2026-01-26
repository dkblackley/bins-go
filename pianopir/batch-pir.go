package pianopir

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

//"encoding/binary"

const (
	RealQueryPerPartition = 2
	QueryPerPartition     = 2
	DefaultValue          = 0xdeadbeef
	ThreadNum             = 1
)

type SimpleBatchPianoPIRConfig struct {
	DBEntryByteNum  uint64 // the number of bytes in a DB entry
	MaxDBEntrySize  uint64 // the number of uint64 in a DB entry
	DBSize          uint64
	BatchSize       uint64
	PartitionNum    uint64
	PartitionSize   uint64
	ThreadNum       uint64
	FailureProbLog2 uint64
	BatchNumNeeded  uint64
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

	FinishedBatchNum        uint64
	QueriesMadeInPartition  uint64
	SupportBatchNum         uint64
	localStorage            uint64  // bytes
	preprocessingTime       float64 // seconds
	commCostPerBatchOnline  uint64  // bytes
	commCostPerBatchOffline uint64  // bytes

	permutation []uint64 // <--- ADD THIS LINE
}

func NewSimpleBatchPianoPIR(DBSize uint64, MaxDBEntrySize uint64, DBEntryByteNum uint64, BatchSize uint64,
	rawDB [][]uint64, FailureProbLog2 uint64, BatchNumNeeded uint64) *SimpleBatchPianoPIR {
	//DBEntrySize := DBEntryByteNum / 8
	//if len(rawDB) != int(DBSize*DBEntrySize) {
	//	log.Fatalf("BatchPIR: len(rawDB) = %v; want %v", len(rawDB), DBSize*DBEntrySize)
	//}

	// create the sub PIR classes
	PartitionNum := BatchSize / RealQueryPerPartition
	//PartitionSize := DBSize / PartitionNum and round up
	PartitionSize := (DBSize + PartitionNum - 1) / PartitionNum

	config := &SimpleBatchPianoPIRConfig{
		DBEntryByteNum:  DBEntryByteNum,
		MaxDBEntrySize:  MaxDBEntrySize,
		DBSize:          DBSize,
		BatchSize:       BatchSize,
		PartitionNum:    PartitionNum,
		PartitionSize:   PartitionSize,
		ThreadNum:       ThreadNum,
		FailureProbLog2: FailureProbLog2,
		BatchNumNeeded:  BatchNumNeeded,
	}

	// --- START INSERTION: Zero-Copy Shuffle Logic ---
	logrus.Infof("Shuffling DB pointers to break spatial locality (Zero-Copy)...")
	seed := int64(12345)
	rng := rand.New(rand.NewSource(seed))

	// 1. Create permutation map
	permInts := rng.Perm(int(DBSize))
	permutation := make([]uint64, DBSize)

	// 2. Create shuffled VIEW of the DB (copies slice headers only, not data)
	shuffledDB := make([][]uint64, DBSize)

	for physicalIdx, logicalIdxInt := range permInts {
		logicalID := uint64(logicalIdxInt)

		// Store map for Query() function
		permutation[logicalID] = uint64(physicalIdx)

		// Point shuffled row to the EXISTING data row (no memory copy)
		shuffledDB[physicalIdx] = rawDB[logicalID]
	}
	// --- END INSERTION ---

	subPIR := make([]*PianoPIR, PartitionNum)

	for i := uint64(0); i < PartitionNum; i++ {
		start := i * PartitionSize
		end := min((i+1)*PartitionSize, DBSize)
		// print start and end
		//fmt.Printf("start: %v, end: %v\n", start, end)

		// assert that the rawDB is of the correct size
		if uint64(len(rawDB)) != DBSize {
			log.Fatalf("Piano PIR len(rawDB) = %v; want %v", len(rawDB), DBSize)
		}

		targetChunkSize := uint64(2 * math.Sqrt(float64(PartitionSize)))
		ChunkSize := uint64(1)
		for ChunkSize < targetChunkSize {
			ChunkSize *= 2
		}
		SetSize := uint64(math.Ceil(float64(PartitionSize) / float64(ChunkSize)))
		// round up to the next mulitple of 4
		SetSize = (SetSize + 3) / 4 * 4

		subConfig := PianoPIRConfig{
			DBEntryByteNum:  DBEntryByteNum,
			MaxDBEntrySize:  MaxDBEntrySize,
			DBSize:          PartitionSize,
			ChunkSize:       ChunkSize,
			SetSize:         SetSize,
			ThreadNum:       ThreadNum,
			FailureProbLog2: FailureProbLog2,
		}

		// subPIR[i] = NewPianoPIR(&subConfig, rawDB[start:end])
		subPIR[i] = NewPianoPIR(&subConfig, shuffledDB[start:end])
	}

	return &SimpleBatchPianoPIR{
		config:                 config,
		subPIR:                 subPIR,
		FinishedBatchNum:       0,
		QueriesMadeInPartition: 0,
		permutation:            permutation,
	}
}

func (p *SimpleBatchPianoPIR) PrintInfo() map[string]string {

	metadata := make(map[string]string)

	fmt.Printf("-----------BatchPIR config --------\n")

	DBSizeInBytes := 0
	totalUint64s := 0

	for i := uint64(0); i < p.config.PartitionNum; i++ {

		rawDB := p.subPIR[i].server.rawDB
		for _, v := range rawDB {
			DBSizeInBytes += len(v) * 8
		}
		totalUint64s += int(p.subPIR[i].server.RetrievalCount)
	}

	fmt.Printf("DB size in MB = %v\n", DBSizeInBytes/1024/1024)
	fmt.Printf("DBSize: %v, DBEntryByteNum: %v, BatchSize: %v, PartitionNum: %v, PartitionSize: %v, ThreadNum: %v,"+
		" FailureProbLog2: %v\n", DBSizeInBytes/8, p.config.DBEntryByteNum, p.config.BatchSize, p.config.PartitionNum,
		p.config.PartitionSize, p.config.ThreadNum, p.config.FailureProbLog2)
	maxQuery := p.subPIR[0].client.MaxQueryNum / QueryPerPartition
	fmt.Printf("max query num = %v\n", maxQuery)
	fmt.Printf("max query per chunk = %v\n", p.subPIR[0].client.maxQueryPerChunk)
	fmt.Printf("total storage = %v MB\n", p.LocalStorageSize()/1024/1024)
	fmt.Printf("comm cost per batch = %v KB\n", p.CommCostPerBatchOnline()/1024)
	fmt.Printf("amortized preprocessing comm cost = %v KB\n", float64(DBSizeInBytes)/float64(maxQuery)/1024)
	fmt.Printf("total amortized comm cost = %v KB\n", float64(DBSizeInBytes)/float64(maxQuery)/1024+float64(p.CommCostPerBatchOnline())/1024)
	fmt.Printf("total uint64s sent = %v\n", totalUint64s)
	fmt.Printf("-----------------------------\n")

	PIR := p.subPIR[0]
	fmt.Printf("-----------PIR config --------\n")
	fmt.Printf("DBSize: %v, DBEntryByteNum: %v, DBEntrySize: %v, ChunkSize: %v, SetSize: %v, ThreadNum: %v, FailureProbLog2: %v\n", PIR.config.DBSize, PIR.config.DBEntryByteNum, PIR.config.MaxDBEntrySize, PIR.config.ChunkSize, PIR.config.SetSize, PIR.config.ThreadNum, PIR.config.FailureProbLog2)
	fmt.Printf("-----------------------------\n")

	metadata["TotalUint64Sent"] = fmt.Sprintf("%d", totalUint64s)
	metadata["DBSizeInBytesMB"] = fmt.Sprintf("%v", DBSizeInBytes/1024/1024)
	metadata["FailureProbLog2"] = fmt.Sprintf("%v", PIR.config.FailureProbLog2)
	metadata["ClientStorageMB"] = fmt.Sprintf("%v", p.LocalStorageSize()/1024/1024)
	metadata["CommCostPerBatchKB"] = fmt.Sprintf("%v", p.CommCostPerBatchOnline()/1024)

	return metadata
}

func (p *SimpleBatchPianoPIR) RecordStats(prepTime float64) {
	p.preprocessingTime = prepTime
	p.localStorage = uint64(p.LocalStorageSize())                 // bytes
	p.commCostPerBatchOnline = uint64(p.CommCostPerBatchOnline()) // bytes
	p.SupportBatchNum = p.subPIR[0].client.MaxQueryNum / QueryPerPartition
	DBSizeInBytes := float64(p.config.DBSize) * float64(p.config.DBEntryByteNum)
	p.commCostPerBatchOffline = uint64(float64(DBSizeInBytes) / float64(p.SupportBatchNum)) // bytes
}

func (p *SimpleBatchPianoPIR) Preprocessing() time.Duration {
	// p.PrintInfo()

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
	logrus.Debugf("Preprocessing time = %v\n", endTime.Sub(startTime))

	p.RecordStats(prepTime)

	return endTime.Sub(startTime)
}

func (p *SimpleBatchPianoPIR) DummyPreprocessing() {
	p.PrintInfo()
	// directly initialize all subPIR
	for i := uint64(0); i < p.config.PartitionNum; i++ {
		p.subPIR[i].DummyPreprocessing()
	}

	log.Printf("Skipping Prep")
	p.RecordStats(0)
}

/// TODO: optimize for multiple batch

func (p *SimpleBatchPianoPIR) Query(idx []uint64) ([][]uint64, error) {

	// --- START INSERTION: Translate Logical -> Physical IDs ---
	shuffledIdx := make([]uint64, len(idx))
	for i, logicalID := range idx {
		if logicalID < uint64(len(p.permutation)) {
			shuffledIdx[i] = p.permutation[logicalID]
		} else {
			shuffledIdx[i] = 0 // Safety fallback
		}
	}
	idx = shuffledIdx // Replace input with shuffled indices
	// --- END INSERTION ---

	// first identify in average how many queries in each partition we need to make

	// Get average num ber partition and ALWAYS ROUND UP!
	// queryNumToMake := (len(idx) + int(p.config.PartitionNum) - 1) / int(p.config.PartitionNum)

	// first arrange the queries into the partitions
	partitionQueries := make([][]uint64, p.config.PartitionNum)
	for i := 0; i < len(idx); i++ {
		partitionIdx := idx[i] / p.config.PartitionSize
		partitionQueries[partitionIdx] = append(partitionQueries[partitionIdx], idx[i])
	}

	maxLen := 0
	for i := range partitionQueries {
		if len(partitionQueries[i]) > maxLen {
			maxLen = len(partitionQueries[i])
		}
	}
	// Get maxium queries we can make
	queryNumToMake := maxLen

	//fmt.Println("partitionQueries: ", partitionQueries)

	// we make a map from index to their responses
	responses := make(map[uint64][]uint64)

	// --- DEBUG START: Analyze Batch Distribution ---
	for i := uint64(0); i < p.config.PartitionNum; i++ {
		if len(partitionQueries[i]) == 0 {
			continue
		}

		// Access the subPIR's config to get ChunkSize
		chunkSize := p.subPIR[i].Config().ChunkSize
		limit := p.subPIR[i].client.maxQueryPerChunk

		// Count how many queries in THIS BATCH map to specific chunks
		chunkCounts := make(map[uint64]int)
		for _, globIdx := range partitionQueries[i] {
			if globIdx == DefaultValue {
				continue
			}

			// Convert global index to partition-local index
			localIdx := globIdx - i*p.config.PartitionSize
			chunkID := localIdx / chunkSize
			chunkCounts[chunkID]++
		}

		// Report any hot chunks
		for cID, count := range chunkCounts {
			// Warning if a SINGLE batch uses more than 50% of the total lifetime budget for a chunk
			if uint64(count) > limit/2 {
				logrus.Warnf("⚠️ HOT CHUNK in Batch! Partition %d, Chunk %d", i, cID)
				logrus.Warnf("   Requests in this batch: %d", count)
				logrus.Warnf("   Max allowed per chunk (lifetime): %d", limit)

				if uint64(count) >= limit {
					logrus.Errorf("🚨 CRITICAL: Batch request count (%d) exceeds lifetime limit (%d)! This query will strictly fail.", count, limit)
				}
			}
		}
	}
	// --- DEBUG END ---

	for i := uint64(0); i < p.config.PartitionNum; i++ {
		//start := i * p.config.PartitionSize
		//end := min((i+1)*p.config.PartitionSize, p.config.DBSize)

		// case 1: if there are not enough queries, just pad with random indices in the partition
		if len(partitionQueries[i]) < queryNumToMake {
			for j := len(partitionQueries[i]); j < queryNumToMake; j++ {
				partitionQueries[i] = append(partitionQueries[i], DefaultValue)
			}
		}

		// now we make queryNumToMake queries to the sub PIR
		for j := uint64(0); j < uint64(queryNumToMake); j++ {
			if partitionQueries[i][j] == DefaultValue {
				_, _ = p.subPIR[i].Query(0, false) // just make a dummy query
			} else {
				query, err := p.subPIR[i].Query(partitionQueries[i][j]-i*p.config.PartitionSize, true)
				if err != nil {
					logrus.Errorf("Not enough hints? SupportBatchnum %d, finBatchNum, %d, ", p.SupportBatchNum, p.FinishedBatchNum)

					logrus.Errorf("the queries to this sub pir is: %v, the offset is %v\n", partitionQueries[i], partitionQueries[i][j]-i*p.config.PartitionSize)
					logrus.Tracef("All the queries are %v\n", partitionQueries)
					logrus.Errorf("SimpleBatchPianoPIR.Query: subPIR[%v].Query(%v) failed: %v\n", i, partitionQueries[i][j], err)

					fmt.Printf("Redo preprocessing. Made %v batches (%v queries in a partition), redo the preprocessing\n", p.FinishedBatchNum, p.QueriesMadeInPartition)
					p.Preprocessing()

					query, err = p.subPIR[i].Query(partitionQueries[i][j]-i*p.config.PartitionSize, true)
					if err != nil {
						logrus.Errorf("Still did not work.... %v", err)
						return nil, err
					}
				}
				responses[partitionQueries[i][j]] = query
			}
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
			// otherwise just make a zero response
			ret[i] = make([]uint64, 1)
			for j := uint64(0); j < 1; j++ {
				ret[i][j] = 0
			}
		}
	}

	// now test if the subPIR has reached the max query num, redo the preprocessing
	// -2 means we want to do the preprocessing before the last query
	if p.QueriesMadeInPartition >= p.subPIR[0].client.MaxQueryNum-2 { //TODO: Remove this once done testing
		fmt.Printf("Redo preprocessing. Made %v batches (%v queries in a partition), redo the preprocessing\n", p.FinishedBatchNum, p.QueriesMadeInPartition)
		p.Preprocessing()
	} else {
		p.FinishedBatchNum += uint64(len(idx) / int(p.config.BatchSize))
		p.QueriesMadeInPartition += uint64(queryNumToMake)
	}

	return ret, nil
}

func (p *SimpleBatchPianoPIR) LocalStorageSize() float64 {
	ret := float64(0)
	for i := uint64(0); i < p.config.PartitionNum; i++ {
		ret += p.subPIR[i].LocalStorageSize()
	}
	return ret
}

func (p *SimpleBatchPianoPIR) CommCostPerBatchOnline() uint64 {
	ret := float64(0)
	for i := uint64(0); i < p.config.PartitionNum; i++ {
		ret += p.subPIR[i].CommCostPerQuery() * float64(QueryPerPartition)
	}
	return uint64(ret)
}

func (p *SimpleBatchPianoPIR) CommCostPerBatchOffline() uint64 {
	return p.commCostPerBatchOffline
}

func (p *SimpleBatchPianoPIR) PreprocessingTime() float64 {
	return p.preprocessingTime
}

func (p *SimpleBatchPianoPIR) Config() *SimpleBatchPianoPIRConfig {
	return p.config
}
