package pianopir

import (
	"math/rand"
	"testing"
	"time"
)

func TestPIRBasic(t *testing.T) {
	// Arrange
	// Set up any necessary data or arguments

	DBSize := uint64(18750)
	DBEntrySize := uint64(4)
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))

	rawDB := make([]uint64, DBEntrySize*DBSize)
	for i := uint64(0); i < DBSize; i++ {
		for j := uint64(0); j < DBEntrySize; j++ {
			rawDB[i*DBEntrySize+j] = rng.Uint64()
		}
	}

	PIR := NewPianoPIR(DBSize, DBEntrySize*8, rawDB, 40)

	// print the config of the PIR
	config := PIR.Config()
	t.Logf("PIR config: %v", config)
	t.Logf("hint num: %v", PIR.client.primaryHintNum)
	t.Logf("max query num: %v", PIR.client.MaxQueryNum)

	maxQueryNum := PIR.client.MaxQueryNum

	PIR.Preprocessing()

	// make 1000 random queries
	for i := 0; i < int(maxQueryNum); i++ {
		idx := rand.Uint64() % DBSize
		query, err := PIR.Query(idx, true)
		if err != nil {
			t.Errorf("PIR.Query(%v) failed: %v", idx, err)
		}

		for j := uint64(0); j < DBEntrySize; j++ {
			if query[j] != rawDB[idx*DBEntrySize+j] {
				t.Errorf("query[%v] = %v; want %v", idx, query[j], rawDB[idx*DBEntrySize+j])
			}
		}

		if i == 0 {
			t.Logf("response = %v", query)
		}

		// just output a message to show the progress
		//t.Logf("PIR.Query(%v) passed", idx)
	}
}

func TestBatchPIRBasic(t *testing.T) {
	// Arrange
	// Set up any necessary data or arguments

	DBSize := uint64(1000000)
	DBEntrySize := uint64(16)
	BatchSize := uint64(32)

	// a seed that's depending on the current time
	//seed := time.Now().UnixNano()
	//rng := rand.New(rand.NewSource(seed))

	rawDB := make([]uint64, DBEntrySize*DBSize)
	for i := uint64(0); i < DBSize; i++ {
		for j := uint64(0); j < DBEntrySize; j++ {
			rawDB[i*DBEntrySize+j] = uint64(i) //rng.Uint64()
		}
	}

	PIR := NewSimpleBatchPianoPIR(DBSize, DBEntrySize*8, BatchSize, rawDB, 20)

	// print the config of the PIR
	config := PIR.Config()
	t.Logf("Batch PIR config: %v", config)

	PIR.Preprocessing()

	// make a single batch query
	// for each partition, make PartitionQueryNum queries
	batchQuery := make([]uint64, 0, BatchSize)

	for i := uint64(0); i < config.PartitionNum; i++ {
		start := i * config.PartitionSize
		end := min((i+1)*config.PartitionSize, DBSize)

		for j := uint64(0); j < QueryPerPartition-1; j++ {
			offset := rand.Uint64() % (end - start)
			// append the query to the batch query
			batchQuery = append(batchQuery, start+offset)
		}
	}

	// now make a batch query
	// they should be all correct

	responses, err := PIR.Query(batchQuery)

	if err != nil {
		t.Errorf("PIR.Query(%v) failed: %v", batchQuery, err)
	}

	for i := 0; i < len(batchQuery); i++ {
		idx := batchQuery[i]
		query := responses[i]
		for j := uint64(0); j < DBEntrySize; j++ {
			if query[j] != rawDB[idx*DBEntrySize+j] {
				t.Errorf("query[%v] = %v; want %v", idx, query[j], rawDB[idx*DBEntrySize+j])
			}
		}
	}

	// we make a batch query with each partition having 4 queries

	batchQuery = make([]uint64, 4*config.PartitionNum)
	for i := 0; i < int(config.PartitionNum); i++ {
		start := i * int(config.PartitionSize)
		end := min((i+1)*int(config.PartitionSize), int(DBSize))

		for j := 0; j < 4; j++ {
			offset := int(rand.Uint64() % uint64(end-start))
			// append the query to the batch query
			batchQuery[i*4+j] = uint64(start + offset)
		}
	}

	responses, err = PIR.Query(batchQuery)

	if err != nil {
		t.Errorf("PIR.Query(%v) failed: %v", batchQuery, err)
	}

	for i := 0; i < len(batchQuery); i++ {
		idx := batchQuery[i]
		query := responses[i]
		for j := uint64(0); j < DBEntrySize; j++ {
			if query[j] != rawDB[idx*DBEntrySize+j] {
				t.Errorf("query[%v] = %v; want %v", idx, query[j], rawDB[idx*DBEntrySize+j])
			}
		}
	}

	//t.Logf("Batch PIR.Query(%v) passed", batchQuery)

	// Make another batch query concentrated in the first partition. Query now
	// expands to the maximum partition occupancy, so every response must be
	// returned instead of silently dropping all but QueryPerPartition entries.

	querySet := make(map[uint64]bool)
	batchQuery = make([]uint64, 0, BatchSize)
	for i := uint64(0); i < BatchSize; i++ {
		idx := rand.Uint64() % config.PartitionSize
		if _, ok := querySet[idx]; ok {
			// resample the index
			i--
			continue
		}
		querySet[idx] = true
		batchQuery = append(batchQuery, idx)
	}

	responses, err = PIR.Query(batchQuery)

	if err != nil {
		t.Errorf("PIR.Query(%v) failed: %v", batchQuery, err)
	}

	for i := uint64(0); i < BatchSize; i++ {
		idx := batchQuery[i]
		query := responses[i]

		for j := uint64(0); j < DBEntrySize; j++ {
			if query[j] != rawDB[idx*DBEntrySize+j] {
				t.Errorf("query[%v] = %v; want %v", idx, query[j], rawDB[idx*DBEntrySize+j])
			}
		}
	}
}

func TestBatchPIRSingleWordEntries(t *testing.T) {
	const (
		dbSize    = uint64(8192)
		batchSize = uint64(32)
	)
	rawDB := make([]uint64, dbSize)
	for i := range rawDB {
		rawDB[i] = uint64(i) + 1
	}

	pir := NewSimpleBatchPianoPIR(dbSize, 8, batchSize, rawDB, 20)
	pir.Preprocessing()
	indices := make([]uint64, pir.Config().PartitionNum)
	for i := range indices {
		indices[i] = uint64(i)*pir.Config().PartitionSize + uint64(i)
	}
	responses, err := pir.Query(indices)
	if err != nil {
		t.Fatalf("single-word BatchPIR query failed: %v", err)
	}
	for i, index := range indices {
		if len(responses[i]) != 1 || responses[i][0] != rawDB[index] {
			t.Fatalf("row %d: got %v, want [%#x]", index, responses[i], rawDB[index])
		}
	}
}

func TestConstantWorkPartitionQueries(t *testing.T) {
	tests := []struct {
		queries    int
		partitions uint64
		want       int
	}{
		{4, 64, 4},
		{512, 64, 27},
		{1600, 64, 55},
		{32, 16, 12},
	}
	for _, test := range tests {
		if got := constantWorkPartitionQueries(test.queries, test.partitions, constantWorkOccupancyFailureBits); got != test.want {
			t.Fatalf("constantWorkPartitionQueries(%d,%d)=%d; want %d", test.queries, test.partitions, got, test.want)
		}
	}
}

func TestBatchPIRConstantWorkRepeatedRows(t *testing.T) {
	const (
		dbSize    = uint64(8192)
		batchSize = uint64(32)
	)
	rawDB := make([]uint64, dbSize)
	for i := range rawDB {
		rawDB[i] = uint64(i) + 17
	}
	pir := NewSimpleBatchPianoPIR(dbSize, 8, batchSize, rawDB, 20)
	pir.SetThreadNum(4)
	pir.SetConstantWork(true)
	pir.Preprocessing()
	if got := pir.Config().ThreadNum; got != 4 {
		t.Fatalf("parallel BatchPIR uses %d threads; want 4", got)
	}
	indices := make([]uint64, pir.Config().PartitionNum)
	for i := range indices {
		indices[i] = uint64(i)*pir.Config().PartitionSize + uint64(i)
	}
	for repetition := 0; repetition < 2; repetition++ {
		serverCallsBefore := uint64(0)
		for _, subPIR := range pir.subPIR {
			serverCallsBefore += subPIR.server.privateQueryCount
		}
		responses, err := pir.Query(indices)
		if err != nil {
			t.Fatalf("constant-work repetition %d failed: %v", repetition, err)
		}
		for i, index := range indices {
			if len(responses[i]) != 1 || responses[i][0] != rawDB[index] {
				t.Fatalf("repetition %d row %d: got %v, want [%#x]", repetition, index, responses[i], rawDB[index])
			}
		}
		serverCallsAfter := uint64(0)
		for _, subPIR := range pir.subPIR {
			serverCallsAfter += subPIR.server.privateQueryCount
		}
		partitionQueries := constantWorkPartitionQueries(
			len(indices), pir.Config().PartitionNum, constantWorkOccupancyFailureBits,
		)
		wantCalls := uint64(partitionQueries) * pir.Config().PartitionNum
		if gotCalls := serverCallsAfter - serverCallsBefore; gotCalls != wantCalls {
			t.Fatalf("repetition %d made %d server queries; want fixed %d", repetition, gotCalls, wantCalls)
		}
	}
}

func TestBatchPIRPerf(t *testing.T) {
	// Arrange
	// Set up any necessary data or arguments

	DBSize := uint64(3201821)
	//DBSize := uint64(100000000)
	//DBSize := uint64(300000)
	DBEntrySize := uint64(112)
	BatchSize := uint64(32)

	// a seed that's depending on the current time
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))

	rawDB := make([]uint64, DBEntrySize*DBSize)
	for i := uint64(0); i < DBSize; i++ {
		for j := uint64(0); j < DBEntrySize; j++ {
			rawDB[i*DBEntrySize+j] = rng.Uint64()
		}
	}

	PIR := NewSimpleBatchPianoPIR(DBSize, DBEntrySize*8, BatchSize, rawDB, 8)

	// print the config of the PIR
	config := PIR.Config()
	t.Logf("Batch PIR config: %v\n", config)
	t.Logf("Batch PIR storage %v MB\n", PIR.LocalStorageSize()/1024/1024)
	t.Logf("Batch PIR max query num%v\n", PIR.subPIR[0].client.MaxQueryNum)
	t.Logf("Sub PIR config: %v\n", PIR.subPIR[0].Config())
	t.Logf("Sub PIR primary hint num: %v\n", PIR.subPIR[0].client.primaryHintNum)
	t.Logf("Sub PIR strorae %v MB\n", PIR.subPIR[0].LocalStorageSize()/1024/1024)
	PIR.subPIR[0].client.PrintStorageBreakdown()

	start := time.Now()
	PIR.Preprocessing()
	end := time.Now()
	t.Logf("Preprocessing time = %v\n", end.Sub(start))

	// now we make 1000 random batchQuery

	queryNum := 300

	start = time.Now()
	for i := 0; i < queryNum; i++ {
		batch := make([]uint64, 0, BatchSize)
		for j := 0; j < int(BatchSize); j++ {
			batch = append(batch, rng.Uint64()%DBSize)
		}
		response, err := PIR.Query(batch)
		if err != nil {
			t.Errorf("PIR.Query(%v) failed: %v", batch, err)
		}
		//we check the first response, either it's all zeros, or it's correct
		for j := uint64(0); j < DBEntrySize; j++ {
			if response[0][j] != 0 && response[0][j] != rawDB[batch[0]*DBEntrySize+j] {
				t.Errorf("response[0][%v] = %v; want %v", j, response[0][j], rawDB[batch[0]*DBEntrySize+j])
			}
		}
	}
	end = time.Now()
	t.Logf("Total query time = %v\n", end.Sub(start))
	t.Logf("Average query time per batch = %v\n", end.Sub(start)/time.Duration(queryNum))
	avgBatchTime := end.Sub(start) / time.Duration(queryNum)

	rtt := time.Duration(50) * time.Millisecond
	parallel := 2
	step := 15

	// now we estimate the average ann latency by (avgBatchTime * parallel + rtt) * step
	annLatency := (avgBatchTime*time.Duration(parallel) + rtt) * time.Duration(step)
	t.Logf("Estimated ANN latency = %v\n", annLatency)
}

func TestXORPerf(t *testing.T) {

	p := make([]uint64, 8)
	q := make([]uint64, 8)
	for i := 0; i < 8; i++ {
		p[i] = 12312312
		q[i] = 12312
	}
	xorSlices(p, q, 8)
	for i := 0; i < 8; i++ {
		if p[i] != 12312312^12312 {
			t.Errorf("p[%v] = %v; want %v", i, p[i], 12312312^12312)
		}
	}

	n := 1000000
	l := 112
	a := make([]uint64, l*n)
	b := make([]uint64, l*n)

	for i := 0; i < n; i++ {
		for j := 0; j < l; j++ {
			a[i*l+j] = 12312312
			b[i*l+j] = 12312
		}
	}

	// naive xor

	start := time.Now()
	for i := 0; i < n; i++ {
		for j := 0; j < l; j++ {
			a[i*l+j] ^= b[i*l+j]
		}
	}
	end := time.Now()
	t.Logf("Naive XOR time = %v\n", end.Sub(start))

	for i := 0; i < l*n; i++ {
		a[i] = 12312312
		b[i] = 12312
	}

	// use XorSlice
	start = time.Now()
	xorSlices(a, b, l*n)
	end = time.Now()
	t.Logf("XorSlices time = %v\n", end.Sub(start))

	// verify the result
	for i := 0; i < l*n; i++ {
		if a[i] != 12312312^12312 {
			t.Errorf("a[%v] = %v; want %v", i, a[i], 12312312^12312)
		}
	}
}

func TestAESPerf(t *testing.T) {

	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	masterKey := RandKey(rng)
	longKey := GetLongKey((*PrfKey128)(&masterKey))

	n := 1000000
	tag := make([]uint64, n)
	results := make([]uint64, n)

	for i := 0; i < n; i++ {
		tag[i] = rng.Uint64()
		results[i] = 0
	}

	start := time.Now()
	for i := 0; i < n; i++ {
		results[i] = PRFEvalWithLongKeyAndTag(longKey, tag[i], uint64(i))
	}
	end := time.Now()
	t.Logf("PRFEvalWithLongKeyAndTag time = %v\n", end.Sub(start))
	t.Logf("average time = %v ns", end.Sub(start).Nanoseconds()/int64(n))

	l := 112
	a := make([]uint64, l*n)
	b := make([]uint64, l*n)

	for i := 0; i < l*n; i++ {
		a[i] = 12312312
		b[i] = 12312
	}

	// use XorSlice
	start = time.Now()

	for i := 0; i < n; i++ {
		xorSlices(a[i*l:(i+1)*l], b[i*l:(i+1)*l], l)
	}

	end = time.Now()
	t.Logf("XorSlices time = %v\n", end.Sub(start))
	t.Logf("average time = %v ns", end.Sub(start).Nanoseconds()/int64(n))
}
