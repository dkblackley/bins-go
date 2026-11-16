package pianopir

import "fmt"

// Simulate turns on simulated PIR (the tree's copy; see pianopir/simulate.go for
// the bins one). A simulated client never builds its hint tables and a simulated
// server never XORs rows: a lookup returns the row itself, which is what a real
// Piano query decodes to. The batching, dummy queries, local cache, per-chunk
// query budget (so re-preprocessing happens at the same points), rounds and byte
// counts run as in a real run. Rows here all have the same size, so bytes sent,
// LAN/WAN times and client storage are exact; only the timings are meaningless
// (main.go writes -1 for them). Set it before building any SimpleBatchPianoPIR.
var Simulate = false

type simClient struct{}

func (c *PianoPIRClient) simInitialization() {
	c.FinishedQueryNum = 0
	c.QueryHistogram = make([]uint64, c.config.SetSize)
	c.localCache = make(map[uint64][]uint64)
}

// simQuery follows Query step for step (same cache, budget checks and errors)
// but skips the hint lookup and the XOR.
func (c *PianoPIRClient) simQuery(idx uint64, server *PianoPIRServer, realQuery bool) ([]uint64, error) {
	ret := make([]uint64, c.config.DBEntrySize)
	if !realQuery {
		server.simPrivateQuery()
		return ret, nil
	}
	if idx >= c.config.DBSize {
		return ret, fmt.Errorf("idx %v is out of range", idx)
	}
	if v, ok := c.localCache[idx]; ok {
		return v, nil
	}
	if c.FinishedQueryNum >= c.MaxQueryNum {
		return ret, fmt.Errorf("exceed the maximum number of queries")
	}
	chunkId := idx / c.config.ChunkSize
	if c.QueryHistogram[chunkId] >= c.maxQueryPerChunk {
		return ret, fmt.Errorf("too many queries in chunk %v", chunkId)
	}

	server.simPrivateQuery()
	w := c.config.DBEntrySize
	copy(ret, server.rawDB[idx*w:(idx+1)*w])

	c.FinishedQueryNum += 1
	c.QueryHistogram[chunkId] += 1
	c.localCache[idx] = ret
	return ret, nil
}

// simPrivateQuery charges the bytes PrivateQuery would: SetSize offsets up, one row down.
func (s *PianoPIRServer) simPrivateQuery() {
	s.privateQueryCount++
	s.RetrievalCount += s.config.SetSize*4 + s.config.DBEntrySize*8
}
