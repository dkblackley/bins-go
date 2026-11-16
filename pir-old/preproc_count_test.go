package pianopir

import (
	"math/rand"
	"testing"
)

// Every offline phase is counted and charged one full stream of the partition, in normal and simulated mode,
// including the refreshes the batch PIR triggers on its own once the hint budget runs out.
func TestPreprocMetricsCountsEveryRefresh(t *testing.T) {
	for _, sim := range []bool{false, true} {
		old := Simulate
		Simulate = sim
		const n, entry, batch = 4096, 32, 8
		raw := make([]uint64, n*entry/8)
		for i := range raw {
			raw[i] = rand.Uint64()
		}
		p := NewSimpleBatchPianoPIR(n, entry, batch, raw, 20)
		p.Preprocessing()
		m := p.PreprocMetrics()
		if m["PreprocCount"] != 1 || m["PreprocBytes"] != n*entry || m["PreprocPartitionCount"] != float64(p.config.PartitionNum) {
			t.Fatalf("sim=%v after the first preprocessing: %v", sim, m)
		}
		rng := rand.New(rand.NewSource(1))
		for q := 0; q < 2000; q++ {
			idx := make([]uint64, batch)
			for i := range idx {
				idx[i] = uint64(rng.Intn(n))
			}
			if _, err := p.Query(idx); err != nil {
				t.Fatalf("sim=%v query %d: %v", sim, q, err)
			}
		}
		m = p.PreprocMetrics()
		full := float64(p.TotalPreProcs)
		if full < 2 {
			t.Fatalf("sim=%v: expected refreshes in 2000 batches, got %v", sim, full)
		}
		if m["PreprocCount"] != full || m["PreprocBytes"] < full*n*entry ||
			m["PreprocPartitionCount"] < full*float64(p.config.PartitionNum) {
			t.Fatalf("sim=%v: metrics %v do not cover %v full refreshes", sim, m, full)
		}
		t.Logf("sim=%v: %v", sim, m)
		Simulate = old
	}
}
