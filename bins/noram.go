package bins

import (
	"github.com/dkblackley/bins-go/diskstore"
	"github.com/dkblackley/bins-go/pianopir"
)

// rowSink collects the rows of a PIR database. Normally they stay on the heap, exactly as before. With -noram they
// go to a file under temp_storage (diskstore) and come back as a read-only mmap, laid out in the order the batch
// PIR shuffles them into, so the PIR code gets the same [][]uint64 either way.
type rowSink struct {
	rows  [][]uint64
	store *diskstore.RowStore
}

func newRowSink(name string, n int) *rowSink {
	if !diskstore.Enabled {
		return &rowSink{rows: make([][]uint64, n)}
	}
	st, err := diskstore.NewRowStore(name, n)
	Must(err)
	return &rowSink{store: st}
}

// Put sets row i. On the heap the sink keeps row itself; on disk it writes a copy. Safe for concurrent use on
// distinct i.
func (s *rowSink) Put(i int, row []uint64) {
	if s.store == nil {
		s.rows[i] = row
		return
	}
	Must(s.store.Put(i, row))
}

func (s *rowSink) onDisk() bool { return s.store != nil }

func (s *rowSink) Finish() [][]uint64 {
	if s.store == nil {
		return s.rows
	}
	// Simulated PIR builds no hints, so nothing ever reads the DB in the shuffled order: map the rows where Put
	// wrote them instead of copying the whole file again (same rows, same answers).
	var order []int
	if !pianopir.Simulate {
		order = pianopir.ShufflePermutation(uint64(s.store.Len()))
	}
	rows, err := s.store.Finish(order)
	Must(err)
	return rows
}
