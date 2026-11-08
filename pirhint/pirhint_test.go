package pirhint

import (
	"testing"

	pianopir "github.com/dkblackley/bins-go/pianopir"
	pirold "github.com/dkblackley/bins-go/pir-old"
)

var sizes = []DB{
	{Rows: 1000, EntryBytes: 16, Batch: 8},
	{Rows: 5183, EntryBytes: 24, Batch: 32},
	{Rows: 70001, EntryBytes: 8, Batch: 128},
	{Rows: 123457, EntryBytes: 64, Batch: 16},
	{Rows: 300000, EntryBytes: 40, Batch: 24},
}

func TestMatchesPirOld(t *testing.T) {
	defer func(c, q float64) { pirold.ChunkSizeFactor, pirold.QueryBudgetScale = c, q }(
		pirold.ChunkSizeFactor, pirold.QueryBudgetScale)
	for _, k := range []Knobs{Default, {1, 1, 20}, {0.5, 0.5, 20}, {2, 0.25, 20}} {
		pirold.ChunkSizeFactor, pirold.QueryBudgetScale = k.ChunkFactor, k.QueryBudget
		for _, db := range sizes {
			raw := make([]uint64, db.Rows*db.EntryBytes/8)
			got := pirold.NewSimpleBatchPianoPIR(db.Rows, db.EntryBytes, db.Batch, raw, k.FailBits).LocalStorageSize()
			want := Compute(db, FlavorOld, k, false).HintBytes
			if got != want {
				t.Errorf("knobs %+v db %+v: pir-old allocates %v, pirhint says %v", k, db, got, want)
			}
		}
	}
}

func TestMatchesPianopir(t *testing.T) {
	for _, db := range sizes {
		words := db.EntryBytes / 8
		raw := make([][]uint64, db.Rows)
		for i := range raw {
			raw[i] = make([]uint64, words)
		}
		got := pianopir.NewSimpleBatchPianoPIR(db.Rows, words, db.EntryBytes, db.Batch, raw, 20, 1).LocalStorageSize()
		want := Compute(db, FlavorNew, Default, false).HintBytes
		if got != want {
			t.Errorf("db %+v: pianopir allocates %v, pirhint says %v", db, got, want)
		}
	}
}

// The scaling claims in the package comment, on concrete numbers.
func TestSplittingCostsMore(t *testing.T) {
	one := Compute(DB{Rows: 4_000_000, EntryBytes: 16, Batch: 128}, FlavorOld, Default, false)
	half := Compute(DB{Rows: 2_000_000, EntryBytes: 16, Batch: 128}, FlavorOld, Default, false)
	if 2*half.HintBytes <= one.HintBytes {
		t.Fatalf("two half-size DBs (%v B) should cost more than one (%v B)", 2*half.HintBytes, one.HintBytes)
	}
	tiny := Compute(DB{Rows: 5000, EntryBytes: 16, Batch: 128}, FlavorOld, Default, true)
	if !tiny.Downloaded || tiny.HintOverDB() < 1 {
		t.Fatalf("a 5000-row DB over 64 partitions should be cheaper to download, hint/DB = %v", tiny.HintOverDB())
	}
}
