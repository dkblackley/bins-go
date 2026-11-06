package pirhint

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// Layout is one way of laying out a method's PIR DBs plus the PIR knobs used.
// DB names start with the stage ("s1/", "s2/", "s3/", "id/", "vec/") so the
// per-stage totals can be reported.
type Layout struct {
	Name     string
	Note     string
	DBs      []DB
	Knobs    Knobs
	Download bool
	Flavor   Flavor
}

type LayoutResult struct {
	Layout
	Results []Result
	Client  float64 // bytes the client stores
	Hints   float64 // bytes of hints alone (before download-if-cheaper)
	Server  float64 // bytes of the DBs
	Stages  map[string]float64
	Online  float64 // estimated bytes one search sends, the other side of the trade-off
	Offline float64 // re-preprocessing download per search, amortized over the refresh period
}

func Evaluate(l Layout) LayoutResult {
	out := LayoutResult{Layout: l, Stages: map[string]float64{}}
	for _, db := range l.DBs {
		r := Compute(db, l.Flavor, l.Knobs, l.Download)
		out.Results = append(out.Results, r)
		stage := strings.SplitN(db.Name, "/", 2)[0]
		out.Stages[stage] += r.ClientBytes()
		out.Online += r.OnlineBytes
		out.Offline += r.OfflineBytes
	}
	out.Client, out.Hints, out.Server = Sum(out.Results)
	return out
}

// StageNames returns the stages of a result set in a fixed order.
func StageNames(rs []LayoutResult) []string {
	seen := map[string]bool{}
	for _, r := range rs {
		for s := range r.Stages {
			seen[s] = true
		}
	}
	var names []string
	for s := range seen {
		names = append(names, s)
	}
	order := map[string]int{"s1": 0, "s2": 1, "s3": 2, "id": 3, "vec": 4, "all": 5}
	sort.Slice(names, func(i, j int) bool { return order[names[i]] < order[names[j]] })
	return names
}

// ---------------------------------------------------------------------------
// Tree (pir-old)
// ---------------------------------------------------------------------------

type TreeParams struct {
	B, R, S     int     // block size, fan-out, sub-blocks: Stage2 values are B/S bytes
	Batch       uint64  // --pir-batch-size, Stage1 and Stage2
	Stage3Batch uint64  // --stage3-batch-size (32 originally)
	LoadFactor  float64 // --cuckoo-load-factor
	EmbedDim    int     // Stage3 embedding dimension
	EmbedBytes  int     // bytes per embedding value: 4 fp32, 2 fp16, 1 int8
	DocsPerRow3 int     // Stage3 docs per PIR row (8)

	// per search, for the online estimate only: query words, nodes kept per level
	// (and leaf blocks scored in Stage2), candidate docs fetched in Stage3
	QueryWords, Beam, Candidates int
}

var DefaultTree = TreeParams{B: 32, R: 32, S: 8, Batch: 128, Stage3Batch: 32, LoadFactor: 0.4,
	EmbedDim: 192, EmbedBytes: 4, DocsPerRow3: 8, QueryWords: 8, Beam: 50, Candidates: 128}

func (p TreeParams) keyProbes() uint64 { return uint64(2 * p.QueryWords * p.Beam) } // 2 Cuckoo probes per key

func (p TreeParams) stage1(shape TreeShape) []DB {
	var dbs []DB
	for level, items := range shape.Stage1Items {
		dbs = append(dbs, DB{Name: fmt.Sprintf("s1/level%d", level),
			Rows: CuckooRows(items, p.LoadFactor), EntryBytes: CuckooEntryBytes(uint64(p.R)), Batch: p.Batch,
			Probes: p.keyProbes()})
	}
	return dbs
}

func (p TreeParams) stage2(shape TreeShape) []DB {
	if shape.Stage2Items == 0 {
		return nil
	}
	return []DB{{Name: "s2/cuckoo", Rows: CuckooRows(shape.Stage2Items, p.LoadFactor),
		EntryBytes: CuckooEntryBytes(uint64(p.B / p.S)), Batch: p.Batch, Probes: p.keyProbes()}}
}

func (p TreeParams) stage3(shape TreeShape) []DB {
	rows := (shape.Docs + uint64(p.DocsPerRow3) - 1) / uint64(p.DocsPerRow3)
	return []DB{{Name: "s3/embeddings", Rows: rows,
		EntryBytes: Words(uint64(p.DocsPerRow3 * p.EmbedDim * p.EmbedBytes)), Batch: p.Stage3Batch,
		Probes: uint64(p.Candidates)}}
}

func (p TreeParams) all(shape TreeShape) []DB {
	return append(append(p.stage1(shape), p.stage2(shape)...), p.stage3(shape)...)
}

// TreeLayouts lists the current tree layout, the DB splits suggested so far,
// and the fixes, each as a full set of DBs so the totals compare directly.
func TreeLayouts(shape TreeShape, p TreeParams) []Layout {
	base := func(name, note string, dbs []DB) Layout {
		return Layout{Name: name, Note: note, DBs: dbs, Knobs: Default, Flavor: FlavorOld}
	}
	var out []Layout
	out = append(out, base("current", "one Cuckoo DB per Stage1 level, one Stage2 DB, Stage3 fp32", p.all(shape)))

	// --- splits / merges of the DBs -------------------------------------
	var total uint64
	for _, n := range shape.Stage1Items {
		total += n
	}
	merged := []DB{{Name: "s1/all-levels", Rows: CuckooRows(total, p.LoadFactor),
		EntryBytes: CuckooEntryBytes(uint64(p.R)), Batch: p.Batch,
		Probes: p.keyProbes() * uint64(len(shape.Stage1Items))}}
	out = append(out, base("s1 one DB for all levels", "every level in one Cuckoo table",
		append(append(merged, p.stage2(shape)...), p.stage3(shape)...)))

	for _, halve := range []bool{false, true} {
		var split []DB
		batch := p.Batch
		if halve {
			batch = max(p.Batch/2, 2)
		}
		for _, db := range p.stage1(shape) {
			for h := 0; h < 2; h++ { // one table per Cuckoo hash function, each probed once per key
				split = append(split, DB{Name: fmt.Sprintf("%s-h%d", db.Name, h),
					Rows: (db.Rows + 1) / 2, EntryBytes: db.EntryBytes, Batch: batch, Probes: db.Probes / 2})
			}
		}
		name, note := "s1 per level per hash", "2 half-size tables per level, same batch each"
		if halve {
			name, note = "s1 per level per hash, batch/2", "2 half-size tables per level, half the batch each"
		}
		out = append(out, base(name, note, append(append(split, p.stage2(shape)...), p.stage3(shape)...)))
	}

	if shape.Stage2Items > 0 {
		// one DB per sub-block position: B/S DBs keyed (term,leaf), 1-byte values.
		// Every one must be probed for every key, or the server learns which sub-block.
		perSub := DB{Name: "s2/per-subblock", Rows: CuckooRows(shape.Stage2Items, p.LoadFactor),
			EntryBytes: CuckooEntryBytes(1), Batch: p.Batch, Copies: uint64(p.B / p.S), Probes: p.keyProbes()}
		out = append(out, base("s2 one DB per sub-block", fmt.Sprintf("%d DBs with 1-byte values", p.B/p.S),
			append(append(p.stage1(shape), perSub), p.stage3(shape)...)))

		// one DB per leaf block keyed by term; every leaf DB must be probed (privacy),
		// a single partition each since each sees only q probes per query
		perLeaf := DB{Name: "s2/per-leaf", Rows: CuckooRows(uint64(math.Ceil(shape.ItemsPerLeaf)), p.LoadFactor),
			EntryBytes: CuckooEntryBytes(uint64(p.B / p.S)), Batch: 2, Copies: shape.Leaves,
			Probes: uint64(2 * p.QueryWords)}
		out = append(out, base("s2 one DB per leaf block", fmt.Sprintf("%d small DBs, 1 partition each", shape.Leaves),
			append(append(p.stage1(shape), perLeaf), p.stage3(shape)...)))
	}

	// --- fixes ----------------------------------------------------------
	dl := base("download if cheaper", "client takes the plain DB when it is smaller than its hints", p.all(shape))
	dl.Download = true
	out = append(out, dl)

	for _, eb := range []int{2, 1} {
		q := p
		q.EmbedBytes = eb
		label := map[int]string{2: "fp16", 1: "int8"}[eb]
		out = append(out, base("s3 "+label, "Stage3 embeddings stored as "+label, q.all(shape)))
	}
	for _, b := range []uint64{16, 8} {
		q := p
		q.Stage3Batch = b
		out = append(out, base(fmt.Sprintf("s3 batch %d", b), fmt.Sprintf("--stage3-batch-size %d (%d partitions)", b, b/2), q.all(shape)))
	}
	for _, b := range []uint64{64, 32} {
		q := p
		q.Batch = b
		out = append(out, base(fmt.Sprintf("s1/s2 batch %d", b), fmt.Sprintf("--pir-batch-size %d", b), q.all(shape)))
	}
	for _, f := range []float64{1, 0.5} {
		l := base(fmt.Sprintf("chunk factor %g", f), fmt.Sprintf("--pir-chunk-factor %g", f), p.all(shape))
		l.Knobs.ChunkFactor = f
		out = append(out, l)
	}
	for _, qb := range []float64{0.5, 0.25} {
		l := base(fmt.Sprintf("query budget %g", qb), fmt.Sprintf("--pir-query-budget %g", qb), p.all(shape))
		l.Knobs.QueryBudget = qb
		out = append(out, l)
	}

	q := p
	q.EmbedBytes = 1
	combo := base("combined", "download if cheaper + s3 int8 + chunk factor 1 + query budget 0.5", q.all(shape))
	combo.Download, combo.Knobs.ChunkFactor, combo.Knobs.QueryBudget = true, 1, 0.5
	out = append(out, combo)

	q.Batch, q.Stage3Batch = min(p.Batch, 32), min(p.Stage3Batch, 16)
	combo2 := base("combined + smaller batches", fmt.Sprintf("combined, --pir-batch-size %d, --stage3-batch-size %d",
		q.Batch, q.Stage3Batch), q.all(shape))
	combo2.Download, combo2.Knobs = true, combo.Knobs
	out = append(out, combo2)
	return out
}

// ---------------------------------------------------------------------------
// Bins (pianopir)
// ---------------------------------------------------------------------------

type BinsParams struct {
	Docs       uint64
	BinSize    uint64 // hash table size (bs)
	DPB        uint64 // docs per bin; the largest bin sets the entry size
	Dim        int
	EmbedBytes int
	Small      bool // the SciFact batch rules in PIR_bins.go (< 10000 docs)
	QueryTerms int  // bins looked up per search (online estimate only)
}

// vecBatch mirrors the Stage2 batch rules in bins/PIR_bins.go.
func (p BinsParams) vecBatch() uint64 {
	t := p.DPB
	if p.Small {
		return min(t, 500)
	}
	return min(3*t, 3000)
}

func (p BinsParams) dbs(vec0 bool, vecBatch uint64) []DB {
	if !vec0 { // bins_old.go: one DB, every bin holds dpb fp32 embeddings
		return []DB{{Name: "all/bins", Rows: p.BinSize,
			EntryBytes: Words(uint64(p.Dim*p.EmbedBytes) * p.DPB), Batch: 14, Probes: uint64(p.QueryTerms)}}
	}
	return []DB{
		{Name: "id/bins", Rows: p.BinSize, EntryBytes: (p.DPB + 1) / 2 * 8, Batch: 24, Probes: uint64(p.QueryTerms)},
		{Name: "vec/embeddings", Rows: p.Docs, EntryBytes: Words(uint64(p.Dim * p.EmbedBytes)), Batch: vecBatch,
			Probes: uint64(p.QueryTerms) * p.DPB},
	}
}

func BinsLayouts(p BinsParams) []Layout {
	base := func(name, note string, dbs []DB) Layout {
		return Layout{Name: name, Note: note, DBs: dbs, Knobs: Default, Flavor: FlavorNew}
	}
	vb := p.vecBatch()
	out := []Layout{
		base("vec1 (one DB)", "ids and embeddings in each bin", p.dbs(false, vb)),
		base("vec0 (ids, then embeddings)", fmt.Sprintf("Vec DB batch %d", vb), p.dbs(true, vb)),
	}
	dl := base("vec0 + download if cheaper", "", p.dbs(true, vb))
	dl.Download = true
	out = append(out, dl)
	for _, eb := range []int{2, 1} {
		q := p
		q.EmbedBytes = eb
		label := map[int]string{2: "fp16", 1: "int8"}[eb]
		out = append(out, base("vec0 + "+label, "Vec DB embeddings as "+label, q.dbs(true, vb)))
	}
	for _, b := range []uint64{vb / 4, vb / 16} {
		if b < 2 || b == vb {
			continue
		}
		out = append(out, base(fmt.Sprintf("vec0 + Vec batch %d", b),
			"fewer partitions; more Piano queries land in each, more sequential work", p.dbs(true, b)))
	}
	return out
}
