package pir

import (
  "log"
  "math"
)

import (
  "github.com/henrycg/simplepir/lwe"
  "github.com/henrycg/simplepir/rand"
  "github.com/henrycg/simplepir/matrix"
)

type DBInfo struct {
	Num       uint64 // number of db entries.
	RowLength uint64 // number of bits per db entry.

	Ne      uint64 // number of Z_p elems per db entry, if db entry size > log(p).

	X uint64 // tunable param that governs communication,
	// must be in range [1, ne] and must be a divisor of ne;
	// represents the number of times the scheme is repeated.

	L uint64 // database height
	M uint64 // database width

	// For in-memory db compression
	Squishing uint64
	Cols      uint64

	Params *lwe.Params
}

type Database[T matrix.Elem] struct {
	Info *DBInfo
	Data *matrix.Matrix[T]
}

func (db *Database[T]) Copy() *Database[T] {
	return &Database[T]{
		Info: db.Info,
		Data: db.Data.Copy(),
	}
}

func (db *Database[T]) Squish() {
	//log.Printf("Original db dims: ")
	//db.Data.Dim()

	// Check that Params allow for this compression
	if !db.Data.CanSquish(db.Info.P()) {
		panic("Bad Params")
	}

	db.Info.Squishing = db.Data.SquishRatio()
	db.Info.Cols = db.Data.Cols()
	db.Data.Squish()
}

// Store the database with entries decomposed into Z_p elements.
// Z_p elements that encode the same database entry are stacked vertically below each other.
func (Info *DBInfo) ReconstructElem(vals []uint64, index uint64) uint64 {
  shortQ := (Info.Params.Logq != 64)

	for i, _ := range vals {
		vals[i] = (vals[i])
    if shortQ {
      vals[i] %= (1 << 32)
    }
		vals[i] = vals[i] % Info.P()
	}

	return Reconstruct_from_base_p(Info.P(), vals)
}

func (db *Database[T]) GetElem(i uint64) uint64 {
	if i >= db.Info.Num {
		panic("Index out of range")
	}

	cols := db.Data.Cols()
	col := i % cols
	row := i / cols

	var vals []uint64
	for j := row * db.Info.Ne; j < (row+1)*db.Info.Ne; j++ {
		vals = append(vals, uint64(db.Data.Get(j, col)))
	}

	return db.Info.ReconstructElem(vals, i)
}

// Returns how many Z_p elements are needed to represent a database of N entries,
// each consisting of row_length bits.
func numEntries(N, row_length, p uint64) (uint64, uint64) {
	if float64(row_length) <= math.Log2(float64(p)) {
		return N, 1
	}

	// use multiple Z_p elems to represent a single DB entry
	ne := Compute_num_entries_base_p(p, row_length)
	return N * ne, ne
}

// Find smallest l, m such that l*m >= N*ne and ne divides l, where ne is
// the number of Z_p elements per db entry determined by row_length and p.
func approxSquareDatabaseDims(dbElems, elemsPerEntry, rowLength, p uint64) (uint64, uint64) {
	l := uint64(math.Floor(math.Sqrt(float64(dbElems))))

	rem := l % elemsPerEntry
	if rem != 0 {
		l += elemsPerEntry - rem
	}

	m := uint64(math.Ceil(float64(dbElems) / float64(l)))

	return l, m
}

func NewDBInfo(logq uint64, num uint64, rowLength uint64) *DBInfo {
	if (num == 0) || (rowLength == 0) {
		panic("Empty database!")
	}
	// Make a guess at plaintext modulus and compute parameters
	tempP := uint64(256)
	dbElems, elemsPerEntry := numEntries(num, rowLength, tempP)
	_, m := approxSquareDatabaseDims(dbElems, elemsPerEntry, rowLength, tempP)

	params := lwe.NewParams(logq, m)
	if params == nil {
		panic("Could not find LWE Params")
	}

	return NewDBInfoFixedParams(num, rowLength, params, false)
}

func NewDBInfoFixedParams(num uint64, rowLength uint64, params *lwe.Params, fixed bool) *DBInfo {
	Info := &DBInfo{
		Num:       num,
		RowLength: rowLength,
		Params:    params,
		M:         params.M,
	}

	// Compute database Info based on real LWE parameters
	dbElems, elemsPerEntry := numEntries(num, rowLength, Info.Params.P)

	Info.L = uint64(math.Ceil(float64(dbElems) / float64(Info.M)))
	if Info.L % elemsPerEntry != 0 {
		Info.L += elemsPerEntry - (Info.L % elemsPerEntry)
	}

	Info.Ne = elemsPerEntry
	Info.X = Info.Ne

	for Info.Ne%Info.X != 0 {
		Info.X += 1
	}

	log.Printf("Total packed db size is ~%f MB\n",
		float64(Info.L*Info.M)*math.Log2(float64(Info.P()))/(1024.0*1024.0*8.0))

	if dbElems > Info.L*Info.M {
		panic("lwe.Params and database size don't match")
	}

	if Info.L%Info.Ne != 0 {
		panic("Number of db elems per entry must divide db height")
	}

	if !fixed {
		// Recompute params based on chosen M
		Info.Params = lwe.NewParams(params.Logq, Info.M)
	}

	if Info.Params == nil {
		panic("Could not find good LWE Params")
	}

	return Info
}

// Number of Z_p elements to represent a DB record
func (Info *DBInfo) RecordSize() uint64 {
	return Info.Ne
}

func (Info *DBInfo) P() uint64 {
	return Info.Params.P
}

func NewDatabaseRandom[T matrix.Elem](prg *rand.BufPRGReader, num, rowLength uint64) *Database[T] {
	info := NewDBInfo(T(0).Bitlen(), num, rowLength)
	return NewDatabaseRandomFixedParams[T](prg, num, rowLength, info.Params)
}

func NewDatabaseRandomFixedParams[T matrix.Elem](prg *rand.BufPRGReader, Num, rowLength uint64, params *lwe.Params) *Database[T] {
	db := new(Database[T])
	db.Info = NewDBInfoFixedParams(Num, rowLength, params, true)

	mod := db.Info.P()
	if ((1 << rowLength) < mod) && (db.Info.Ne == 1) {
		mod = (1 << rowLength)
	}

  maxSize := db.Info.P()
  if float64(rowLength) < math.Log2(float64(db.Info.P())) {
    maxSize = (1 << rowLength)
  }
	db.Data = matrix.Rand[T](prg, db.Info.L, db.Info.M, maxSize)

	// clear overflow cols
	row := db.Info.L - 1
	for i := Num; i < db.Info.L*db.Info.M; i++ {
		col := i % db.Info.M
		db.Data.Set(row, col, 0)
	}

	for i := uint64(0); i < db.Info.L*db.Info.M; i++ {
		col := i % db.Info.M
		if db.Data.Get(row, col) >= T(db.Info.P()) {
      panic("bad")
    }
	}

	return db
}

func NewDatabase[T matrix.Elem](num, rowLength uint64, vals []uint64) *Database[T] {
	info := NewDBInfo(T(0).Bitlen(), num, rowLength)
	return NewDatabaseFixedParams[T](num, rowLength, vals, info.Params)
}

func NewDatabaseFixedParams[T matrix.Elem](Num, rowLength uint64, vals []uint64, params *lwe.Params) *Database[T] {
	db := new(Database[T])
	db.Info = NewDBInfoFixedParams(Num, rowLength, params, true)
	db.Data = matrix.Zeros[T](db.Info.L, db.Info.M)

	if uint64(len(vals)) != Num {
		panic("Bad input db")
	}

  // Use multiple Z_p elems to represent each db elem
  for i, elem := range vals {
    for j := uint64(0); j < db.Info.Ne; j++ {
      db.Data.Set((uint64(i)/db.Info.M)*db.Info.Ne+j,
        uint64(i)%db.Info.M,
        T(Base_p(db.Info.P(), elem, j)))
    }
  }

	return db
}
