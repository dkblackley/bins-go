package pir

import (
	"github.com/henrycg/simplepir/lwe"
	"github.com/henrycg/simplepir/rand"
	"github.com/henrycg/simplepir/matrix"
)

type Server[T matrix.Elem] struct {
	params       *lwe.Params
	matrixAseed  *rand.PRGKey

	db           *Database[T]
	hint         *matrix.Matrix[T]
}

func NewServer[T matrix.Elem](db *Database[T]) *Server[T] {
	return setupServer(db, rand.RandomPRGKey())
}

func NewServerSeed[T matrix.Elem](db *Database[T], seed *rand.PRGKey) *Server[T] {
	return setupServer(db, seed)
}

func setupServer[T matrix.Elem](db *Database[T], matrixAseed *rand.PRGKey) *Server[T] {
	src := rand.NewBufPRG(rand.NewPRG(matrixAseed))
  matrixA := matrix.Rand[T](src, db.Info.M, db.Info.Params.N, 0)

	s := &Server[T]{
		params:      db.Info.Params,
		matrixAseed: matrixAseed,
		db:          db.Copy(),
		//hint:        matrix.MulSeededRight(db.Data, matrixAseeded),
		hint:        matrix.Mul(db.Data, matrixA),
	}

	s.db.Squish()

	return s
}

func (s *Server[T]) Hint() *matrix.Matrix[T] {
	return s.hint
}

func (s *Server[T]) DropHint() {
	s.hint = &matrix.Matrix[T]{}
}

func (s *Server[T]) MatrixA() *rand.PRGKey {
	return s.matrixAseed
}

func (s *Server[T]) Params() *lwe.Params {
	return s.params
}

func (s *Server[T]) DBInfo() *DBInfo {
	return s.db.Info
}

func (s *Server[T]) Get(i uint64) uint64 {
	return s.db.GetElem(i)
}

func (s *Server[T]) Answer(query *Query[T]) *Answer[T] {
	return &Answer[T]{ matrix.MulVecPacked(s.db.Data, query.Query) }
}

