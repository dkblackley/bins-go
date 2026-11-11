package pir

import (
  //"log"
)

import (
	"github.com/henrycg/simplepir/matrix"
)


type Query[T matrix.Elem] struct {
	Query *matrix.Matrix[T]
}

type Secret[T matrix.Elem] struct {
	query  *matrix.Matrix[T]
	secret *matrix.Matrix[T]
	interm *matrix.Matrix[T]
	index  uint64
}

type Answer[T matrix.Elem] struct {
	Answer *matrix.Matrix[T]
}

func (q *Query[T]) SelectRows(start, num, squishing uint64) *Query[T] {
	res := new(Query[T])
	res.Query = q.Query.RowsDeepCopy(start, num)

	r, c := res.Query.Rows(), res.Query.Cols()
	if (r * c) % squishing != 0 {
		res.Query.AppendZeros(squishing - ((r * c) % squishing)) 
	}

	return res
}
