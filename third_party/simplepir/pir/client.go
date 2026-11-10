package pir

import (
	"github.com/henrycg/simplepir/lwe"
	"github.com/henrycg/simplepir/rand"
	"github.com/henrycg/simplepir/matrix"
)

type Client[T matrix.Elem] struct {
	prg          *rand.BufPRGReader

	params       *lwe.Params
	dbinfo       *DBInfo
	hint         *matrix.Matrix[T]

	matrixAseeds []rand.PRGKey
	matrixArows  []uint64
}

func NewClient[T matrix.Elem](hint *matrix.Matrix[T], matrixAseed *rand.PRGKey, dbinfo *DBInfo) *Client[T] {
	return NewClientDistributed(hint, []rand.PRGKey { *matrixAseed}, []uint64{ dbinfo.M }, dbinfo)
}

func NewClientDistributed[T matrix.Elem](hint *matrix.Matrix[T], matrixAseeds []rand.PRGKey, matrixArows []uint64, dbinfo *DBInfo) *Client[T] {
  c := &Client[T]{
		prg: rand.NewRandomBufPRG(),

		params: dbinfo.Params,
		dbinfo:  dbinfo,

		matrixAseeds: matrixAseeds, // Warning: not copied
		matrixArows: matrixArows,   // Warning: not copied
	}

  	if hint != nil {
		c.hint = hint.Copy()
  	}

  	return c
}

func (c *Client[T]) Hint() *matrix.Matrix[T] {
	return c.hint
}

func (c *Client[T]) GenerateSecret() *matrix.Matrix[T] {
        //log.Printf("Warning! Using ternary secrets for SimplePIR LHE.")
        return matrix.Ternary[T](c.prg, c.params.N, 1)
}

func (c *Client[T]) PreprocessQuery() *Secret[T] {
	inSecret := c.GenerateSecret()
	return c.PreprocessQueryGivenSecret(inSecret)
}

func (c *Client[T]) PreprocessQueryGivenSecret(inSecret *matrix.Matrix[T]) *Secret[T] {
	s := &Secret[T]{
		secret: inSecret,
	}

	// Compute H * s
	if c.hint != nil {
		s.interm = matrix.Mul(c.hint, s.secret)
	}

	src := make([]matrix.IoRandSource, len(c.matrixAseeds))
	for i, seed := range c.matrixAseeds {
		src[i] = rand.NewBufPRG(rand.NewPRG(&seed))
	}
        matrixAseeded := matrix.NewSeeded[T](src, c.matrixArows, c.params.N)

	err := matrix.Gaussian[T](c.prg, c.dbinfo.M, 1)

	// Compure A * s + e
	query := matrix.MulSeededLeft(matrixAseeded, s.secret)
	query.Add(err)

	// Pad the query to match the dimensions of the compressed DB
	if c.dbinfo.M%c.dbinfo.Squishing != 0 {
		query.AppendZeros(c.dbinfo.Squishing - (c.dbinfo.M % c.dbinfo.Squishing))
	}

	s.query = query

	return s
}

func (c *Client[T]) QueryPreprocessed(i uint64, s *Secret[T]) *Query[T] {
	s.index = i
	s.query.AddAt(i%c.dbinfo.M, 0, T(c.params.Delta))
	return &Query[T]{ s.query }
}

func (c *Client[T]) Query(i uint64) (*Secret[T], *Query[T]) {
	s := c.PreprocessQuery()
	q := c.QueryPreprocessed(i, s)
	return s, q
}

func (c *Client[T]) Decode(ans *matrix.Matrix[T], index uint64) uint64 {
	var vals []uint64
	row := index / c.dbinfo.M

	// Recover each Z_p element that makes up the desired database entry
	for j := row * c.dbinfo.Ne; j < (row+1)*c.dbinfo.Ne; j++ {
		noised := uint64(ans.Get(j, 0))
		denoised := c.params.Round(noised)
		vals = append(vals, denoised)
		//log.Printf("Reconstructing row %d: %d\n", j, denoised)
	}

	return c.dbinfo.ReconstructElem(vals, index)
}

func (c *Client[T]) Recover(s *Secret[T], ansIn *Answer[T]) uint64 {
	if s.interm == nil {
		s.interm = matrix.Mul(c.hint, s.secret)
	}

	ans := ansIn.Answer.Copy()
	ans.Sub(s.interm)

	return c.Decode(ans, s.index)
}

func (c *Client[T]) DecodeMany(ans *matrix.Matrix[T]) []uint64 {
	num_values := (ans.Rows() / c.dbinfo.Ne)
	out := make([]uint64, num_values)
	for row := uint64(0); row < ans.Rows(); row += c.dbinfo.Ne {
		var vals []uint64
		// Recover each Z_p element that makes up the desired database entry
		for j := uint64(0); j < c.dbinfo.Ne; j++ {
			noised := uint64(ans.Get(row + j, 0))
			denoised := c.params.Round(noised)
			vals = append(vals, denoised)
		}

		out[row] = c.dbinfo.ReconstructElem(vals, 0)
		//log.Printf("Reconstructing row %d: %d\n", row, out[row])
	}

	return out
}

func (c *Client[T]) RecoverMany(s *Secret[T], ansIn *Answer[T]) []uint64 {
	if s.interm == nil {
		s.interm = matrix.Mul(c.hint, s.secret)
	}

	ans := ansIn.Answer.Copy()
	ans.Sub(s.interm)

	return c.DecodeMany(ans)
}

func (c *Client[T]) GetM() uint64 {
	return c.dbinfo.M
}

func (c *Client[T]) GetL() uint64 {
	return c.dbinfo.L
}

func (c *Client[T]) GetP() uint64 {
	return c.params.P
}

func (c *Client[T]) GetSecurityParam() uint64 {
	return c.params.N
}

func (c *Client[T]) ClearHint() {
	c.hint = nil
}
