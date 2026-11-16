package pir

import (
	"testing"
	"fmt"

	"github.com/henrycg/simplepir/matrix"
	"github.com/henrycg/simplepir/lwe"
	"github.com/henrycg/simplepir/rand"
)

func runLHE[T matrix.Elem](t *testing.T, client *Client[T], server *Server[T], db *Database[T], arr *matrix.Matrix[T]) {
	//secret, query := client.QueryLHE(arr)
	secret := client.PreprocessQueryLHE()
	query := client.QueryLHEPreprocessed(arr, secret)
	answer := server.Answer(query)

	vals := client.RecoverManyLHE(secret, answer)
  
	shouldBe := matrix.Mul(db.Data, arr)
	shouldBe.ModConst(T(db.Info.P()))

	at := uint64(0)
	for i := uint64(0); i < uint64(vals.Rows()); i++ {
		should_be := uint64(0)
		for j := uint64(0); (j < uint64(arr.Rows())) && (at < db.Info.Num); j++ {
			should_be += uint64(arr.Get(j, 0)) * db.GetElem(at)
			at += 1
		}
		should_be %= db.Info.P()

		if should_be != uint64(vals.Get(i, 0)) {
			fmt.Printf("Row %d: Got %d instead of %d (mod %d) -- %d\n",
			            i, uint64(vals.Get(i, 0)), should_be, db.Info.P(), shouldBe.Get(i, 0))
			t.Fail()
		}
	}

	if !shouldBe.Equals(vals) {
		fmt.Printf("should be: %v \n", shouldBe)
		fmt.Printf("got : %v\n", vals)
		t.Fail()
	}
}

func testLHE[T matrix.Elem](t *testing.T, N uint64, d uint64) {
	prg := rand.NewRandomBufPRG()
	params := lwe.NewParamsFixedP(T(0).Bitlen(), N, 512)
	db := NewDatabaseRandomFixedParams[T](prg, N, d, params)
  	arr := matrix.Rand[T](prg, db.Info.M, 1, params.P)

	server := NewServer(db)
	client := NewClient(server.Hint(), server.MatrixA(), db.Info)

	runLHE(t, client, server, db, arr)
}



func TestLHE32(t *testing.T) {
	testLHE[matrix.Elem32](t, uint64(1<<7)+3, uint64(9))
}

func TestLHE64(t *testing.T) {
	testLHE[matrix.Elem64](t, uint64(1<<8)+5, uint64(9))
}

func TestLHE32_2(t *testing.T) {
	testLHE[matrix.Elem32](t, uint64(1<<13), uint64(8))
}

func TestLHE64_2(t *testing.T) {
	testLHE[matrix.Elem64](t, uint64(1<<13), uint64(8))
}

func TestLHE32_3(t *testing.T) {
	testLHE[matrix.Elem32](t, uint64(1<<13), uint64(6))
}

func TestLHE64_3(t *testing.T) {
	testLHE[matrix.Elem64](t, uint64(1<<13), uint64(6))
}

func TestLHEBigDB32(t *testing.T) {
	testLHE[matrix.Elem32](t, uint64(1<<14), uint64(9))
}

func TestLHEBigDB64(t *testing.T) {
	testLHE[matrix.Elem64](t, uint64(1<<14), uint64(9))
}

