package pir

import (
	"bytes"
	"encoding/gob"
	"fmt"
	//"log"
	"testing"

	"github.com/henrycg/simplepir/matrix"
	"github.com/henrycg/simplepir/rand"
)

const SEC_PARAM = uint64(1 << 10)

func TestGobQuery(t *testing.T) {
	m := matrix.New[matrix.Elem32](5, 5)
	m.Set(1, 0, 0)
	m.Set(2, 0, 1)
	m.Set(3, 0, 2)

	q := Query[matrix.Elem32]{
		Query: m,
	}

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err := enc.Encode(&q)
	if err != nil {
		fmt.Println(err)
		panic("Encoding failed")
	}

	dec := gob.NewDecoder(&buf)
	var q2 Query[matrix.Elem32]
	err = dec.Decode(&q2)
	if err != nil {
		fmt.Println(err)
		panic("Decoding failed")
	}

	if !q.Query.Equals(q2.Query) {
		q.Query.Print()
		q2.Query.Print()
		panic("Objects are not equal")
	}
}

func TestGobAnswer(t *testing.T) {
	m := matrix.New[matrix.Elem32](5, 5)
	m.Set(1, 0, 0)
	m.Set(2, 0, 1)
	m.Set(3, 0, 2)

	a := Answer[matrix.Elem32]{
		Answer: m,
	}

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err := enc.Encode(&a)
	if err != nil {
		fmt.Println(err)
		panic("Encoding failed")
	}

	dec := gob.NewDecoder(&buf)
	var a2 Answer[matrix.Elem32]
	err = dec.Decode(&a2)
	if err != nil {
		fmt.Println(err)
		panic("Decoding failed")
	}

	if !a.Answer.Equals(a2.Answer) {
		a.Answer.Print()
		a2.Answer.Print()
		panic("Objects are not equal")
	}
}

func testServerEncode[T matrix.Elem](t *testing.T, N, d uint64) {
	prg := rand.NewRandomBufPRG()
	db := NewDatabaseRandom[T](prg, N, d)
	server := NewServer(db)

	var b bytes.Buffer
	enc := gob.NewEncoder(&b)
	err := enc.Encode(server)
	if err != nil {
		t.Fatal("Encoding failed")
	}

	dec := gob.NewDecoder(&b)
	var server2 Server[T]
	err = dec.Decode(&server2)
	if err != nil {
		t.Fatal("Decoding failed")
	}

	if *server2.params != *server.params {
		t.Fatal("Parameter mismatch")
	}
	if !server2.hint.Equals(server.hint) {
		t.Fatal("Hint mismatch")
	}

	if server2.db.Info.Num != server.db.Info.Num {
		t.Fatal("DB info mismatch")
	}
	if server2.db.Info.Params.N != server.db.Info.Params.N {
		t.Fatal("DB info mismatch")
	}
	if !server2.db.Data.Equals(server.db.Data) {
		t.Fatal("DB mismatch")
	}
}

func TestServerEncode32(t *testing.T) {
	testServerEncode[matrix.Elem32](t, uint64(1<<20), uint64(8))
}

func TestServerEncode64(t *testing.T) {
	testServerEncode[matrix.Elem64](t, uint64(1<<20), uint64(8))
}

// Run full PIR scheme (offline + online phases).
func runPIR[T matrix.Elem](t *testing.T, client *Client[T], server *Server[T], db *Database[T], i uint64) {
	//secret, query := client.Query(i)
	secret := client.PreprocessQuery()
	query := client.QueryPreprocessed(i, secret)

	answer := server.Answer(query)
	val := client.Recover(secret, answer)

	if db.GetElem(i) != val {
		t.Fatalf("(querying index %d): Got %d instead of %d\n",
			i, val, db.GetElem(i))
	}
}

func runPIRmany[T matrix.Elem](t *testing.T, client *Client[T], server *Server[T], db *Database[T], i uint64) {
	//secret, query := client.Query(i)
	secret := client.PreprocessQuery()
	query := client.QueryPreprocessed(i, secret)

	answer := server.Answer(query)

	vals := client.RecoverMany(secret, answer)

	col_index := i % db.Info.M
  //log.Printf("Rowlen: %v Ne: %v\n", len(vals), db.Info.Ne)
  //log.Printf("vals: %v \n", vals)
	for row := uint64(0); row < uint64(len(vals)); row++ {
		index := row*db.Info.M + col_index
		if db.GetElem(index) != vals[row] {
			t.Fatalf("Querying index %d: Got %d instead of %d\n",
				index, vals[row], db.GetElem(index))
		}
	}
}


func testSimplePir[T matrix.Elem](t *testing.T, N uint64, d uint64, index uint64) {
	prg := rand.NewRandomBufPRG()
	db := NewDatabaseRandom[T](prg, N, d)
  //for i := uint64(0); i<db.Data.Rows(); i++ {
  //  for j := uint64(0); j<db.Data.Cols(); j++ {
  //    db.Data.Set(i,j,T(0))//-T(db.Info.P()/2))
  //  }
    //db.Data.Set(i,i,T(0000))
  //}
  //db.Data.AddConst(T(db.Info.P()/2))
  //db.Data.Print()
  //log.Printf("===%v", db.Data.Get(0,0))

	server := NewServer(db)
	client := NewClient(server.Hint(), server.MatrixA(), db.Info)

	runPIR(t, client, server, db, index)
}

func testSimplePirMany[T matrix.Elem](t *testing.T, N uint64, d uint64, index uint64) {
	prg := rand.NewRandomBufPRG()
	db := NewDatabaseRandom[T](prg, N, d)
  //log.Printf("packing: %v", db.Info.Packing)

	server := NewServer(db)
	client := NewClient(server.Hint(), server.MatrixA(), db.Info)

	runPIRmany(t, client, server, db, index)
}


func testSimplePirCompressed[T matrix.Elem](t *testing.T, N uint64, d uint64, index uint64) {
	prg := rand.NewRandomBufPRG()
	db := NewDatabaseRandom[T](prg, N, d)

	seed := rand.RandomPRGKey()
	server := NewServerSeed(db, seed)
	client := NewClient(server.Hint(), server.MatrixA(), db.Info)

	runPIR(t, client, server, db, index)
}

func testSimplePirCompressedMany[T matrix.Elem](t *testing.T, N uint64, d uint64, index uint64) {
	prg := rand.NewRandomBufPRG()
	db := NewDatabaseRandom[T](prg, N, d)

	seed := rand.RandomPRGKey()
	server := NewServerSeed(db, seed)
	client := NewClient(server.Hint(), server.MatrixA(), db.Info)

	runPIRmany(t, client, server, db, index)
}


// Test SimplePIR correctness on DB with short entries.
func TestSimplePir32(t *testing.T) {
	testSimplePir[matrix.Elem32](t, uint64(1<<20), uint64(8), 262144)
}

//func TestSimplePirSmall64(t *testing.T) {
//	testSimplePir[matrix.Elem64](t, uint64(1<<8), uint64(3), 34)
//}


func TestSimplePir64(t *testing.T) {
	testSimplePir[matrix.Elem64](t, uint64(1<<20), uint64(12), 262144)
}

func TestSimplePirMany32(t *testing.T) {
	testSimplePirMany[matrix.Elem32](t, uint64(1<<20), uint64(8), 262144)
}

func TestSimplePirMany64(t *testing.T) {
	testSimplePirMany[matrix.Elem64](t, uint64(1<<20), uint64(17), 262144)
}


//func TestSimplePirCompressed32(t *testing.T) {
//	testSimplePirCompressed[matrix.Elem32](t, uint64(1<<20), uint64(4), 262144)
//}

func TestSimplePirCompressed64(t *testing.T) {
	testSimplePirCompressed[matrix.Elem64](t, uint64(1<<20), uint64(10), 262144)
}

func TestSimplePirCompressedMany32(t *testing.T) {
	testSimplePirCompressedMany[matrix.Elem32](t, uint64(1<<20), uint64(7), 262144)
}

func TestSimplePirCompressedMany64(t *testing.T) {
	testSimplePirCompressedMany[matrix.Elem64](t, uint64(1<<20), uint64(15), 262144)
}

// Test SimplePIR correctness on DB with long entries
func TestSimplePirLongRow32(t *testing.T) {
	testSimplePir[matrix.Elem32](t, uint64(1<<20), uint64(32), 1)
}

func TestSimplePirLongRow64(t *testing.T) {
	testSimplePir[matrix.Elem64](t, uint64(1<<20), uint64(64), 1)
}

func TestSimplePirLongRowCompressed32(t *testing.T) {
	testSimplePirCompressed[matrix.Elem32](t, uint64(1<<20), uint64(32), 1)
}

func TestSimplePirLongRowCompressed64(t *testing.T) {
	testSimplePirCompressed[matrix.Elem64](t, uint64(1<<20), uint64(64), 1)
}

// Test SimplePIR correctness on big DB
func TestSimplePirBigDB32(t *testing.T) {
	testSimplePir[matrix.Elem32](t, uint64(1<<25), uint64(9), 0)
}

func TestSimplePirBigDB64(t *testing.T) {
	testSimplePir[matrix.Elem64](t, uint64(1<<25), uint64(18), 0)
}

func TestSimplePirBigDBmany32(t *testing.T) {
	testSimplePirMany[matrix.Elem32](t, uint64(1<<25), uint64(9), 0)
}

func TestSimplePirBigDBmany64(t *testing.T) {
	testSimplePirMany[matrix.Elem64](t, uint64(1<<25), uint64(18), 0)
}

func TestSimplePirBigDBCompressed32(t *testing.T) {
	testSimplePirCompressed[matrix.Elem32](t, uint64(1<<25), uint64(9), 0)
}

func TestSimplePirBigDBCompressed64(t *testing.T) {
	testSimplePirCompressed[matrix.Elem64](t, uint64(1<<25), uint64(18), 0)
}

func TestSimplePirBigDBCompressedMany32(t *testing.T) {
	testSimplePirCompressedMany[matrix.Elem32](t, uint64(1<<25), uint64(9), 0)
}

func TestSimplePirBigDBCompressedMany64(t *testing.T) {
	testSimplePirCompressedMany[matrix.Elem64](t, uint64(1<<25), uint64(18), 2)
}

