package pir

import (
  "testing"

	"github.com/henrycg/simplepir/matrix"
)

func testDBInit[T matrix.Elem](t *testing.T, N uint64, d uint64, vals []uint64) *Database[T] {
	db := NewDatabase[T](N, d, vals)

	for i := uint64(0); i < N; i++ {
		if db.GetElem(i) != (i + 1) {
			t.Fatalf("Reconstruct failed! %v != %v", db.GetElem(i), i+1)
		}
	}

	return db
}

// Test that DB packing methods are correct, when each database entry is ~ 1 Z_p elem.
func testDBMediumEntries[T matrix.Elem](t *testing.T) *Database[T] {
	vals := []uint64{1, 2, 3, 4}
	return testDBInit[T](t, uint64(4), uint64(7), vals)
}

func TestDBMediumEntries32(t *testing.T) {
  db := testDBMediumEntries[matrix.Elem32](t)
  if db.Info.Ne != 1 {
    t.Fail()
  }
}

func TestDBMediumEntries64(t *testing.T) {
  db := testDBMediumEntries[matrix.Elem64](t)
  if db.Info.Ne != 1 {
    t.Fail()
  }
}

// Test that DB packing methods are correct, when multiple database entries fit in 1 Z_p elem.
func testDBSmallEntries[T matrix.Elem](t *testing.T) {
	vals := []uint64{1, 2, 3, 4}
	db := testDBInit[T](t, uint64(4), uint64(3), vals)

	if db.Info.Ne != 1 {
		t.Fail()
	}
}

func TestDBSmallEntries32(t *testing.T) {
  testDBSmallEntries[matrix.Elem32](t)
}

func TestDBSmallEntries64(t *testing.T) {
  testDBSmallEntries[matrix.Elem64](t)
}

// Test that DB packing methods are correct, when each database entry requires multiple Z_p elems.
func testDBLargeEntries[T matrix.Elem](t *testing.T) {
	vals := []uint64{1, 2, 3, 4}
	db := testDBInit[T](t, uint64(4), uint64(21), vals)

	if db.Info.Ne <= 1 {
		t.Fatal()
	}
}

func TestDBLargeEntries32(t *testing.T) {
  testDBLargeEntries[matrix.Elem64](t)
}

func TestDBLargeEntries64(t *testing.T) {
  testDBLargeEntries[matrix.Elem64](t)
}
