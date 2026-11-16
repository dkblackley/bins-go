package diskstore

import (
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestRowStoreAndCleanup(t *testing.T) {
	base := t.TempDir()
	dir, err := Init(base, "test")
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	const n = 1000
	want := make([][]uint64, n)
	st, err := NewRowStore("db", n)
	if err != nil {
		t.Fatal(err)
	}
	for _, i := range rng.Perm(n) { // any order
		if i%7 == 0 {
			continue // empty rows
		}
		want[i] = make([]uint64, 1+rng.Intn(20))
		for k := range want[i] {
			want[i][k] = rng.Uint64()
		}
		if err := st.Put(i, want[i]); err != nil {
			t.Fatal(err)
		}
	}
	got, err := st.Finish(rng.Perm(n))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatal("rows differ")
	}
	tab := ZeroTable("hints", 10, 4)
	tab[3][2] = 42 // writable
	if tab[3][2] != 42 || tab[4][0] != 0 {
		t.Fatal("hint table")
	}
	if files, _ := filepath.Glob(filepath.Join(dir, "*")); len(files) != 3 { // owner, db, hints (staging removed)
		t.Fatalf("files in run dir: %v", files)
	}
	Cleanup()
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("run dir still there: %v", err)
	}
}

// Finish(nil) maps the staging file in place: same rows, no second file.
func TestRowStoreFinishInPlace(t *testing.T) {
	dir, err := Init(t.TempDir(), "test")
	if err != nil {
		t.Fatal(err)
	}
	defer Cleanup()
	st, err := NewRowStore("db", 50)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(2))
	want := make([][]uint64, 50)
	for _, i := range rng.Perm(50) { // any Put order
		if i%7 == 3 {
			continue // empty row
		}
		want[i] = make([]uint64, 1+rng.Intn(20))
		for k := range want[i] {
			want[i][k] = rng.Uint64()
		}
		if err := st.Put(i, want[i]); err != nil {
			t.Fatal(err)
		}
	}
	got, err := st.Finish(nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatal("rows differ")
	}
	if files, _ := filepath.Glob(filepath.Join(dir, "*")); len(files) != 2 { // owner, staging
		t.Fatalf("files in run dir: %v", files)
	}
}
