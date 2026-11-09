package diskstore

import (
	"path/filepath"
	"testing"

	"github.com/kshedden/gonpy"
)

func TestMmapNpyMatchesGonpy(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.npy")
	w, err := gonpy.NewFileWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	const n, dim = 37, 6
	data := make([]float32, n*dim)
	for i := range data {
		data[i] = float32(i)*0.25 - 3
	}
	w.Shape = []int{n, dim}
	if err := w.WriteFloat32(data); err != nil {
		t.Fatal(err)
	}
	got, err := MmapNpyFloat32(path, n-5, dim) // first n-5 rows, like -n smaller than the file
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != n-5 {
		t.Fatalf("rows %d", len(got))
	}
	for i, row := range got {
		for k, v := range row {
			if v != data[i*dim+k] {
				t.Fatalf("[%d][%d] = %v, want %v", i, k, v, data[i*dim+k])
			}
		}
	}
	if _, err := MmapNpyFloat32(path, n, dim+1); err == nil {
		t.Fatal("wrong dim accepted")
	}
}
