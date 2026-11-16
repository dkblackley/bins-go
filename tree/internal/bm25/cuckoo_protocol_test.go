package bm25

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type fixtureRow struct {
	termID uint32
	nodeID uint32
	value  []byte
}

func writeCuckooFixture(t *testing.T, directory, prefix string, rows []fixtureRow, vocab map[string]int) (string, string, string) {
	t.Helper()
	dataPath := filepath.Join(directory, prefix+"_data.bin")
	idmapPath := filepath.Join(directory, prefix+"_idmap.bin")
	vocabPath := filepath.Join(directory, prefix+"_vocab.json")
	var data bytes.Buffer
	var idmap bytes.Buffer
	for rowIndex, row := range rows {
		data.Write(row.value)
		if err := binary.Write(&idmap, binary.LittleEndian, row.termID); err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(&idmap, binary.LittleEndian, row.nodeID); err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(&idmap, binary.LittleEndian, uint64(rowIndex)); err != nil {
			t.Fatal(err)
		}
	}
	vocabJSON, err := json.Marshal(vocab)
	if err != nil {
		t.Fatal(err)
	}
	for path, content := range map[string][]byte{
		dataPath: data.Bytes(), idmapPath: idmap.Bytes(), vocabPath: vocabJSON,
	} {
		if err := os.WriteFile(path, content, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dataPath, idmapPath, vocabPath
}

func TestStage1SeparateCuckooLevels(t *testing.T) {
	directory := t.TempDir()
	vocab := make(map[string]int)
	rows := make([]fixtureRow, 0, 64)
	for term := 0; term < 32; term++ {
		vocab["t"+string(rune('a'+term))] = term
		rows = append(rows, fixtureRow{
			termID: uint32(term), nodeID: 1,
			value: []byte{byte(term + 1), byte(term + 2), byte(term + 3), byte(term + 4)},
		})
		rows = append(rows, fixtureRow{
			termID: uint32(term), nodeID: uint32(4 + term%4),
			value: []byte{byte(term + 5), byte(term + 6), byte(term + 7), byte(term + 8)},
		})
	}
	dataPath, idmapPath, vocabPath := writeCuckooFixture(t, directory, "stage1", rows, vocab)
	oldStrict := StrictPIR
	StrictPIR = true
	defer func() { StrictPIR = oldStrict }()
	db, err := NewStage1PIRDBWithCuckoo(
		dataPath, idmapPath, vocabPath, 4, true, 8, DefaultCuckooBuildConfig(),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if len(db.Levels) != 2 {
		t.Fatalf("Stage1 databases=%d; want 2", len(db.Levels))
	}
	if db.Levels[0].Table.ItemCount != 32 || db.Levels[1].Table.ItemCount != 32 {
		t.Fatalf("Stage1 level counts=(%d,%d); want (32,32)", db.Levels[0].Table.ItemCount, db.Levels[1].Table.ItemCount)
	}

	root := db.SumChildScoresAtLevel(0, []string{"ta", "__padding__"}, []int{1})
	if got, want := root[1], []int{1, 2, 3, 4}; !equalInts(got, want) {
		t.Fatalf("root scores=%v; want %v", got, want)
	}
	children := db.SumChildScoresAtLevel(1, []string{"ta", "__padding__"}, []int{4})
	if got, want := children[4], []int{5, 6, 7, 8}; !equalInts(got, want) {
		t.Fatalf("level-1 scores=%v; want %v", got, want)
	}
}

func TestStage2CuckooProtocolLookup(t *testing.T) {
	directory := t.TempDir()
	vocab := make(map[string]int)
	rows := make([]fixtureRow, 0, 32)
	for term := 0; term < 32; term++ {
		vocab["t"+string(rune('a'+term))] = term
		rows = append(rows, fixtureRow{
			termID: uint32(term), nodeID: uint32(16 + term),
			value: []byte{byte(term + 1), byte(term + 2)},
		})
	}
	dataPath, idmapPath, vocabPath := writeCuckooFixture(t, directory, "stage2", rows, vocab)
	oldStrict := StrictPIR
	StrictPIR = true
	defer func() { StrictPIR = oldStrict }()
	pre, err := NewStage2PrecomputedWithCuckoo(
		dataPath, idmapPath, vocabPath, 4, 2, 10, true, 8, DefaultCuckooBuildConfig(),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer pre.Close()
	got := pre.SumBoundsForTermsWithPIR([]string{"ta", "__padding__"}, []int{16, 17})
	if want := []int{1, 2}; !equalInts(got[16], want) {
		t.Fatalf("leaf 16 scores=%v; want %v", got[16], want)
	}
	if want := []int{0, 0}; !equalInts(got[17], want) {
		t.Fatalf("leaf 17 scores=%v; want %v", got[17], want)
	}
}

func equalInts(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
