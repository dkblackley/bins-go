package bm25

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadVocabularyAcceptsStringAndNumericIDs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vocab.json")
	if err := os.WriteFile(path, []byte(`{"alpha":"0","beta":1,"gamma":"2660823"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := loadVocabulary(path)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]int{"alpha": 0, "beta": 1, "gamma": 2_660_823}
	for term, termID := range want {
		if got[term] != termID {
			t.Fatalf("vocabulary[%q]=%d; want %d", term, got[term], termID)
		}
	}
}

func TestLoadVocabularyRejectsNonDecimalID(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vocab.json")
	if err := os.WriteFile(path, []byte(`{"alpha":"not-an-id"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadVocabulary(path); err == nil {
		t.Fatal("loadVocabulary accepted a non-decimal term ID")
	}
}
