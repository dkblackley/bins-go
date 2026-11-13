package bins

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/blugelabs/bluge"
)

// The bins vocabulary is made of strictEnglishAnalyzer terms (Porter stems: "studies" -> "studi"), and each word's bin
// is filled with searchTopDocs(word). So the index has to hold the same stemmed terms, or a stem that isn't itself a
// surface word ("studi", "happi", "comput") finds no documents and its bin stays empty.
func TestBinsIndexMatchesStemmedVocab(t *testing.T) {
	dir := t.TempDir()
	texts := []string{
		"Recent studies of happy computers",
		"The computer was studied by happier students",
		"Generally speaking, generations of running dogs",
	}
	f, _ := os.Create(filepath.Join(dir, "corpus.jsonl"))
	for i, s := range texts {
		b, _ := json.Marshal(map[string]string{"_id": fmt.Sprintf("d%d", i), "title": "", "text": s})
		fmt.Fprintln(f, string(b))
	}
	f.Close()
	Must(BuildBlugeIndexFromJSONL(filepath.Join(dir, "corpus.jsonl"), filepath.Join(dir, "index")))
	reader, err := bluge.OpenReader(bluge.DefaultConfig(filepath.Join(dir, "index")))
	Must(err)
	defer reader.Close()

	// every vocab term, made exactly as MakeUnigramDB makes it, must find the documents it came from
	an := strictEnglishAnalyzer()
	for i, s := range texts {
		for _, tok := range an.Analyze([]byte(" " + s)) {
			word := string(tok.Term)
			found := false
			for _, id := range searchTopDocs(reader, word, 10) {
				found = found || id == fmt.Sprintf("d%d", i)
			}
			if !found {
				t.Errorf("vocab term %q (from d%d) does not find d%d in the bins index", word, i, i)
			}
		}
	}

	// raw query text (the plaintext BM25 path in bins_util.go) must still match after the change
	for _, q := range []string{"studies", "Computers", "happiness"} {
		it, err := reader.Search(t.Context(), bluge.NewTopNSearch(10, binsTextQuery(q)))
		Must(err)
		if m, _ := it.Next(); m == nil {
			t.Errorf("raw query %q matches nothing", q)
		}
	}
}
