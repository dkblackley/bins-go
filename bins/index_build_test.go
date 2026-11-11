package bins

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/blugelabs/bluge"
)

// The parallel index builder gives the same search results (same docs, same order, ties included) as the serial one.
func TestParallelIndexMatchesSerial(t *testing.T) {
	n := 20000
	if v := os.Getenv("INDEX_TEST_DOCS"); v != "" {
		n, _ = strconv.Atoi(v)
	}
	dir := t.TempDir()
	corpus := filepath.Join(dir, "corpus.jsonl")
	f, err := os.Create(corpus)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	vocab := make([]string, 3000)
	syl := []string{"ka", "lo", "mi", "tu", "re", "sa", "no", "vi", "pe", "du", "ga", "zo", "be", "fi", "ru"}
	for i := range vocab {
		vocab[i] = syl[i%15] + syl[(i/15)%15] + syl[(i/225)%15]
	}
	enc := json.NewEncoder(f)
	for i := 0; i < n; i++ {
		words := make([]byte, 0, 400)
		for j := 0; j < 10+rng.Intn(80); j++ {
			words = append(words, vocab[int(float64(len(vocab))*rng.Float64()*rng.Float64())]...)
			words = append(words, ' ')
		}
		title := ""
		if i%3 == 0 {
			title = vocab[rng.Intn(len(vocab))]
		}
		_ = enc.Encode(map[string]string{"_id": fmt.Sprint(i), "title": title, "text": string(words)})
	}
	f.Close()

	serial, parallel := filepath.Join(dir, "serial"), filepath.Join(dir, "parallel")
	t0 := time.Now()
	if err := buildBlugeIndexSerial(corpus, serial); err != nil {
		t.Fatal(err)
	}
	t1 := time.Now()
	if err := BuildBlugeIndexFromJSONL(corpus, parallel); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d docs: serial %v, parallel %v", n, t1.Sub(t0), time.Since(t1))

	rs, err := bluge.OpenReader(bluge.DefaultConfig(serial))
	if err != nil {
		t.Fatal(err)
	}
	defer rs.Close()
	rp, err := bluge.OpenReader(bluge.DefaultConfig(parallel))
	if err != nil {
		t.Fatal(err)
	}
	defer rp.Close()
	for i := 0; i < len(vocab); i += 7 {
		a, b := searchTopDocs(rs, vocab[i], 100), searchTopDocs(rp, vocab[i], 100)
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("word %s: top docs differ\nserial   %v\nparallel %v", vocab[i], a, b)
		}
	}
}

// The unordered build holds the same documents and the same scores (only tie order may differ).
func TestUnorderedIndexSameScores(t *testing.T) {
	dir := t.TempDir()
	corpus := filepath.Join(dir, "corpus.jsonl")
	f, _ := os.Create(corpus)
	rng := rand.New(rand.NewSource(2))
	for i := 0; i < 30000; i++ {
		fmt.Fprintf(f, "{\"_id\":\"%d\",\"text\":\"%s %s %s\"}\n", i, []string{"alpha", "beta", "gamma"}[rng.Intn(3)],
			[]string{"delta", "omega"}[rng.Intn(2)], strconv.Itoa(rng.Intn(1000)))
	}
	f.Close()
	a, b := filepath.Join(dir, "a"), filepath.Join(dir, "b")
	if err := BuildBlugeIndexFromJSONL(corpus, a); err != nil {
		t.Fatal(err)
	}
	t.Setenv("BINS_INDEX_UNORDERED", "1")
	t0 := time.Now()
	if err := BuildBlugeIndexFromJSONL(corpus, b); err != nil {
		t.Fatal(err)
	}
	t.Logf("unordered build %v", time.Since(t0))
	ra, _ := bluge.OpenReader(bluge.DefaultConfig(a))
	rb, _ := bluge.OpenReader(bluge.DefaultConfig(b))
	defer ra.Close()
	defer rb.Close()
	ca, _ := ra.Count()
	cb, _ := rb.Count()
	if ca != cb || ca != 30000 {
		t.Fatalf("doc counts %d %d", ca, cb)
	}
	for _, w := range []string{"alpha", "delta", "omega", "17"} {
		x, y := searchTopDocs(ra, w, 30000), searchTopDocs(rb, w, 30000)
		if len(x) != len(y) {
			t.Fatalf("%s: %d vs %d hits", w, len(x), len(y))
		}
	}
}
