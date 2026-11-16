package bins

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/dkblackley/bins-go/globals"
)

// A tiny corpus end to end: build the text bins, query, decode, BM25 re-rank.
func TestTextBinsEndToEnd(t *testing.T) {
	dir := t.TempDir()
	topics := []string{"volcano eruption lava magma", "football goal striker league", "piano sonata composer orchestra",
		"python compiler programming language", "bushfire smoke evacuation highway", "glacier ice melting climate"}
	corpus, _ := os.Create(filepath.Join(dir, "corpus.jsonl"))
	queries, _ := os.Create(filepath.Join(dir, "queries.jsonl"))
	qrels, _ := os.Create(filepath.Join(dir, "qrels.tsv"))
	n := 0
	for rep := 0; rep < 10; rep++ {
		for ti, topic := range topics {
			id := fmt.Sprintf("d%d", n)
			text := fmt.Sprintf("The %s story number %d was written about the %s.", topic, rep, topic)
			b, _ := json.Marshal(map[string]string{"_id": id, "title": "", "text": text})
			fmt.Fprintln(corpus, string(b))
			if rep == 0 {
				b, _ = json.Marshal(map[string]string{"_id": fmt.Sprintf("q%d", ti), "text": topic})
				fmt.Fprintln(queries, string(b))
				fmt.Fprintf(qrels, "q%d\t%s\t1\n", ti, id)
			}
			n++
		}
	}
	corpus.Close()
	queries.Close()
	qrels.Close()
	Must(BuildBlugeIndexFromJSONL(filepath.Join(dir, "corpus.jsonl"), filepath.Join(dir, "index")))

	config := &globals.Args{
		DBSize: uint(n), BinSize: 200, DocsPerBin: 20, DChoice: 1, K: 3,
		Metadata: map[string]string{},
		DatasetMeta: globals.DatasetMetadata{Name: "tiny", IndexDir: filepath.Join(dir, "index"),
			OriginalDir: filepath.Join(dir, "corpus.jsonl"), Queries: filepath.Join(dir, "queries.jsonl"),
			Qrels: filepath.Join(dir, "qrels.tsv")},
		Text: globals.TextConf{Enabled: true, Format: "bow", Order: 2, IDFBits: 12, TrainDocs: 30, K1: 0.9, B: 0.4},
	}
	db := MakeTextDb(config)
	db.Preprocess()
	config.DocIDMapPacmann, _ = MakeDocIDAndQueryIDMap(config.DatasetMeta)

	for ti := range topics {
		res, err := db.DoSearch(fmt.Sprintf("q%d", ti), 3)
		if err != nil {
			t.Fatal(err)
		}
		te := res.(*TextEntry)
		if got := len(te.Decode(config)); got < 10 {
			t.Fatalf("q%d: decoded only %d docs, want all 10 on the topic", ti, got)
		}
		for _, id := range te.Rank(config, 3) {
			var idx int
			fmt.Sscanf(id, "d%d", &idx)
			if idx%len(topics) != ti {
				t.Fatalf("q%d: top doc %s is off topic", ti, id)
			}
		}
	}
	if db.GetMetaData()["TextBytesPerDoc"] == "" {
		t.Fatal("missing text stats in metadata")
	}
}
