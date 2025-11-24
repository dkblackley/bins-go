// debug_scifact_qrels.go
package bins

import (
	"context"
	"github.com/blugelabs/bluge"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

// debugScifactFull prints:
//   - total queries vs. total qrels entries
//   - a sample of qrels keys and their doc-counts
//   - for the first N queries that *do* have qrels:
//     – the Query text
//     – the list of relevant docs
//     – the top-K retrieved docs with scores and rel-flags
func debugScifactFull(
	idxPath, queriesPath, qrelsPath string,
	sampleQ, topK int,
) {
	// load everything
	qs, err := LoadQueries(queriesPath)
	Must(err)
	rels, err := loadQrels(qrelsPath)
	Must(err)

	// summary counts
	logrus.Debugf("Loaded %d total queries, %d queries with qrels\n\n",
		len(qs), len(rels))

	// sample some qrels keys
	logrus.Debugf("Sample qrels entries:")
	i := 0
	for qid, docs := range rels {
		logrus.Debugf("  • QID=%-5s → %3d relevant docs\n", qid, len(docs))
		i++
		if i >= 5 {
			break
		}
	}
	logrus.Debugf("\n")

	// evaluator setup
	reader, err := bluge.OpenReader(bluge.DefaultConfig(idxPath))
	Must(err)
	defer reader.Close()

	bar := progressbar.Default(int64(sampleQ), "debug")

	printed := 0
	for _, q := range qs {
		if printed >= sampleQ {
			break
		}
		docs, has := rels[q.ID]
		if !has {
			continue
		}
		printed++

		// show the Query and its ground-truth
		logrus.Debugf("=== Query #%d: ID=%s\n  \"%s\"\n", printed, q.ID, q.Text)
		logrus.Debugf("  Relevant docs: %v\n", mapKeys(docs))

		// run the same BM25 search
		boolean := bluge.NewBooleanQuery().
			AddShould(bluge.NewMatchQuery(q.Text).SetField("title")).
			AddShould(bluge.NewMatchQuery(q.Text).SetField("body"))
		req := bluge.NewTopNSearch(topK, boolean)
		it, _ := reader.Search(context.Background(), req)

		logrus.Debugf("  Top retrieved:")
		for rank := 1; rank <= topK; rank++ {
			match, err := it.Next()
			if err != nil || match == nil {
				break
			}
			// extract the stored "_id" field
			var docID string
			Must(match.VisitStoredFields(func(f string, v []byte) bool {
				if f == "_id" {
					docID = string(v)
					return false
				}
				return true
			}))
			isRel := docs[docID] > 0
			logrus.Debugf("    %2d. %-12s  score=%.4f  rel=%t\n",
				rank, docID, match.Score, isRel)
		}
		logrus.Debugf("\n")
		bar.Add(1)
	}
}

// mapKeys extracts the keys from a map[string]int
func mapKeys(m map[string]int) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	return ks
}
