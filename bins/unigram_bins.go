// unigram_simple.go
// Minimal testing helper: treat every vocab term as a unigram (n=1),
// and either (A) build a simple token->docs lookup using single-term BM25,
// or (B) assign each token to D hash bins without any scoring.
//
// Drop this next to your existing files (package main). If you already
// declared the BM25 interface or NgramIndex elsewhere, delete the duplicates.

package bins

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/blugelabs/bluge"
	"github.com/blugelabs/bluge/analysis"
	"github.com/blugelabs/bluge/analysis/char"
	"github.com/blugelabs/bluge/analysis/lang/en"
	"github.com/blugelabs/bluge/analysis/token"
	"github.com/blugelabs/bluge/analysis/tokenizer"
	"github.com/dkblackley/bins-go/globals"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

type Config struct {
	K         uint
	D         uint
	MaxBins   uint
	Filenames bool
	Threshold uint
}

func doBM25Search(queries []string, path_to_corpus string) {

}

func strictEnglishAnalyzer() *analysis.Analyzer {
	return &analysis.Analyzer{
		// Optional: normalize punctuation BEFORE tokenizing (e.g., turn periods/commas into spaces)
		CharFilters: []analysis.CharFilter{
			char.NewRegexpCharFilter(regexp.MustCompile(`[.,]+`), []byte(" ")),
		},
		// Critical: letters-only tokenizer (drops digits/punct)
		Tokenizer: tokenizer.NewLetterTokenizer(),
		TokenFilters: []analysis.TokenFilter{
			en.NewPossessiveFilter(),
			token.NewLowerCaseFilter(),
			token.NewStopTokensFilter(en.StopWords()),
			en.StemmerFilter(),
			token.NewLengthFilter(2, 40), // tune min/max token length
		},
	}
}

// TODO: Replace bluge.reader with a generic implements
func MakeUnigramDB(reader *bluge.Reader, dataset globals.DatasetMetadata, config *globals.Args) [][]string {

	//tokeniser := en.NewAnalyzer()

	// tokeniser := strictEnglishAnalyzer()

	//logrus.Info("Making Unigram Database")
	//queries, er := LoadQueries(dataset.Queries)
	//Must(er)
	//qrels, er := LoadQrels(dataset.Qrels)
	// Must(er)
	docs, er := LoadCorpus(dataset.OriginalDir)
	Must(er)

	bar := progressbar.Default(int64(len(docs)), fmt.Sprintf("Scanning Vocab for %s", dataset.Name))

	// No sets in go, gotta make my own...
	set := make(map[string]struct{})

	nw := runtime.GOMAXPROCS(0)
	local := make([]map[string]struct{}, nw)
	chunk := (len(docs) + nw - 1) / nw
	var vwg sync.WaitGroup
	for w := 0; w < nw; w++ {
		lo, hi := min(w*chunk, len(docs)), min((w+1)*chunk, len(docs))
		vwg.Add(1)
		go func(w, lo, hi int) {
			defer vwg.Done()
			an := strictEnglishAnalyzer()
			m := make(map[string]struct{})
			for _, doc := range docs[lo:hi] {
				for _, t := range an.Analyze([]byte(doc.Title + " " + doc.Text)) {
					m[string(t.Term)] = struct{}{}
				}
				bar.Add(1)
			}
			local[w] = m
		}(w, lo, hi)
	}
	vwg.Wait()
	for _, m := range local {
		for k := range m {
			set[k] = struct{}{}
		}
	}
	total_items_in_set := len(set)

	bar.Finish()

	logrus.Infof("Total items in vocab: %d", total_items_in_set)
	config.Metadata["VocabSize"] = strconv.Itoa(total_items_in_set)

	// realBinSize := uint(float64(total_items_in_set) * config.BinSize)
	realBinSize := config.BinSize

	logrus.Infof("Size of/number of bins: %d ", realBinSize)
	config.Metadata["RealBinSize"] = strconv.Itoa(int(realBinSize))

	//// Very 'hacky' a mapping to a 'set' which is a mapping to globals. Is converted into a regular bin at the end.
	//setsBins := make(map[uint]map[string]struct{})
	// Can't use the set because round robin requires structure and set is unorganised
	binsOrder := make(map[uint][]string)
	// bin -> how many unigrams have landed here. Drives the round-robin stride.
	binHits := make(map[uint]uint)

	bar = progressbar.Default(int64(total_items_in_set), fmt.Sprintf("Putting items into bins %s", dataset.Name))

	words := make([]string, 0, len(set))
	for w := range set {
		words = append(words, w)
	}
	sort.Strings(words)
	set = nil
	docs = nil

	// Search the words in chunks (in parallel) and merge each chunk into the bins (in word order) before searching the
	// next, so only one chunk's top docs are in memory at once instead of the whole vocabulary's (V x docsPerBin doc
	// IDs: ~33 GB on nq at docsPerBin 2500). The merge order is the same as before, so the bins are identical.
	const searchChunk = 1 << 16
	topDocs := make([][]string, min(searchChunk, len(words)))
	for lo := 0; lo < len(words); lo += searchChunk {
		hi := min(lo+searchChunk, len(words))
		var next atomic.Int64
		next.Store(int64(lo))
		var wg sync.WaitGroup
		for w := 0; w < runtime.GOMAXPROCS(0); w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					i := int(next.Add(1) - 1)
					if i >= hi {
						return
					}
					topDocs[i-lo] = searchTopDocs(reader, words[i], int(config.DocsPerBin))
					bar.Add(1)
				}
			}()
		}
		wg.Wait()

		for i := lo; i < hi; i++ {
			for d := uint(0); d <= config.DChoice; d++ {
				var bin_index = hashTokenChoice(words[i], d)
				mergeIntoBin(binsOrder, binHits, uint(bin_index)%realBinSize, topDocs[i-lo], config.DocsPerBin)
			}
			topDocs[i-lo] = nil
		}
	}
	topDocs = nil

	bar.Finish()
	binsSlice := make([][]string, realBinSize)
	for bin, ids := range binsOrder {
		binsSlice[int(bin)] = ids
	}

	//for bin, set := range setsBins {
	//	idx := int(bin)
	//
	//	// Pre-size capacity to avoid re-allocs while appending
	//	binsSlice[idx] = make([]string, 0, len(set))
	//	for w := range set {
	//		binsSlice[idx] = append(binsSlice[idx], w)
	//	}
	//}

	return binsSlice

}

// binsAnalyzer is the analyzer of the bins' Bluge index (title and body, see index_build.go) and of raw query text
// searched in it: the same strictEnglishAnalyzer the vocabulary and the query words are made with. It used to be
// Bluge's standard analyzer (lowercase only), so a vocabulary stem that isn't itself a word ("studi", "dog" from
// "dogs") found no documents and its bin stayed empty. Its tokenizer and filters keep no state, so one is shared.
var binsAnalyzer = strictEnglishAnalyzer()

// binsTextField is a title / body field of the bins index
func binsTextField(name, value string) *bluge.TermField {
	return bluge.NewTextField(name, value).WithAnalyzer(binsAnalyzer)
}

// binsTextQuery matches raw text (a query) against title and body, analyzed like the index
func binsTextQuery(text string) bluge.Query {
	return bluge.NewBooleanQuery().
		AddShould(bluge.NewMatchQuery(text).SetField("title").SetAnalyzer(binsAnalyzer)).
		AddShould(bluge.NewMatchQuery(text).SetField("body").SetAnalyzer(binsAnalyzer))
}

// searchTopDocs is the top n docs of one vocabulary word. The word is already an analyzed term (a stem), so it is
// looked up as is: analyzing it again would stem the stem.
func searchTopDocs(reader *bluge.Reader, word string, n int) []string {
	boolean := bluge.NewBooleanQuery().
		AddShould(bluge.NewTermQuery(word).SetField("title")).
		AddShould(bluge.NewTermQuery(word).SetField("body"))

	it, err := reader.Search(context.Background(), bluge.NewTopNSearch(n, boolean))
	Must(err)

	// Not make(.., 0, n): most words match far fewer than n docs, and n slots per word for the whole vocabulary was
	// most of the memory at large docsPerBin. Grow as needed, then return an exact-size copy.
	doc_ids := make([]string, 0, min(n, 16))
	for {
		match, err := it.Next()
		if err != nil || match == nil {
			break
		}
		var docID string
		Must(match.VisitStoredFields(func(field string, value []byte) bool {
			if field == "_id" {
				docID = string(value)
				return false
			}
			return true
		}))
		doc_ids = append(doc_ids, docID)
	}
	if cap(doc_ids) > len(doc_ids) {
		doc_ids = append(make([]string, 0, len(doc_ids)), doc_ids...)
	}
	return doc_ids
}

// mergeIntoBin interleaves incoming (BM25 rank order) with whatever is already
// in the bin
func mergeIntoBin(binsOrder map[uint][]string, binHits map[uint]uint, bin uint, incoming []string, threshold uint) {
	binHits[bin]++
	keep := binHits[bin] - 1 // existing docs kept between each incoming doc

	existing := binsOrder[bin]

	seen := make(map[string]struct{}, len(existing)+len(incoming))
	// at most threshold docs, and never more than the two inputs hold (with millions of bins, a full threshold-sized
	// array per bin was GBs of empty slots)
	merged := make([]string, 0, min(threshold, uint(len(existing)+len(incoming))))

	push := func(id string) {
		if _, dup := seen[id]; dup {
			return
		}
		seen[id] = struct{}{}
		merged = append(merged, id)
	}

	i, j := 0, 0
	for uint(len(merged)) < threshold && (i < len(existing) || j < len(incoming)) {
		for k := uint(0); k < keep && i < len(existing) && uint(len(merged)) < threshold; k++ {
			push(existing[i])
			i++
		}
		if j < len(incoming) {
			push(incoming[j])
			j++
		} else if i < len(existing) { // incoming exhausted, drain the rest
			push(existing[i])
			i++
		} else {
			break
		}
	}

	binsOrder[bin] = merged
}

//func add(sets map[uint]map[string]struct{}, bin uint, word string) {
//	if sets[bin] == nil {
//		sets[bin] = make(map[string]struct{})
//	}
//	sets[bin][word] = struct{}{}
//}

func hashTokenChoice(tokens string, i uint) uint64 {
	// Join all strings into a single byte sequence
	// joined := strings.Join(tokens, "|")
	data := []byte(tokens)

	// Append integer i in big-endian form
	var buf [4]byte
	binary.BigEndian.PutUint32(buf[:], uint32(i))
	data = append(data, buf[:]...)

	// Hash with SHA-256
	sum := sha256.Sum256(data)

	// Take the first 8 bytes as uint64
	return binary.BigEndian.Uint64(sum[0:8])
}
