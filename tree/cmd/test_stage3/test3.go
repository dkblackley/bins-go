package bm25

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"

	run_tree "github.com/dkblackley/bins-go/tree/cmd"
	"github.com/dkblackley/bins-go/tree/internal/bm25"
	"github.com/sirupsen/logrus"
)

func TestStage3() {
	//docEmbed := flag.String("doc-embed", "../../../../../../datasets/tree/stage3_doc_aligned_shuffle.npy", "Path to document embeddings")
	//docIDMap := flag.String("doc-id-map", "../../../../../../datasets/tree/stage3_doc_aligned.npy.ids", "Path to document ID map")
	//queryEmbed := flag.String("query-embed", "../../../../../../datasets/tree/stage3_query.npy", "Path to query embeddings")

	docEmbed := flag.String("doc-embed", "../../../../../../datasets/Son/my_vectors_192.npy", "Path to document embeddings")
	docIDMap := flag.String("doc-id-map", "../../../../../../datasets/Son/my_vectors_192.npy.ids", "Path to document ID map")
	queryEmbed := flag.String("query-embed", "../../../../../../datasets/Son/query_192_float32.npy", "Path to query embeddings")
	embedDim := flag.Int("embed-dim", 192, "Embedding dimensionality")
	enablePIR := flag.Bool("pir", false, "Enable PIR for private embedding access")
	pirBatchSize := flag.Int64("pir-batch-size", 4096, "Batch size for PIR queries")

	qrels := "../../../../../../datasets/tree/qrels.dev.small.tsv"

	// Load qrels (optional) for evaluation and MRR calculation
	var qrelsMap map[string]map[string]bool
	if qrels != "" {
		qrelsMap = run_tree.LoadQrels(qrels)

		fmt.Printf("DEBUG: loaded %d qids from qrels (%s)\n", len(qrelsMap), qrels)
		if v, ok := qrelsMap["1049085"]; ok {
			fmt.Printf("DEBUG: qrels[1049085] has %d entries\n", len(v))
		} else {
			i := 0
			for k := range qrelsMap {
				if i == 0 {
					fmt.Printf("DEBUG: sample qid from qrels: %s\n", k)
				} else if i < 3 {
					fmt.Printf("DEBUG: another qid: %s\n", k)
				} else {
					break
				}
				i++
			}
		}

	}

	flag.Parse()

	fmt.Println("=== Stage3 Reranker Test ===")
	fmt.Printf("Loading embeddings:\n")
	fmt.Printf("  Doc embeddings: %s\n", *docEmbed)
	fmt.Printf("  Doc ID map: %s\n", *docIDMap)
	fmt.Printf("  Query embeddings: %s\n", *queryEmbed)
	fmt.Printf("  Embedding dim: %d\n", *embedDim)
	fmt.Printf("  Enable PIR: %v\n", *enablePIR)
	fmt.Printf("  PIR batch size: %d\n", *pirBatchSize)

	// Read analyzed queries JSON file
	// data, err := os.ReadFile("../../../datasets/tree/queries.dev.small.analyzed.json")
	data, err := os.ReadFile("../../../../../../datasets/tree/queries.dev.small.analyzed.json")

	if err != nil {
		logrus.Errorf("Error reading analyzed queries file: %v\n", err)
		os.Exit(1)
	}

	var analyzedQueries []run_tree.AnalyzedQuery
	if err := json.Unmarshal(data, &analyzedQueries); err != nil {
		logrus.Errorf("Error parsing analyzed queries JSON: %v\n", err)
		os.Exit(1)
	}
	var stage3Idx = make([]string, len(analyzedQueries))

	queryMap := make(map[string]run_tree.AnalyzedQuery)
	for i, query := range analyzedQueries {
		queryMap[query.ID] = query
		stage3Idx[i] = query.ID
	}
	// Initialize Stage3 reranker
	// sr, err := bm25.NewStage3Reranker(*docEmbed, *docIDMap, *queryEmbed, *embedDim, *enablePIR, uint64(*pirBatchSize))
	// sr, err := bm25.NewStage3Reranker(*docEmbed, *docIDMap, *queryEmbed, "../../../../../../datasets/tree/block_permutations.json", 192, *enablePIR, uint64(32), stage3Idx)
	sr, err := bm25.NewStage3Reranker(*docEmbed, *docIDMap, *queryEmbed, "", 192, *enablePIR, uint64(32), 8, stage3Idx)
	if err != nil {
		fmt.Printf("ERROR: Failed to initialize Stage3 reranker: %v\n", err)
		os.Exit(1)
	}
	defer sr.Close()

	fmt.Println("\n=== Stage3 Successfully Initialized ===")
	fmt.Printf("Number of documents: %d\n", len(sr.DocIDMap))
	fmt.Printf("Embedding dimension: %d\n", sr.EmbedDim)
	fmt.Printf("Bytes per element: %d\n", sr.BytesPerElem)
	fmt.Printf("Number of queries: %d\n", len(sr.QueryEmbeddings))

	// Test 1: Verify document count matches ID map
	if len(sr.DocIDMap) > 0 {
		fmt.Printf("\n=== Test 1: Doc ID Map ===")
		fmt.Printf("First 5 doc IDs:\n")
		for i := 0; i < 5 && i < len(sr.DocIDMap); i++ {
			fmt.Printf("  [%d] %s\n", i, sr.DocIDMap[i])
		}
	}

	// Test 2: Retrieve and check embeddings
	fmt.Printf("\n=== Test 2: Embedding Access ===")
	if len(sr.DocIDMap) > 0 {
		testIdx := 0
		emb := sr.GetDocEmbedding(testIdx)
		fmt.Printf("Doc %d embedding (first 5 values): ", testIdx)
		for i := 0; i < 5 && i < len(emb); i++ {
			fmt.Printf("%.6f ", emb[i])
		}
		fmt.Printf("\n")
	}

	// Test 3: Query embeddings
	fmt.Printf("\n=== Test 3: Query Embeddings ===")
	fmt.Printf("Number of query embeddings loaded: %d\n", len(sr.QueryEmbeddings))
	if len(sr.QueryEmbeddings) > 0 {
		fmt.Printf("Query 0 embedding (first 5 values): ")
		for i := 0; i < 5 && i < len(sr.QueryEmbeddings[0]); i++ {
			fmt.Printf("%.6f ", sr.QueryEmbeddings[0][i])
		}
		fmt.Printf("\n")
	}

	// First: QID, third: Docid
	// 300674 0 7067032 1
	//125705 0 7067056 1
	//94798 0 7067181 1
	//9083 0 7067274 1
	//174249 0 7067348 1
	//320792 0 7067677 1
	//1090270 0 7067796 1
	//1101279 0 7067891 1
	//201376 0 7068066 1
	//54544 0 7068203 1

	var queryToTest [10]string

	//queryToTest[0] = "300674"
	//queryToTest[1] = "125705"
	//queryToTest[2] = "94798"
	//queryToTest[3] = "9083"
	//queryToTest[4] = "174249"
	//queryToTest[5] = "320792"
	//queryToTest[6] = "1090270"
	//queryToTest[7] = "1101279"
	//queryToTest[8] = "201376"
	//queryToTest[9] = "54544"

	queryToTest[0] = "1048585"
	queryToTest[1] = "2"
	queryToTest[2] = "524332"
	queryToTest[3] = "1048642"
	queryToTest[4] = "524447"
	queryToTest[5] = "786674"
	queryToTest[6] = "1048876"
	queryToTest[7] = "1048917"
	queryToTest[8] = "786786"
	queryToTest[9] = "524699"

	for _, qid := range queryToTest {
		qEmbIdx := sr.QueryIDMap[qid]
		var mostSimilar string
		var matchingID int
		maxScore := float32(0)
		var trueGold string
		singleKey, _ := qrelsMap[qid]
		for k := range singleKey {
			trueGold = k
			if !singleKey[k] {
				fmt.Printf("ERROR: qrelsMap[%s] does not contain %s\n", qid, k)
				fmt.Printf("qrelsMap[%s]: %v\n", qid, singleKey)
				os.Exit(1)
			}
			break
		}
		for intDocID, extDocID := range sr.DocIDMap {
			// fmt.Printf("%s %d %s %d\n", qid, intDocID, extDocID, sr.QueryIDMap[qid])
			docEmb := sr.GetDocEmbedding(intDocID)
			queryEmb := sr.QueryEmbeddings[qEmbIdx]
			similarity := bm25.CosineSimilarity(queryEmb, docEmb)
			if similarity > maxScore {
				matchingID = intDocID
				mostSimilar = extDocID
				maxScore = similarity
			}
		}
		fmt.Printf("\n\nMost similar doc for %s: %s (internal docid %d)\n", qid, mostSimilar, matchingID)
		fmt.Printf("Score: %.6f\n", maxScore)

		i, _ := strconv.Atoi(trueGold)
		fmt.Printf("True gold doc for %s: %s (or %d)\n", qid, trueGold, i)
		docEmb := sr.GetDocEmbedding(i)
		queryEmb := sr.QueryEmbeddings[qEmbIdx]
		similarity := bm25.CosineSimilarity(queryEmb, docEmb)
		fmt.Printf("Score for GOLD DOC INTERNAL: %.6f\n", similarity)

		externalIDGold := sr.DocIDMap[i]
		i, _ = strconv.Atoi(externalIDGold)
		fmt.Printf("True EXTERNAL gold doc for %s: %s (or %d)\n", qid, externalIDGold, i)
		docEmb = sr.GetDocEmbedding(i)
		queryEmb = sr.QueryEmbeddings[qEmbIdx]
		similarity = bm25.CosineSimilarity(queryEmb, docEmb)
		fmt.Printf("Score for GOLD DOC EXTERNAL: %.6f\n", similarity)

	}

	//os.Exit(1)
	//
	//// Test 4: Cosine similarity calculation
	//fmt.Printf("\n=== Test 4: Cosine Similarity ===")
	//if len(sr.QueryEmbeddings) > 0 && len(sr.DocIDMap) > 0 {
	//	qEmbIdx := sr.QueryIDMap["300674"] // should be docid 7067032
	//	v, _ := qrelsMap["300674"]
	//	fmt.Printf("v: %v\n", v)
	//	if !v["7067032"] {
	//		fmt.Printf("ERROR: qrelsMap[300674] does not contain 7067032\n")
	//		os.Exit(1)
	//	}
	//	queryEmb := sr.QueryEmbeddings[qEmbIdx]
	//	docEmb := sr.GetDocEmbedding(7067032)
	//	similarity := bm25.CosineSimilarity(queryEmb, docEmb)
	//	fmt.Printf("Cosine similarity between query 300674 and doc 7067032: %.6f\n", similarity)
	//
	//	// Test a few more documents
	//	fmt.Printf("Similarities for doc 0-4 with query 0:\n")
	//	for i := 0; i < 5 && i < len(sr.DocIDMap); i++ {
	//		docEmb := sr.GetDocEmbedding(i)
	//		sim := bm25.CosineSimilarity(queryEmb, docEmb)
	//		fmt.Printf("  Doc %d: %.6f\n", i, sim)
	//	}
	//}
	//
	//// Test 5: Batch retrieval
	//fmt.Printf("\n=== Test 5: Batch Embedding Retrieval ===")
	//if len(sr.DocIDMap) >= 10 {
	//	docIndices := []int{0, 1, 2, 5, 10}
	//	embeddings := sr.GetDocEmbeddingBatch(docIndices)
	//	fmt.Printf("Retrieved %d embeddings in batch\n", len(embeddings))
	//	for _, idx := range docIndices {
	//		if emb, ok := embeddings[idx]; ok {
	//			fmt.Printf("  Doc %d: retrieved OK (first value: %.6f)\n", idx, emb[0])
	//		} else {
	//			fmt.Printf("  Doc %d: FAILED to retrieve\n", idx)
	//		}
	//	}
	//}

	fmt.Println("\n=== All Tests Completed Successfully ===")
}
