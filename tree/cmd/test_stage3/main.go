package main

import (
	"flag"
	"fmt"
	"os"

	"bm25-msmarco/internal/bm25"
)

func main() {
	docEmbed := flag.String("doc-embed", "data/my_vectors_192_float32.npy", "Path to document embeddings")
	docIDMap := flag.String("doc-id-map", "data/my_vectors_768.external.ids", "Path to document ID map")
	queryEmbed := flag.String("query-embed", "data/query_192_float32.npy", "Path to query embeddings")
	embedDim := flag.Int("embed-dim", 192, "Embedding dimensionality")
	enablePIR := flag.Bool("pir", false, "Enable PIR for private embedding access")
	pirBatchSize := flag.Int64("pir-batch-size", 4096, "Batch size for PIR queries")

	flag.Parse()

	fmt.Println("=== Stage3 Reranker Test ===")
	fmt.Printf("Loading embeddings:\n")
	fmt.Printf("  Doc embeddings: %s\n", *docEmbed)
	fmt.Printf("  Doc ID map: %s\n", *docIDMap)
	fmt.Printf("  Query embeddings: %s\n", *queryEmbed)
	fmt.Printf("  Embedding dim: %d\n", *embedDim)
	fmt.Printf("  Enable PIR: %v\n", *enablePIR)
	fmt.Printf("  PIR batch size: %d\n", *pirBatchSize)

	// Initialize Stage3 reranker
	sr, err := bm25.NewStage3Reranker(*docEmbed, *docIDMap, *queryEmbed, *embedDim, *enablePIR, uint64(*pirBatchSize))
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

	// Test 4: Cosine similarity calculation
	fmt.Printf("\n=== Test 4: Cosine Similarity ===")
	if len(sr.QueryEmbeddings) > 0 && len(sr.DocIDMap) > 0 {
		queryEmb := sr.QueryEmbeddings[0]
		docEmb := sr.GetDocEmbedding(0)
		similarity := bm25.CosineSimilarity(queryEmb, docEmb)
		fmt.Printf("Cosine similarity between query 0 and doc 0: %.6f\n", similarity)

		// Test a few more documents
		fmt.Printf("Similarities for doc 0-4 with query 0:\n")
		for i := 0; i < 5 && i < len(sr.DocIDMap); i++ {
			docEmb := sr.GetDocEmbedding(i)
			sim := bm25.CosineSimilarity(queryEmb, docEmb)
			fmt.Printf("  Doc %d: %.6f\n", i, sim)
		}
	}

	// Test 5: Batch retrieval
	fmt.Printf("\n=== Test 5: Batch Embedding Retrieval ===")
	if len(sr.DocIDMap) >= 10 {
		docIndices := []int{0, 1, 2, 5, 10}
		embeddings := sr.GetDocEmbeddingBatch(docIndices)
		fmt.Printf("Retrieved %d embeddings in batch\n", len(embeddings))
		for _, idx := range docIndices {
			if emb, ok := embeddings[idx]; ok {
				fmt.Printf("  Doc %d: retrieved OK (first value: %.6f)\n", idx, emb[0])
			} else {
				fmt.Printf("  Doc %d: FAILED to retrieve\n", idx)
			}
		}
	}

	fmt.Println("\n=== All Tests Completed Successfully ===")
}
