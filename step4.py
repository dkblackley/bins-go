import argparse
from collections import defaultdict
import time
from typing import List, Set
import torch
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
def load_qrels(qrels_path):
    qrels = {}
    first = True
    with open(qrels_path, 'r') as f:
        for line in f:
            if first:
                first = False
                continue
            qid, docid, rel = line.strip().split()
            if int(rel) > 0:
                if qid not in qrels:
                    qrels[qid] = set()
                qrels[qid].add(docid)
    return qrels

def load_output(step3_output_path):
    output = defaultdict(list)
    with open(step3_output_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            qid = parts[0]
            doc_id = parts[1]  # Adjust based on actual format
            output[qid].append(doc_id)
    return output

def load_embeddings(embeddings_path, id_path):
    # embedding in npy format
    x =  np.load(embeddings_path, mmap_mode='r')

    with open(id_path, 'r') as fid:
        idmap = np.array([int(line.strip()) for line in fid])
    return x, idmap

def calculate_mrr_at_k(reranked_doc_ids: List[str], gold_doc_ids: Set[str], k: int) -> float:
    """Calculates the reciprocal rank for a single query."""
    for i, docid in enumerate(reranked_doc_ids[:k]):
        rank = i + 1
        if docid in gold_doc_ids:
            return 1.0 / rank
    return 0.0

def load_queries(queries_path):
    queries = {}
    with open(queries_path, 'r') as f:
        for line in f:
            qid, qtext = line.strip().split('\t', 1)
            queries[qid] = qtext
    return queries

def load_queries_embeddings(queries_emb_path):# queries_id_path):
    x =  np.load(queries_emb_path, mmap_mode='r')

    # with open(queries_id_path, 'r') as fid:
    #     idmap = np.array([line.strip() for line in fid])
    return x, None

def load_documents(documents_path):
    # documents_path is collection.tsv
    documents = {}
    with open(documents_path, 'r') as f:
        for line in f:
            docid, doctext = line.strip().split('\t', 1)
            documents[docid] = doctext
    return documents

def main(args):
    qrels = load_qrels(args.qrels)
    queries = load_queries(args.queries)
    documents = load_documents(args.documents)
    step3 = load_output(args.step3_output)
    print(len(documents))
    # preview documents:
    for docid, doctext in list(documents.items())[:5]:
        print(f"DocID: {docid}\nText: {doctext}\n")


    recalls = []
    for qid, docids in step3.items():
        if qid in qrels:
            predicts = set(docids)
            bl = qrels.get(qid, set())
            gold = {did for did in bl}
            inter = len(gold & predicts)
            rec = inter / len(gold)
            recalls.append(rec)
    if recalls:
        arr = np.array(recalls)
        print(f"Recall@k: mean={arr.mean():.4f} median={np.median(arr):.4f} p90={np.quantile(arr, 0.9):.4f}")
    else:
        print("No queries with relevant documents were evaluated.")

    # step 4: load embeddings and rerank
    # pca = faiss.read_VectorTransform("pca_768_to_192.faiss")
    query_embeddings, _ = load_queries_embeddings('../datasets/Son/query_192_float32.npy')#, 'queries_768.npy.ids')
    mrrs = []
    document_embeddings, idmap = load_embeddings('../datasets/Son/my_vectors_192_float32.npy', '../datasets/Son/my_vectors_768.npy.ids')
    print(document_embeddings.shape, idmap.shape)
    database = {}
    for id, embedding in tqdm(zip(idmap, document_embeddings), total=len(idmap)):
        database[str(id)] = embedding
    i = 0
    for qid, docids in tqdm(step3.items()):
        # print(docids,qid)
        gold = qrels.get(qid, set())
        if not gold:
            continue
        # t = time.time()
        topk_embeddings = [database[docid] for docid in docids if docid in database]
        # print("Time taken to fetch doc embeddings:", time.time()-t)
        if not topk_embeddings:
            assert False, f"No embeddings found for docids: {docids}"
        # sort topk embeddings by cosine similarity to query embedding
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3', device=device)
        #query_embedding = model.encode([queries[qid]], convert_to_numpy=True)[0]
        # pca transform
        #query_embedding = pca.apply_py(query_embedding.reshape(1, -1))[0]
        query_embedding = query_embeddings[i]#np.where(query_idmap == qid)[0][0]]
        i+=1
        # compute cosine similarity
        # t = time.time()
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
        topk_embeddings_norm = [emb / np.linalg.norm(emb) for emb in topk_embeddings]
        sims = [np.dot(query_embedding_norm, emb_norm) for emb_norm in topk_embeddings_norm]
        # rerank docids by sims
        scored_docids = list(zip(docids, sims))
        scored_docids.sort(key=lambda x: x[1], reverse=True)
        reranked_docids = [docid for docid, score in scored_docids]
        step3[qid] = reranked_docids  # update with reranked
        # print("Time taken to rerank:", time.time()-t)
        mrr = calculate_mrr_at_k(reranked_docids, gold, k=args.k)
        mrrs.append(mrr)
    print(f"Mean MRR@{args.k}: {np.mean(mrrs):.4f}")

    # save reranked results
    with open('step4_reranked_output.tsv', 'w') as f:
        for qid, docids in step3.items():
            for rank, docid in enumerate(docids):
                f.write(f"{qid}\t{docid}\t{rank+1}\n")
    if mrrs:
        arr = np.array(mrrs)
        print(f"MRR@10: mean={arr.mean():.4f} median={np.median(arr):.4f} p90={np.quantile(arr, 0.9):.4f}")
    else:
        print("No queries with relevant documents were evaluated for MRR.")

if __name__=="__main__":
    ap = argparse.ArgumentParser(description='Optimized Cache‑free PP top‑k simulation on MS MARCO')
    ap.add_argument('--qrels', type=str, required=True, help='Path to qrels file')
    ap.add_argument('--step3-output', type=str, required=True, help='Path to step3 output file')
    ap.add_argument('--queries', type=str, required=True, help='Path to queries file')
    ap.add_argument('--documents', type=str, required=True, help='Path to documents file')
    ap.add_argument('--k', type=int, default=10, help='Value of k for MRR@k')
    main(ap.parse_args())