# Datasets

All standard retrieval benchmarks with relevance judgements, so MRR and Recall are computed the same way on each.

| `-name` | Docs | Queries | Words/query |
|---|---|---|---|
| `scifact` | 5,183 | 300 | 12.4 |
| `nq` | 2,681,468 | 3,452 | 9.2 |
| `dbpedia-entity` | 4,635,922 | 400 | 5.4 |
| `fever` | 5,416,568 | 6,666 | 8.1 |
| `msmarco` (v1 passage) | 8,841,823 | 6,980 (dev small) | 6.0 |
| `msmarco-v2-25m` / `-50m` / `-100m` | 25M / 50M / 100M | 3,903 | ~6 |
| `msmarco-v2` (passage) | 138,364,198 | 3,903 (dev1) | ~6 |
| `miracl-en` | 32,893,221 | 799 (dev) | short |

The MS MARCO v2 subsamples keep every document that is relevant to some query and fill the rest at random (seed 42),
so scores stay comparable across sizes. `python datasets/query_stats.py <dataset dir>` prints the query-length
distribution.

## Preparing a dataset

`local_run.sh` (repo root) runs these steps for every dataset in `DATASETS`; this is what each one does.

```bash
bash datasets/download.sh nq ../datasets/raw
python datasets/prepare_dataset.py --format beir --src ../datasets/raw/nq --out ../datasets/nq --split test
python collection_preproc.py --documents ../datasets/nq/corpus.jsonl --output ../datasets/nq/nq_corpus_embed_f32.npy \
    --pca ../datasets/nq/pca_e5_768_to_192.faiss --num-docs "$(cat ../datasets/nq/num_docs.txt)"
python query_preproc.py --queries ../datasets/nq/queries.jsonl --output ../datasets/nq/nq_query_embed_f32.npy \
    --pca ../datasets/nq/pca_e5_768_to_192.faiss
./pir_app -t index -name nq -dataset ../datasets          # keyword index for PILLAR-Bin
bash datasets/prep_tree.sh nq                             # PILLAR-Tree artifacts for every (B, r, s) of the grid
```

Layout under the datasets folder:

```
raw/<name>/                         downloads
<name>/corpus.jsonl                 doc IDs renumbered 0..N-1 (docid_map.tsv keeps the originals)
<name>/queries.jsonl, qrels/test.tsv, num_docs.txt, stats.json
<name>/pca_e5_768_to_192.faiss      one PCA per dataset
<name>/<name>_corpus_embed_f32.npy  N x 192 float32 (e5-base-v2 + PCA)
<name>/<name>_query_embed_f32.npy
index_<name>/                       keyword index (PILLAR-Bin)
tiptoe/<name>_dpc*_m*_b*/           Tiptoe databases
```

`datasets/prep_tree.sh` writes the tree's artifacts to `data/<name>_tree/` in the repo: a single-threaded Lucene
index, the Stage 1/2 tables per `B<B>_r<r>_s<s>/`, the Stage 3 embeddings in index order and the analysed queries
(`tree/build_tree_artifacts.py`).

## Notes

- Doc IDs are renumbered to `0..N-1` so PIR ID entries are short and fixed width.
- Each dataset has its own PCA.
- Corpora of 25M+ documents are embedded in fp16 and stored as float32; MS MARCO v2 is ~106 GB of vectors.
- The subsamples copy msmarco-v2's embedding rows, so embed msmarco-v2 first.
