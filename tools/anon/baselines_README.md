# Tiptoe and Coeus baselines

Both run on the same datasets and write the same `metadata.json` keys as the Go methods, so `plotting/` loads them
as the methods `tiptoe` and `coeus`. Both report **exact retrieval quality** and **exact (or lower-bound)
communication**, with the LAN/WAN times derived from it. Server compute time is not reported for either.

| | Upstream code | Here | What is exact |
|---|---|---|---|
| **Tiptoe** (SOSP'23) | ahenzinger/tiptoe (Go + SEAL) | `baselines/tiptoe/preprocess_tiptoe.py` builds the clustered, bundled database; `baselines/tiptoe/simulate_tiptoe.py` runs the protocol in plaintext. The upstream protocol itself is wrapped in `tiptoe/tiptoe.go` (`go build -tags tiptoe`, needs SEAL 4.1.1, `baselines/setup_baselines.sh tiptoe`). | Quality and returned documents (the homomorphic computation decrypts to the true values, so the client's logic is replayed in plaintext). Database shapes (Tiptoe's own packing, ported). Bytes and client storage are lower bounds from the message formats (payload only). |
| **Coeus** (SOSP'21) | ishtiyaque/Coeus_artifact (C++ + SEAL 3.x) | `baselines/coeus/run_coeus.py` | Quality: Coeus ranks by `q · tfidf(d)` under BFV, which is exact, so its top-k equals plaintext TF-IDF top-k. Communication: the serialised ciphertext sizes from the artifact's parameters (256 KiB response ciphertexts; ~270 MiB per query on MS MARCO). |

Both use the same 192-dim embeddings as PILLAR and PACMANN (Tiptoe's search embeddings; for Coeus only the shared
cosine re-rank).

**Rule used everywhere:** when a cost cannot be known exactly, the number that is best for the baseline is taken
(payload-only sizes, keys counted once). Each run writes a `TiptoeCostNote` / `CoeusCostNote` listing what was left out.

## Run

`local_run.sh tiptoe coeus` (repo root) runs both sweeps. By hand:

```bash
python baselines/tiptoe/preprocess_tiptoe.py --dataset ../datasets --name nq --docs-per-cluster 2500 \
    --multi-assign 2 --bundle-size 160 --out ../datasets/tiptoe/nq_dpc2500_m2_b160 --kmeans-cache ../datasets/tiptoe/kmeans_cache
python baselines/tiptoe/simulate_tiptoe.py --dataset ../datasets --name nq --tiptoe-dir ../datasets/tiptoe/nq_dpc2500_m2_b160 \
    --k 10 --emb-hint-mb 500 --results-root results
python baselines/coeus/run_coeus.py --dataset ../datasets --name nq --k 10 --features 32768 --tf log --results-root results
```

Result folders: `tiptoe_<dataset>_c<clusters>_m<multi-assign>_b<bundle>_h<hint MB>_k<K>[_<precision>]` and
`coeus_<dataset>_k<K>_f<features>_tf<raw|log>`.

## Swept parameters

**Tiptoe:** docs per cluster {250, 2500, 40000} (Tiptoe: ~2,500) x multi-assign {1, 2} (Tiptoe: 2) x bundle size
{40, 320} (Tiptoe: 160) x embedding hint {125, 500} MB (Tiptoe: 500) x k {10, 100} x embedding precision
{fp32, fp8, fp1}. Tiptoe probes exactly one cluster and uses 5-bit embeddings, as in the protocol. It caps the
number of clusters at 25,196, so on the largest corpora several docs-per-cluster values give the same database.

**Coeus:** dictionary size F {8192, 32768, 262144} x tf weighting {raw, log} x k {10, 100}. Upload is `ceil(F/8192)`
query ciphertexts; download is `ceil(N/8192)` response ciphertexts, one encrypted score per document, whatever F.
An F larger than the vocabulary only adds empty ciphertexts, so those configs are skipped.

## metadata.json

The same keys as the Go methods (`MRR`, `Recall`, `MRRPreReRank`, `ClientStorageMB`, `TotalByteSent`,
`TotalLANTime`, `TotalWANTime`, `DBSizeInBytesMB`, `NumQueries`, ...), plus `Tiptoe*` / `Coeus*` details.

* Tiptoe client storage: the hint metadata the client keeps plus the cluster centroids. Its ~32 MiB per-query token
  is counted in `TotalByteSent`; LAN/WAN time covers the two online rounds, the token's is in
  `TiptoeOfflineLANTime` / `TiptoeOfflineWANTime`.
* Coeus client storage: the secret key. The Galois keys are counted once as a setup upload (`CoeusSetupUploadMB`).
* Neither baseline downloads document text, matching PILLAR and PACMANN.

`go.mod` replaces Tiptoe's `underhood` dependency (its pseudo-version timestamp is wrong) and uses
`third_party/simplepir`, a copy of SimplePIR with a two-line type fix for Go >= 1.24.
