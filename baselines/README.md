# Tiptoe and Coeus baselines

Reviewer 1 asked what PILLAR adds over "just combining Tiptoe and Coeus", and Reviewer 3 asked how PILLAR's client
storage and latency compare with Tiptoe's. Both systems have public code, and this folder runs them on our datasets.
Each run writes the same `metadata.json` keys as the Go methods, so `plotting/load_results.py` loads them as the
methods `tiptoe` and `coeus` and every existing plot picks them up.

| | Code | How it runs here | What's real, what's modelled |
|---|---|---|---|
| **Tiptoe** (SOSP'23) | [ahenzinger/tiptoe](https://github.com/ahenzinger/tiptoe) (Go + SEAL, MIT) | `pir_app_tiptoe -t tiptoe`: upstream Tiptoe imported as a Go module, driven in-process by `tiptoe/tiptoe.go` | Everything is real: k-means clustering, the SimplePIR-based encrypted ranking, URL PIR, and the per-query "underhood" token that removes the client hint. We measure time, bytes, client state and MRR/Recall the same way as for PILLAR. |
| **Coeus** (SOSP'21) | [ishtiyaque/Coeus_artifact](https://github.com/ishtiyaque/Coeus_artifact) (C++ + SEAL 3.x) | `baselines/coeus/run_coeus.py` | **Quality is exact**: Coeus ranks by `q · tfidf(d)` under BFV, which has no approximation error, so its top-k equals plaintext TF-IDF top-k. **Communication** is computed from the encryption parameters hard-coded in the artifact. **Server time** comes from running the artifact's own `opt2` benchmark on a few small matrices on your machine and extrapolating linearly. The artifact only benchmarks random matrices on a 96-machine AWS cluster, so it can't be run end to end on our data. |

Both use the same 192-dim embeddings as PILLAR and PACMANN. For Tiptoe they are the search embeddings; for Coeus
they are only used in the shared cosine rerank that every method gets in `main.go`. Tiptoe's own text embeddings
are also 192-dim, so no PCA step is needed.

## 1. Setup (once, on a login node)

```bash
module load gnu12                         # Tiptoe's rlwe.go hard-codes -std=c++20, so GCC >= 10 is needed
./baselines/setup_baselines.sh tiptoe     # SEAL 4.1.1 in ~/opt/seal-4.1, then builds ./pir_app_tiptoe
./baselines/setup_baselines.sh coeus      # SEAL 3.5.6 in ~/opt/seal-3.5, then builds Coeus' matmult/bin/opt2
```

The normal `go build` / `build.sh` is unchanged. Tiptoe is only compiled with `-tags tiptoe`, and a plain
`pir_app` exits with a clear message if you pass `-t tiptoe`. Python needs numpy; `faiss` is used for k-means if
installed (it is much faster on MS MARCO), otherwise sklearn, otherwise plain numpy.

## 2. Run

```bash
sbatch run_tiptoe.slurm                   # 2 datasets x 7 cluster sizes x k in {10,100}
sbatch --array=0 run_coeus.slurm          # first task calibrates opt2 and caches the fit
sbatch --array=1-7 run_coeus.slurm        # the other tasks reuse the cached fit
```

Result folders: `tiptoe_<dataset>_c<clusters>_k<K>` and `coeus_<dataset>_k<K>_f<features>`.

Quick check without SEAL: `baselines/tiptoe/simulate_tiptoe_quality.py` replays Tiptoe's ranking in plaintext.
The crypto is exact, so this gives the same MRR/Recall (`MRRPreReRank`) as a real run, apart from the order of
tied scores.

## 3. The knobs we sweep, and why

**Tiptoe: number of clusters.** This is the main trade-off in the Tiptoe paper. The client scores only the cluster
whose centroid is nearest to the query, then fetches one URL bundle (up to 160 docs) holding its best doc. Bigger
clusters give better recall and a bigger download; more clusters give a bigger upload (`clusters x 192`). The
slurm sweeps 250 to 40,000 docs per cluster (Tiptoe itself used about 2,500). Everything else matches Tiptoe's
code: 5-bit embeddings, boundary docs assigned to 2 clusters (the 20th-percentile rule), and bundles of 160.

**Coeus: dictionary size F.** Upload is `ceil(F/8192)` ciphertexts. Download is `ceil(N/8192)` ciphertexts of
384 KB each, because Coeus returns an encrypted score for every document. That download term is what makes Coeus
expensive on MS MARCO: 1,080 ciphertexts, about 405 MiB per query. The paper used F = 65,536.

## 4. What lands in metadata.json

The same keys as the Go methods are used (`MRR`, `Recall`, `MRRPreReRank`, `ClientStorageMB`, `TotalByteSent`,
`TotalLANTime`, `TotalWANTime`, `TotalAnswerTime`, `CommCostPerBatchKB`, `DBSizeInBytesMB`, `NumQueries`), plus
`Tiptoe*` / `Coeus*` details.

* **Tiptoe client storage** = the hint metadata the client keeps (DB parameters and index maps, with the hint
  matrix dropped as Tiptoe's coordinator does) plus the cluster centroids. The per-query token is counted in
  `TotalByteSent` and `MaintainenceTime`, like PIR preprocessing for the other methods. LAN/WAN time covers only
  the 2 online rounds; the token's network time is in `TiptoeOfflineLANTime/WANTime`.
* **Coeus client storage** = secret key + Galois keys. Those keys are uploaded once (`CoeusSetupUploadMB`) and are
  not counted per query. The step-1 server time is the `opt2` fit divided by `--parallelism` (the slurm uses the
  task's CPU count, since `opt2` is single-threaded and Coeus scales almost linearly across workers, per their
  Fig. 9). The step-2 metadata-PIR server time is not included, which flatters Coeus slightly. Without
  `--opt2-bin`, `TotalAnswerTime` is left out rather than guessed.
* Neither baseline downloads document text, matching PILLAR and PACMANN.

## 5. Things to know

* Tiptoe pins `underhood` at a pseudo-version whose timestamp doesn't match the commit, so `go.mod` has a
  `replace` to the same commit with the right timestamp.
* Upstream `simplepir` doesn't compile with Go >= 1.24 because it defines methods on cgo types. `third_party/simplepir`
  is a copy with a two-line type fix (see `third_party/README.md`).
* Tiptoe caps the number of clusters at 25,196 (`TOTAL_NUM_CLUSTERS`). At about 2,500 docs per cluster that covers
  up to roughly 60M docs. For a 100M corpus, use `--docs-per-cluster 4000` or more.
* The Tiptoe Go wrapper has been compile-checked against the upstream packages. It couldn't be run in the
  development sandbox because SEAL couldn't be built there, so the first real run is on your cluster.
  `simulate_tiptoe_quality.py` is the cross-check for its MRR.
