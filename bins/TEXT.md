# Bins text mode (`-text`)

`pir_app -t bins -text ...` puts the compressed text of each bin's documents in the PIR DB instead of their
embeddings, and the client re-ranks what it retrieves with BM25. No embedding files are read. Every dataset that
`GetDatasets` knows works (msmarco, scifact, nq, hotpotqa, fever, msmarco-v2*, ...). Slurm: `run_bins_text.slurm`.

| flag | default | |
|---|---|---|
| `-text` | off | turn the mode on (bins only) |
| `-text-format` | `bow` | `bow`: sorted stems + tf (all BM25 needs). `seq`: stems in document order |
| `-text-order` | `3` | characters of context in the coder's model (0-3) |
| `-text-idf-bits` | `20` | client IDF sketch: 2 x 2^bits one-byte cells (0 = no idf) |
| `-text-train-docs` | `200000` | docs used to train the model |
| `-k1`, `-b` | 0.9, 0.4 | BM25 parameters of the client re-rank |

## What is stored

1. Each doc goes through the analyzer the bins and queries already use: letters only, lowercase, English stopwords
   removed, Porter stemmed (`strictEnglishAnalyzer`).
2. `bow`: the unique stems are sorted and front-coded (each stem stores how many leading letters it shares with
   the previous one), and each stem's end symbol carries its tf (1, 2, 3, or 4+ then Exp-Golomb).
3. That symbol stream is arithmetic coded (`bins/textcodec`) with a static order-3 character model: the frequency
   of each next letter given the previous 3 letters of the stem, trained once on a sample of the corpus.
   Decoding is integer arithmetic over those counts. There is no word list or codeword table on the client.
4. Codes are self-delimiting (a decoder knows where the code ended), so a bin is just: doc count, then for each doc
   (sorted by index) an Exp-Golomb gap to its doc index and its code. About 2 bytes of index per doc.

The client keeps the model (~1.3 MB at order 3, 48 KB at order 2) and the IDF sketch (2 MB at 20 bits). Both are
added to `ClientStorageMB` (the PIR-only number is `PIRClientStorageMB`). BM25 needs the doc length, which is the
sum of the tfs, and the average doc length, which is one public number.

## Sizes (proxy corpus, see below)

`go run ./bins/cmd/textbench -corpus <corpus.jsonl> -train 200000 -test 200000` prints this table for a real
corpus. These numbers are from a 10.8K-passage proxy (news + Wikipedia cut into 56-word passages, the MS MARCO
average), trained on 5.4K and measured on the other 5.4K, because the sandbox this was written in could not
download MS MARCO. Run textbench on Hopper for the real numbers; with 200K training docs order 3 should do better.

| encoding | bytes/doc | vs 192-dim fp32 embedding |
|---|---|---|
| embedding (192 x float32) | 768.0 | 100% |
| raw text (UTF-8) | 357.3 | 46.5% |
| stems, ASCII + space | 223.6 | 29.1% |
| stems, 5-bit alphabet | 140.4 | 18.3% |
| stems, static Huffman (order 0) | 119.1 | 15.5% |
| arithmetic, stems in order, order 2 / 3 | 91.9 / 81.6 | 12.0% / 10.6% |
| arithmetic, sorted stems + tf, order 2 / 3 (**default**) | 75.6 / 67.2 | 9.8% / 8.8% |
| (reference) word-ID entropy, needs the vocabulary on the client | 64.8 | 8.4% |

Plus about 18 bits of doc index per doc in a bin (MS MARCO, 100 docs/bin). The word-ID row is the best a word
dictionary could do with a unigram model, and it is only ~4% smaller while needing a vocabulary lookup on the
client, which is not allowed.

For MS MARCO (8.84M passages, ~56 words each) that is roughly 70 B/doc with the index, so ~0.6 GB of text versus
~6.3 GiB of 192-dim embeddings, and a bin of 100 docs is ~7 KB instead of 75 KB. The Piano hint and every PIR
response scale with the entry size, so both shrink by about the same factor.

## Padding accounting (also affects the embedding runs)

Bins have different sizes. The DB already stored short entries without padding, but `pianopir` charged every
response and every hint slot `MaxDBEntrySize`. Now:

- `TotalByteSent` / `TotalWANTime` / `TotalLANTime`: a response is the XOR of SetSize entries, so it is as long as
  the longest of them. The server knows every length and which entries it XORed, so sending only that prefix
  (plus a 4-byte length) leaks nothing. The old numbers are kept as `*Padded`.
- `ClientStorageMB`: each stored parity / replacement entry is charged its length without trailing zero words,
  plus a 4-byte length. The old number is `ClientStoragePaddedMB`.

Fixed-size DBs (PACMANN, the tree's pir-old) are unaffected.

## The tree

Not implemented. The tree's PIR (pir-old, Cuckoo tables) needs equal-size rows, so Stage 3 could hold text only by
padding every row to the largest one, and that padding would be real cost.
