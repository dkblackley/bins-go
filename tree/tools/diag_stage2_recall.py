#!/usr/bin/env python3
"""
Diagnostic: check whether qrel gold doc appears in Stage2 hitSubs for a given qid

Usage:
  ./diag_stage2_recall.py --qid <QID> --stage2-hits ../stage2_hits.json --qrels ../qrels.dev.small.tsv \
      [--docmap /path/to/docmap.bin] [--docidmap /path/to/doc-id-map.txt] [--sample 50]

The script attempts to map internal doc indices from hitSubs to external doc IDs
using (in order): lucene `docmap.bin` (8-byte little-endian ints), a `doc-id-map` file
(one ID per line loaded by Stage3), or falling back to the numeric internal id string.
It then reports whether any gold external ID from qrels is present and prints diagnostics.
"""
import argparse
import json
import os
import struct
from typing import Dict, List


def load_qrels(path: str) -> Dict[str, List[str]]:
    m = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            qid = parts[0]
            docid = parts[2]
            m.setdefault(qid, []).append(docid)
    return m


def load_stage2_hits(path: str) -> Dict[str, List[Dict]]:
    with open(path, 'r') as f:
        return json.load(f)


def load_docmap(path: str) -> List[int]:
    # docmap.bin assumed to be sequence of little-endian uint64 per internal doc id
    size = os.path.getsize(path)
    count = size // 8
    res = []
    with open(path, 'rb') as f:
        for _ in range(count):
            b = f.read(8)
            if len(b) < 8:
                break
            res.append(struct.unpack('<Q', b)[0])
    return res


def load_docid_map_txt(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [line.strip() for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qid', required=True)
    ap.add_argument('--stage2-hits', required=True)
    ap.add_argument('--qrels', required=True)
    ap.add_argument('--docmap', default='')
    ap.add_argument('--docidmap', default='')
    ap.add_argument('--sample', type=int, default=50)
    args = ap.parse_args()

    qid = args.qid
    stage2_hits = load_stage2_hits(args.stage2_hits)
    qrels = load_qrels(args.qrels)

    golds = qrels.get(qid, [])
    if not golds:
        print(f'No gold qrels found for qid={qid} in {args.qrels}')
        return
    print(f'Gold docs for qid={qid}: {golds}')

    if qid not in stage2_hits:
        print(f'No Stage2 hits entry for qid={qid} in {args.stage2_hits}')
        return

    hit_blocks = stage2_hits[qid]
    total_docs = sum(h['End'] - h['Start'] for h in hit_blocks)
    print(f'Loaded {len(hit_blocks)} hit blocks containing {total_docs} docs')

    docmap = None
    if args.docmap and os.path.exists(args.docmap):
        print(f'Loading docmap from {args.docmap} ...')
        docmap = load_docmap(args.docmap)
        print(f'  docmap entries: {len(docmap)}')

    docidmap = None
    if args.docidmap and os.path.exists(args.docidmap):
        print(f'Loading doc-id map from {args.docidmap} ...')
        docidmap = load_docid_map_txt(args.docidmap)
        print(f'  doc-id-map entries: {len(docidmap)}')

    found = False
    found_examples = []
    sample_list = []

    # iterate candidates, try mapping and check against golds
    # stop early if found but still collect sample entries
    max_sample = args.sample
    sample_count = 0

    for blk_idx, blk in enumerate(hit_blocks):
        start = int(blk['Start'])
        end = int(blk['End'])
        for docidx in range(start, end):
            ext = None
            if docmap is not None and docidx < len(docmap):
                ext = str(docmap[docidx])
            elif docidmap is not None and docidx < len(docidmap):
                ext = docidmap[docidx]
            else:
                ext = str(docidx)

            if sample_count < max_sample:
                sample_list.append((docidx, ext))
                sample_count += 1

            if ext in golds:
                found = True
                found_examples.append({'blk_idx': blk_idx, 'docidx': docidx, 'ext': ext})
                # we continue to gather a few examples
                if len(found_examples) >= 10:
                    break
        if len(found_examples) >= 10:
            break

    if found:
        print(f'FOUND gold doc(s) among Stage2 candidates for qid={qid} (examples):')
        for e in found_examples:
            print(f"  Block#{e['blk_idx']} internal={e['docidx']} external={e['ext']}")
    else:
        print(f'Gold docs NOT FOUND in Stage2 candidates for qid={qid}')

    print('\nSample of first %d candidate mappings (internal -> external):' % len(sample_list)
    )
    for internal, ext in sample_list:
        print(f'  {internal} -> {ext}')


if __name__ == '__main__':
    main()
