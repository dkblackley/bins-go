import numpy as np
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligned', required=True, help='Path to aligned_embeddings.npy')
    parser.add_argument('--output_data', required=True, help='Path to shuffled_embeddings.npy')
    parser.add_argument('--output_map', required=True, help='Path to block_permutation.json')
    parser.add_argument('--block_size', type=int, default=8, help='Number of rows per block')
    args = parser.parse_args()

    # 1. Load Aligned Data
    print(f"Loading {args.aligned}...")
    data = np.load(args.aligned, mmap_mode='r')
    N, dim = data.shape
    
    # 2. Pad to full block size if needed
    remainder = N % args.block_size
    if remainder != 0:
        pad_rows = args.block_size - remainder
        print(f"Padding data with {pad_rows} rows to fit block size...")
        padding = np.zeros((pad_rows, dim), dtype=data.dtype)
        # We can't append to mmap, so we'll handle this during write
        # Logic below handles logical indexing
        effective_N = N + pad_rows
    else:
        effective_N = N

    num_blocks = effective_N // args.block_size
    print(f"Total Blocks: {num_blocks}")

    # 3. Generate Random Permutation
    # perm[i] = The original block index that should go to position 'i'
    # We want to shuffle the LOCATION of blocks.
    rng = np.random.default_rng(seed=42) # Fixed seed for reproducibility
    perm = rng.permutation(num_blocks)
    
    # We need the inverse mapping for the Client:
    # Client wants Block B. Where is it? It's at PermInv[B].
    perm_inv = np.zeros(num_blocks, dtype=int)
    perm_inv[perm] = np.arange(num_blocks)

    # 4. Write Shuffled Data
    print("Writing shuffled data...")
    os.makedirs(os.path.dirname(args.output_data), exist_ok=True)
    out = np.lib.format.open_memmap(args.output_data, mode='w+', dtype=data.dtype, shape=(effective_N, dim))
    
    # Write block by block (or chunk by chunk for speed)
    # Target Block 'i' comes from Source Block 'perm[i]'
    chunk_size = 1000 # process 1000 blocks at a time
    for i in range(0, num_blocks, chunk_size):
        end = min(i + chunk_size, num_blocks)
        target_indices = np.arange(i, end)
        source_indices = perm[target_indices]
        
        # Construct source batch
        # This is tricky with raw numpy rows, easier to do row-math
        # But for speed, let's do a loop or advanced indexing if memory allows
        # Advanced indexing on mmap can be slow, but usually fine for sequential-ish writes
        
        # Actually, let's write linear Target, reading random Source
        for tgt_idx, src_idx in zip(target_indices, source_indices):
            # Source slice
            s_start = src_idx * args.block_size
            s_end = s_start + args.block_size
            
            # Handle padding case for source reading
            if s_end > N:
                # Part real, part pad
                real_rows = N - s_start
                block_data = np.zeros((args.block_size, dim), dtype=data.dtype)
                block_data[:real_rows] = data[s_start:N]
            else:
                block_data = data[s_start:s_end]
            
            # Target slice
            t_start = tgt_idx * args.block_size
            t_end = t_start + args.block_size
            
            out[t_start:t_end] = block_data
            
        if i % 10000 == 0:
            print(f"Processed {i}/{num_blocks} blocks...")

    del out # flush
    print(f"Shuffled data saved to {args.output_data}")

    # 5. Save Map for Client
    # Client needs: "I want Block X, give me Shuffled Index Y"
    # This is perm_inv.
    print(f"Saving permutation map to {args.output_map}...")
    with open(args.output_map, 'w') as f:
        json.dump(perm_inv.tolist(), f)
    print("Done.")

if __name__ == "__main__":
    main()