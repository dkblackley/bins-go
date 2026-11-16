package pianopir

import (
	"math/rand"
	"time"
)

// -noram mode for the client hint tables.
//
// The parity tables (primaryParity, backupParity) are the client's storage and, for the bins, about as big as the
// database itself: every row is MaxDBEntrySize words. With HintStore set they live outside the Go heap (diskstore
// puts them in an mmap'd file), and preprocessing fills them one block of rows at a time (preprocessBlocked). The
// normal loop (UpdatePreprocessing) touches every hint row for every chunk, which is fine in RAM but would read and
// write the whole table from disk once per chunk. Blocked, each block of rows stays in RAM while the partition's DB
// is streamed past it, so the table is written once and the DB is read once per block.
//
// The parities are XORs, so the order they are built in does not change them: answers, byte counts and the padded
// storage numbers are the same as a normal run. Only the timings differ.

// HintStore, if set, returns a rows x width table of zeros that is not on the Go heap. Set by main for -noram.
var HintStore func(name string, rows, width uint64) [][]uint64

// HintDiskMinBytes: a client whose parity tables are smaller than this keeps them on the heap even with HintStore
// set (small DBs, e.g. the bins' ID DB, gain nothing from the disk).
var HintDiskMinBytes uint64 = 64 << 20

// HintBlockBytes is how much of a parity table preprocessBlocked works on at once, per partition. Up to ThreadNum
// partitions preprocess at the same time, so about ThreadNum x this is hot in RAM. Bigger = fewer passes over the DB.
var HintBlockBytes uint64 = 512 << 20

func parityBytes(config *PianoPIRConfig, primaryHintNum, maxQueryPerChunk uint64) uint64 {
	return (primaryHintNum + config.SetSize*maxQueryPerChunk) * config.MaxDBEntrySize * 8
}

func useDiskHints(config *PianoPIRConfig, primaryHintNum, maxQueryPerChunk uint64) bool {
	return HintStore != nil && parityBytes(config, primaryHintNum, maxQueryPerChunk) >= HintDiskMinBytes
}

func (c *PianoPIRClient) diskHints() bool {
	return useDiskHints(c.config, c.primaryHintNum, c.maxQueryPerChunk)
}

// newParityTable is the primary parity table NewPianoPIRClient starts with.
func newParityTable(config *PianoPIRConfig, primaryHintNum, maxQueryPerChunk uint64) [][]uint64 {
	if useDiskHints(config, primaryHintNum, maxQueryPerChunk) {
		return HintStore("hints_primary", primaryHintNum, config.MaxDBEntrySize)
	}
	t := make([][]uint64, primaryHintNum)
	for i := range t {
		t[i] = make([]uint64, config.MaxDBEntrySize)
	}
	return t
}

// resetPrimaryParity is resetRows for the primary table. A table on disk is not zeroed here: preprocessBlocked
// zeroes each row right before it fills it, so the file is only written once.
func (c *PianoPIRClient) resetPrimaryParity() [][]uint64 {
	if !c.diskHints() {
		return resetRows(c.primaryParity, c.primaryHintNum, c.config.MaxDBEntrySize)
	}
	if uint64(len(c.primaryParity)) != c.primaryHintNum {
		return HintStore("hints_primary", c.primaryHintNum, c.config.MaxDBEntrySize)
	}
	return c.primaryParity
}

// resetBackupParity is resetRows for chunk i's backup table. On disk all chunks share one file, made when chunk 0
// is first reset (Initialization resets chunks in order).
func (c *PianoPIRClient) resetBackupParity(i uint64) [][]uint64 {
	m, w := c.maxQueryPerChunk, c.config.MaxDBEntrySize
	if !c.diskHints() {
		return resetRows(c.backupParity[i], m, w)
	}
	if uint64(len(c.backupParity[i])) != m {
		all := HintStore("hints_backup", c.config.SetSize*m, w)
		for k := uint64(0); k < c.config.SetSize; k++ {
			c.backupParity[k] = all[k*m : (k+1)*m : (k+1)*m]
		}
	}
	return c.backupParity[i]
}

// preprocessBlocked computes the same hint table as calling UpdatePreprocessing for every chunk, but loops over
// blocks of hint rows on the outside and chunks on the inside. rawDB is already padded to SetSize*ChunkSize rows.
func (c *PianoPIRClient) preprocessBlocked(rawDB [][]uint64) {
	cs, sets, m := c.config.ChunkSize, c.config.SetSize, c.maxQueryPerChunk
	block := max(HintBlockBytes/(max(c.config.MaxDBEntrySize, 1)*8), 1)

	// primary hints: one entry from every chunk
	for lo := uint64(0); lo < c.primaryHintNum; lo += block {
		hi := min(lo+block, c.primaryHintNum)
		for i := lo; i < hi; i++ {
			clear(c.primaryParity[i])
		}
		for chunkId := uint64(0); chunkId < sets; chunkId++ {
			chunk := rawDB[chunkId*cs : (chunkId+1)*cs]
			for i := lo; i < hi; i++ {
				offset := PRFEvalWithLongKeyAndTag(c.longKey, c.primaryShortTag[i], chunkId) & (cs - 1)
				EntryXor(c.primaryParity[i:i+1], chunk[offset:offset+1])
			}
		}
	}

	// backup hints of group g: one entry from every chunk except g. Row f of the flat table is group f/m, slot f%m.
	total := sets * m
	for lo := uint64(0); lo < total; lo += block {
		hi := min(lo+block, total)
		for f := lo; f < hi; f++ {
			clear(c.backupParity[f/m][f%m])
		}
		for chunkId := uint64(0); chunkId < sets; chunkId++ {
			chunk := rawDB[chunkId*cs : (chunkId+1)*cs]
			for f := lo; f < hi; f++ {
				g, j := f/m, f%m
				if g == chunkId {
					continue
				}
				offset := PRFEvalWithLongKeyAndTag(c.longKey, c.backupShortTag[g][j], chunkId) & (cs - 1)
				EntryXor(c.backupParity[g][j:j+1], chunk[offset:offset+1])
			}
		}
	}

	// replacement entries: random rows of each chunk (pointers into the DB, as in UpdatePreprocessing)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	for chunkId := uint64(0); chunkId < sets; chunkId++ {
		for j := uint64(0); j < m; j++ {
			offset := rng.Uint64() & (cs - 1)
			c.replacementIdx[chunkId][j] = offset + chunkId*cs
			c.replacementVal[chunkId][j] = rawDB[chunkId*cs+offset]
		}
	}
}
