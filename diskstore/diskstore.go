// Package diskstore is the -noram mode: the big read-only arrays (the PIR databases and the corpus embeddings) live
// in files under temp_storage/<run id>/ and are mmap'd, instead of being held on the Go heap. The kernel pages them
// in as they are touched and can drop them again under memory pressure, so a job only needs RAM for what it is
// using right now (plus the client hint tables, which stay on the heap; see README "Low-RAM mode").
//
// Results are the same as a normal run: the same bytes, in the same [][]uint64 shape, go into the same PIR code.
// Only the timings change (disk reads instead of RAM reads), which is why main.go tags these runs "noram": "true".
package diskstore

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/sirupsen/logrus"
)

var (
	// Enabled is set by Init. Code that builds a big array checks it to decide between the heap and a file.
	Enabled bool
	// Dir is this run's own folder, temp_storage/<run id>.
	Dir string

	mu       sync.Mutex
	mappings [][]byte
	files    int
	cleaned  bool
)

// StaleAfter: Init deletes other run folders under the same base whose owner file is older than this. A run that
// was killed (OOM, scancel -9, node failure) never gets to clean up after itself, and bins jobs never run this long.
const StaleAfter = 48 * time.Hour

// Init creates base/<tag>_<date>_<host>_<pid>_<random>/ and turns the mode on. Cleanup removes the folder again;
// it also runs on SIGTERM / SIGINT (Slurm sends SIGTERM at the time limit and on scancel).
func Init(base, tag string) (string, error) {
	if err := os.MkdirAll(base, 0o755); err != nil {
		return "", fmt.Errorf("noram: create %s: %w", base, err)
	}
	sweepStale(base)

	var rnd [6]byte
	if _, err := rand.Read(rnd[:]); err != nil {
		return "", err
	}
	host, _ := os.Hostname()
	host = strings.SplitN(host, ".", 2)[0]
	name := fmt.Sprintf("%s_%s_%s_%d_%s", sanitize(tag), time.Now().Format("20060102-150405"), sanitize(host),
		os.Getpid(), hex.EncodeToString(rnd[:]))
	dir := filepath.Join(base, name)
	if err := os.Mkdir(dir, 0o755); err != nil { // Mkdir, not MkdirAll: fails if the name is somehow taken
		return "", fmt.Errorf("noram: create %s: %w", dir, err)
	}
	owner := fmt.Sprintf("host=%s pid=%d slurm_job=%s_%s started=%s\n", host, os.Getpid(),
		os.Getenv("SLURM_ARRAY_JOB_ID"), os.Getenv("SLURM_ARRAY_TASK_ID"), time.Now().Format(time.RFC3339))
	if err := os.WriteFile(filepath.Join(dir, "owner"), []byte(owner), 0o644); err != nil {
		return "", err
	}

	Dir, Enabled = dir, true
	StartPeakSampler("")
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		s := <-sig
		logrus.Warnf("noram: got %v, removing %s", s, Dir)
		Cleanup()
		os.Exit(143)
	}()
	logrus.Infof("noram: big arrays go to %s (mmap'd, not on the heap)", dir)
	return dir, nil
}

// Cleanup unmaps everything and deletes the run folder. Safe to call more than once. Nothing may touch a slice
// handed out by this package afterwards (it would segfault), so main calls it only once the run is finished.
func Cleanup() {
	mu.Lock()
	defer mu.Unlock()
	if !Enabled || cleaned {
		return
	}
	cleaned = true
	for _, m := range mappings {
		_ = syscall.Munmap(m)
	}
	mappings = nil
	if err := os.RemoveAll(Dir); err != nil {
		logrus.Warnf("noram: could not remove %s: %v", Dir, err)
	}
}

var peakAnonKB uint64 // guarded by mu

var (
	samplerOnce sync.Once
	peakFile    string // guarded by mu
	started     = time.Now()
)

// StartPeakSampler starts tracking the peak RssAnon (once per process; later calls only set the file). With a file,
// it also rewrites "<peak MB> <seconds since start>" there every 10 s, so a run that is killed (OOM, time limit)
// still leaves how much heap it had reached and when, for the next memory estimate (slurm_common.sh).
func StartPeakSampler(file string) {
	mu.Lock()
	if file != "" {
		peakFile = file
	}
	mu.Unlock()
	samplerOnce.Do(func() { go sampleAnon() })
}

// sampleAnon tracks the peak of RssAnon (heap, stacks: the memory that cannot be paged back to a file). The usual
// peak RSS (VmHWM, /usr/bin/time, sacct MaxRSS) also counts the mmap'd file pages that happen to be resident, so
// for a -noram run (or a tree run that mmaps its Cuckoo cache) it shows the job's memory limit, not what it needs.
func sampleAnon() {
	for tick := 0; ; tick++ {
		if b, err := os.ReadFile("/proc/self/status"); err == nil {
			for _, line := range strings.Split(string(b), "\n") {
				if strings.HasPrefix(line, "RssAnon:") {
					var kb uint64
					fmt.Sscanf(strings.TrimSpace(strings.TrimPrefix(line, "RssAnon:")), "%d", &kb)
					mu.Lock()
					peakAnonKB = max(peakAnonKB, kb)
					mu.Unlock()
				}
			}
		}
		if tick%10 == 0 {
			mu.Lock()
			f, kb := peakFile, peakAnonKB
			mu.Unlock()
			if f != "" {
				_ = os.WriteFile(f, []byte(fmt.Sprintf("%.0f %.0f\n", float64(kb)/1024, time.Since(started).Seconds())), 0o644)
			}
		}
		time.Sleep(time.Second)
	}
}

// PeakAnonMB is the highest RssAnon seen so far, in MB (sampled every second).
func PeakAnonMB() float64 {
	mu.Lock()
	defer mu.Unlock()
	return float64(peakAnonKB) / 1024
}

func sweepStale(base string) {
	entries, err := os.ReadDir(base)
	if err != nil {
		return
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		st, err := os.Stat(filepath.Join(base, e.Name(), "owner"))
		if err != nil || time.Since(st.ModTime()) < StaleAfter {
			continue
		}
		logrus.Infof("noram: removing stale run folder %s (older than %v)", e.Name(), StaleAfter)
		_ = os.RemoveAll(filepath.Join(base, e.Name()))
	}
}

func sanitize(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '/' || r == ' ' || r == os.PathSeparator {
			return '-'
		}
		return r
	}, s)
}

func newFile(name string) (*os.File, error) {
	mu.Lock()
	files++
	path := filepath.Join(Dir, fmt.Sprintf("%03d_%s", files, sanitize(name)))
	mu.Unlock()
	return os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o644)
}

// mmapReadOnly maps the first size bytes of f. PROT_READ on purpose: the PIR code must never write into a DB row
// (replacement values point straight at rows), and with a read-only map a bug like that crashes instead of
// silently changing the database.
func mmapReadOnly(f *os.File, size int64) ([]byte, error) {
	if size == 0 {
		return nil, nil
	}
	b, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("noram: mmap %s (%d bytes): %w", f.Name(), size, err)
	}
	mu.Lock()
	mappings = append(mappings, b)
	mu.Unlock()
	return b, nil
}

func words(b []byte, off, n int64) []uint64 {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*uint64)(unsafe.Pointer(&b[off])), n)[:n:n]
}

// RowStore builds a [][]uint64 whose rows live in a file. Rows can be added in any order (and from several
// goroutines); Finish writes them out in the order the PIR will read them and maps the result.
type RowStore struct {
	name    string
	staging *os.File
	mu      sync.Mutex
	end     int64   // bytes written to staging
	off     []int64 // staging offset of logical row i, -1 = empty
	n       []uint32
}

// NewRowStore makes a store for rows 0..rows-1, all empty until Put.
func NewRowStore(name string, rows int) (*RowStore, error) {
	f, err := newFile(name + ".staging")
	if err != nil {
		return nil, err
	}
	s := &RowStore{name: name, staging: f, off: make([]int64, rows), n: make([]uint32, rows)}
	for i := range s.off {
		s.off[i] = -1
	}
	return s, nil
}

// Len is the number of rows.
func (s *RowStore) Len() int { return len(s.off) }

// Put stores row i (a copy; the caller may reuse row). Each row may be put once.
func (s *RowStore) Put(i int, row []uint64) error {
	if len(row) == 0 {
		return nil
	}
	b := unsafe.Slice((*byte)(unsafe.Pointer(&row[0])), len(row)*8)
	s.mu.Lock()
	at := s.end
	s.end += int64(len(b))
	s.off[i], s.n[i] = at, uint32(len(row))
	s.mu.Unlock()
	_, err := s.staging.WriteAt(b, at) // WriteAt is safe to call concurrently on distinct ranges
	return err
}

// Finish returns the rows, backed by a read-only mmap. physOrder[p] is the logical row the PIR stores at physical
// position p (pianopir.ShufflePermutation): writing the rows in that order makes the PIR's preprocessing, which
// walks each partition chunk by chunk in physical order, read the file front to back instead of jumping around.
// nil physOrder keeps the staging file as it is: it is mapped in place, with no second copy (the rows are wherever
// Put wrote them; only the read pattern of a real preprocessing pass cares about the layout, and a simulated run has
// none). Otherwise the rows are copied out in that order and the staging file is deleted.
func (s *RowStore) Finish(physOrder []int) ([][]uint64, error) {
	rows := len(s.off)
	if physOrder == nil {
		// Rewriting meant reading the whole staging file once more (in shuffled row order, so mostly random page
		// faults) and writing it again: for a 150 GB vec1 DB on NFS that alone took longer than the 30 min jobs.
		data, err := mmapReadOnly(s.staging, s.end)
		_ = s.staging.Close() // the mapping keeps the file; Cleanup removes it with the run folder
		if err != nil {
			return nil, err
		}
		ret := make([][]uint64, rows)
		for i := 0; i < rows; i++ {
			if s.off[i] >= 0 {
				ret[i] = words(data, s.off[i], int64(s.n[i]))
			}
		}
		logrus.Infof("noram: %s: %d rows, %.2f GB on disk in %s (staging order, not copied)", s.name, rows,
			float64(s.end)/1e9, Dir)
		s.off, s.n = nil, nil
		return ret, nil
	}
	defer func() {
		_ = s.staging.Close()
		_ = os.Remove(s.staging.Name())
	}()
	src, err := mmapReadOnly(s.staging, s.end)
	if err != nil {
		return nil, err
	}

	out, err := newFile(s.name)
	if err != nil {
		return nil, err
	}
	defer out.Close()
	final := make([]int64, rows) // offset in the final file
	w := bufio.NewWriterSize(out, 8<<20)
	var at int64
	for p := 0; p < rows; p++ {
		i := p
		if physOrder != nil {
			i = physOrder[p]
		}
		final[i] = at
		if s.off[i] < 0 {
			continue
		}
		nb := int64(s.n[i]) * 8
		if _, err := w.Write(src[s.off[i] : s.off[i]+nb]); err != nil {
			return nil, fmt.Errorf("noram: write %s: %w", out.Name(), err)
		}
		at += nb
	}
	if err := w.Flush(); err != nil {
		return nil, fmt.Errorf("noram: write %s: %w", out.Name(), err)
	}
	// the staging copy is no longer needed: unmap it now so its disk space is freed by the Remove above
	mu.Lock()
	for k, m := range mappings {
		if len(m) > 0 && len(src) > 0 && &m[0] == &src[0] {
			mappings = append(mappings[:k], mappings[k+1:]...)
			break
		}
	}
	mu.Unlock()
	if src != nil {
		_ = syscall.Munmap(src)
	}

	data, err := mmapReadOnly(out, at)
	if err != nil {
		return nil, err
	}
	ret := make([][]uint64, rows)
	for i := 0; i < rows; i++ {
		if s.off[i] >= 0 {
			ret[i] = words(data, final[i], int64(s.n[i]))
		}
	}
	logrus.Infof("noram: %s: %d rows, %.2f GB on disk in %s", s.name, rows, float64(at)/1e9, Dir)
	s.off, s.n = nil, nil
	return ret, nil
}

// ZeroTable returns a rows x width table of zeros backed by a new sparse file, mapped read-write. Used for the
// client hint tables (pianopir.HintStore); the kernel writes dirty pages back to the file and can drop them under
// memory pressure, so the table does not have to fit in RAM.
func ZeroTable(name string, rows, width uint64) [][]uint64 {
	ret := make([][]uint64, rows)
	size := int64(rows * width * 8)
	if size == 0 {
		for i := range ret {
			ret[i] = []uint64{}
		}
		return ret
	}
	f, err := newFile(name)
	if err != nil {
		logrus.Fatalf("noram: %v", err)
	}
	defer f.Close()
	if err := f.Truncate(size); err != nil {
		logrus.Fatalf("noram: size %s to %d bytes: %v", f.Name(), size, err)
	}
	b, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		logrus.Fatalf("noram: mmap %s (%d bytes): %v", f.Name(), size, err)
	}
	mu.Lock()
	mappings = append(mappings, b)
	mu.Unlock()
	for i := uint64(0); i < rows; i++ {
		ret[i] = words(b, int64(i*width*8), int64(width))
	}
	return ret
}
