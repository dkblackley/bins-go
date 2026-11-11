package diskstore

import (
	"encoding/binary"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"unsafe"
)

var npyShape = regexp.MustCompile(`'shape':\s*\(\s*(\d+)\s*,\s*(\d+)\s*,?\s*\)`)

// MmapNpyFloat32 maps a 2-D little-endian float32 .npy file (what the dataset scripts write) and returns its first
// n rows as views into the map, with the same shape checks as globals.LoadFloat32MatrixFromNpy. Nothing is copied
// to the heap: a 138M x 192 corpus (106 GB) costs only the row headers (3.3 GB).
func MmapNpyFloat32(filename string, n, dim int) ([][]float32, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close() // the mapping stays valid after the file is closed

	var pre [12]byte
	if _, err := f.ReadAt(pre[:], 0); err != nil {
		return nil, fmt.Errorf("%s: %w", filename, err)
	}
	if string(pre[:6]) != "\x93NUMPY" {
		return nil, fmt.Errorf("%s: not a .npy file", filename)
	}
	var hdrLen, hdrStart int64
	switch pre[6] {
	case 1:
		hdrLen, hdrStart = int64(binary.LittleEndian.Uint16(pre[8:10])), 10
	case 2, 3:
		hdrLen, hdrStart = int64(binary.LittleEndian.Uint32(pre[8:12])), 12
	default:
		return nil, fmt.Errorf("%s: unsupported .npy version %d", filename, pre[6])
	}
	hdr := make([]byte, hdrLen)
	if _, err := f.ReadAt(hdr, hdrStart); err != nil {
		return nil, fmt.Errorf("%s: %w", filename, err)
	}
	h := string(hdr)
	if !strings.Contains(h, "'descr': '<f4'") {
		return nil, fmt.Errorf("%s: dtype is not little-endian float32: %s", filename, strings.TrimSpace(h))
	}
	if !strings.Contains(h, "'fortran_order': False") {
		return nil, fmt.Errorf("%s: fortran_order arrays are not supported", filename)
	}
	m := npyShape.FindStringSubmatch(h)
	if m == nil {
		return nil, fmt.Errorf("%s: not a 2-D array: %s", filename, strings.TrimSpace(h))
	}
	rows, _ := strconv.Atoi(m[1])
	cols, _ := strconv.Atoi(m[2])
	if rows < n || cols != dim {
		fmt.Printf("Invalid shape: (%d, %d)\nExpected shape: (%d, %d)\n", rows, cols, n, dim)
		return nil, fmt.Errorf("invalid shape: [%d %d]", rows, cols)
	}
	dataOff := hdrStart + hdrLen
	if dataOff%4 != 0 {
		return nil, fmt.Errorf("%s: data offset %d is not 4-byte aligned", filename, dataOff)
	}
	size := dataOff + int64(rows)*int64(cols)*4
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() < size {
		return nil, fmt.Errorf("%s: file is %d bytes, header says %d", filename, st.Size(), size)
	}
	b, err := mmapReadOnly(f, size)
	if err != nil {
		return nil, err
	}
	ret := make([][]float32, n)
	if n == 0 || dim == 0 {
		return ret, nil
	}
	all := unsafe.Slice((*float32)(unsafe.Pointer(&b[dataOff])), n*dim)
	for i := 0; i < n; i++ {
		ret[i] = all[i*dim : (i+1)*dim : (i+1)*dim]
	}
	return ret, nil
}
