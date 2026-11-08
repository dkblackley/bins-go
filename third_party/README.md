# third_party

## simplepir

Copy of github.com/henrycg/simplepir at v0.0.0-20230920020624-026ee7bd6783 (MIT, see `simplepir/LICENSE`),
used by Tiptoe. `go.mod` points at it with a `replace`.

One change: Go 1.24 refuses to compile methods on cgo types ("cannot define new methods on non-local type
Elem32"), and upstream declares `type Elem32 = C.Elem32` and then gives it methods. So in `matrix/matrix.go`:

    type Elem32 = C.Elem32   ->   type Elem32 uint32
    type Elem64 = C.Elem64   ->   type Elem64 uint64

and the eight calls in `matrix/math.go` that pass these slices to C now cast to `*C.Elem32` / `*C.Elem64`.
`C.Elem32` is `uint32_t` (see `matrix/matrix.h`), so the memory layout and the C code are unchanged.
`performance.png` was dropped to keep the repo small.
