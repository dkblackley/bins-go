package pir

import (
  "crypto/elliptic"
  "crypto/rand"
  "testing"
)

func BenchmarkCompress(b *testing.B) {
  p256 := elliptic.P256()  

  _, x, y, err := elliptic.GenerateKey(p256, rand.Reader)
  if err != nil {
    panic("fail")
  }

  b.ResetTimer()
  for k := 0; k < b.N; k++ {
    for i := 0; i < 1024; i++ {
      for j := 0; j < 8; j++ {
        x, y = p256.Double(x, y)
      }
    }
  }
}
