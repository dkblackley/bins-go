package bm25

import (
	"bytes"
	"fmt"
	"testing"

	pianopir "github.com/dkblackley/bins-go/pir-old"
)

func testCuckooSource(items int, valueBytes int, called *bool) cuckooSource {
	return func(yield func(key uint64, value []byte) error) error {
		*called = true
		for i := 0; i < items; i++ {
			value := make([]byte, valueBytes)
			for j := range value {
				value[j] = byte(i*17 + j)
			}
			key := packedCuckooKey(uint32(i+1), uint32(i+100))
			if err := yield(key, value); err != nil {
				return err
			}
		}
		return nil
	}
}

func assertTestCuckooRows(t *testing.T, table *CuckooTable, items, valueBytes int) {
	t.Helper()
	occupied := 0
	for slot := uint64(0); slot < table.SlotCount; slot++ {
		if table.Data[slot*table.EntryWords] != 0 {
			occupied++
		}
	}
	if occupied != items {
		t.Fatalf("occupied slots=%d; want %d", occupied, items)
	}
	for i := 0; i < items; i++ {
		key := packedCuckooKey(uint32(i+1), uint32(i+100))
		first, second := table.positions(key)
		var row []uint64
		if table.Data[first*table.EntryWords] == key {
			row = table.Data[first*table.EntryWords : (first+1)*table.EntryWords]
		} else if table.Data[second*table.EntryWords] == key {
			row = table.Data[second*table.EntryWords : (second+1)*table.EntryWords]
		} else {
			t.Fatalf("key %#x is in neither hash location", key)
		}
		got, ok := cuckooResponseValue(row, key, valueBytes)
		if !ok {
			t.Fatalf("stored key %#x did not match itself", key)
		}
		want := make([]byte, valueBytes)
		for j := range want {
			want[j] = byte(i*17 + j)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("key %#x value=%v; want %v", key, got, want)
		}
	}
}

func TestCuckooBuildAndCache(t *testing.T) {
	const (
		items      = 10_000
		valueBytes = 13
	)
	config := DefaultCuckooBuildConfig()
	config.CacheDir = t.TempDir()
	called := false
	table, err := buildOrLoadCuckooTable(
		"test", items, valueBytes, 1234, config, "test.cuckoo",
		testCuckooSource(items, valueBytes, &called),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("builder source was not called")
	}
	assertTestCuckooRows(t, table, items, valueBytes)
	if table.cacheMmap == nil {
		t.Fatal("a freshly built and cached table should be served from the cache file (mmap), not the heap")
	}
	if err := table.Close(); err != nil {
		t.Fatal(err)
	}

	called = false
	loaded, err := buildOrLoadCuckooTable(
		"test", items, valueBytes, 1234, config, "test.cuckoo",
		testCuckooSource(items, valueBytes, &called),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	if called {
		t.Fatal("valid cache unexpectedly rebuilt from source")
	}
	assertTestCuckooRows(t, loaded, items, valueBytes)
}

func TestCuckooRowsThroughBatchPIR(t *testing.T) {
	const (
		items      = 512
		valueBytes = 5
	)
	config := DefaultCuckooBuildConfig()
	called := false
	table, err := buildOrLoadCuckooTable(
		"pir-test", items, valueBytes, 99, config, "", testCuckooSource(items, valueBytes, &called),
	)
	if err != nil {
		t.Fatal(err)
	}
	pir := pianopir.NewSimpleBatchPianoPIR(table.SlotCount, table.EntryWords*8, 32, table.Data, 20)
	pir.Preprocessing()

	indices := make([]uint64, 0, 32)
	keys := make([]uint64, 0, 16)
	for i := 0; i < 16; i++ {
		key := packedCuckooKey(uint32(i+1), uint32(i+100))
		first, second := table.positions(key)
		keys = append(keys, key)
		indices = append(indices, first, second)
	}
	responses, err := pir.Query(indices)
	if err != nil {
		t.Fatal(err)
	}
	for i, key := range keys {
		value, ok := cuckooResponseValue(responses[2*i], key, valueBytes)
		if !ok {
			value, ok = cuckooResponseValue(responses[2*i+1], key, valueBytes)
		}
		if !ok {
			t.Fatalf("PIR responses do not contain key %#x", key)
		}
		want := make([]byte, valueBytes)
		for j := range want {
			want[j] = byte(i*17 + j)
		}
		if !bytes.Equal(value, want) {
			t.Fatalf("PIR key %#x value=%v; want %v", key, value, want)
		}
	}

	missing := packedCuckooKey(0x80000001, 999)
	first, second := table.positions(missing)
	missResponses, err := pir.Query([]uint64{first, second})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := cuckooResponseValue(missResponses[0], missing, valueBytes); ok {
		t.Fatal("missing key matched its first slot")
	}
	if _, ok := cuckooResponseValue(missResponses[1], missing, valueBytes); ok {
		t.Fatal("missing key matched its second slot")
	}
}

func TestStage1ParentLevel(t *testing.T) {
	tests := []struct {
		node  uint32
		level int
	}{
		{1, 0},
		{128, 1},
		{255, 1},
		{16_384, 2},
		{32_767, 2},
		{2_097_152, 3},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("node_%d", test.node), func(t *testing.T) {
			level, ok := stage1ParentLevel(test.node, 128)
			if !ok || level != test.level {
				t.Fatalf("stage1ParentLevel(%d)=(%d,%v); want (%d,true)", test.node, level, ok, test.level)
			}
		})
	}
	if _, ok := stage1ParentLevel(2, 128); ok {
		t.Fatal("node 2 should not be in the r=128 artifact tree")
	}
}
