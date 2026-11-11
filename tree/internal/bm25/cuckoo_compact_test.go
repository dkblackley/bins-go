package bm25

import "testing"

// Every key must be found at one of its probes, decode to an upper bound of its
// value (exact for 8 bits), and survive a reload from the cache.
func TestCompactCuckooLayouts(t *testing.T) {
	const (
		items      = 20_000
		valueBytes = 13
	)
	layouts := []struct {
		hashes, keyBytes, valueBits int
		loadFactor                  float64
	}{
		{3, 8, 8, 0.9},
		{2, 2, 8, 0.4},
		{2, 8, 4, 0.4},
		{3, 2, 4, 0.9},
		{3, 1, 2, 0.85},
	}
	for _, layout := range layouts {
		config := DefaultCuckooBuildConfig()
		config.CacheDir = t.TempDir()
		config.Hashes, config.KeyBytes, config.ValueBits = layout.hashes, layout.keyBytes, layout.valueBits
		config.LoadFactor = layout.loadFactor
		if config.IsDefaultLayout() {
			t.Fatal("test layout is the default one")
		}
		for reload := 0; reload < 2; reload++ {
			called := false
			table, err := buildOrLoadCuckooTable("test", items, valueBytes, 99, config, "c"+config.CacheSuffix()+".cuckoo",
				testCuckooSource(items, valueBytes, &called))
			if err != nil {
				t.Fatalf("%+v: %v", layout, err)
			}
			if called == (reload == 1) {
				t.Fatalf("%+v reload %d: source called=%v", layout, reload, called)
			}
			if want := compactEntryWords(layout.keyBytes, layout.valueBits, valueBytes); table.EntryWords != want {
				t.Fatalf("%+v: entry words %d; want %d", layout, table.EntryWords, want)
			}
			falseMatches := 0
			for i := 0; i < items; i++ {
				key := packedCuckooKey(uint32(i+1), uint32(i+100))
				probes := table.probes(key)
				if len(probes) != layout.hashes {
					t.Fatalf("%d probes; want %d", len(probes), layout.hashes)
				}
				var merged []byte
				for _, slot := range probes {
					row := table.Data[slot*table.EntryWords : (slot+1)*table.EntryWords]
					if value, ok := table.decodeValue(row, key); ok {
						merged = maxValue(merged, value)
					}
				}
				found := merged != nil
				if found {
					for j, v := range merged {
						want := byte(i*17 + j)
						if v < want || (layout.valueBits == 8 && layout.keyBytes == 8 && v != want) {
							t.Fatalf("%+v key %d byte %d: decoded %d for %d", layout, i, j, v, want)
						}
					}
				}
				if !found {
					t.Fatalf("%+v: key %d not at any probe", layout, i)
				}
				missing := packedCuckooKey(uint32(i+1), uint32(i+7_000_000))
				for _, slot := range table.probes(missing) {
					if _, ok := table.decodeValue(table.Data[slot*table.EntryWords:(slot+1)*table.EntryWords], missing); ok {
						falseMatches++
					}
				}
			}
			// full keys never match a missing key; 1-byte fingerprints may (about 3/255)
			if layout.keyBytes == 8 && falseMatches != 0 {
				t.Fatalf("%+v: %d false matches with full keys", layout, falseMatches)
			}
			if layout.keyBytes == 2 && falseMatches > 10 {
				t.Fatalf("%+v: %d false matches with 2-byte fingerprints", layout, falseMatches)
			}
			table.Close()
		}
	}
}

func TestCompactConfigValidation(t *testing.T) {
	config := DefaultCuckooBuildConfig()
	if config.CacheSuffix() != "" || !config.IsDefaultLayout() {
		t.Fatal("default config must keep the original layout and cache names")
	}
	config.LoadFactor = 0.6
	if validateCompactConfig(config) == nil {
		t.Fatal("two hashes at load 0.6 accepted")
	}
	config.Hashes = 3
	if validateCompactConfig(config) != nil {
		t.Fatal("three hashes at load 0.6 rejected")
	}
	config.Hashes = 4
	if validateCompactConfig(config) == nil {
		t.Fatal("four hashes accepted")
	}
}

func TestValueStepIsUpperBound(t *testing.T) {
	for bits := 1; bits <= 8; bits++ {
		step := valueStep(bits)
		for v := 0; v < 256; v++ {
			code := (v + step - 1) / step
			if code >= 1<<bits || min(255, code*step) < v {
				t.Fatalf("bits %d v %d: code %d decodes to %d", bits, v, code, min(255, code*step))
			}
		}
	}
}
