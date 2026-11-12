package bm25

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// vocabularyTermID accepts both the numeric values emitted by the newer
// builders and the quoted decimal values in the existing MS MARCO artifact.
type vocabularyTermID int

func (termID *vocabularyTermID) UnmarshalJSON(encoded []byte) error {
	encoded = bytes.TrimSpace(encoded)
	if len(encoded) >= 2 && encoded[0] == '"' && encoded[len(encoded)-1] == '"' {
		encoded = encoded[1 : len(encoded)-1]
	}
	if len(encoded) == 0 {
		return fmt.Errorf("empty term ID")
	}
	value := uint64(0)
	for _, digit := range encoded {
		if digit < '0' || digit > '9' {
			return fmt.Errorf("term ID %q is not a nonnegative decimal integer", encoded)
		}
		value = value*10 + uint64(digit-'0')
		if value > math.MaxInt32-1 {
			return fmt.Errorf("term ID %q exceeds the Cuckoo key range", encoded)
		}
	}
	*termID = vocabularyTermID(value)
	return nil
}

// loadVocabulary streams the JSON object instead of first reading the whole
// file. The production vocabulary has millions of entries, so this avoids
// retaining an additional copy of its JSON representation during startup.
func loadVocabulary(path string) (map[string]int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	capacity := 0
	if stat, statErr := file.Stat(); statErr == nil && stat.Size() > 0 {
		// The current artifact averages about 25 encoded bytes per entry.
		capacity = int(stat.Size() / 24)
	}
	vocabulary := make(map[string]int, capacity)
	decoder := json.NewDecoder(bufio.NewReaderSize(file, 1<<20))
	opening, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	if delimiter, ok := opening.(json.Delim); !ok || delimiter != '{' {
		return nil, fmt.Errorf("vocabulary must be a JSON object")
	}
	for decoder.More() {
		keyToken, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		term, ok := keyToken.(string)
		if !ok {
			return nil, fmt.Errorf("vocabulary key is not a string")
		}
		var termID vocabularyTermID
		if err := decoder.Decode(&termID); err != nil {
			return nil, fmt.Errorf("term %q: %w", term, err)
		}
		vocabulary[term] = int(termID)
	}
	closing, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	if delimiter, ok := closing.(json.Delim); !ok || delimiter != '}' {
		return nil, fmt.Errorf("vocabulary JSON object is not terminated")
	}
	return vocabulary, nil
}
