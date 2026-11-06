package pir

import (
	"bytes"
	"encoding/gob"
)

// Warning: does not store the A matrix (to save space)
func (s *Server[T]) GobEncode() ([]byte, error) {
	buf := new(bytes.Buffer)
	enc := gob.NewEncoder(buf)
	err := enc.Encode(s.params)
	if err != nil {
		return buf.Bytes(), err
	}

	err = enc.Encode(s.db)
	if err != nil {
		return buf.Bytes(), err
	}

	err = enc.Encode(s.hint)
	return buf.Bytes(), err
}

func (s *Server[T]) GobDecode(buf []byte) error {
	b := bytes.NewBuffer(buf)
	dec := gob.NewDecoder(b)
	err := dec.Decode(&s.params)
	if err != nil {
		return err
	}

	err = dec.Decode(&s.db)
	if err != nil {
		return err
	}

	err = dec.Decode(&s.hint)
	return err
}
