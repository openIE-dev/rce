// Reference syntax verdicts for data.json from Go 1.24 encoding/json.
//
// For every file of vectors/data/json/JSONTestSuite_test_parsing and every
// input of vectors/data/json/syntax_corpus.json (written by json_gen.py
// corpus: seeded mutations, and string escapes around lone surrogates),
// json.Unmarshal into a json.RawMessage runs Go's full syntax check
// (checkValid: the RFC 8259 grammar, nesting up to 10000 levels) and
// nothing else, so its *json.SyntaxError gives the verdict and the offset.
// Go's Offset counts the bytes read including the offending one; at the
// end of input Go steps one virtual space, so the Offset is then the length.
//
//	cd stdlib && go run scripts/gen/json_go.go > vectors/data/json/go1.24_syntax.json
package main

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
)

type verdict struct {
	Valid  bool   `json:"valid"`
	Offset int64  `json:"offset"`
	Msg    string `json:"msg"`
}

func check(b []byte) verdict {
	var m json.RawMessage
	err := json.Unmarshal(b, &m)
	if err == nil {
		return verdict{Valid: true, Offset: -1}
	}
	var se *json.SyntaxError
	if errors.As(err, &se) {
		return verdict{Valid: false, Offset: se.Offset, Msg: se.Error()}
	}
	return verdict{Valid: false, Offset: -2, Msg: err.Error()}
}

func main() {
	out := map[string]any{"go": runtime.Version()}
	suite := map[string]verdict{}
	files, err := filepath.Glob("vectors/data/json/JSONTestSuite_test_parsing/*.json")
	if err != nil || len(files) == 0 {
		fmt.Fprintln(os.Stderr, "no suite files (run from stdlib/)")
		os.Exit(1)
	}
	sort.Strings(files)
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			panic(err)
		}
		suite[filepath.Base(f)] = check(b)
	}
	out["suite"] = suite
	var corpus map[string][]string
	cb, err := os.ReadFile("vectors/data/json/syntax_corpus.json")
	if err != nil {
		panic(err)
	}
	if err := json.Unmarshal(cb, &corpus); err != nil {
		panic(err)
	}
	for _, k := range []string{"mutations", "lone"} {
		var cv []verdict
		for _, h := range corpus[k] {
			b, err := hex.DecodeString(h)
			if err != nil {
				panic(err)
			}
			cv = append(cv, check(b))
		}
		out[k] = cv
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", " ")
	if err := enc.Encode(out); err != nil {
		panic(err)
	}
}
