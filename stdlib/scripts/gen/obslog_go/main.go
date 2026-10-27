// Reference outputs for obs.log (scripts/gen/obslog_gen.py runs this with
// `go run .` and feeds it JSON on stdin): go-logfmt v0.6.0 encoding and
// decoding, and the zap v1.27.0 sampler decisions.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/go-logfmt/logfmt"
	"go.uber.org/zap/zapcore"
)

type pair [3]string // key, type (s, i, b, n), value

// Entries are [time ns, zap level, message].
type sampleIn struct {
	Tick       int64           `json:"tick"`
	First      int             `json:"first"`
	Thereafter int             `json:"thereafter"`
	Raw        [][]interface{} `json:"entries"`
}

type input struct {
	Encode     [][]pair   `json:"encode"`
	EncodeSkip [][]pair   `json:"encode_skip"`
	Decode     []string   `json:"decode"`
	Streams    []string   `json:"streams"`
	Sample     []sampleIn `json:"sample"`
}

type synErr struct {
	Line int    `json:"line"`
	Pos  int    `json:"pos"`
	Msg  string `json:"msg"`
}

type decoded struct {
	Pairs [][2]string `json:"pairs"`
	Err   *synErr     `json:"err"`
}

type streamOut struct {
	Records [][][2]string `json:"records"`
	Err     *synErr       `json:"err"`
}

type output struct {
	Encode     []*string   `json:"encode"`
	EncodeSkip []string    `json:"encode_skip"`
	Decode     []decoded   `json:"decode"`
	Streams    []streamOut `json:"streams"`
	Sample     [][]bool    `json:"sample"`
}

func value(p pair) interface{} {
	switch p[1] {
	case "s":
		return p[2]
	case "i":
		v, err := strconv.ParseInt(p[2], 10, 64)
		if err != nil {
			panic(err)
		}
		return v
	case "b":
		return p[2] == "true"
	case "n":
		return nil
	}
	panic("bad type " + p[1])
}

func toSyn(err error) *synErr {
	var se *logfmt.SyntaxError
	if errors.As(err, &se) {
		return &synErr{Line: se.Line, Pos: se.Pos, Msg: se.Msg}
	}
	if err != nil {
		panic(err)
	}
	return nil
}

func decodeAll(s string) streamOut {
	dec := logfmt.NewDecoderSize(strings.NewReader(s), 4<<20)
	out := streamOut{Records: [][][2]string{}}
	for dec.ScanRecord() {
		rec := [][2]string{}
		for dec.ScanKeyval() {
			rec = append(rec, [2]string{string(dec.Key()), string(dec.Value())})
		}
		out.Records = append(out.Records, rec)
		if dec.Err() != nil {
			break
		}
	}
	out.Err = toSyn(dec.Err())
	return out
}

func main() {
	raw, err := io.ReadAll(os.Stdin)
	if err != nil {
		panic(err)
	}
	var in input
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	if err := d.Decode(&in); err != nil {
		panic(err)
	}
	var out output
	for _, kvs := range in.Encode {
		args := []interface{}{}
		for _, p := range kvs {
			args = append(args, p[0], value(p))
		}
		b, err := logfmt.MarshalKeyvals(args...)
		if err != nil {
			out.Encode = append(out.Encode, nil)
			continue
		}
		s := string(b)
		out.Encode = append(out.Encode, &s)
	}
	// Each pair through EncodeKeyval; a pair refused for its key writes
	// nothing and the record goes on.
	for _, kvs := range in.EncodeSkip {
		var buf bytes.Buffer
		enc := logfmt.NewEncoder(&buf)
		for _, p := range kvs {
			if err := enc.EncodeKeyval(p[0], value(p)); err != nil && err != logfmt.ErrInvalidKey {
				panic(err)
			}
		}
		out.EncodeSkip = append(out.EncodeSkip, buf.String())
	}
	for _, line := range in.Decode {
		so := decodeAll(line)
		dd := decoded{Pairs: [][2]string{}, Err: so.Err}
		if len(so.Records) > 0 {
			dd.Pairs = so.Records[0]
		}
		out.Decode = append(out.Decode, dd)
	}
	for _, s := range in.Streams {
		out.Streams = append(out.Streams, decodeAll(s))
	}
	for _, sc := range in.Sample {
		core := zapcore.NewCore(zapcore.NewJSONEncoder(zapcore.EncoderConfig{}), zapcore.AddSync(io.Discard), zapcore.DebugLevel)
		var got []bool
		hook := zapcore.SamplerHook(func(_ zapcore.Entry, dec zapcore.SamplingDecision) {
			got = append(got, dec&zapcore.LogSampled != 0)
		})
		s := zapcore.NewSamplerWithOptions(core, time.Duration(sc.Tick), sc.First, sc.Thereafter, hook)
		for _, e := range sc.Raw {
			t, _ := e[0].(json.Number).Int64()
			lv, _ := e[1].(json.Number).Int64()
			msg := e[2].(string)
			s.Check(zapcore.Entry{Level: zapcore.Level(lv), Time: time.Unix(0, t), Message: msg}, nil)
		}
		out.Sample = append(out.Sample, got)
	}
	w := bufio.NewWriter(os.Stdout)
	e := json.NewEncoder(w)
	e.SetEscapeHTML(false)
	if err := e.Encode(out); err != nil {
		panic(err)
	}
	w.Flush()
}
