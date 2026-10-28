// SPDX-License-Identifier: MIT OR Apache-2.0
// Base64 encoding written with shifts and masks, checked against
// encoding/base64 and decoded back.
package main

import (
	"encoding/base64"
	"fmt"
)

const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

func encode(b []byte) string {
	out := make([]byte, 0, (len(b)+2)/3*4)
	for i := 0; i < len(b); i += 3 {
		var n uint32
		k := 0
		for j := 0; j < 3; j++ {
			n <<= 8
			if i+j < len(b) {
				n |= uint32(b[i+j])
				k++
			}
		}
		for j := 0; j < 4; j++ {
			if j <= k {
				out = append(out, alphabet[(n>>(18-6*j))&63])
			} else {
				out = append(out, '=')
			}
		}
	}
	return string(out)
}

func main() {
	for _, s := range []string{"", "f", "fo", "foo", "foob", "fooba", "foobar", "héllo wörld", "\x00\xff\xfe"} {
		e := encode([]byte(s))
		d, err := base64.StdEncoding.DecodeString(e)
		fmt.Printf("%q %s %v %v %v\n", s, e, e == base64.StdEncoding.EncodeToString([]byte(s)), string(d) == s, err)
	}
}
