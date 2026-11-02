#!/usr/bin/env python3
"""Unit tests of the E1 harness and write-up: python3 -m unittest test_e1 (in this directory).

The Mach-O files are built from bytes here, so the tests run on any system.
"""
import copy
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e1  # noqa: E402
import report  # noqa: E402

BUILD = "/private/var/folders/xy/T/e1-abc/a/build"
UUID = bytes(range(16))


def name16(s):
    return s.encode().ljust(16, b"\0")


def macho(oso_path=BUILD + "/hello.o", oso_mtime=1735689600, uuid=UUID, so_dir=BUILD + "/", text=b"\x1f\x20\x03\xd5" * 4):
    """A minimal 64-bit arm64 executable: __TEXT with one section, __LINKEDIT with
    a symbol table (N_SO, N_SO, N_OSO), its string table, a code signature and
    8 bytes that no load command locates; load commands LC_SEGMENT_64 (2),
    LC_SYMTAB, LC_UUID, LC_CODE_SIGNATURE."""
    strtab = b"\0" + so_dir.encode() + b"\0" + b"hello.c\0" + oso_path.encode() + b"\0"
    strtab += b"\0" * (-len(strtab) % 8)
    s_dir, s_file = 1, 2 + len(so_dir)
    s_oso = s_file + len(b"hello.c\0")
    cmds_size = (72 + 80) + 72 + 24 + 24 + 16
    text_off = 0x200
    link_off = text_off + len(text)
    symoff, nsyms = link_off, 3
    stroff = symoff + 16 * nsyms
    sig_off = stroff + len(strtab)
    sig = b"\xfa\xde\x0c\xc0" + b"\x11" * 28
    end = sig_off + len(sig) + 8
    hdr = struct.pack("<8I", 0xFEEDFACF, 0x0100000C, 0, 2, 5, cmds_size, 0, 0)
    text_seg = struct.pack("<II16sQQQQIIII", 0x19, 72 + 80, name16("__TEXT"), 0x100000000, 0x4000, 0, link_off, 5, 5, 1, 0)
    text_sec = struct.pack("<16s16sQQIIIIIIII", name16("__text"), name16("__TEXT"), 0x100000000 + text_off, len(text), text_off, 2, 0, 0, 0x80000400, 0, 0, 0)
    link_seg = struct.pack("<II16sQQQQIIII", 0x19, 72, name16("__LINKEDIT"), 0x100004000, 0x4000, link_off, end - link_off, 1, 1, 0, 0)
    symtab = struct.pack("<6I", 0x2, 24, symoff, nsyms, stroff, len(strtab))
    lc_uuid = struct.pack("<II", 0x1B, 24) + uuid
    lc_sig = struct.pack("<4I", 0x1D, 16, sig_off, len(sig))
    data = hdr + text_seg + text_sec + link_seg + symtab + lc_uuid + lc_sig
    data += b"\0" * (text_off - len(data)) + text
    for strx, ntype, value in ((s_dir, 0x64, 0), (s_file, 0x64, 0), (s_oso, 0x66, oso_mtime)):
        data += struct.pack("<IBBHQ", strx, ntype, 0 if ntype == 0x66 else 1, 1 if ntype == 0x66 else 0, value)
    data += strtab + sig + b"\0" * 8
    assert len(data) == end
    return data


class MachORegions(unittest.TestCase):
    def test_partition_covers_every_byte_once(self):
        d = macho()
        regs = e1.macho_regions(d)
        self.assertEqual(regs[0][1], 0)
        self.assertEqual(regs[-1][2], len(d))
        for (_n1, _s1, e1_), (_n2, s2, _e2) in zip(regs, regs[1:]):
            self.assertEqual(e1_, s2)
        names = [r[0] for r in regs]
        for want in ("header", "load command LC_SEGMENT_64 __TEXT", "LC_UUID", "load command LC_SYMTAB",
                     "load command LC_CODE_SIGNATURE", "__TEXT,__text", "symbol table", "string table",
                     "code signature", "segment __TEXT, other", "segment __LINKEDIT, other"):
            self.assertIn(want, names)

    def test_not_macho(self):
        self.assertIsNone(e1.macho_regions(b"\x7fELF" + b"\0" * 60))
        self.assertIsNone(e1.macho_regions(b""))
        self.assertIsNone(e1.diff_macho(b"abc", macho()))
        self.assertIsNone(e1.macho_debug_map(b"\xca\xfe\xba\xbe" + b"\0" * 40))

    def test_bad_load_commands(self):
        d = bytearray(macho())
        struct.pack_into("<I", d, 20, len(d) + 1)  # sizeofcmds past the end
        self.assertIsNone(e1.macho_regions(bytes(d)))
        d = bytearray(macho())
        struct.pack_into("<I", d, 32 + 4, 4)  # a load command smaller than its own header
        self.assertIsNone(e1.macho_regions(bytes(d)))

    def test_uuid_change_is_attributed_to_lc_uuid(self):
        self.assertEqual(e1.diff_macho(macho(), macho(uuid=bytes(b ^ 0xFF for b in UUID))), {"LC_UUID": 16})

    def test_object_mtime_is_attributed_to_the_symbol_table(self):
        self.assertEqual(e1.diff_macho(macho(), macho(oso_mtime=1735689601)), {"symbol table": 1})

    def test_longer_path_counts_only_the_string_table(self):
        a = macho()
        # only the last string grows, so no string index in the symbol table moves
        b = macho(oso_path="/private/var/folders/xy/T/e1-abc/another-directory/build/hello.o")
        got = e1.diff_macho(a, b)
        self.assertEqual(set(got) - {"code signature", "load command LC_CODE_SIGNATURE", "load command LC_SYMTAB",
                                     "load command LC_SEGMENT_64 __LINKEDIT", "segment __LINKEDIT, other"},
                         {"string table"})
        self.assertNotIn("symbol table", got)
        self.assertNotIn("__TEXT,__text", got)

    def test_debug_map(self):
        dm = e1.macho_debug_map(macho(), [(BUILD, "{dir}")])
        self.assertEqual(dm["uuid"], UUID.hex())
        self.assertEqual(dm["so"], ["{dir}/", "hello.c"])
        self.assertEqual(dm["oso"], [["{dir}/hello.o", 1735689600]])


class Compare(unittest.TestCase):
    def test_records_regions_and_debug_maps_of_differing_macho_files(self):
        a = {"hello": macho(), "notes.txt": b"x"}
        b = {"hello": macho(oso_mtime=1735689602), "notes.txt": b"y"}
        res = e1.compare((a, None), (b, None), [(BUILD, "{dir}")])
        self.assertEqual(res["status"], "differs")
        rows = {r["file"]: r for r in res["files"]}
        self.assertEqual(rows["hello"]["macho_regions"], {"symbol table": 1})
        self.assertEqual(rows["hello"]["macho_debug_map"]["a"]["oso"], [["{dir}/hello.o", 1735689600]])
        self.assertEqual(rows["hello"]["macho_debug_map"]["b"]["oso"], [["{dir}/hello.o", 1735689602]])
        self.assertNotIn("macho_regions", rows["notes.txt"])
        self.assertNotIn("sections", rows["notes.txt"])

    def test_identical_macho_files_record_nothing_more(self):
        res = e1.compare(({"hello": macho()}, None), ({"hello": macho()}, None))
        self.assertEqual(res["status"], "identical")
        self.assertNotIn("macho_regions", res["files"][0])


class Machine(unittest.TestCase):
    def test_macos_facts(self):
        out = {
            ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M5 Max",
            ("sysctl", "-n", "hw.model"): "Mac17,7",
            ("sw_vers", "-productName"): "macOS",
            ("sw_vers", "-productVersion"): "27.2",
            ("sw_vers", "-buildVersion"): "27C100",
        }
        with mock.patch.object(e1.platform, "system", return_value="Darwin"), \
                mock.patch.object(e1, "command_output", side_effect=lambda argv: out[tuple(argv)]):
            m = e1.machine()
        self.assertEqual((m["cpu"], m["model"], m["os"]), ("Apple M5 Max", "Mac17,7", "macOS 27.2 (27C100)"))

    def test_command_output_of_a_missing_command(self):
        self.assertEqual(e1.command_output(["/nonexistent/e1-no-such-command"]), "")


def results(schema, system, files):
    return {
        "schema": schema, "date_utc": "2026-11-02T00:00:00Z",
        "machine": {"system": system, "release": "", "machine": "arm64", "cpu": "", "cpus": 1, "os": "", "python": "3"},
        "environment": {"LC_ALL": "C"}, "harness": {"e1.py": "0" * 64, "matrix.json": "0" * 64},
        "toolchains": {"clang": "clang 1"}, "absent": [],
        "rows": [{
            "config": "clang -O2 -g", "tool": "clang", "language": "C", "program": "c/hello.c",
            "same_dir": {"status": "differs", "files": files},
            "cross_dir": {"status": "identical", "files": []},
            "remapped": {"status": "not tested", "error": ""},
            "same_dir_clock": {"status": "identical", "files": []},
            "path_copies_cross": 0, "path_copies_remapped": None, "sha256": {}, "sha256_remapped": None,
        }],
    }


class Report(unittest.TestCase):
    def render(self, res):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "r.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(res, f)
            return report.render([p])

    def test_schema_1_macos_file_is_marked(self):
        md = self.render(results("rce-compile-e1-results/1", "Darwin",
                                 [{"file": "hello", "size_a": 9, "size_b": 9, "differing_bytes": 3}]))
        self.assertIn("written without Mach-O attribution", md)
        self.assertIn("| 3 | without Mach-O attribution |", md)

    def test_schema_1_linux_file_is_not_marked(self):
        md = self.render(results("rce-compile-e1-results/1", "Linux",
                                 [{"file": "hello", "size_a": 9, "size_b": 9, "differing_bytes": 3}]))
        self.assertNotIn("without Mach-O attribution", md)

    def test_schema_2_regions_and_debug_map(self):
        a = e1.macho_debug_map(macho(), [(BUILD, "{dir}")])
        b = copy.deepcopy(a)
        b["oso"][0][1] += 1
        md = self.render(results("rce-compile-e1-results/2", "Darwin", [{
            "file": "hello", "size_a": 9, "size_b": 9, "differing_bytes": 1,
            "macho_regions": {"symbol table": 1}, "macho_debug_map": {"a": a, "b": b}}]))
        self.assertNotIn("without Mach-O attribution", md)
        self.assertIn("`symbol table` 1", md)
        self.assertIn("same: `%s`" % UUID.hex(), md)
        self.assertIn("a only: `{dir}/hello.o` mtime 1735689600<br>b only: `{dir}/hello.o` mtime 1735689601<br>0 in both", md)
        self.assertIn("| 2 in both |", md)

    def test_unknown_schema_is_refused(self):
        with self.assertRaises(SystemExit):
            self.render(results("rce-compile-e1-results/9", "Linux", []))

    def test_committed_write_up_is_current(self):
        paths = sorted(os.path.join(HERE, "results", f) for f in os.listdir(os.path.join(HERE, "results")) if f.endswith(".json"))
        with open(os.path.join(HERE, "RESULTS.md"), encoding="utf-8") as f:
            self.assertEqual(f.read(), report.render(paths))


if __name__ == "__main__":
    unittest.main()
