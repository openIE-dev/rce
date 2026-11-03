#!/usr/bin/env python3
"""Unit tests of E5: the alpha-equivalence hash, the WARP-style function ID, the renamer and the report.

The two properties PROTOCOL.md requires of the IR-level hash are tested on
modules built here instruction by instruction: renaming every identifier
(every bound index, and the names section) leaves the hash unchanged, and
changing any one opcode or constant changes it.

    python3 test_e5.py
"""
import glob
import hashlib
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import irhash  # noqa: E402
import machine  # noqa: E402
import rename  # noqa: E402
import report  # noqa: E402
import wasm_eps  # noqa: E402

I32, I64 = 0x7F, 0x7E
E4_PROGRAMS = os.path.join(HERE, "..", "e4-energy", "programs")


def u(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def s(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        done = (n == 0 and not b & 0x40) or (n == -1 and b & 0x40)
        out.append(b | (0 if done else 0x80))
        if done:
            return bytes(out)


def vec(items):
    return u(len(items)) + b"".join(items)


def section(sid, payload):
    return bytes([sid]) + u(len(payload)) + payload


def name(t):
    b = t.encode()
    return u(len(b)) + b


OPS = {"local.get": 0x20, "local.set": 0x21, "local.tee": 0x22, "global.get": 0x23, "global.set": 0x24,
       "call": 0x10, "i32.const": 0x41, "i64.const": 0x42, "i32.add": 0x6A, "i32.sub": 0x6B, "i32.mul": 0x6C,
       "i32.lt_u": 0x49, "i32.eqz": 0x45, "i32.load": 0x28, "i32.store": 0x36, "if": 0x04, "else": 0x05,
       "end": 0x0B, "loop": 0x03, "block": 0x02, "br_if": 0x0D, "br": 0x0C, "return": 0x0F, "drop": 0x1A,
       "i64.extend_i32_u": 0xAD, "i64.mul": 0x7E, "i32.wrap_i64": 0xA7, "nop": 0x01, "select": 0x1B}


def asm(code):
    out = bytearray()
    for ins in code:
        op = ins[0]
        out.append(OPS[op])
        if op in ("local.get", "local.set", "local.tee", "global.get", "global.set", "call", "br_if", "br"):
            out += u(ins[1])
        elif op == "i32.const":
            out += s(ins[1] - (1 << 32) if ins[1] >= 1 << 31 else ins[1])
        elif op == "i64.const":
            out += s(ins[1] - (1 << 64) if ins[1] >= 1 << 63 else ins[1])
        elif op in ("i32.load", "i32.store"):
            out += u(2) + u(ins[1])
        elif op in ("if", "loop", "block"):
            out.append(ins[1])
    return bytes(out)


class Spec:
    """A module written as data, so that its indices can be permuted."""

    def __init__(self):
        # types: 0 = (i32) -> i32, 1 = (i32, i32) -> i32
        self.types = [([I32], [I32]), ([I32, I32], [I32])]
        # globals: (type, mut, init value)
        self.globals = [(I32, 1, 66560), (I32, 1, 7), (I32, 0, 1024)]
        # functions: (type, locals [types], code); function 0 is exported as run
        self.funcs = [
            (0, [I32, I32, I32], [
                ("local.get", 0), ("i32.const", 3), ("call", 1), ("local.set", 1),
                ("global.get", 1), ("local.get", 1), ("call", 2), ("local.tee", 2),
                ("i32.const", 2000), ("i32.load", 4), ("i32.add",), ("local.set", 3),
                ("local.get", 3), ("local.get", 2), ("i32.lt_u",),
                ("if", I32), ("local.get", 3), ("else",), ("local.get", 2), ("end",),
                ("global.get", 1), ("i32.const", 1), ("i32.add",), ("global.set", 1),
                ("end",)]),
            (1, [I32], [
                ("local.get", 0), ("local.get", 1), ("i32.mul",), ("local.set", 2),
                ("block", 0x40), ("loop", 0x40), ("local.get", 2), ("i32.eqz",), ("br_if", 1),
                ("local.get", 2), ("i32.const", 1), ("i32.sub",), ("local.set", 2), ("br", 0), ("end",), ("end",),
                ("local.get", 0), ("i64.extend_i32_u",), ("i64.const", 0x100000001), ("i64.mul",),
                ("i32.wrap_i64",), ("end",)]),
            (1, [I64], [
                ("local.get", 0), ("local.get", 1), ("i32.sub",), ("global.get", 0), ("drop",),
                ("i32.const", 2004), ("local.get", 0), ("i32.store", 8), ("end",)]),
        ]
        self.exports = [("run", 0, 0), ("memory", 2, 0)]
        self.datas = [(2000, b"\x01\x02\x03\x04"), (2008, b"\x09\x08")]
        self.names = ["run", "helper", "store"]

    def encode(self, names_section=True):
        out = b"\0asm\x01\0\0\0"
        out += section(1, vec([b"\x60" + vec([bytes([p]) for p in ps]) + vec([bytes([r]) for r in rs])
                               for ps, rs in self.types]))
        out += section(3, vec([u(t) for t, _l, _c in self.funcs]))
        out += section(5, vec([b"\x00" + u(1)]))
        out += section(6, vec([bytes([t, m]) + b"\x41" + s(v) + b"\x0b" for t, m, v in self.globals]))
        out += section(7, vec([name(n) + bytes([k]) + u(i) for n, k, i in self.exports]))
        bodies = []
        for _t, locs, code in self.funcs:
            body = vec([u(1) + bytes([t]) for t in locs]) + asm(code)
            bodies.append(u(len(body)) + body)
        out += section(10, vec(bodies))
        out += section(11, vec([b"\x00\x41" + s(off) + b"\x0b" + u(len(b)) + b for off, b in self.datas]))
        if names_section:
            fn = vec([u(i) + name(n) for i, n in enumerate(self.names)])
            out += section(0, name("name") + bytes([1]) + u(len(fn)) + fn)
        return out

    def permuted(self, seed):
        """The same module with every bound index permuted and every name changed."""
        rnd = random.Random(seed)
        sp = Spec.__new__(Spec)
        # types
        tperm = list(range(len(self.types)))
        rnd.shuffle(tperm)
        sp.types = [None] * len(self.types)
        for old, new in enumerate(tperm):
            sp.types[new] = self.types[old]
        # functions
        fperm = list(range(len(self.funcs)))
        rnd.shuffle(fperm)
        gperm = list(range(len(self.globals)))
        rnd.shuffle(gperm)
        sp.globals = [None] * len(self.globals)
        for old, new in enumerate(gperm):
            sp.globals[new] = self.globals[old]
        sp.funcs = [None] * len(self.funcs)
        for old, (t, locs, code) in enumerate(self.funcs):
            np = len(self.types[t][0])
            lperm = list(range(len(locs)))
            rnd.shuffle(lperm)
            nlocs = [None] * len(locs)
            for o, n in enumerate(lperm):
                nlocs[n] = locs[o]
            ncode = []
            for ins in code:
                if ins[0] in ("local.get", "local.set", "local.tee") and ins[1] >= np:
                    ins = (ins[0], np + lperm[ins[1] - np])
                elif ins[0] in ("global.get", "global.set"):
                    ins = (ins[0], gperm[ins[1]])
                elif ins[0] == "call":
                    ins = (ins[0], fperm[ins[1]])
                ncode.append(ins)
            sp.funcs[fperm[old]] = (tperm[t], nlocs, ncode)
        sp.exports = [(n, k, fperm[i] if k == 0 else i) for n, k, i in reversed(self.exports)]
        sp.datas = list(reversed(self.datas))
        sp.names = ["f%d_%d" % (seed, k) for k in range(len(self.funcs))]
        return sp


class IRHashTest(unittest.TestCase):
    def setUp(self):
        self.spec = Spec()
        self.base = irhash.r_hash(irhash.Module(self.spec.encode()))[0]

    def test_module_parses_and_counts_like_e4(self):
        data = self.spec.encode()
        self.assertEqual(irhash.Module(data).ir_nodes(), wasm_eps.Module(data).ir_nodes())

    def test_renaming_every_identifier_keeps_the_hash(self):
        for seed in range(40):
            sp = self.spec.permuted(seed)
            m = irhash.Module(sp.encode())
            self.assertEqual(irhash.r_hash(m)[0], self.base, "seed %d" % seed)
            self.assertEqual(irhash.r_hash(m, "run")[0], irhash.r_hash(irhash.Module(self.spec.encode()), "run")[0])

    def test_the_permuted_module_is_a_different_byte_string(self):
        self.assertNotEqual(self.spec.permuted(3).encode(), self.spec.encode())

    def test_names_section_is_not_hashed(self):
        self.assertEqual(irhash.r_hash(irhash.Module(self.spec.encode(names_section=False)))[0], self.base)

    def test_permuted_module_runs_the_same(self):
        for seed in range(5):
            a = wasm_eps.count(self.spec.encode(), "run", [5])["result"]
            b = wasm_eps.count(self.spec.permuted(seed).encode(), "run", [5])["result"]
            self.assertEqual(a, b)

    def test_any_opcode_or_constant_change_changes_the_hash(self):
        m = irhash.Module(self.spec.encode())
        muts = irhash.mutations(m)
        kinds = {k for _f, _i, k in muts}
        self.assertEqual(kinds, {"opcode", "constant", "offset"})
        for mut in muts:
            undo = irhash.apply_mutation(m, mut)
            self.assertNotEqual(irhash.r_hash(m)[0], self.base, str(mut))
            undo()
        self.assertEqual(irhash.r_hash(m)[0], self.base)

    def test_every_opcode_change_at_the_byte_level_changes_the_hash(self):
        """Rewrite each opcode byte of the encoded module to every other opcode of its class."""
        for fi, (_t, _l, code) in enumerate(self.spec.funcs):
            for ii, ins in enumerate(code):
                op = OPS[ins[0]]
                if op not in irhash.CLASS_OF:
                    continue
                for other in irhash.CLASS_OF[op]:
                    if other == op:
                        continue
                    name_of = {v: k for k, v in OPS.items()}
                    sp = Spec()
                    new = (name_of.get(other, None),) + tuple(ins[1:])
                    if new[0] is None:
                        continue
                    sp.funcs[fi][2][ii] = new
                    self.assertNotEqual(irhash.r_hash(irhash.Module(sp.encode()))[0], self.base, str((fi, ii, new)))

    def test_global_initial_value_and_data_bytes_change_the_hash(self):
        sp = Spec()
        sp.globals[1] = (I32, 1, 8)
        self.assertNotEqual(irhash.r_hash(irhash.Module(sp.encode()))[0], self.base)
        sp = Spec()
        sp.datas[0] = (2000, b"\x01\x02\x03\x05")
        self.assertNotEqual(irhash.r_hash(irhash.Module(sp.encode()))[0], self.base)
        sp = Spec()
        sp.exports[0] = ("run2", 0, 0)
        self.assertNotEqual(irhash.r_hash(irhash.Module(sp.encode()))[0], self.base)

    def test_swapping_which_function_is_called_changes_the_hash(self):
        sp = Spec()
        code = sp.funcs[0][2]
        code[2], code[6] = ("call", 2), ("call", 1)
        self.assertNotEqual(irhash.r_hash(irhash.Module(sp.encode()))[0], self.base)

    def test_entry_hash_ignores_unreachable_functions(self):
        sp = Spec()
        sp.funcs.append((0, [], [("local.get", 0), ("end",)]))
        sp.names.append("dead")
        m = irhash.Module(sp.encode())
        self.assertNotEqual(irhash.r_hash(m)[0], self.base)
        self.assertEqual(irhash.r_hash(m, "run")[0], irhash.r_hash(irhash.Module(self.spec.encode()), "run")[0])

    def test_encoding_is_prefix_free(self):
        self.assertNotEqual(irhash.enc(("ab", "c")), irhash.enc(("a", "bc")))
        self.assertNotEqual(irhash.enc((1, (2,))), irhash.enc(((1,), 2)))
        self.assertNotEqual(irhash.enc(b"1"), irhash.enc(1))


# -- the machine-code level --------------------------------------------------------------

class X86Test(unittest.TestCase):
    def insns(self, code):
        out, p = [], 0
        while p < len(code):
            n, k, t = machine.x86_insn(code, p)
            out.append((p, n, k, t))
            p += n
        return out

    def test_nops(self):
        for b in (b"\x90", b"\x66\x90", b"\x0f\x1f\x00", b"\x0f\x1f\x40\x00", b"\x0f\x1f\x44\x00\x00",
                  b"\x66\x0f\x1f\x44\x00\x00", b"\x0f\x1f\x80\x00\x00\x00\x00", b"\x0f\x1f\x84\x00\x00\x00\x00\x00",
                  b"\x66\x2e\x0f\x1f\x84\x00\x00\x00\x00\x00", b"\x66\x66\x2e\x0f\x1f\x84\x00\x00\x00\x00\x00"):
            self.assertEqual(self.insns(b), [(0, len(b), "nop", None)], b.hex())
        self.assertEqual(self.insns(b"\xf3\x90")[0][2], "op")    # PAUSE
        self.assertEqual(self.insns(b"\x41\x90")[0][2], "op")    # XCHG eax, r8d

    def test_self_moves(self):
        self.assertEqual(self.insns(b"\x48\x89\xc0")[0][2], "selfmove")  # mov rax, rax
        self.assertEqual(self.insns(b"\x4d\x89\xc0")[0][2], "selfmove")  # mov r8, r8
        self.assertEqual(self.insns(b"\x49\x89\xc0")[0][2], "op")        # mov r8, rax
        self.assertEqual(self.insns(b"\x89\xff")[0][2], "op")            # mov edi, edi clears the upper half

    def test_lengths_and_branches(self):
        code = (b"\x55"                      # push rbp
                b"\x48\x89\xe5"              # mov rbp, rsp
                b"\x83\xff\x02"              # cmp edi, 2
                b"\x72\x05"                  # jb +5
                b"\x8d\x04\xbd\x00\x00\x00\x00"  # lea eax, [rdi*4+0]
                b"\xc3"                      # ret
                b"\x0f\x84\xf0\xff\xff\xff"  # je -16
                b"\xe8\x00\x00\x00\x00"      # call rel32
                b"\x48\xb8" + bytes(8) +     # movabs rax, imm64
                b"\xc7\x45\xfc\x01\x00\x00\x00"  # mov dword [rbp-4], 1
                b"\x66\x0f\x6f\x05\x00\x00\x00\x00"  # movdqa xmm0, [rip+0]
                b"\x66\x0f\x3a\x0f\xc1\x08"  # palignr xmm0, xmm1, 8
                b"\xc3")
        got = self.insns(code)
        self.assertEqual([n for _p, n, _k, _t in got], [1, 3, 3, 2, 7, 1, 6, 5, 10, 7, 8, 6, 1])
        self.assertEqual(got[3][2:], ("jcc", 14))
        self.assertEqual(got[6][2:], ("jcc", 17 + 6 - 16))

    def test_function_id_drops_nops_and_zeroes_relocated_instructions(self):
        f = b"\x55\xe8\x00\x00\x00\x00\x5d\xc3"
        base = machine.function_id(f, machine.native_blocks(f, machine.x86_insn), [2])[0]
        g = b"\x55\x0f\x1f\x00\xe8\x11\x22\x33\x44\x5d\x90\xc3"
        self.assertEqual(machine.function_id(g, machine.native_blocks(g, machine.x86_insn), [5])[0], base)
        h = b"\x55\xe8\x00\x00\x00\x00\x5e\xc3"   # pop rsi instead of pop rbp
        self.assertNotEqual(machine.function_id(h, machine.native_blocks(h, machine.x86_insn), [2])[0], base)
        # without the relocation entry the call's bytes are hashed
        self.assertNotEqual(machine.function_id(g, machine.native_blocks(g, machine.x86_insn), [])[0], base)

    def test_blocks_are_ordered_by_start_address(self):
        f = b"\x85\xff\x74\x01\xc3\xc3"   # test edi,edi; je +1; ret; ret
        blocks = machine.native_blocks(f, machine.x86_insn)
        self.assertEqual([b[0][0] for b in blocks], [0, 4, 5])


class Arm64Test(unittest.TestCase):
    def test_decoding(self):
        def dec(word, at):
            return machine.arm64_insn(bytes(at) + word.to_bytes(4, "little"), at)
        self.assertEqual(dec(0xD503201F, 0)[1], "nop")
        self.assertEqual(dec(0xAA0103E1, 0)[1], "selfmove")   # mov x1, x1
        self.assertEqual(dec(0xAA0203E1, 0)[1], "op")         # mov x1, x2
        self.assertEqual(dec(0x2A0103E1, 0)[1], "op")         # mov w1, w1 is kept
        self.assertEqual(dec(0xD65F03C0, 0)[1], "ret")
        self.assertEqual(dec(0x14000002, 8)[1:], ("jmp", 16))
        self.assertEqual(dec(0x54FFFFE0, 8)[1:], ("jcc", 4))   # b.eq -4
        self.assertEqual(dec(0x34000040, 0)[1:], ("jcc", 8))   # cbz w0, +8
        self.assertEqual(dec(0x94000000, 0)[1], "op")          # bl does not end a block


def have(*tools):
    return all(shutil.which(t) for t in tools)


@unittest.skipUnless(have("clang"), "clang is not on PATH")
class ObjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(prefix="e5-test-")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def build(self, argv, out):
        p = subprocess.run(argv + ["-o", os.path.join(self.d, out)], capture_output=True)
        if p.returncode:
            self.skipTest("build failed: %s" % p.stderr.decode()[-200:])
        return open(os.path.join(self.d, out), "rb").read()

    def test_elf_object_functions_and_relocations(self):
        data = self.build(["clang", "-c", "-O0", os.path.join(E4_PROGRAMS, "c", "fib.c")], "fib.o")
        res = machine.object_ids(data)
        self.assertEqual(res["status"], "hashed")
        self.assertEqual(sorted(f["name"] for f in res["functions"]), ["fib", "run"])
        # a call within the section to a static function needs no relocation; sieve's globals do
        res = machine.object_ids(self.build(["clang", "-c", "-O0", os.path.join(E4_PROGRAMS, "c", "sieve.c")], "s.o"))
        self.assertGreater(res["functions"][0]["relocs"], 0)

    def test_no_symbol_table(self):
        data = bytearray(self.build(["clang", "-c", "-O0", os.path.join(E4_PROGRAMS, "c", "fib.c")], "fib.o"))
        shoff, shnum = int.from_bytes(data[0x28:0x30], "little"), int.from_bytes(data[0x3C:0x3E], "little")
        for i in range(shnum):
            h = shoff + 64 * i
            if int.from_bytes(data[h + 4:h + 8], "little") == 2:
                data[h + 4:h + 8] = (0).to_bytes(4, "little")
        res = machine.object_ids(bytes(data))
        self.assertEqual(res["status"], "functions cannot be located: no symbol table")
        self.assertIsNone(machine.module_id(res))

    def test_macho_object(self):
        data = self.build(["clang", "--target=arm64-apple-macos11", "-ffreestanding", "-c", "-O1",
                           os.path.join(E4_PROGRAMS, "c", "fib.c")], "fib-macho.o")
        res = machine.object_ids(data)
        self.assertEqual((res["format"], res["arch"], res["status"]), ("mach-o", "arm64", "hashed"))
        self.assertIn("_run", [f["name"] for f in res["functions"]])
        self.assertIsNotNone(machine.entry_id(res, "run"))

    def test_wasm_object(self):
        data = self.build(["clang", "--target=wasm32", "-c", "-O0", os.path.join(E4_PROGRAMS, "c", "fib.c")], "fib-w.o")
        res = machine.object_ids(data)
        self.assertEqual((res["format"], res["status"]), ("wasm", "hashed"))
        self.assertEqual(sorted(f["name"] for f in res["functions"]), ["fib", "run"])
        self.assertTrue(any(f["relocs"] for f in res["functions"]))

    def test_same_object_in_two_directories(self):
        src = open(os.path.join(E4_PROGRAMS, "c", "sort.c"), "rb").read()
        ids = []
        for sub in ("a", "b"):
            os.makedirs(os.path.join(self.d, sub), exist_ok=True)
            p = os.path.join(self.d, sub, "sort.c")
            open(p, "wb").write(src)
            ids.append(machine.module_id(machine.object_ids(self.build(["clang", "-c", "-O2", p], sub + "/sort.o"))))
        self.assertEqual(ids[0], ids[1])

    @unittest.skipUnless(have("llvm-objdump"), "llvm-objdump is not on PATH")
    def test_x86_lengths_agree_with_llvm_objdump(self):
        import e5
        n = 0
        for k in ("sumsq", "collatz", "sieve", "sort", "crc32", "matmul", "fib", "popcount"):
            for o in ("-O0", "-O2", "-O3", "-Oz"):
                out = "%s%s.o" % (k, o)
                data = self.build(["clang", "-c", o, os.path.join(E4_PROGRAMS, "c", k + ".c")], out)
                agr = e5.objdump_agreement(os.path.join(self.d, out), data)
                if agr is None:
                    self.skipTest("not an x86-64 host")
                self.assertEqual(agr["agree"], agr["functions"], out)
                n += agr["functions"]
        self.assertGreater(n, 0)

    @unittest.skipUnless(have("wasm-ld"), "wasm-ld is not on PATH")
    def test_renamed_kernel_has_the_same_ir_hash(self):
        for k in ("fib", "sieve", "matmul"):
            text = open(os.path.join(E4_PROGRAMS, "c", k + ".c")).read()
            rtext, mapping = rename.rename("c", text)
            self.assertTrue(mapping)
            hs = []
            for tag, t in (("o", text), ("r", rtext)):
                p = os.path.join(self.d, "%s-%s.c" % (k, tag))
                open(p, "w").write(t)
                data = self.build(["clang", "--target=wasm32", "-nostdlib", "-Wl,--no-entry", "-Wl,--export=run", "-O2",
                                   p], "%s-%s.wasm" % (k, tag))
                hs.append(irhash.r_hash(irhash.Module(data))[0])
            self.assertEqual(hs[0], hs[1], k)


class RenameTest(unittest.TestCase):
    def test_c(self):
        text = "#include <stdint.h>\n#define CAP 4u\nstatic uint32_t a[CAP];\n/* note */\n" \
               "uint32_t run(uint32_t n) { uint32_t s = 0; for (uint32_t i = 0; i < n; i++) s += a[i]; return s; }\n"
        out, mapping = rename.rename("c", text)
        self.assertEqual(set(mapping), {"CAP", "a", "n", "s", "i"})
        self.assertIn("uint32_t run(", out)
        self.assertIn("#include <stdint.h>", out)
        self.assertIn("#define n0_", out)
        self.assertNotIn("note", out)
        self.assertEqual(rename.rename("c", text), (out, mapping))

    def test_rust_keeps_paths_methods_macros_and_captures(self):
        text = ('fn area(x: u32) -> u32 { x }\nfn main() { let k = 3u32; let v = k.wrapping_add(1); '
                'println!("{k} {}", v); std::process::exit(0); let s = 0..k; }\n')
        out, mapping = rename.rename("rust", text)
        self.assertEqual(set(mapping), {"area", "x", "v", "s"})
        for kept in ("wrapping_add", "println!", "std::process::exit", "{k}", "0..k"):
            self.assertIn(kept, out)

    def test_e4_sources_rename(self):
        for p in sorted(glob.glob(os.path.join(E4_PROGRAMS, "*", "*.*"))):
            lang = "c" if p.endswith(".c") else "rust"
            out, mapping = rename.rename(lang, open(p).read())
            self.assertIn("run", out)
            self.assertNotIn("fn main", out.replace("fn main()", "")) if lang == "rust" else None


class ReportTest(unittest.TestCase):
    def test_committed_results_render(self):
        paths = sorted(glob.glob(os.path.join(HERE, "results", "*.json")))
        if not paths:
            self.skipTest("no results file")
        text = report.render(paths)
        self.assertTrue(text.startswith("# E5 results"))


if __name__ == "__main__":
    unittest.main()
