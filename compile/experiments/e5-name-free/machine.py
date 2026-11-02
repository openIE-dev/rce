#!/usr/bin/env python3
"""E5's machine-code-level R hash: a function ID in the style of Binary Ninja's WARP.

WARP's README (github.com/Vector35/warp, read for E5) defines a function GUID
as the UUIDv5 of its basic-block GUIDs sorted from the highest start address
to the lowest, and a basic-block GUID as the UUIDv5 of the bytes of its
instructions in execution order, with relocatable instructions zeroed, NOPs
left out and instructions that set a register to itself left out when they
are effectively NOPs. This module implements that definition for unlinked
object files (the outputs of `-c`) in three formats, PROTOCOL.md, "The
machine-code-level hash":

- ELF relocatable files for x86-64, with the relocations of `.rela.*`;
- Mach-O object files for arm64, with each section's relocation entries,
  read with the Mach-O load-command parser of E1's e1.py;
- WebAssembly object files (wasm32), with the `reloc.CODE` section.

Functions are located from the symbol table (ELF `STT_FUNC` symbols; Mach-O
section symbols of the text section; the code section of a WebAssembly
object, which delimits every function body). A file with no symbol table
gives the status "functions cannot be located" and no hash.

An instruction is *relocatable* when a relocation entry of the object
patches any of its bytes; all of its bytes are replaced by zero bytes. The
NOPs and self-moves left out are declared in NOPS below, per architecture.
A block whose instructions are all left out is itself left out.

    python3 machine.py OBJECT
"""
import hashlib
import os
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e1-determinism"))

import e1  # noqa: E402  (Mach-O load commands)
import irhash  # noqa: E402  (WebAssembly decoding)

# the namespaces of WARP's README
BLOCK_NS = uuid.UUID("0192a178-7a5f-7936-8653-3cbaa7d6afe7")
FUNC_NS = uuid.UUID("0192a179-61ac-7cef-88ed-012296e9492f")

NOPS = {
    "x86_64": "opcode 0x90 without REX.B and without an F3 prefix (F3 90 is PAUSE); 0F 1F /0 (multi-byte NOP, any "
              "prefixes); self-move left out: MOV r64, r64 (REX.W 89 or 8B, register form) with the same source and "
              "destination; MOV r32, r32 to itself is kept, because it clears the upper 32 bits",
    "arm64": "NOP (0xD503201F); self-moves left out: MOV Xd, Xd (ORR Xd, XZR, Xd, 0xAA0003E0 with Rd = Rm) and "
             "MOV SP, SP (ADD SP, SP, #0, 0x910003FF); MOV Wd, Wd is kept, because it clears the upper 32 bits",
    "wasm32": "nop (0x01); no self-move: a WebAssembly instruction does not name a source and destination register",
}


class Undecoded(Exception):
    pass


# -- x86-64 instruction lengths ------------------------------------------------------------

PREFIXES = {0xF0, 0xF2, 0xF3, 0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65, 0x66, 0x67}
# one-byte map: opcodes with a ModRM byte
MODRM1 = set()
for _b in range(0x00, 0x40, 8):
    MODRM1.update((_b, _b + 1, _b + 2, _b + 3))
MODRM1.update([0x63, 0x69, 0x6B, 0x80, 0x81, 0x83] + list(range(0x84, 0x90)) +
              [0xC0, 0xC1, 0xC6, 0xC7, 0xD0, 0xD1, 0xD2, 0xD3] + list(range(0xD8, 0xE0)) + [0xF6, 0xF7, 0xFE, 0xFF])
IMM8_1 = {0x04, 0x0C, 0x14, 0x1C, 0x24, 0x2C, 0x34, 0x3C, 0x6A, 0x6B, 0x80, 0x83, 0xA8, 0xC0, 0xC1, 0xC6, 0xCD,
          0xE4, 0xE5, 0xE6, 0xE7} | set(range(0xB0, 0xB8))
IMMZ_1 = {0x05, 0x0D, 0x15, 0x1D, 0x25, 0x2D, 0x35, 0x3D, 0x68, 0x69, 0x81, 0xA9, 0xC7}
REL8 = set(range(0x70, 0x80)) | {0xE0, 0xE1, 0xE2, 0xE3, 0xEB}
INVALID64 = {0x06, 0x07, 0x0E, 0x16, 0x17, 0x1E, 0x1F, 0x27, 0x2F, 0x37, 0x3F, 0x60, 0x61, 0x62, 0x82, 0x9A,
             0xD4, 0xD5, 0xD6, 0xEA}
NOMODRM2 = {0x05, 0x06, 0x07, 0x08, 0x09, 0x0B, 0x0E, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x37, 0x77, 0xA0, 0xA1,
            0xA2, 0xA8, 0xA9, 0xAA} | set(range(0x80, 0x90)) | set(range(0xC8, 0xD0))
IMM8_2 = {0x70, 0x71, 0x72, 0x73, 0xA4, 0xAC, 0xBA, 0xC2, 0xC4, 0xC5, 0xC6}


def _modrm_len(code, p):
    """Bytes of ModRM, SIB and displacement starting at p, and the ModRM byte."""
    if p >= len(code):
        raise Undecoded("truncated ModRM")
    modrm = code[p]
    mod, rm = modrm >> 6, modrm & 7
    n = 1
    if mod != 3:
        if rm == 4:
            if p + 1 >= len(code):
                raise Undecoded("truncated SIB")
            sib = code[p + 1]
            n += 1
            if mod == 0 and sib & 7 == 5:
                n += 4
        if mod == 0 and rm == 5:
            n += 4
        elif mod == 1:
            n += 1
        elif mod == 2:
            n += 4
    return n, modrm


def x86_insn(code, p):
    """Decode one x86-64 instruction at p: (length, kind, target offset or None).

    kind: "nop", "selfmove", "jcc", "jmp", "ret", "stop" (no fall-through, no known target),
    "ijmp" (indirect jump), or "op".
    """
    start = p
    pfx = set()
    while p < len(code) and code[p] in PREFIXES:
        pfx.add(code[p])
        p += 1
    rex = 0
    if p < len(code) and 0x40 <= code[p] <= 0x4F:
        rex = code[p]
        p += 1
    if p >= len(code):
        raise Undecoded("truncated instruction at 0x%x" % start)
    op = code[p]
    p += 1
    opsize16 = 0x66 in pfx
    immz = 2 if opsize16 else 4
    if op in (0xC4, 0xC5):
        if op == 0xC5:
            vmap, p = 1, p + 1
        else:
            vmap, p = code[p] & 0x1F, p + 2
        vop = code[p]
        p += 1
        if vmap == 1 and vop == 0x77:
            return p - start, "op", None
        n, _m = _modrm_len(code, p)
        p += n
        if vmap == 3 or (vmap == 1 and vop in IMM8_2):
            p += 1
        return p - start, "op", None
    if op == 0x62:
        raise Undecoded("EVEX instruction at 0x%x" % start)
    if op == 0x0F:
        op2 = code[p]
        p += 1
        if op2 == 0x38:
            p += 1
            n, _m = _modrm_len(code, p)
            return p + n - start, "op", None
        if op2 == 0x3A:
            p += 1
            n, _m = _modrm_len(code, p)
            return p + n + 1 - start, "op", None
        if op2 == 0x0F:
            raise Undecoded("3DNow! instruction at 0x%x" % start)
        if 0x80 <= op2 <= 0x8F:
            rel = int.from_bytes(code[p:p + 4], "little", signed=True)
            p += 4
            return p - start, "jcc", p + rel
        if op2 == 0x0B:
            return p - start, "stop", None
        if op2 in NOMODRM2:
            return p - start, "op", None
        n, modrm = _modrm_len(code, p)
        p += n
        if op2 in IMM8_2:
            p += 1
        if op2 == 0x1F and (modrm >> 3) & 7 == 0:
            return p - start, "nop", None
        return p - start, "op", None
    if op in INVALID64:
        raise Undecoded("opcode 0x%02x is not valid in 64-bit mode (at 0x%x)" % (op, start))
    if op in REL8:
        rel = int.from_bytes(code[p:p + 1], "little", signed=True)
        p += 1
        kind = "jmp" if op == 0xEB else "jcc"
        return p - start, kind, p + rel
    if op in (0xE8, 0xE9):
        rel = int.from_bytes(code[p:p + 4], "little", signed=True)
        p += 4
        return p - start, ("jmp" if op == 0xE9 else "op"), (p + rel if op == 0xE9 else None)
    if op in (0xC3, 0xCB):
        return p - start, "ret", None
    if op in (0xC2, 0xCA):
        return p + 2 - start, "ret", None
    if op in (0xCC, 0xF4):
        return p - start, "stop", None
    if op == 0x90:
        if rex & 1 or 0xF3 in pfx:
            return p - start, "op", None
        return p - start, "nop", None
    if 0xB8 <= op <= 0xBF:
        return p + (8 if rex & 8 else immz) - start, "op", None
    if 0xA0 <= op <= 0xA3:
        return p + (4 if 0x67 in pfx else 8) - start, "op", None
    if op == 0xC8:
        return p + 3 - start, "op", None
    if op in MODRM1:
        n, modrm = _modrm_len(code, p)
        p += n
        reg = (modrm >> 3) & 7
        if op in IMM8_1:
            p += 1
        elif op in IMMZ_1:
            p += immz
        elif op == 0xF6 and reg in (0, 1):
            p += 1
        elif op == 0xF7 and reg in (0, 1):
            p += immz
        if op in (0x89, 0x8B) and rex & 8 and modrm >> 6 == 3:
            r = reg | (8 if rex & 4 else 0)
            b = (modrm & 7) | (8 if rex & 1 else 0)
            if r == b:
                return p - start, "selfmove", None
        if op == 0xFF and reg in (4, 5):
            return p - start, "ijmp", None
        return p - start, "op", None
    if op in IMM8_1:
        return p + 1 - start, "op", None
    if op in IMMZ_1:
        return p + immz - start, "op", None
    return p - start, "op", None


def arm64_insn(code, p):
    """Decode one arm64 instruction at p: (4, kind, target or None)."""
    if p + 4 > len(code):
        raise Undecoded("truncated arm64 instruction at 0x%x" % p)
    w = int.from_bytes(code[p:p + 4], "little")
    if w == 0xD503201F:
        return 4, "nop", None
    if (w & 0xFFE0FFE0) == 0xAA0003E0 and (w >> 16) & 31 == w & 31:
        return 4, "selfmove", None
    if w == 0x910003FF:
        return 4, "selfmove", None

    def sx(v, bits):
        return v - (1 << bits) if v >> (bits - 1) & 1 else v
    if w & 0xFC000000 == 0x14000000:
        return 4, "jmp", p + 4 * sx(w & 0x3FFFFFF, 26)
    if w & 0xFF000010 == 0x54000000:
        return 4, "jcc", p + 4 * sx((w >> 5) & 0x7FFFF, 19)
    if w & 0x7E000000 == 0x34000000:
        return 4, "jcc", p + 4 * sx((w >> 5) & 0x7FFFF, 19)
    if w & 0x7E000000 == 0x36000000:
        return 4, "jcc", p + 4 * sx((w >> 5) & 0x3FFF, 14)
    if w & 0xFFFFFC1F == 0xD65F0000:
        return 4, "ret", None
    if w & 0xFFFFFC1F == 0xD61F0000:
        return 4, "ijmp", None
    if w & 0xFFE0001F == 0xD4200000:  # BRK
        return 4, "stop", None
    return 4, "op", None


def wasm_blocks(body, base):
    """Split a WebAssembly body into blocks at control instructions: [(start, [(start, end, kind)])]."""
    insns = []
    for ins in body["code"]:
        kind = "nop" if ins.op == 0x01 else "op"
        insns.append((ins.start - base, ins.end - base, kind, ins.op))
    blocks, cur = [], []
    for s, e, kind, op in insns:
        if op in (0x02, 0x03, 0x04, 0x05) and cur:
            blocks.append(cur)
            cur = []
        cur.append((s, e, kind))
        if op in (0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x00):
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


# -- the WARP-style ID ----------------------------------------------------------------------

def native_blocks(code, decoder):
    """Decode code (one function) and split it into basic blocks: [[(start, end, kind)]]."""
    insns, p = [], 0
    while p < len(code):
        n, kind, target = decoder(code, p)
        insns.append((p, p + n, kind, target))
        p += n
    leaders = {0}
    for s, e, kind, target in insns:
        if kind in ("jcc", "jmp", "ret", "stop", "ijmp"):
            leaders.add(e)
            if target is not None and 0 <= target < len(code):
                leaders.add(target)
    blocks, cur = [], []
    for s, e, kind, _t in insns:
        if s in leaders and cur:
            blocks.append(cur)
            cur = []
        cur.append((s, e, kind))
    if cur:
        blocks.append(cur)
    return blocks


def function_id(code, blocks, reloc_offsets):
    """The WARP-style GUID of one function: code is its bytes, offsets relative to its start."""
    guids = []
    for blk in blocks:
        b = bytearray()
        for s, e, kind in blk:
            if kind in ("nop", "selfmove"):
                continue
            if any(s <= r < e for r in reloc_offsets):
                b += bytes(e - s)
            else:
                b += code[s:e]
        if b:
            guids.append((blk[0][0], uuid.uuid5(BLOCK_NS, bytes(b))))
    guids.sort(key=lambda x: -x[0])
    return str(uuid.uuid5(FUNC_NS, b"".join(g.bytes for _s, g in guids))), len(guids)


# -- object formats ---------------------------------------------------------------------------

def _u(d, o, n):
    return int.from_bytes(d[o:o + n], "little")


def elf_functions(data):
    """Functions of a 64-bit little-endian ELF relocatable file: (arch, [(name, code, reloc offsets)])."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise Undecoded("not a 64-bit little-endian ELF file")
    machine = _u(data, 18, 2)
    arch = {62: "x86_64", 183: "arm64"}.get(machine)
    if arch is None:
        raise Undecoded("ELF machine %d is not decoded" % machine)
    shoff, shnum = _u(data, 0x28, 8), _u(data, 0x3C, 2)
    secs = []
    for i in range(shnum):
        h = shoff + 64 * i
        secs.append({"type": _u(data, h + 4, 4), "flags": _u(data, h + 8, 8), "off": _u(data, h + 24, 8),
                     "size": _u(data, h + 32, 8), "link": _u(data, h + 40, 4), "info": _u(data, h + 44, 4),
                     "entsize": _u(data, h + 56, 8)})
    symtab = next((s for s in secs if s["type"] == 2), None)
    if symtab is None:
        return arch, None
    strtab = secs[symtab["link"]]
    relocs = {}
    for s in secs:
        if s["type"] == 4:  # SHT_RELA
            lst = relocs.setdefault(s["info"], [])
            for k in range(s["size"] // 24):
                lst.append(_u(data, s["off"] + 24 * k, 8))
        elif s["type"] == 9:  # SHT_REL
            lst = relocs.setdefault(s["info"], [])
            for k in range(s["size"] // 16):
                lst.append(_u(data, s["off"] + 16 * k, 8))
    syms = []
    for k in range(symtab["size"] // 24):
        o = symtab["off"] + 24 * k
        name_off, info, shndx = _u(data, o, 4), data[o + 4], _u(data, o + 6, 2)
        value, size = _u(data, o + 8, 8), _u(data, o + 16, 8)
        if info & 0xF != 2 or shndx == 0 or shndx >= len(secs) or not secs[shndx]["flags"] & 4:
            continue
        end = data.find(b"\0", strtab["off"] + name_off)
        syms.append((shndx, value, size, data[strtab["off"] + name_off:end].decode("utf-8", "replace")))
    out, seen = [], set()
    syms.sort()
    for i, (shndx, value, size, name) in enumerate(syms):
        if (shndx, value) in seen:
            continue
        seen.add((shndx, value))
        sec = secs[shndx]
        if size == 0:
            nxt = [v for sh, v, _sz, _n in syms if sh == shndx and v > value]
            size = (min(nxt) if nxt else sec["size"]) - value
        code = data[sec["off"] + value: sec["off"] + value + size]
        rel = [r - value for r in relocs.get(shndx, []) if value <= r < value + size]
        out.append((name, code, rel))
    return arch, out


def macho_functions(data):
    """Functions of a thin 64-bit Mach-O object: (arch, [(name, code, reloc offsets)])."""
    cmds = e1.macho_load_commands(data)
    if cmds is None:
        raise Undecoded("not a thin 64-bit little-endian Mach-O file")
    cputype = e1._u32(data, 4)
    arch = {0x0100000C: "arm64", 0x01000007: "x86_64"}.get(cputype)
    if arch is None:
        raise Undecoded("Mach-O CPU type 0x%x is not decoded" % cputype)
    sects, symtab = [], None
    for cmd, off, size in cmds:
        if cmd == 0x19:
            for i in range(e1._u32(data, off + 64)):
                h = off + 72 + 80 * i
                sects.append({"name": e1._name16(data, h), "seg": e1._name16(data, h + 16),
                              "addr": e1._u64(data, h + 32), "size": e1._u64(data, h + 40),
                              "off": e1._u32(data, h + 48), "reloff": e1._u32(data, h + 56),
                              "nreloc": e1._u32(data, h + 60), "flags": e1._u32(data, h + 64)})
        elif cmd == 0x2:
            symtab = [e1._u32(data, off + 8 + 4 * k) for k in range(4)]
    if symtab is None:
        return arch, None
    symoff, nsyms, stroff, _strsize = symtab
    by_sect = {}
    for k in range(nsyms):
        o = symoff + 16 * k
        strx, ntype, nsect, value = e1._u32(data, o), data[o + 4], data[o + 5], e1._u64(data, o + 8)
        if ntype & 0xE0 or ntype & 0x0E != 0x0E or nsect == 0:
            continue
        end = data.find(b"\0", stroff + strx)
        name = data[stroff + strx:end].decode("utf-8", "replace")
        if name.startswith(("l", "L")):  # assembler-local labels such as ltmp0
            continue
        by_sect.setdefault(nsect, []).append((value, name))
    out = []
    for nsect, syms in sorted(by_sect.items()):
        sec = sects[nsect - 1]
        if not sec["flags"] & 0x80000000:  # S_ATTR_PURE_INSTRUCTIONS
            continue
        rel = []
        for k in range(sec["nreloc"]):
            a = e1._u32(data, sec["reloff"] + 8 * k)
            if not a & 0x80000000:
                rel.append(a)
        syms = sorted(set(syms))
        starts = sorted(set(v for v, _n in syms))
        named = {}
        for v, n in syms:
            named.setdefault(v, n)
        for i, v in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else sec["addr"] + sec["size"]
            lo = v - sec["addr"]
            code = data[sec["off"] + lo: sec["off"] + lo + (end - v)]
            out.append((named[v], code, [r - lo for r in rel if lo <= r < lo + (end - v)]))
    return arch, out


WASM_ADDEND = {3, 4, 5, 8, 9, 11, 14, 15, 16, 17, 21, 22, 23, 25}


def wasm_object_functions(data):
    """Functions of a WebAssembly object: ("wasm32", [(name, code, blocks, reloc offsets)])."""
    m = irhash.Module(data)
    code_sec = next(k for k, (sid, _s, _e) in enumerate(m.sections) if sid == 10) if m.code_start is not None else None
    rel = []
    for nm, p, end in m.customs:
        if nm != "reloc.CODE":
            continue
        r = irhash.Reader(data, p, end)
        if r.u() != code_sec:
            continue
        for _ in range(r.u()):
            t = r.u()
            off = r.u()
            r.u()
            if t in WASM_ADDEND:
                r.s(32)
            rel.append(m.code_start + off)
    names = {}
    for nm, p, end in m.customs:
        if nm != "linking":
            continue
        r = irhash.Reader(data, p, end)
        r.u()
        while r.p < end:
            sub = r.byte()
            size = r.u()
            send = r.p + size
            if sub == 8:  # WASM_SYMBOL_TABLE
                for _ in range(r.u()):
                    kind = r.byte()
                    flags = r.u()
                    if kind in (0, 2, 4, 5):  # function, global, event/tag, table
                        idx = r.u()
                        if not flags & 0x10 or flags & 0x40:  # defined, or undefined with an explicit name
                            nme = r.name()
                            if kind == 0:
                                names.setdefault(idx, nme)
                    elif kind == 1:  # data
                        r.name()
                        if not flags & 0x10:
                            r.u()
                            r.u()
                            r.u()
                    elif kind == 3:  # section
                        r.u()
            r.p = send
    nf = m.n_imported_funcs()
    out = []
    for k, b in enumerate(m.bodies):
        code = data[b["start"]:b["end"]]
        rels = [x - b["start"] for x in rel if b["start"] <= x < b["end"]]
        blocks = wasm_blocks(b, b["start"])
        if blocks and b["body_start"] > b["start"]:  # the local declarations belong to the first block
            s0 = blocks[0]
            blocks[0] = [(0, b["body_start"] - b["start"], "op")] + s0
        out.append((names.get(nf + k, "#%d" % k), code, blocks, rels))
    has_symbols = any(nm == "linking" for nm, _p, _e in m.customs)
    return "wasm32", out, has_symbols


def object_ids(data):
    """WARP-style IDs of every function of an object file.

    Returns {"format", "arch", "status", "functions": [{"name", "id", "blocks", "bytes", "relocs"}], "nops"}.
    """
    fmt = ("elf" if data[:4] == b"\x7fELF" else "mach-o" if data[:4] == b"\xcf\xfa\xed\xfe"
           else "wasm" if data[:4] == b"\0asm" else None)
    res = {"format": fmt, "arch": None, "functions": []}
    try:
        if fmt == "wasm":
            arch, funcs, has_sym = wasm_object_functions(data)
            res["arch"] = arch
            if not has_sym:
                res["status"] = "functions located by the code section; no symbol table, so no names"
            for name, code, blocks, rels in funcs:
                fid, nb = function_id(code, blocks, rels)
                res["functions"].append({"name": name, "id": fid, "blocks": nb, "bytes": len(code), "relocs": len(rels)})
        elif fmt in ("elf", "mach-o"):
            arch, funcs = elf_functions(data) if fmt == "elf" else macho_functions(data)
            res["arch"] = arch
            if funcs is None:
                res["status"] = "functions cannot be located: no symbol table"
                return res
            dec = x86_insn if arch == "x86_64" else arm64_insn
            for name, code, rels in funcs:
                fid, nb = function_id(code, native_blocks(code, dec), rels)
                res["functions"].append({"name": name, "id": fid, "blocks": nb, "bytes": len(code), "relocs": len(rels)})
        else:
            res["status"] = "not an object format this harness reads"
            return res
    except (Undecoded, irhash.Unsupported) as e:
        res["status"] = "not decoded: %s" % e
        res["functions"] = []
        return res
    res.setdefault("status", "hashed")
    res["nops"] = NOPS[res["arch"]]
    return res


def module_id(res):
    """The ID of a whole object: SHA-256 over its sorted function IDs (names are not used)."""
    if not res["functions"]:
        return None
    return hashlib.sha256("\n".join(sorted(f["id"] for f in res["functions"])).encode()).hexdigest()


def entry_id(res, entry):
    """The ID of the function named entry (or _entry on Mach-O), else None."""
    for f in res["functions"]:
        if f["name"] in (entry, "_" + entry):
            return f["id"]
    return None


def main():
    a = sys.argv[1:]
    if len(a) != 1:
        sys.exit("usage: python3 machine.py OBJECT")
    res = object_ids(open(a[0], "rb").read())
    print(res["format"], res["arch"], res["status"], module_id(res))
    for f in res["functions"]:
        print("  %s %s blocks=%d bytes=%d relocs=%d" % (f["id"], f["name"], f["blocks"], f["bytes"], f["relocs"]))


if __name__ == "__main__":
    main()
