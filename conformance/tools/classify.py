"""Area of a test, from its diagnostic code, origin and constructs."""
import re
AREAS = ["lexing-parsing", "types-checking", "evaluation-fuel", "numbers", "text-unicode",
         "forms-log", "effects-capabilities", "host-contract", "concurrency", "sandbox", "emission-parity"]
CODE = {
    "lexing-parsing": "LEX SYN VER",
    "types-checking": "TY NAME MOD OWN REG TRAIT MATCH PAT STATE STEP REF LIN SAFE VIS",
    "evaluation-fuel": "INT LOOP REC ITER",
    "concurrency": "PAR BUD LVAR CRDT SUP SES DL ATOM GRP SCOPE ND NODE RT INTR SPAWN",
    "effects-capabilities": "EFF CAP FFI DEV SEC CT CTX SIMD",
}
def by_code(code):
    if not code: return None
    m = re.match(r"W-([A-Z]+)", code)
    for a, ks in CODE.items():
        if m and m.group(1) in ks.split(): return a
    return None
KW = [
    ("sandbox", r"sandbox|seccomp"),
    ("host-contract", r"/host|host_|\bhost\b"),
    ("concurrency", r"\b(par|lvar|freeze|budget|spawn|chan|actor|session|supervisor|select|group|device|scope)\b"),
    ("numbers", r"\b(f64|f32|u64|i32|u8|checked_|wrapping_|saturating_|nan|inf)\b|float"),
    ("text-unicode", r"\\u\{|\.chars\(\)|\.words\(\)|\btext\b.*\[|unicode|utf8"),
    ("forms-log", r"^state |^event |^step |\bexample\b"),
]
def classify(code, origin, src):
    a = by_code(code)
    if a: return a
    for a, rx in KW:
        if re.search(rx, origin): return a
    for a, rx in KW[2:]:
        if re.search(rx, src, re.M): return a
    return "evaluation-fuel"
