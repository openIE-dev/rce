// External scanner for RCE: the layout pass of spec 2.6 (same algorithm as
// the Rust reference lexer in spec/checker), bracket depth, and NAME with reserved
// keywords (spec 2.4).
#include "tree_sitter/parser.h"
#include <string.h>
#include <stdint.h>

enum { NL, INDENT, DEDENT, NAME, LPAR, RPAR, LSQB, RSQB, LBRACE, RBRACE, SKIP_NL, INT, FLOAT, SEND, BAD, SENTINEL };

#define MAXD 120
typedef struct {
  uint16_t depth;
  uint8_t n;            // indent stack size
  uint16_t stack[MAXD];
  uint8_t pend_indent;
  uint8_t pend_dedent;
  uint8_t bad;
  uint8_t eof_nl;       // the end-of-file NEWLINE of spec 2.6 rule 4 was emitted
  uint8_t started;      // a token other than the first line's comment was seen
  uint8_t v06;          // the module declares `# rce-version: 0.6` or `0.7` (spec 2.7)
} S;

static const char *KW[] = {"actor","after","and","arena","as","bound","break","catch","chan","check","comptime",
 "const","continue","default","device","elif","else","emit","error","event","example","extern","fail","false","fn","for","forall",
 "from","if","impl","in","intent","iter","kind","let","linear","loop","match","module","mut","none","not","on","or","pass",
 "pub","recv","refused","return","select","self","send","spawn","state","step","struct","then","to","trait","true",
 "try","type","use","var","while","with","yield",
 0,
 // draft 0.6 and 0.6b (Appendix C): reserved only in a version 0.6 module (spec 2.7)
 "par","branch","nondet","budget","deadline","priority",
 "replicated","interrupt","session","lvar","group","lanes","barrier","scope","atomic",
 "protocol","offer","choose","node","supervisor","strategy","child","restart", 0};

void *tree_sitter_rce_external_scanner_create(void) { S *s = calloc(1, sizeof(S)); s->n = 1; s->stack[0] = 0; return s; }
void tree_sitter_rce_external_scanner_destroy(void *p) { free(p); }
unsigned tree_sitter_rce_external_scanner_serialize(void *p, char *b) { memcpy(b, p, sizeof(S)); return sizeof(S); }
void tree_sitter_rce_external_scanner_deserialize(void *p, const char *b, unsigned n) {
  S *s = p;
  if (n == sizeof(S)) memcpy(s, b, n); else { memset(s, 0, sizeof(S)); s->n = 1; }
}

// Index of the first draft 0.6 keyword in KW (after the 0.5 terminator).
static int kw06_start(void) { int i = 0; while (KW[i]) i++; return i + 1; }

// Read one comment (lookahead at '#') to the end of the line. On the first
// line of the module, record whether it is the version 0.6 declaration.
static void comment(TSLexer *lx, S *s) {
  char b[48]; int n = 0;
  int col0 = lx->get_column(lx) == 0;
  lx->advance(lx, false);
  while (!lx->eof(lx) && lx->lookahead != '\n' && lx->lookahead != '\r') {
    if (n < 47) b[n++] = (char)lx->lookahead;
    lx->advance(lx, false);
  }
  if (s->started || !col0) return;
  b[n] = 0;
  int i = 0;
  while (b[i] == ' ') i++;
  if (strncmp(b + i, "rce-version:", 12) != 0) return;
  i += 12;
  while (b[i] == ' ') i++;
  int e = n;
  while (e > i && b[e - 1] == ' ') e--;
  if (e - i == 3 && (strncmp(b + i, "0.6", 3) == 0 || strncmp(b + i, "0.7", 3) == 0)) s->v06 = 1;
}

static int is_id0(int32_t c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
static int is_id(int32_t c) { return is_id0(c) || (c >= '0' && c <= '9'); }

static int isdig(int32_t c, int base) {
  if (base == 16) return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
  if (base == 8) return c >= '0' && c <= '7';
  if (base == 2) return c == '0' || c == '1';
  return c >= '0' && c <= '9';
}
// After one digit has been consumed (and marked): (_?d)* with mark_end at each digit.
static void digits(TSLexer *lx, int base) {
  for (;;) {
    if (lx->lookahead == '_') { lx->advance(lx, false); if (!isdig(lx->lookahead, base)) return; }
    if (!isdig(lx->lookahead, base)) return;
    lx->advance(lx, false); lx->mark_end(lx);
  }
}
// Lark tries FLOAT (priority 2) first, then INT's alternatives hex|oct|bin|dec.
static bool number(TSLexer *lx) {
  int32_t c0 = lx->lookahead;
  lx->advance(lx, false); lx->mark_end(lx);
  if (c0 == '0' && (lx->lookahead == 'x' || lx->lookahead == 'o' || lx->lookahead == 'b')) {
    int base = lx->lookahead == 'x' ? 16 : lx->lookahead == 'o' ? 8 : 2;
    lx->advance(lx, false);
    if (isdig(lx->lookahead, base)) { lx->advance(lx, false); lx->mark_end(lx); digits(lx, base); }
    lx->result_symbol = INT; return true;   // "0" alone when the prefix has no digit
  }
  digits(lx, 10);
  lx->result_symbol = INT;
  if (lx->lookahead == '.') {
    lx->advance(lx, false);
    if (!isdig(lx->lookahead, 10)) return true;
    lx->advance(lx, false); lx->mark_end(lx); digits(lx, 10);
    lx->result_symbol = FLOAT;
    if (lx->lookahead == 'e' || lx->lookahead == 'E') {
      lx->advance(lx, false);
      if (lx->lookahead == '+' || lx->lookahead == '-') lx->advance(lx, false);
      if (isdig(lx->lookahead, 10)) { while (isdig(lx->lookahead, 10)) lx->advance(lx, false); lx->mark_end(lx); }
    }
    return true;
  }
  if (lx->lookahead == 'e' || lx->lookahead == 'E') {
    lx->advance(lx, false);
    if (lx->lookahead == '+' || lx->lookahead == '-') lx->advance(lx, false);
    if (isdig(lx->lookahead, 10)) { while (isdig(lx->lookahead, 10)) lx->advance(lx, false); lx->mark_end(lx); lx->result_symbol = FLOAT; }
  }
  return true;
}

bool tree_sitter_rce_external_scanner_scan(void *p, TSLexer *lx, const bool *valid) {
  S *s = p;
  if (valid[SENTINEL]) return false;  // error recovery: let the internal lexer run
  if (s->bad) { lx->result_symbol = BAD; return true; }
  if (s->pend_indent) { s->pend_indent = 0; lx->result_symbol = INDENT; return true; }
  if (s->pend_dedent) { s->pend_dedent--; lx->result_symbol = DEDENT; return true; }
  while (lx->lookahead == ' ') lx->advance(lx, true);
  int32_t c = lx->lookahead;
  if (lx->eof(lx)) {
    // Spec 2.6 rule 4: at end of file emit NEWLINE if needed (a last line
    // without a final line end), then one DEDENT per stack entry above 0.
    if (valid[NL] && !s->eof_nl && s->depth == 0) { s->eof_nl = 1; lx->mark_end(lx); lx->result_symbol = NL; return true; }
    if (s->n > 1) { s->n--; lx->result_symbol = DEDENT; return true; }
    return false;
  }
  if (c == '\n' || c == '\r' || c == '#') {
    // _NL: (/\r?\n[ ]*/ | COMMENT)+ ; indentation = spaces after the last newline
    int ind = 0, sawnl = 0;
    for (;;) {
      if (lx->lookahead == '#') { comment(lx, s); continue; }
      if (lx->lookahead == '\r') { lx->advance(lx, false); if (lx->lookahead != '\n') { s->bad = 1; break; } }
      if (lx->lookahead == '\n') {
        lx->advance(lx, false); sawnl = 1; ind = 0; s->started = 1;
        while (lx->lookahead == ' ') { lx->advance(lx, false); ind++; }
        continue;
      }
      break;
    }
    lx->mark_end(lx);
    if (s->depth > 0) { lx->result_symbol = SKIP_NL; return true; }
    lx->result_symbol = NL;
    if (!sawnl) return true;  // a comment at EOF without newline (no final line end)
    int top = s->stack[s->n - 1];
    if (ind > top) { if (s->n < MAXD) { s->stack[s->n++] = ind; s->pend_indent = 1; } else s->bad = 1; }
    else if (ind < top) {
      int k = 0;
      while (s->n > 1 && s->stack[s->n - 1] > ind) { s->n--; k++; }
      if (s->stack[s->n - 1] != ind) s->bad = 1;  // DedentError
      s->pend_dedent = k;
    }
    return true;
  }
  switch (c) {
    case '(': case '[': case '{':
      lx->advance(lx, false); s->depth++;
      lx->result_symbol = c == '(' ? LPAR : c == '[' ? LSQB : LBRACE; return true;
    case ')': case ']': case '}':
      lx->advance(lx, false); if (s->depth > 0) s->depth--;
      lx->result_symbol = c == ')' ? RPAR : c == ']' ? RSQB : RBRACE; return true;
  }
  if (c >= '0' && c <= '9') return number(lx);
  if (c == '<') {
    lx->advance(lx, false);
    if (lx->lookahead == '-') { lx->advance(lx, false); lx->mark_end(lx); lx->result_symbol = SEND; return true; }
    return false;
  }
  if (is_id0(c)) {
    char w[64]; int n = 0;
    while (is_id(lx->lookahead)) { if (n < 63) w[n] = (char)lx->lookahead; n++; lx->advance(lx, false); }
    if (n < 64) {
      w[n] = 0;
      if (strcmp(w, "_") == 0) return false;
      for (int i = 0; KW[i]; i++) if (strcmp(w, KW[i]) == 0) return false;
      if (s->v06)
        for (int i = kw06_start(); KW[i]; i++) if (strcmp(w, KW[i]) == 0) return false;
    }
    lx->mark_end(lx);
    s->started = 1;
    lx->result_symbol = NAME; return true;
  }
  return false;
}
