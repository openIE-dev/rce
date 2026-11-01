// External scanner for RCE: the layout pass of spec 2.6 (same algorithm as
// the Rust reference lexer in spec/checker), bracket depth, and NAME with reserved
// keywords (spec 2.4).
#include "tree_sitter/parser.h"
#include <string.h>
#include <stdint.h>

enum { NL, INDENT, DEDENT, NAME, LPAR, RPAR, LSQB, RSQB, LBRACE, RBRACE, SKIP_NL, INT, FLOAT, SEND, BAD, COMMENT, PARFOR, TRAP, SENTINEL, KEPT, APPLIED };

#define MAXD 120
typedef struct {
  uint16_t depth;
  uint8_t n;            // indent stack size
  uint16_t stack[MAXD];
  uint8_t pend_indent;
  uint8_t pend_dedent;
  uint8_t bad;
  uint8_t eof_nl;       // the end-of-file NEWLINE of spec 2.6 rule 4 was emitted
  uint8_t seen;         // a name, number or bracket token was scanned (the file's first word is exempt from the line-start check)
  uint8_t started;      // a token other than the first line's comment was seen
  uint8_t cont;         // the last line end stopped before a comment line; the layout
                        // decision for the following code line was already made
} S;

static const char *KW[] = {"actor","after","and","arena","as","bound","break","catch","chan","check","comptime",
 "const","continue","default","device","elif","else","emit","error","event","example","extern","fail","false","fn","for","forall",
 "from","if","impl","in","intent","iter","kind","let","linear","loop","match","module","mut","none","not","on","or","pass",
 "pub","recv","return","select","self","send","spawn","state","step","struct","then","to","trait","true",
 "try","type","use","var","while","with","yield",
 // the concurrency words (Appendix C): one language, one reserved list (spec 2.4)
 "par","branch","nondet","budget","deadline","priority",
 "replicated","interrupt","session","lvar","group","lanes","barrier","scope","atomic",
 "protocol","offer","choose","node","supervisor","strategy","child","restart", 0};

// Words that can only begin a declaration or a statement, never continue a line.
static const char *LINE_START[] = {"event","step","state","kind","type","struct","impl","use","intent","example","check","law",
 "extern","error","module","pub","return","emit","fail","break","continue","pass","yield","let","var","while","par","actor","trait","const","iter","protocol","supervisor","node","comptime","fn",0};

void *tree_sitter_rce_external_scanner_create(void) { S *s = calloc(1, sizeof(S)); s->n = 1; s->stack[0] = 0; return s; }
void tree_sitter_rce_external_scanner_destroy(void *p) { free(p); }
unsigned tree_sitter_rce_external_scanner_serialize(void *p, char *b) { memcpy(b, p, sizeof(S)); return sizeof(S); }
void tree_sitter_rce_external_scanner_deserialize(void *p, const char *b, unsigned n) {
  S *s = p;
  if (n == sizeof(S)) memcpy(s, b, n); else { memset(s, 0, sizeof(S)); s->n = 1; }
}

// Read one comment (lookahead at '#') to the end of the line. A `# rce-version:` line is a release
// label: tolerated and ignored (spec 2.7), so there is nothing to record.
static void comment(TSLexer *lx, S *s) {
  lx->advance(lx, false);
  while (!lx->eof(lx) && lx->lookahead != '\n' && lx->lookahead != '\r') {
    if (lx->lookahead == '\t') s->bad = 1;  // W-LEX-1: a text with a tab requires change, in a comment too
    lx->advance(lx, false);
  }
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
  int first_col = !s->started && lx->get_column(lx) == 0;
  while (lx->lookahead == ' ') lx->advance(lx, true);
  int32_t c = lx->lookahead;
  // indentation on the first line of the file (an indent with no block) is an error
  if (first_col && c != ' ' && lx->get_column(lx) > 0 && c != '\n' && c != '\r' && c != '#' && !lx->eof(lx)) { lx->result_symbol = BAD; return true; }
  if (lx->eof(lx)) {
    // Spec 2.6 rule 4: at end of file emit NEWLINE if needed (a last line
    // without a final line end), then one DEDENT per stack entry above 0.
    if (valid[NL] && !s->eof_nl && s->depth == 0) { s->eof_nl = 1; lx->mark_end(lx); lx->result_symbol = NL; return true; }
    if (s->n > 1) { s->n--; lx->result_symbol = DEDENT; return true; }
    return false;
  }
  if (c == '#' && valid[COMMENT]) {
    // A comment is an extra token. It never includes the line end, so the
    // layout pass below still sees every line end.
    comment(lx, s);
    lx->mark_end(lx);
    lx->result_symbol = COMMENT;
    return true;
  }
  if (c == '\n' || c == '\r' || c == '#') {
    // _NL: (/\r?\n[ ]*/ | COMMENT)+ ; indentation = spaces after the last newline.
    // When comments are tokens, the line end token stops before the first
    // comment line, but the scan continues past comment lines (lookahead
    // only, no mark_end) to find the indentation of the next code line.
    int ind = 0, sawnl = 0, stopped = 0;
    for (;;) {
      if (lx->lookahead == '#') {
        if (!stopped && sawnl && valid[COMMENT]) { lx->mark_end(lx); stopped = 1; }
        comment(lx, s);
        continue;
      }
      if (lx->lookahead == '\r') { lx->advance(lx, false); if (lx->lookahead != '\n') { s->bad = 1; break; } }
      if (lx->lookahead == '\n') {
        lx->advance(lx, false); sawnl = 1; ind = 0; s->started = 1;
        while (lx->lookahead == ' ') { lx->advance(lx, false); ind++; }
        continue;
      }
      break;
    }
    // a last line of spaces only is blank: it neither indents nor dedents
    if (lx->eof(lx) && ind > 0) ind = 0;
    if (!stopped) lx->mark_end(lx);
    int was_cont = s->cont;
    s->cont = stopped;
    // Inside brackets, or the rest of a run whose layout was already decided:
    // the line ends are an extra.
    if (s->depth > 0 || was_cont) { lx->result_symbol = SKIP_NL; return true; }
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
      lx->advance(lx, false); s->depth++; s->seen = 1;
      lx->result_symbol = c == '(' ? LPAR : c == '[' ? LSQB : LBRACE; return true;
    case ')': case ']': case '}':
      lx->advance(lx, false); if (s->depth > 0) s->depth--;
      lx->result_symbol = c == ')' ? RPAR : c == ']' ? RSQB : RBRACE; return true;
  }
  if (c >= '0' && c <= '9') { s->seen = 1; return number(lx); }
  if (c == '<') {
    lx->advance(lx, false);
    if (lx->lookahead == '-') { lx->advance(lx, false); lx->mark_end(lx); lx->result_symbol = SEND; return true; }
    return false;
  }
  if (is_id0(c)) {
    unsigned wcol = lx->get_column(lx);  // a word at column 0 follows a line end: it is not glued to anything
    char w[64]; int n = 0;
    while (is_id(lx->lookahead)) { if (n < 63) w[n] = (char)lx->lookahead; n++; lx->advance(lx, false); }
    if (n < 64) {
      w[n] = 0;
      if (strcmp(w, "_") == 0) return false;
      // contextual words of a law (3.11 R10): as the whole outcome of an `every step`
      // line (APPLIED is valid only there), `applied` before the line end and `kept`
      // before `:` or the line end are the outcome words; elsewhere they are NAMEs
      if (valid[APPLIED] && !valid[SENTINEL] && (strcmp(w, "applied") == 0 || strcmp(w, "kept") == 0)) {
        int kept = w[0] == 'k';
        lx->mark_end(lx);
        while (lx->lookahead == ' ') lx->advance(lx, false);
        int32_t c2 = lx->lookahead;
        int end = c2 == '\n' || c2 == '\r' || c2 == '#' || lx->eof(lx);
        s->started = 1; s->seen = 1;
        if (end || (kept && c2 == ':')) { lx->result_symbol = kept ? KEPT : APPLIED; return true; }
        lx->result_symbol = NAME; return true;
      }
      // contextual word (COMP-2): `kept` is the outcome keyword only where the
      // grammar allows an outcome and a `:` follows; elsewhere it is a NAME
      if (valid[KEPT] && strcmp(w, "kept") == 0) {
        lx->mark_end(lx);
        while (lx->lookahead == ' ') lx->advance(lx, false);
        if (lx->lookahead == ':') { s->started = 1; s->seen = 1; lx->result_symbol = KEPT; return true; }
        s->started = 1; s->seen = 1;
        lx->result_symbol = NAME; return true;
      }
      // `par` after the iterable of a `for` (valid only there): its own token, so a `par` glued after
      // any other expression is an error and not a missing line end
      if (valid[PARFOR] && strcmp(w, "par") == 0) { lx->mark_end(lx); lx->result_symbol = PARFOR; return true; }
      // `kept: trap("...")` (section 3 Outcome): after `kept:`, `trap` followed by `(` is the
      // trap outcome; any other word there, and `trap` anywhere else, is a NAME
      if (valid[TRAP] && !valid[SENTINEL] && strcmp(w, "trap") == 0) {
        lx->mark_end(lx);
        while (lx->lookahead == ' ') lx->advance(lx, true);
        s->started = 1; s->seen = 1;
        lx->result_symbol = lx->lookahead == '(' ? TRAP : NAME; return true;
      }
      // contextual word (5.1 refined types): `where` is the keyword only
      // where a NAME cannot stand, i.e. after the type of a type declaration
      if (strcmp(w, "where") == 0 && !valid[NAME]) return false;
      // contextual words of a law (3.11 R10): `always`, `every` and `attest` begin a law
      // line, where a NAME cannot stand; elsewhere they are NAMEs
      if (!valid[NAME] && !valid[SENTINEL] && (strcmp(w, "always") == 0 || strcmp(w, "every") == 0 || strcmp(w, "attest") == 0)) return false;
      // F.6.1: `t` directly before a quote is the template prefix, not a name
      if (n == 1 && w[0] == 't' && lx->lookahead == '"') return false;
      // A declaration or statement starts a line (spec 2.6): where a line end is
      // required (`_NL` is valid) and none was read, such a word is an error,
      // not a missing `_NL` the parser could insert. The first token of the file
      // is exempt.
      if (valid[NL] && !valid[SENTINEL] && wcol > 0) {
        for (int i = 0; LINE_START[i]; i++) if (strcmp(w, LINE_START[i]) == 0) { lx->result_symbol = BAD; return true; }
      }
      // an ordinary name where a line end is required and cannot continue the line
      // (`)Neg` after a closing bracket): an error, not a missing line end
      if (valid[NL] && !valid[NAME] && !valid[SENTINEL] && wcol > 0) {
        int kw = 0;
        for (int i = 0; KW[i]; i++) if (strcmp(w, KW[i]) == 0) kw = 1;
        if (!kw && strcmp(w, "where") && strcmp(w, "by") && strcmp(w, "label") && strcmp(w, "release") && strcmp(w, "endorse") &&
            strcmp(w, "ct") && strcmp(w, "requires") && strcmp(w, "ensures") && strcmp(w, "ghost") && strcmp(w, "decreases") && strcmp(w, "law")) {
          lx->result_symbol = BAD; return true;
        }
      }
      for (int i = 0; KW[i]; i++) if (strcmp(w, KW[i]) == 0) return false;
      // contextual word (F.13): `by` is the keyword only where a NAME cannot stand
      if (strcmp(w, "by") == 0 && !valid[NAME]) return false;
      if ((strcmp(w, "label") == 0 || strcmp(w, "release") == 0 || strcmp(w, "endorse") == 0 || strcmp(w, "ct") == 0 || strcmp(w, "requires") == 0 || strcmp(w, "ensures") == 0 || strcmp(w, "ghost") == 0 || strcmp(w, "decreases") == 0 || strcmp(w, "law") == 0)) return false;
    }
    lx->mark_end(lx);
    s->started = 1; s->seen = 1;
    lx->result_symbol = NAME; return true;
  }
  return false;
}
