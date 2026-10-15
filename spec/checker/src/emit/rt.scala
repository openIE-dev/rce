// RCE runtime for emitted Scala 3 (spec 10.5). It mirrors the reference
// evaluator value by value, as rt.js does: integers are BigInt with range
// traps, f64 is the JVM double (strict IEEE 754 binary64 since Java 17,
// JEP 306: no contraction, round to nearest even), f32 is rounded once from
// f64, text is a JVM string measured in UTF-8 bytes, maps keep insertion
// order, values are never mutated in place.
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.control.ControlThrowable
import java.nio.charset.StandardCharsets.UTF_8

class Flow extends RuntimeException(null, null, false, false)
final class Trap(val m: String) extends Flow
final class Fail(val v: Any) extends Flow { var outs: Vector[Any] = Vector(); var slf: Any = null }
final class Stop extends Flow
final class Ret(val v: Any) extends Flow
final class Cancel(val f: Any) extends Flow
final class Refuse(val r: String) extends RuntimeException(null, null, false, false)
// loop control of the emitted code (not an RCE flow)
final class BrkEx extends ControlThrowable
final class ContEx extends ControlThrowable
val BRK = new BrkEx
val CONT = new ContEx
def trap(m: String): Nothing = throw new Trap(m)
def V(xs: Any*): Vector[Any] = xs.toVector
def at(x: Any, i: Int): Any = x match {
  case v: Vector[?] => v(i)
  case v: ArrayBuffer[?] => v(i)
  case v: Array[?] => v(i)
  case (a, b) => if (i == 0) a else b
  case _ => trap("bad index")
}
def atOr(x: Any, i: Int): Any = x match {
  case v: Vector[?] => if (i < v.length) v(i) else null
  case _ => null
}
def argat(a: Vector[Any], i: Int): Any = if (i < a.length) a(i) else UNIT
def vlen(x: Any): Int = x match { case v: Vector[?] => v.length; case v: ArrayBuffer[?] => v.length; case _ => 0 }
def vec(x: Any): Vector[Any] = x match {
  case v: Vector[?] => v.asInstanceOf[Vector[Any]]
  case v: ArrayBuffer[?] => v.toVector.asInstanceOf[Vector[Any]]
  case null => Vector()
  case _ => trap("bad vector")
}
def vpush(x: Any, v: Any): Vector[Any] = vec(x) :+ v

// Fuel (6.7, 10.5)
var FUEL_LIMIT: BigInt = BigInt(50000000)
var FUEL_TRACE = false
var FUEL: Long = 50000000L
val KC = mutable.HashMap[String, Any]()
def init_args(args: Seq[String]): Unit = {
  for (a <- args) if (a.startsWith("--fuel=")) FUEL_LIMIT = BigInt(a.substring(7))
  FUEL_TRACE = args.contains("--fuel-trace")
  FUEL = FUEL_LIMIT.toLong
}
def fuel(): Unit = { if (FUEL == 0L) trap("comptime fuel exhausted"); FUEL -= 1 }
// `bound n` (6.7): active calls of each bounded function, per thread
val DEPTH: ThreadLocal[scala.collection.mutable.HashMap[String, Long]] = ThreadLocal.withInitial(() => scala.collection.mutable.HashMap[String, Long]())
def depth_enter(k: String, n: Long): Unit = { val d = DEPTH.get.getOrElse(k, 0L); if (d >= n) trap("recursion bound"); DEPTH.get(k) = d + 1 }
def depth_exit(k: String): Unit = { val d = DEPTH.get.getOrElse(k, 0L); if (d > 0) DEPTH.get(k) = d - 1 }
def fuel_reset(): Unit = { FUEL = FUEL_LIMIT.toLong; KC.clear(); DEPTH.get.clear() }
def fuel_report(): Unit = if (FUEL_TRACE) say("fuel " + (FUEL_LIMIT - BigInt(FUEL)))
def is_flow(e: Throwable): Boolean = e.isInstanceOf[Flow]

def P2(n: Int): BigInt = BigInt(1) << n
val RANGES: Map[String, (BigInt, BigInt)] = Map(
  "i8" -> (BigInt(-128), BigInt(127)), "i16" -> (BigInt(-32768), BigInt(32767)),
  "i32" -> (-P2(31), P2(31) - 1), "i64" -> (-P2(63), P2(63) - 1),
  "u8" -> (BigInt(0), BigInt(255)), "u16" -> (BigInt(0), BigInt(65535)), "u32" -> (BigInt(0), P2(32) - 1),
  "u64" -> (BigInt(0), P2(64) - 1), "usize" -> (BigInt(0), P2(64) - 1))
def irange(t: String): (BigInt, BigInt) = RANGES.getOrElse(t, RANGES("i64"))
val I128: (BigInt, BigInt) = (-P2(127), P2(127) - 1)

final case class I(v: BigInt, t: String)
final case class F(v: Double)
final class F32(val v: Double) { override def equals(o: Any) = o match { case x: F32 => x.v == v; case _ => false }; override def hashCode = v.hashCode }
object UnitT
val UNIT = UnitT
final case class Tup(xs: Vector[Any])
final case class Lst(xs: Vector[Any])
final case class Mp(ps: Vector[(Any, Any)])
final class Opt(val x: Any)
val NONE = new Opt(null)
final class Rec(val n: String, val fs: Vector[(String, Any)])
final class Err(val e: Any)
final class FnV(val n: String)
final class Clo(val f: Vector[Any] => Any)
final class Rng(val a: BigInt, val b: BigInt, val t: String)
final class Gen(val run: Int => (Vector[Any], Boolean, Throwable))
final class Chan(val id: String, val cap: BigInt)
final class Task(val id: String)
final class Actor(val id: String, val ty: String)
final class Session(val out: String, val inn: String, val cap: BigInt)
final class LVar(var v: Any, val top: Any) { var frz: Boolean = false }
def some(x: Any): Opt = new Opt(x)
def lit(v: Long): I = I(BigInt(v), "lit")
def lit(v: BigInt): I = I(v, "lit")
def lit(v: Int): I = I(BigInt(v), "lit")
def isBool(v: Any): Boolean = v.isInstanceOf[Boolean]
def isStr(v: Any): Boolean = v.isInstanceOf[String]
def recfs(xs: Vector[Any]): Vector[(String, Any)] = xs.map(p => (at(p, 0).asInstanceOf[String], at(p, 1)))
def mkrec(n: String, fs: Vector[Any]): Rec = new Rec(n, recfs(fs))

// ---- formatting (Rust Display / Debug) ----
// Java's Double.toString and Float.toString give the shortest round-trip
// digits since JDK 19 (JDK-4511638); only the layout is changed here.
def digits_of(s0: String): (String, Int) = {
  val s = s0.replace('E', 'e')
  var m = s; var e = 0
  val k = s.indexOf('e')
  if (k >= 0) { m = s.substring(0, k); e = s.substring(k + 1).toInt }
  var d = ""; var p = 0
  val i = m.indexOf('.')
  if (i >= 0) { d = m.substring(0, i) + m.substring(i + 1); p = i + e } else { d = m; p = m.length + e }
  var z = 0
  while (z < d.length - 1 && d(z) == '0') z += 1
  d = d.substring(z); p -= z
  var j = d.length
  while (j > 0 && d(j - 1) == '0') j -= 1
  d = if (j == 0) "0" else d.substring(0, j)
  (d, p)
}
// Java prints at least two significant digits ("4.9E-324" for the least
// subnormal), so a two-digit result is checked for a one-digit round-trip
// form, choosing the nearer one as Rust does ("5e-324").
def sdigits(a: Double): (String, Int) = {
  val (d, p) = digits_of(java.lang.Double.toString(a))
  if (d.length != 2) return (d, p)
  val exact = new java.math.BigDecimal(a)
  val c0 = d(0) - '0'
  val cands = Seq((c0.toString, p), (if (c0 == 9) ("1", p + 1) else ((c0 + 1).toString, p)))
    .filter { case (c, q) => java.lang.Double.parseDouble("0." + c + "e" + q) == a }
  if (cands.isEmpty) (d, p)
  else cands.minBy { case (c, q) => new java.math.BigDecimal("0." + c + "e" + q).subtract(exact).abs }
}
def plain(neg: Boolean, d: String, p: Int): String = {
  val r =
    if (d == "0") "0"
    else if (p <= 0) "0." + "0" * (-p) + d
    else if (p >= d.length) d + "0" * (p - d.length)
    else d.substring(0, p) + "." + d.substring(p)
  (if (neg) "-" else "") + r
}
def negz(x: Double): Boolean = x < 0 || (x == 0 && 1.0 / x < 0)
def fdisplay(x: Double): String = {
  if (x == 0) return "0"
  val a = math.abs(x); val (d, p) = sdigits(a)
  if (a >= 1e-5 && a < 1e16) return plain(x < 0, d, p)
  val e = p - 1
  (if (x < 0) "-" else "") + d(0) + (if (d.length > 1) "." + d.substring(1) else "") + "e" + (if (e >= 0) "+" else "") + e
}
def fdebug(x: Double): String = {
  if (x == 0) return if (negz(x)) "-0.0" else "0.0"
  val a = math.abs(x)
  val (d, p) = sdigits(a)
  if (a >= 1e-4 && a < 1e16) { val s = plain(x < 0, d, p); return if (s.contains(".")) s else s + ".0" }
  (if (x < 0) "-" else "") + d(0) + (if (d.length > 1) "." + d.substring(1) else "") + "e" + (p - 1)
}
def f32display(x: Double): String = {
  if (x == 0) return if (negz(x)) "-0" else "0"
  val (d, p) = digits_of(java.lang.Float.toString(math.abs(x).toFloat))
  plain(x < 0, d, p)
}
def fround32(x: Double): Double = x.toFloat.toDouble
def int_to_f32(n: BigInt): Double = {
  val a = n.abs
  if (a < P2(24)) return n.toDouble
  val sh = a.bitLength - 24
  var m = a >> sh
  val rem = a - (m << sh); val half = BigInt(1) << (sh - 1)
  if (rem > half || (rem == half && m.testBit(0))) m += 1
  val r = fround32(m.toDouble * math.pow(2, sh))
  if (n < 0) -r else r
}
def rdebug(s: String): String = {
  val o = new StringBuilder("\"")
  s.codePoints().forEach { k =>
    if (k == '\t') o ++= "\\t" else if (k == '\r') o ++= "\\r" else if (k == '\n') o ++= "\\n"
    else if (k == '\\') o ++= "\\\\" else if (k == '"') o ++= "\\\"" else if (k == 0) o ++= "\\0"
    else if (k < 0x20 || (k >= 0x7f && k <= 0x9f) || (k >= 0x300 && k <= 0x36f)) o ++= "\\u{" + Integer.toHexString(k) + "}"
    else o.appendAll(Character.toChars(k))
  }
  o.append('"').toString
}
def show(v: Any): String = v match {
  case b: Boolean => if (b) "true" else "false"
  case s: String => rdebug(s)
  case x: I => x.v.toString
  case x: LVar => "lvar(" + show(x.v) + ")"
  case x: F => fdisplay(x.v)
  case UnitT => "()"
  case x: Tup => "(" + x.xs.map(show).mkString(", ") + ")"
  case x: Lst => "[" + x.xs.map(show).mkString(", ") + "]"
  case x: Mp => "{" + x.ps.map(p => show(p._1) + ": " + show(p._2)).mkString(", ") + "}"
  case x: Opt => if (x.x == null) "none" else "some(" + show(x.x) + ")"
  case x: Rec => if (x.fs.isEmpty) x.n else x.n + "(" + x.fs.map(p => p._1 + ": " + show(p._2)).mkString(", ") + ")"
  case x: Err => "error " + show(x.e)
  case x: FnV => x.n
  case _: Clo => "<fn>"
  case x: Rng => s"${x.a}..${x.b}"
  case x: F32 => f32display(x.v)
  case _: Gen => "<iter>"
  case _: Chan => "<chan>"
  case _: Task => "<task>"
  case x: Actor => "<actor " + x.ty + ">"
  case _: Session => "<session>"
  case _ => "?"
}
def veq(a: Any, b: Any): Boolean = (a, b) match {
  case (x: Boolean, y: Boolean) => x == y
  case (_: Boolean, _) | (_, _: Boolean) => false
  case (x: String, y: String) => x == y
  case (_: String, _) | (_, _: String) => false
  case (x: I, y: I) => x.v == y.v
  case (x: F, y: F) => x.v == y.v
  case (x: F32, y: F32) => x.v == y.v
  case (UnitT, UnitT) => true
  case (x: Tup, y: Tup) => x.xs.length == y.xs.length && x.xs.indices.forall(i => veq(x.xs(i), y.xs(i)))
  case (x: Lst, y: Lst) => x.xs.length == y.xs.length && x.xs.indices.forall(i => veq(x.xs(i), y.xs(i)))
  case (x: Mp, y: Mp) => x.ps.length == y.ps.length && x.ps.forall(p => y.ps.exists(q => veq(p._1, q._1) && veq(p._2, q._2)))
  case (x: Opt, y: Opt) => if (x.x == null || y.x == null) x.x == null && y.x == null else veq(x.x, y.x)
  case (x: Rec, y: Rec) => x.n == y.n && x.fs.length == y.fs.length && x.fs.indices.forall(i => x.fs(i)._1 == y.fs(i)._1 && veq(x.fs(i)._2, y.fs(i)._2))
  case (x: Err, y: Err) => veq(x.e, y.e)
  case (x: Chan, y: Chan) => x.id == y.id
  case (x: Task, y: Task) => x.id == y.id
  case (x: Actor, y: Actor) => x.id == y.id
  case (x: Session, y: Session) => x.out == y.out && x.inn == y.inn
  case (x: LVar, y: LVar) => x eq y
  case _ => false
}
def c3[T](x: T, y: T)(using o: Ordering[T]): Int = { val c = o.compare(x, y); if (c > 0) 1 else if (c < 0) -1 else 0 }
def cpcmp(a: String, b: String): Int = {
  val x = a.codePoints().toArray; val y = b.codePoints().toArray
  var i = 0
  while (i < math.min(x.length, y.length)) { val r = c3(x(i), y(i)); if (r != 0) return r; i += 1 }
  c3(x.length, y.length)
}
def cmp(a: Any, b: Any): Int = (a, b) match {
  case (x: I, y: I) => c3(x.v, y.v)
  case (x: F, y: F) => c3(x.v, y.v)
  case (x: F32, y: F32) => c3(x.v, y.v)
  case (x: String, y: String) => cpcmp(x, y)
  case (x: Boolean, y: Boolean) => c3(if (x) 1 else 0, if (y) 1 else 0)
  case (x: Tup, y: Tup) => seqcmp(x.xs, y.xs)
  case (x: Lst, y: Lst) => seqcmp(x.xs, y.xs)
  case _ => trap("values are not ordered")
}
def seqcmp(a: Vector[Any], b: Vector[Any]): Int = {
  var i = 0
  while (i < math.min(a.length, b.length)) { val r = cmp(a(i), b(i)); if (r != 0) return r; i += 1 }
  c3(a.length, b.length)
}
def truthy(v: Any): Boolean = v match { case b: Boolean => b; case _ => trap("expected bool, found " + show(v)) }
def chk(x: BigInt, t: String): I = {
  val tt = if (t == "lit") "i64" else t
  val (lo, hi) = irange(tt)
  if (x < lo || x > hi) trap("overflow: " + x + " is outside " + tt)
  I(x, t)
}
def fin(r: Double): F = if (r.isFinite) F(r) else trap("non-finite float")
def fin32(r: Double): F32 = if (r.isFinite) new F32(r) else trap("non-finite float")
def jsdiv(x: BigInt, y: BigInt): BigInt = x / y // BigInt `/` truncates toward zero, `%` has the dividend's sign
def arith(op: String, a: Any, b: Any): Any = (a, b) match {
  case (ai: I, bi: I) =>
    val x = ai.v; val y = bi.v; val ty = if (ai.t == "lit") bi.t else ai.t
    val bits = ty match { case "i8" | "u8" => 8; case "i16" | "u16" => 16; case "i32" | "u32" => 32; case _ => 64 }
    op match {
      case "+" => chk(x + y, ty)
      case "-" => chk(x - y, ty)
      case "*" => chk(x * y, ty)
      case "/" => if (y == 0) trap("division by zero"); chk(x / y, ty)
      case "%" => if (y == 0) trap("division by zero"); chk(x / y, ty); chk(x % y, ty)
      case "&" => chk(x & y, ty)
      case "|" => chk(x | y, ty)
      case "^" => chk(x ^ y, ty)
      case "<<" =>
        if (y < 0 || y >= bits) trap("shift amount out of range")
        val r = (x << y.toInt) & (P2(bits) - 1)
        chk(if (ty.startsWith("i") && r >= P2(bits - 1)) r - P2(bits) else r, ty)
      case ">>" => if (y < 0 || y >= bits) trap("shift amount out of range"); chk(x >> y.toInt, ty)
      case _ => trap("bad operator " + op)
    }
  case (x: F, y: F) => fop(op, x.v, y.v, false)
  case (x: F32, y: F32) => fop(op, x.v, y.v, true)
  case (x: Lst, y: Lst) if Set("+", "-", "*", "/", "%").contains(op) =>
    if (x.xs.length != y.xs.length) trap("lane count mismatch")
    Lst(x.xs.indices.map(i => arith(op, x.xs(i), y.xs(i))).toVector)
  case (x: F, y: I) => arith(op, x, F(y.v.toDouble))
  case (x: I, y: F) => arith(op, F(x.v.toDouble), y)
  case (x: String, y: String) if op == "+" => x + y
  case _ => trap("cannot apply " + op + " to " + show(a) + " and " + show(b))
}
def fop(op: String, x: Double, y: Double, f3: Boolean): Any = {
  val r = op match { case "+" => x + y; case "-" => x - y; case "*" => x * y; case "/" => x / y; case _ => trap("bad float operator") }
  if (f3) fin32(fround32(r)) else fin(r)
}
def contains(c: Any, x: Any): Boolean = c match {
  case l: Lst => l.xs.exists(y => veq(y, x))
  case t: Tup => t.xs.exists(y => veq(y, x))
  case m: Mp => m.ps.exists(p => veq(p._1, x))
  case s: String => x match { case y: String => s.contains(y); case _ => false }
  case r: Rng => x match { case i: I => r.a <= i.v && i.v < r.b; case _ => false }
  case _ => trap("`in` needs a collection")
}
def binop(op: String, x: Any, y: Any): Any = op match {
  case "==" => veq(x, y)
  case "!=" => !veq(x, y)
  case "<" => cmp(x, y) < 0
  case "<=" => cmp(x, y) <= 0
  case ">" => cmp(x, y) > 0
  case ">=" => cmp(x, y) >= 0
  case "in" => contains(y, x)
  case "not in" => !contains(y, x)
  case ".." | "..=" => (x, y) match {
    case (a: I, b: I) => new Rng(a.v, if (op == "..") b.v else b.v + 1, if (a.t == "lit") b.t else a.t)
    case _ => trap("bad range")
  }
  case _ => arith(op, x, y)
}
def map_set(ps: Vector[(Any, Any)], k: Any, v: Any): Vector[(Any, Any)] = {
  val i = ps.indexWhere(p => veq(p._1, k))
  if (i >= 0) ps.updated(i, (ps(i)._1, v)) else { val j = ps.indexWhere(p => canon_cmp(p._1, k) > 0); if (j < 0) ps :+ ((k, v)) else ps.patch(j, Seq((k, v)), 0) }
}
def mk_map(pairs: Vector[Any]): Mp = { var ps = Vector[(Any, Any)](); for (p <- pairs) ps = map_set(ps, at(p, 0), at(p, 1)); Mp(ps) }
def list_repeat(x: Any, n: Any): Lst = n match {
  case i: I if i.v >= 0 && i.v <= 10000000 => charge(BigInt(4) + i.v); Lst(Vector.fill(i.v.toInt)(x))
  case _ => trap("bad repeat count")
}
def u8s(s: String): Array[Byte] = s.getBytes(UTF_8)
// SHA-256 (FIPS 180-4) as 64 lowercase hex digits (5.15 `sha256`), from the JDK
def sha256hex(m: Array[Byte]): String = java.security.MessageDigest.getInstance("SHA-256").digest(m).map(b => f"${b & 0xff}%02x").mkString
def index(c: Any, i: Any): Any = (c, i) match {
  case (l: Lst, k: I) => if (k.v >= 0 && k.v < l.xs.length) l.xs(k.v.toInt) else trap("index " + k.v + " out of range")
  case (s: String, k: I) =>
    val b = u8s(s)
    if (k.v >= 0 && k.v < b.length) I(BigInt(b(k.v.toInt) & 0xff), "u8") else trap("index " + k.v + " out of range")
  case (m: Mp, _) => m.ps.find(p => veq(p._1, i)) match { case Some(p) => p._2; case None => trap("missing key " + show(i)) }
  case _ => trap("cannot index " + show(c))
}
def slice(c: Any, ta: () => Any, tb: () => Any): Any = {
  charge(1)
  var xs: Vector[Any] = null; var text: Array[Byte] = null
  c match { case l: Lst => xs = l.xs; case s: String => text = u8s(s); case _ => trap("cannot slice " + show(c)) }
  val n = BigInt(if (text != null) text.length else xs.length)
  val av = if (ta != null) ta() else null; val bv = if (tb != null) tb() else null
  def norm(v: Any, d: BigInt): BigInt = v match {
    case null => d
    case i: I => var k = i.v; if (k < 0) k = n + k; if (k < 0 || k > n) trap("slice index out of range"); k
    case _ => trap("bad slice bound")
  }
  val lo = norm(av, 0).toInt; val hi = norm(bv, n).toInt
  if (lo > hi) trap("slice bounds reversed")
  if (text != null) {
    val N = text.length
    def bd(k: Int) = k == 0 || k == N || (text(k) & 0xc0) != 0x80
    if (!bd(lo) || !bd(hi)) trap("not a char boundary")
    return new String(text.slice(lo, hi), UTF_8)
  }
  Lst(xs.slice(lo, hi))
}
def field(v: Any, m: String): Any = v match {
  case r: Rec => r.fs.find(_._1 == m) match { case Some(p) => p._2; case None => trap("no field " + m) }
  case _ => trap(show(v) + " has no field " + m)
}
def tmember(v: Any, i: Int): Any = v match { case t: Tup if i < t.xs.length => t.xs(i); case _ => trap("bad tuple index") }
def neg(v: Any): Any = v match {
  case x: I => chk(-x.v, x.t)
  case x: F => F(-x.v)
  case x: F32 => new F32(-x.v)
  case _ => trap("bad negation")
}
def bitnot(v: Any): Any = v match {
  case x: I => val (lo, hi) = irange(if (x.t == "lit") "i64" else x.t); I(if (lo == 0) hi - x.v else ~x.v, x.t)
  case _ => trap("bad ~")
}
def not_(v: Any): Boolean = !truthy(v)
def sat128(x: BigInt): BigInt = if (x < I128._1) I128._1 else if (x > I128._2) I128._2 else x
def trunc_big(f: Double): BigInt = sat128(BigDecimal(f).setScale(0, BigDecimal.RoundingMode.DOWN).toBigInt)
def cast(v: Any, T: Vector[Any]): Any = {
  val toF32 = T(0) == "f32"; val n: String = if (T(0) == "int") T(1).asInstanceOf[String] else null
  v match {
    case x: I if n == null && toF32 => new F32(int_to_f32(x.v))
    case x: F if n == null && toF32 => val r = fround32(x.v); if (!r.isFinite) trap("non-finite float"); new F32(r)
    case x: F32 if n == null && !toF32 => F(x.v)
    case x: F32 if n != null => chk(trunc_big(x.v), n)
    case x: I if n != null => chk(x.v, n)
    case x: F if n != null => if (!x.v.isFinite) trap("non-finite float"); chk(trunc_big(x.v), n)
    case x: Boolean if n != null => chk(if (x) 1 else 0, n)
    case x: I if n == null => F(x.v.toDouble)
    case _ => v
  }
}
def ctx_lit(s: Any): Any = { charge(4); s }
def ctx_encode(c: String, s0: Any): Any = {
  charge(BigInt(4) + size_of(s0))
  val s = s0 match { case x: String => x; case _ => trap("context text") }
  if (c == "Html") return s.flatMap { case '&' => "&amp;"; case '<' => "&lt;"; case '>' => "&gt;"; case '"' => "&quot;"; case '\'' => "&#39;"; case ch => ch.toString }
  if (c == "Url") return u8s(s).map { b0 => val b = b0 & 0xff
    if ((b >= 48 && b <= 57) || (b >= 65 && b <= 90) || (b >= 97 && b <= 122) || b == 45 || b == 46 || b == 95 || b == 126) b.toChar.toString else "%" + f"$b%02X" }.mkString
  if (c == "Sql") return "'" + s.replace("'", "''") + "'"
  val o = new StringBuilder("\"")
  s.codePoints().forEach { u =>
    if (u == '"') o ++= "\\\"" else if (u == '\\') o ++= "\\\\" else if (u == '\n') o ++= "\\n" else if (u == '\r') o ++= "\\r" else if (u == '\t') o ++= "\\t"
    else if (u < 0x20) o ++= "\\u" + f"$u%04x" else o.appendAll(Character.toChars(u))
  }
  o.append('"').toString
}
def asBig(x: Any): BigInt = x match { case b: BigInt => b; case i: Int => BigInt(i); case l: Long => BigInt(l); case _ => null }
def ptm(t: Vector[Any], v: Any): BigInt = {
  if (t(0) == "lit") return asBig(t(1))
  if (t(0) == "self") return v match { case i: I => i.v; case _ => null }
  v match { case s: String => BigInt(u8s(s).length); case l: Lst => BigInt(l.xs.length); case m: Mp => BigInt(m.ps.length); case _ => null }
}
def pholds(p: Vector[Any], v: Any): Boolean = {
  val k = p(0)
  if (k == "cmp") {
    val x = ptm(vec(p(2)), v); val y = ptm(vec(p(3)), v)
    if (x == null || y == null) return false
    return p(1) match { case "<" => x < y; case "<=" => x <= y; case ">" => x > y; case ">=" => x >= y; case "==" => x == y; case _ => x != y }
  }
  if (k == "cls") {
    v match {
      case s: String =>
        return p(1) match {
          case "isdigit" => s.nonEmpty && s.forall(c => c >= '0' && c <= '9')
          case "isalpha" => s.nonEmpty && s.forall(c => (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
          case "isspace" => s.nonEmpty && s.forall(c => " \t\n\r\f".contains(c))
          case _ => s == ""
        }
      case _ => return false
    }
  }
  if (k == "and") return pholds(vec(p(1)), v) && pholds(vec(p(2)), v)
  if (k == "or") return pholds(vec(p(1)), v) || pholds(vec(p(2)), v)
  !pholds(vec(p(1)), v)
}
def fit(v0: Any, T0: Any): Any = {
  val T = vec(T0)
  var v = v0
  if (T(0) == "refn") { v = fit(v, T(1)); if (!pholds(vec(T(3)), v)) trap("refinement: value is outside " + T(2)); return v }
  val k = T(0)
  v match {
    case x: I =>
      if (k == "int") { val t = T(1).asInstanceOf[String]; val (lo, hi) = irange(t); if (x.v < lo || x.v > hi) trap("overflow: " + x.v + " is outside " + t); return I(x.v, t) }
      if (k == "f64") return F(x.v.toDouble)
      if (k == "f32") return new F32(int_to_f32(x.v))
    case _ =>
  }
  v match {
    case o: Opt if o.x != null && k == "opt" => return new Opt(fit(o.x, T(1)))
    case l: Lst if k == "list" || k == "arr" => return Lst(l.xs.map(x => fit(x, T(1))))
    case m: Mp if k == "map" => return Mp(m.ps.map(p => (fit(p._1, T(1)), fit(p._2, T(2)))))
    case t: Tup if k == "tup" && t.xs.length == vlen(T(1)) => val ts = vec(T(1)); return Tup(t.xs.indices.map(i => fit(t.xs(i), ts(i))).toVector)
    case _: Err => return v
    case _ =>
  }
  if (k == "union" || k == "ref") return fit(v, T(1))
  v
}
def fitall(a: Vector[Any], Ts: Vector[Any]): Vector[Any] = a.take(Ts.length).indices.map(i => fit(a(i), Ts(i))).toVector
def wrapsat(m: String, x: BigInt, t: String, y0: Any): I = {
  val y = y0 match { case i: I => i.v; case _ => trap(m) }
  val r = if (m.endsWith("add")) x + y else if (m.endsWith("sub")) x - y else x * y
  val (lo, hi) = irange(t)
  if (m.startsWith("wrapping")) { val w = hi - lo + 1; var q = (r - lo) % w; if (q < 0) q += w; return I(q + lo, t) }
  I(if (r < lo) lo else if (r > hi) hi else r, t)
}
val WS: Set[Int] = "\t\n\u000b\u000c\r \u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000".map(_.toInt).toSet
def cps(s: String): Array[Int] = s.codePoints().toArray
def fromCps(c: Array[Int], i: Int, j: Int): String = new String(c, i, j - i)
def rtrim(s: String): String = { val c = cps(s); var i = 0; var j = c.length; while (i < j && WS(c(i))) i += 1; while (j > i && WS(c(j - 1))) j -= 1; fromCps(c, i, j) }
def rwords(s: String): Vector[Any] = {
  val out = ArrayBuffer[Any](); val cur = new java.lang.StringBuilder()
  for (c <- cps(s)) { if (WS(c)) { if (cur.length > 0) { out += cur.toString; cur.setLength(0) } } else cur.appendCodePoint(c) }
  if (cur.length > 0) out += cur.toString
  out.toVector
}
def jsplit(s: String, sep: String): Vector[Any] = {
  val out = ArrayBuffer[Any](); var i = 0
  var k = s.indexOf(sep, i)
  while (k >= 0) { out += s.substring(i, k); i = k + sep.length; k = s.indexOf(sep, i) }
  out += s.substring(i)
  out.toVector
}
def rlines(s: String): Vector[Any] = {
  if (s == "") return Vector()
  var p = jsplit(s, "\n")
  if (p.last == "") p = p.dropRight(1)
  p.map { x => val t = x.asInstanceOf[String]; if (t.endsWith("\r")) t.dropRight(1) else t }
}
def aupper(s: String): String = s.map(c => if (c >= 'a' && c <= 'z') (c - 32).toChar else c)
def alower(s: String): String = s.map(c => if (c >= 'A' && c <= 'Z') (c + 32).toChar else c)
def toint(s: String): I = {
  val t = rtrim(s); val u = if (t.startsWith("+") || t.startsWith("-")) t.substring(1) else t
  if (u == "" || !u.forall(c => c >= '0' && c <= '9')) trap("not an integer: " + rdebug(s))
  val n = BigInt(if (t.startsWith("+")) u else t)
  if (n < I128._1 || n > I128._2) trap("not an integer: " + rdebug(s))
  chk(n, "i64")
}
def fround(x: Double): Double = { var r = if (x < 0) math.ceil(x) else math.floor(x); if (math.abs(x - r) >= 0.5) r += math.signum(x); r }
def sort_list(xs: Vector[Any]): Vector[Any] = {
  var err: Trap = null
  val r = xs.sortWith { (p, q) => try cmp(p, q) < 0 catch { case e: Trap => if (err == null) err = e; false } }
  if (err != null) throw err
  r
}
def apply_(f: Any, vals: Vector[Any]): Any = f match {
  case c: Clo => c.f(vals)
  case g: FnV => RAW(g.n)(vals)
  case _ => trap(show(f) + " is not callable")
}
def call_value(f: Any, vals: Vector[Any]): Any = f match {
  case g: FnV => fuel(); FNS(g.n)(null, vals)(0)
  case c: Clo => c.f(vals)
  case _ => trap(show(f) + " is not callable")
}
def a0(a: Vector[Any]): Any = if (a.nonEmpty) a(0) else UNIT
def builtin(r: Any, m: String, a: Vector[Any]): (Any, Any) = {
  r match {
    case c: Chan if m == "close" => comm(V("close", c)); return (UNIT, null)
    case t: Task if m == "join" => val rp = comm(V("join", t.id)); return (if (rp(0) == "val") rp(1) else UNIT, null)
    case s: Session if m == "offer" =>
      charge(8)
      val rp = comm(V("recv", new Chan(s.inn, s.cap)))
      rp(1) match {
        case o: Opt if rp(0) == "val" && o.x != null =>
          return (o.x match { case v: Rec if v.fs.isEmpty => new Rec("$label:" + v.n, Vector()); case v => v }, null)
        case _ => trap("session closed")
      }
    case s: Session if m == "close" => comm(V("close", new Chan(s.out, s.cap))); return (UNIT, null)
    case lv: LVar =>
      if (m == "put") { lvar_put(lv, a0(a)); return (UNIT, null) }
      if (m == "wait") return (lvar_wait(lv, a0(a)), null)
      if (m == "freeze") { lv.frz = true; return (lv.v, null) }
    case _ =>
  }
  r match {
    case ri: I =>
      val x = ri.v; val ty = ri.t
      m match {
        case "checked_add" | "checked_sub" | "checked_mul" | "checked_div" =>
          val op = m match { case "checked_add" => "+"; case "checked_sub" => "-"; case "checked_mul" => "*"; case _ => "/" }
          try return (new Opt(arith(op, r, a(0))), null) catch { case _: Trap => return (NONE, null) }
        case "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "saturating_add" | "saturating_sub" | "saturating_mul" => return (wrapsat(m, x, ty, a(0)), null)
        case "tostr" => return (x.toString, null)
        case "abs" => return (chk(x.abs, ty), null)
        case "min" | "max" =>
          val y = a(0) match { case i: I => i.v; case _ => trap(m) }
          return (I(if (m == "min") (if (x < y) x else y) else (if (x > y) x else y), ty), null)
        case _ =>
      }
    case rf: F =>
      val x = rf.v
      m match {
        case "sqrt" => if (x < 0) trap("non-finite float"); return (F(math.sqrt(x)), null)
        case "abs" => return (F(math.abs(x)), null)
        case "min" | "max" =>
          val y = a(0) match { case f: F => f.v; case _ => trap(m) }
          return (F(if (m == "min") (if (x <= y) x else y) else (if (x >= y) x else y)), null)
        case "floor" => return (F(math.floor(x)), null)
        case "ceil" => return (F(math.ceil(x)), null)
        case "round" => return (F(fround(x)), null)
        case "tostr" => return (fdisplay(x), null)
        case _ =>
      }
    case b: Boolean if m == "tostr" => return (if (b) "true" else "false", null)
    case s: String if m == "len" => return (I(BigInt(u8s(s).length), "usize"), null)
    case f: F32 =>
      if (m == "abs") return (new F32(math.abs(f.v)), null)
      if (m == "tostr") return (f32display(f.v), null)
    case g: Gen if m == "collect" =>
      val (items, _, err) = g.run(Int.MaxValue)
      if (err != null) throw err
      return (Lst(items), null)
    case _ =>
  }
  r match {
    case s: String =>
      def sa: String = a(0) match { case x: String => x; case _ => trap(m) }
      m match {
        case "strip" => return (rtrim(s), null)
        case "upper" => return (aupper(s), null)
        case "sha256" => return (sha256hex(u8s(s)), null)
        case "lower" => return (alower(s), null)
        case "reverse" => val c = cps(s).reverse; return (fromCps(c, 0, c.length), null)
        case "tostr" | "raw" => return (s, null)
        case "chars" => return (Lst(cps(s).map(c => new String(Character.toChars(c))).toVector), null)
        case "words" => return (Lst(rwords(s)), null)
        case "lines" => return (Lst(rlines(s)), null)
        case "split" =>
          val sep = a(0) match { case x: String => x; case _ => trap("split") }
          if (sep == "") return (Lst(("" +: cps(s).map(c => new String(Character.toChars(c))).toVector) :+ ""), null)
          return (Lst(jsplit(s, sep)), null)
        case "isdigit" => return (s.nonEmpty && s.forall(c => c >= '0' && c <= '9'), null)
        case "isalpha" => return (s.nonEmpty && s.forall(c => (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')), null)
        case "isspace" => return (s.nonEmpty && s.forall(c => " \t\n\u000c\r".contains(c)), null)
        case "isempty" => return (s == "", null)
        case "contains" => return (s.contains(sa), null)
        case "startswith" => return (s.startsWith(sa), null)
        case "endswith" => return (s.endsWith(sa), null)
        case "toint" => return (toint(s), null)
        case "join" =>
          a(0) match { case l: Lst => return (l.xs.map { case x: String => x; case x => show(x) }.mkString(s), null); case _ => trap("join") }
        case _ =>
      }
    case l: Lst =>
      val xs = l.xs
      m match {
        case "len" => return (I(BigInt(xs.length), "usize"), null)
        case "isempty" => return (xs.isEmpty, null)
        case "append" | "push" => return (UNIT, Lst(xs :+ a(0)))
        case "pop" => return if (xs.isEmpty) (NONE, Lst(Vector())) else (new Opt(xs.last), Lst(xs.dropRight(1)))
        case "clear" => return (UNIT, Lst(Vector()))
        case "sort" => return (UNIT, Lst(sort_list(xs)))
        case "with" => return (Lst(xs :+ a(0)), null)
        case "reverse" => return (Lst(xs.reverse), null)
        case "contains" => return (xs.exists(x => veq(x, a(0))), null)
        case "get" =>
          val k = a(0) match { case i: I => i.v; case _ => trap("get") }
          return (if (k >= 0 && k < xs.length) new Opt(xs(k.toInt)) else NONE, null)
        case "utf8" =>
          val bs = xs.map { case i: I if i.v >= 0 && i.v <= 255 => i.v.toByte; case _ => trap("utf8: byte outside 0..255") }.toArray
          val dec = java.nio.charset.StandardCharsets.UTF_8.newDecoder().onMalformedInput(java.nio.charset.CodingErrorAction.REPORT).onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT)
          try return (dec.decode(java.nio.ByteBuffer.wrap(bs)).toString, null) catch { case _: java.nio.charset.CharacterCodingException => trap("utf8: invalid UTF-8") }
        case "sum" => var acc: Any = I(0, "lit"); for (x <- xs) acc = arith("+", acc, x); return (acc, null)
        case "map" => return (Lst(xs.map { x => fuel(); apply_(a(0), Vector(x)) }), null)
        case "filter" => return (Lst(xs.filter { x => fuel(); apply_(a(0), Vector(x)) == true }), null)
        case "pscan" => return (pscan(xs, a(0), a(1), false), null)
        case "pscan_incl" => return (pscan(xs, a(0), a(1), true), null)
        case "fold" => var acc = a(0); for (x <- xs) { fuel(); acc = apply_(a(1), Vector(acc, x)) }; return (acc, null)
        case _ =>
      }
    case mp: Mp =>
      val ps = mp.ps
      m match {
        case "len" => return (I(BigInt(ps.length), "usize"), null)
        case "get" => ps.find(p => veq(p._1, a(0))) match { case Some(p) => return (p._2, null); case None => return (new Err(new Rec("Missing", Vector())), null) }
        case "contains" => return (ps.exists(p => veq(p._1, a(0))), null)
        case "insert" => return (UNIT, Mp(map_set(ps, a(0), a(1))))
        case "remove" =>
          val i = ps.indexWhere(p => veq(p._1, a(0)))
          if (i < 0) return (NONE, Mp(ps))
          return (new Opt(ps(i)._2), Mp(ps.take(i) ++ ps.drop(i + 1)))
        case "keys" => return (Lst(ps.map(_._1)), null)
        case "values" => return (Lst(ps.map(_._2)), null)
        case _ =>
      }
    case o: Opt =>
      if (m == "isnone") return (o.x == null, null)
      if (m == "issome") return (o.x != null, null)
      if (m == "or") return (if (o.x == null) a(0) else o.x, null)
    case _ =>
  }
  trap(show(r) + " has no method " + m + " at comptime")
}
// C.6.4 method work, as conc::method_work in the reference evaluator.
val LIN_TEXT = Set("strip", "upper", "lower", "reverse", "tostr", "chars", "words", "lines", "split", "isdigit", "isalpha", "isspace", "contains", "startswith", "endswith", "toint", "join", "sha256")
val LIN_LIST = Set("reverse", "contains", "sum", "map", "filter", "fold", "utf8")
val LIN_MAP = Set("get", "contains", "insert", "remove", "keys", "values")
def size_of(v: Any): BigInt = v match { case s: String => BigInt(u8s(s).length); case l: Lst => BigInt(l.xs.length); case m: Mp => BigInt(m.ps.length); case _ => BigInt(0) }
def ceil_log2(n: BigInt): BigInt = if (n <= 1) 0 else BigInt((n - 1).bitLength)
def method_work(recv: Any, m: String, args: Vector[Any]): BigInt = {
  val n = size_of(recv); val k = if (args.nonEmpty) size_of(args(0)) else BigInt(0)
  if ((isStr(recv) && LIN_TEXT(m)) || (recv.isInstanceOf[Lst] && LIN_LIST(m)) || (recv.isInstanceOf[Mp] && LIN_MAP(m))) return n + k
  if (recv.isInstanceOf[Lst] && m == "sort") return n * ceil_log2(n)
  0
}
def head_of(recv: Any): String = recv match { case r: Rec => KIND_OF.getOrElse(r.n, r.n); case _: Lst => "List"; case _ => null }
def call_method(recv: Any, m: String, args: Vector[Any], sized: Boolean): Vector[Any] = {
  val head = head_of(recv)
  if (head != null && METHODS.contains(head + "." + m)) { fuel(); return METHODS(head + "." + m)(recv, args) }
  charge(BigInt(4) + (if (sized) method_work(recv, m, args) else BigInt(0)))
  val (v, s) = builtin(recv, m, args)
  Vector(v, Vector(), s)
}
def construct(name: String, fields: Vector[Any], args: Vector[Any]): Rec = {
  val n = fields.length; val vals = Array.fill[Any](n)(null)
  for ((p, i) <- args.zipWithIndex) {
    val lb = at(p, 0); val v = at(p, 1)
    var k = i
    if (lb != null) { val j = fields.indexWhere(f => at(f, 0) == lb); if (j >= 0) k = j }
    if (k < n) vals(k) = fit(v, at(fields(k), 1))
  }
  new Rec(name, fields.indices.map { k =>
    val f = fields(k); var v = vals(k)
    if (v == null) { val init = at(f, 2); if (init == null) trap("missing field " + at(f, 0)); v = fit(init.asInstanceOf[() => Any](), at(f, 1)) }
    (at(f, 0).asInstanceOf[String], v)
  }.toVector)
}
type Binds = mutable.HashMap[String, Any]
def new_binds(): Binds = mutable.HashMap[String, Any]()
def getb(b: Any, n: String): Any = b.asInstanceOf[Binds].getOrElse(n, UNIT)
def lastseg(s: String): String = s.split("::", -1).last
def pmatch(P0: Any, v: Any, b: Binds): Boolean = {
  val P = vec(P0)
  P(0) match {
    case "wild" => true
    case "bind" =>
      v match { case r: Rec if r.fs.isEmpty && r.n.startsWith("$label:") => return r.n.substring(7) == lastseg(P(1).asInstanceOf[String]); case _ => }
      b(P(1).asInstanceOf[String]) = v; true
    case "tag" =>
      v match {
        case r: Rec if r.fs.isEmpty && r.n.startsWith("$label:") => r.n.substring(7) == lastseg(P(1).asInstanceOf[String])
        case r: Rec => r.n == P(1)
        case _ => false
      }
    case "some" => v match { case o: Opt if o.x != null => pmatch(P(1), o.x, b); case _ => false }
    case "ctor" =>
      v match {
        case r: Rec if r.n == P(1) =>
          val ps = vec(P(2))
          for (i <- ps.indices) {
            val lb = at(ps(i), 0); val q = at(ps(i), 1)
            var fv: Any = null
            if (lb != null) { r.fs.find(_._1 == lb) match { case Some(f) => fv = f._2; case None => } } else if (i < r.fs.length) fv = r.fs(i)._2
            if (fv == null || !pmatch(q, fv, b)) return false
          }
          true
        case _ => false
      }
    case "lit" => veq(P(1).asInstanceOf[() => Any](), v)
    case "range" =>
      val x = P(1).asInstanceOf[() => Any](); val y = P(3).asInstanceOf[() => Any]()
      cmp(v, x) >= 0 && (if (P(2) == true) cmp(v, y) <= 0 else cmp(v, y) < 0)
    case "tup" =>
      val ps = vec(P(1))
      v match {
        case t: Tup if t.xs.length == ps.length => ps.indices.forall(i => pmatch(ps(i), t.xs(i), b))
        case _ => false
      }
    case "alt" => vec(P(1)).exists(q => pmatch(q, v, b))
    case _ => false
  }
}
def pm(P: Any, v: Any): Binds = { val b = new_binds(); if (pmatch(P, v, b)) b else null }
val YIELDS = ArrayBuffer[(ArrayBuffer[Any], Int)]()
def yield_(v: Any): Unit = {
  if (YIELDS.isEmpty) trap("yield outside an iterator")
  val (buf, cap) = YIELDS.last
  if (buf.length >= LOOP_LIMIT) trap("loop bound")
  buf += v
  if (buf.length >= cap) throw new Stop
}
def gen_run(body: () => Unit, cap: Int): (Vector[Any], Boolean, Throwable) = {
  YIELDS += ((ArrayBuffer[Any](), cap))
  var fin = true; var err: Throwable = null
  try body() catch {
    case _: Stop => fin = false
    case e: Trap => err = e
    case e: Fail => err = e
    case e: Ret => ()
    case e: Throwable => YIELDS.remove(YIELDS.length - 1); throw e
  }
  val items = YIELDS.remove(YIELDS.length - 1)._1
  (items.toVector, fin, err)
}
def iter_items(v: Any): Vector[Any] = v match {
  case l: Lst => l.xs
  case r: Rng =>
    if (r.b - r.a > LOOP_LIMIT * 10) trap("loop bound")
    val out = ArrayBuffer[Any](); var i = r.a
    while (i < r.b) { out += I(i, r.t); i += 1 }
    out.toVector
  case s: String => cps(s).map(c => new String(Character.toChars(c))).toVector
  case m: Mp => m.ps.map(p => Tup(Vector(p._1, p._2)))
  case g: Gen => val (items, _, err) = g.run(Int.MaxValue); if (err != null) throw err; items
  case _ => trap("cannot iterate over " + show(v))
}
def iterate(v: Any): Iterator[Any] = v match {
  case g: Gen => new Iterator[Any] {
    var done = 0; var cap = 64; var buf: Vector[Any] = Vector(); var pos = 0; var fin = false; var err: Throwable = null; var started = false
    def fill(): Unit = {
      while (pos >= buf.length && !(started && (fin || err != null))) {
        if (started) cap *= 2
        started = true
        val (items, f, e) = g.run(cap)
        buf = items.drop(done); pos = 0; fin = f; err = e
      }
    }
    def hasNext: Boolean = { fill(); if (pos < buf.length) true else { if (err != null) { val e = err; err = null; fin = true; throw e }; false } }
    def next(): Any = { val x = buf(pos); pos += 1; done += 1; x }
  }
  case _ => iter_items(v).iterator
}
def listcomp(itv: Any, P: Any, names: Vector[Any], body: Vector[Any] => Any, cond: Vector[Any] => Any): Lst = {
  val out = ArrayBuffer[Any]()
  for (x <- iter_items(itv)) {
    fuel()
    val b = new_binds()
    if (pmatch(P, x, b)) {
      val args = names.map(n => b.getOrElse(n.asInstanceOf[String], UNIT))
      if (cond == null || truthy(cond(args))) out += body(args)
    }
  }
  charge(BigInt(4) + out.length)
  Lst(out.toVector)
}
def catch_(x: Any, h: Any => Any): Any = x match { case e: Err => h(e.e); case _ => x }
def try_(x: Any, els: () => Any): Any = x match {
  case e: Err => throw new Fail(if (els != null) els() else e.e)
  case o: Opt => if (o.x == null) { if (els != null) throw new Fail(els()); trap("try on none") }; o.x
  case _ => x
}
def try_eval(th: () => Any): Any = try th() catch { case _: Flow => null }
def compound(op: String, cur: Any, r: Any): Any = { charge(op_cost(op, cur)); compound_uncharged(op, cur, r) }
def compound_uncharged(op: String, cur: Any, r: Any): Any = {
  val t = cur match { case i: I => i.t; case _ => "lit" }
  arith(op, cur, r) match { case x: I => chk(x.v, t); case x => x }
}
def fixtag(cur: Any, v: Any): Any = (cur, v) match { case (c: I, x: I) if c.t != "lit" => chk(x.v, c.t); case _ => v }
def set_field(rec: Any, m: String, v: Any): Rec = rec match {
  case r: Rec => val i = r.fs.indexWhere(_._1 == m); if (i < 0) trap("no field"); new Rec(r.n, r.fs.updated(i, (m, v)))
  case _ => trap("member assignment on a non-record")
}
def set_tuple(t: Any, i: Int, v: Any): Tup = t match { case x: Tup if i < x.xs.length => Tup(x.xs.updated(i, v)); case _ => trap("bad tuple assignment") }
def set_index(c: Any, k: Any, v: Any): Any = (c, k) match {
  case (l: Lst, i: I) => if (i.v < 0 || i.v >= l.xs.length) trap("index " + i.v + " out of range"); Lst(l.xs.updated(i.v.toInt, v))
  case (m: Mp, _) => Mp(map_set(m.ps, k, v))
  case _ => trap("bad indexed assignment")
}
def loop_bound(v: Any): BigInt = v match { case i: I => i.v; case _ => LOOP_LIMIT }
def arena_bound(v: Any): BigInt = v match { case i: I => i.v; case _ => trap("bad arena bound") }
def check_(v: Any): Unit = if (!truthy(v)) trap("check failed")
def jesc(s: String): Int = { var n = 2; s.codePoints().forEach { c => if ("\"\\\n\r\t".indexOf(c) >= 0) n += 2 else if (c < 0x20) n += 6 else n += new String(Character.toChars(c)).getBytes(UTF_8).length }; n }
def json_size(v: Any): Int = v match {
  case true => 4
  case false => 5
  case x: I => x.v.toString.length
  case x: F => fdebug(x.v).length
  case x: F32 => fdebug(x.v).length
  case s: String => jesc(s)
  case UnitT => 4
  case o: Opt => if (o.x == null) 4 else json_size(o.x)
  case e: Err => json_size(e.e)
  case t: Tup => seqsz(t.xs.map(json_size), t.xs.length)
  case l: Lst => seqsz(l.xs.map(json_size), l.xs.length)
  case m: Mp => seqsz(m.ps.map(p => 3 + json_size(p._1) + json_size(p._2)), m.ps.length)
  case r: Rec => 2 + jesc(r.n) + 1 + seqsz(r.fs.map(p => jesc(p._1) + 1 + json_size(p._2)), r.fs.length)
  case r: Rng => 3 + r.a.toString.length + r.b.toString.length
  case f: FnV => jesc(f.n)
  case _ => 2
}
def seqsz(xs: Vector[Int], n: Int): Int = 2 + xs.sum + math.max(n - 1, 0)
def arena_check(bound: BigInt, vals: Vector[Any]): Unit = if (BigInt(vals.map(json_size).sum) > bound) trap("size limit")
def why(e: Throwable): String = e match {
  case t: Trap => "trap: " + t.m
  case f: Fail => "failed with " + show(f.v)
  case _: Stop => "Stop"
  case _ => "Ret"
}
def dbgV(v: Any): String = v match {
  case b: Boolean => "Bool(" + b + ")"
  case s: String => "Text(" + rdebug(s) + ")"
  case x: I => "Int(" + x.v + ", " + rdebug(x.t) + ")"
  case x: F => "F(" + fdebug_full(x.v) + ")"
  case x: F32 => "F32(" + fdebug_full(x.v) + ")"
  case UnitT => "Unit"
  case t: Tup => "Tuple([" + t.xs.map(dbgV).mkString(", ") + "])"
  case l: Lst => "List([" + l.xs.map(dbgV).mkString(", ") + "])"
  case m: Mp => "Map([" + m.ps.map(p => "(" + dbgV(p._1) + ", " + dbgV(p._2) + ")").mkString(", ") + "])"
  case o: Opt => if (o.x == null) "Opt(None)" else "Opt(Some(" + dbgV(o.x) + "))"
  case r: Rec => "Rec(" + rdebug(r.n) + ", [" + r.fs.map(p => "(" + rdebug(p._1) + ", " + dbgV(p._2) + ")").mkString(", ") + "])"
  case e: Err => "Err(" + dbgV(e.e) + ")"
  case f: FnV => "Fn(" + rdebug(f.n) + ")"
  case r: Rng => "Range(" + r.a + ", " + r.b + ", " + rdebug(r.t) + ")"
  case c: Chan => "Chan(" + rdebug(c.id) + ", " + c.cap + ")"
  case t: Task => "Task(" + rdebug(t.id) + ")"
  case a: Actor => "Actor(" + rdebug(a.id) + ", " + rdebug(a.ty) + ")"
  case s: Session => "Session(" + rdebug(s.out) + ", " + rdebug(s.inn) + ", " + s.cap + ")"
  case _: Clo => "Closure(..)"
  case _: Gen => "Gen(..)"
  case _ => "LVar(..)"
}
def fdebug_full(x: Double): String = if (x.isNaN) "NaN" else if (x == Double.PositiveInfinity) "inf" else if (x == Double.NegativeInfinity) "-inf" else fdebug(x)
def dbg_flow(e: Throwable): String = e match {
  case r: Ret => "Ret(" + dbgV(r.v) + ")"
  case f: Fail => "Fail(" + dbgV(f.v) + ")"
  case t: Trap => "Trap(" + rdebug(t.m) + ")"
  case _: Stop => "Stop"
  case c: Cancel => "Cancel(" + c.f + ")"
  case _ => "?"
}
def refusal(e: Throwable): String = e match {
  case f: Fail => f.v match { case r: Rec => r.n; case _ => "?" }
  case t: Trap => "trap(" + t.m + ")"
  case _ => "trap(" + dbg_flow(e) + ")"
}
def actor_refusal(e: Throwable): String = refusal(e)
val OUT = ArrayBuffer[Any]()
def emit(v: Any): Unit = { if (TASK != null) { comm(V("emit", v)); return }; OUT += v }
val LINES = ArrayBuffer[String]()
def say(s: String): Unit = LINES += s
def obs(label: String, th: () => Any): Unit = {
  fuel_reset()
  try say(label + " = " + show(th())) catch { case e: Flow => say(label + " " + why(e)) }
  fuel_report()
}
def show_state(s: Vector[Any]): String = s.map { r => at(r, 0).toString + " { " + vec(at(r, 1)).map(f => at(f, 0).toString + " = " + show(at(f, 1))).mkString(", ") + " }" }.mkString("; ")
def delta(s: Vector[Any], ev: Any): (Vector[Any], Vector[Any]) = {
  val nxt = ArrayBuffer[Any](); val outs = ArrayBuffer[Any]()
  for (reg <- s) {
    val name = at(reg, 0).asInstanceOf[String]; val fs = vec(at(reg, 1))
    var got: (Any, Vector[Any]) = null
    val it = STEPS.getOrElse(name, Vector()).iterator
    while (got == null && it.hasNext) {
      val st = it.next()
      OUT.clear()
      val r = try st(fs, ev) catch { case e: Flow => throw new Refuse(refusal(e)) }
      if (r != null) got = (r, OUT.toVector)
    }
    if (got == null) nxt += V(name, fs) else { nxt += V(name, got._1); outs ++= got._2 }
  }
  (nxt.toVector, outs.toVector)
}
def run_example(line: Int, events: Vector[Any]): Unit = {
  fuel_reset()
  try run_example1(line, events) finally fuel_report()
}
def run_example1(line: Int, events: Vector[Any]): Unit = {
  say("example " + line)
  var s: Vector[Any] = null
  try s = initial_state() catch { case e: Flow => say("initial " + why(e)); return }
  say("0 (initial) " + show_state(s))
  for (k <- 1 to events.length) {
    val name = at(events(k - 1), 0).toString; val th = at(events(k - 1), 1).asInstanceOf[() => Any]
    var ev: Any = null
    try ev = th() catch { case e: Flow => say(s"$k $name " + why(e)); return }
    if (ev == null) { say(s"$k $name not an event"); return }
    var r: (Vector[Any], Vector[Any]) = null
    try r = delta(s, ev) catch { case e: Refuse => say(s"$k " + show(ev) + " refused: " + e.r); return }
    s = r._1
    say(s"$k " + show(ev) + " " + show_state(s))
    for (o <- r._2) say("  emit " + show(o))
  }
}
def run_main(f: (Any, Vector[Any]) => Vector[Any]): Unit = {
  fuel_reset()
  try run_main1(f) finally fuel_report()
}
def run_main1(f: (Any, Vector[Any]) => Vector[Any]): Unit = {
  say("main")
  OUT.clear()
  try {
    fuel()
    val v = f(null, Vector())(0)
    for (o <- OUT) say("emit " + show(o))
    say("main = " + show(v))
  } catch { case e: Flow =>
    for (o <- OUT) say("emit " + show(o))
    say("main " + why(e))
  }
}

// ---- draft 0.6 (C.5.2, C.11.2)
def par_ident(op: String, b: Any): Any = {
  b match {
    case x: I =>
      if (op == "+" || op == "|" || op == "^") return I(0, x.t)
      if (op == "*") return I(1, x.t)
      if (op == "&") { val (lo, hi) = irange(if (x.t == "lit") "i64" else x.t); return I(if (lo == 0) hi else -1, x.t) }
    case _: F => if (op == "+") return F(-0.0); if (op == "*") return F(1.0)
    case _: F32 => if (op == "+") return new F32(-0.0); if (op == "*") return new F32(1.0)
    case _: Boolean => if (op == "&") return true; if (op == "|" || op == "^") return false
    case _ =>
  }
  trap("no identity for " + op + "= on " + show(b))
}
def par_combine(op: String, a: Any, b: Any): Any = {
  charge(op_cost(op, a))
  (a, b) match {
    case (x: Boolean, y: Boolean) if op == "&" => x && y
    case (x: Boolean, y: Boolean) if op == "|" => x || y
    case (x: Boolean, y: Boolean) if op == "^" => x != y
    case _ => compound_uncharged(op, a, b)
  }
}
def par_tree(op: String, c0: Any, i: Any): Any = {
  val c = vec(c0)
  if (c.isEmpty) return i
  if (c.length == 1) return c(0)
  val h = (c.length + 1) / 2
  val l = par_tree(op, c.take(h), i)
  val r = par_tree(op, c.drop(h), i)
  par_combine(op, l, r)
}
def pscan(xs: Vector[Any], f: Any, e: Any, inclusive: Boolean): Lst = {
  val n = xs.length
  if (n == 0) return Lst(Vector())
  val sums = mutable.HashMap[(Int, Int), Any]()
  def up(lo: Int, hi: Int): Any = {
    val v = if (hi - lo == 1) xs(lo) else { val h = lo + (hi - lo + 1) / 2; val l = up(lo, h); val r = up(h, hi); apply_(f, Vector(l, r)) }
    sums((lo, hi)) = v
    v
  }
  val out = Array.fill[Any](n)(null)
  def down(lo: Int, hi: Int, off: Any): Unit = {
    if (hi - lo == 1) { out(lo) = off; return }
    val h = lo + (hi - lo + 1) / 2
    val ls = sums((lo, h))
    val roff = if (off == null) ls else apply_(f, Vector(off, ls))
    down(lo, h, off)
    down(h, hi, roff)
  }
  up(0, n)
  down(0, n, null)
  Lst(out.indices.map(i => if (inclusive) (if (out(i) == null) xs(i) else apply_(f, Vector(out(i), xs(i)))) else (if (out(i) == null) e else out(i))).toVector)
}
// ---- cost accounting for `with budget` (C.6.1, C.6.4), as interp `charge`
val BUD = ArrayBuffer[Array[BigInt]]()
def charge(u0: BigInt): Unit = {
  if (BUD.isEmpty) return
  for (b <- BUD) if (b(0) + u0 > b(1)) trap("budget")
  for (b <- BUD) b(0) += u0
}
def charge(u: Int): Unit = if (BUD.nonEmpty) charge(BigInt(u))
def budget_enter(n: Any): Unit = BUD += Array(BigInt(0), marker_int(n))
var BUD_LAST: BigInt = BigInt(0)
def budget_exit(): Unit = { BUD_LAST = BUD.remove(BUD.length - 1)(0) }
def budget_last(): Any = new I(BUD_LAST, "u64")
def op_cost(op: String, x: Any): BigInt = x match {
  case _: F | _: F32 => if (op == "+" || op == "-" || op == "*") 2 else if (op == "/" || op == "%") 8 else 1
  case _ => 1
}
def cindex(c: Any, i: Any): Any = { charge(1); index(c, i) }
def cfield(v: Any, m: String): Any = { charge(1); field(v, m) }
def ctmember(v: Any, i: Int): Any = { charge(1); tmember(v, i) }
def cneg(v: Any): Any = { charge(1); neg(v) }
def cbitnot(v: Any): Any = { charge(1); bitnot(v) }
def cnot(v: Any): Any = { charge(1); not_(v) }
def cbinop(op: String, x: Any, y: Any): Any = { charge(op_cost(op, x)); binop(op, x, y) }
def pre_method(recv: Any, m: String): Any = { val h = head_of(recv); if (h != null && METHODS.contains(h + "." + m)) charge(4); recv }
def pre_value(f: Any): Any = { if (f.isInstanceOf[FnV]) charge(4); f }
def marker_int(v: Any): BigInt = v match { case i: I => i.v; case _ => trap("expected an integer, got " + show(v)) }

// ---- LVars (C.11.8) and the built-in Merge instances (C.11.5)
def lvar_new(a: Vector[Any]): LVar = new LVar(if (a.nonEmpty) a(0) else UNIT, if (a.length > 1) a(1) else null)
def crank(v: Any): Int = v match {
  case UnitT => 0
  case o: Opt if o.x == null => 0
  case _: Boolean => 1
  case _: I | _: F | _: F32 => 2
  case _: String => 3
  case _: Lst | _: Tup => 4
  case _ => 5
}
def bcmp(a: String, b: String): Int = {
  val x = u8s(a); val y = u8s(b); var i = 0
  while (i < math.min(x.length, y.length)) { if (x(i) != y(i)) return c3(x(i) & 0xff, y(i) & 0xff); i += 1 }
  c3(x.length, y.length)
}
def numv(v: Any): Double = v match { case f: F => f.v; case f: F32 => f.v; case i: I => i.v.toDouble; case _ => 0.0 }
def canon_cmp(a: Any, b: Any): Int = {
  val ra = crank(a); val rb = crank(b)
  if (ra != rb) return c3(ra, rb)
  (a, b) match {
    case (x: Boolean, y: Boolean) => return c3(if (x) 1 else 0, if (y) 1 else 0)
    case (x: I, y: I) => return c3(x.v, y.v)
    case _ =>
  }
  if (ra == 2) return c3(numv(a), numv(b))
  if (ra == 3) return bcmp(a.asInstanceOf[String], b.asInstanceOf[String])
  if (ra == 4) {
    val xs = a match { case l: Lst => l.xs; case t: Tup => t.xs; case _ => Vector() }
    val ys = b match { case l: Lst => l.xs; case t: Tup => t.xs; case _ => Vector() }
    var i = 0
    while (i < math.min(xs.length, ys.length)) { val c = canon_cmp(xs(i), ys(i)); if (c != 0) return c; i += 1 }
    return c3(xs.length, ys.length)
  }
  (a, b) match {
    case (x: Opt, y: Opt) if x.x != null && y.x != null => return canon_cmp(x.x, y.x)
    case (x: Rec, y: Rec) =>
      if (x.n != y.n) return bcmp(x.n, y.n)
      var i = 0
      while (i < math.min(x.fs.length, y.fs.length)) { var c = bcmp(x.fs(i)._1, y.fs(i)._1); if (c == 0) c = canon_cmp(x.fs(i)._2, y.fs(i)._2); if (c != 0) return c; i += 1 }
      return c3(x.fs.length, y.fs.length)
    case _ =>
  }
  bcmp(show(a), show(b))
}
val canonOrd: Ordering[Any] = (x: Any, y: Any) => canon_cmp(x, y)
def merge_builtin(a: Any, b: Any): Any = (a, b) match {
  case (x: I, y: I) => I(if (x.v > y.v) x.v else y.v, if (x.t == "lit") y.t else x.t)
  case (x: Boolean, y: Boolean) => x || y
  case (x: Lst, y: Lst) =>
    val v = (x.xs ++ y.xs).sorted(canonOrd); val out = ArrayBuffer[Any]()
    for (e <- v) if (out.isEmpty || canon_cmp(out.last, e) != 0) out += e
    Lst(out.toVector)
  case (x: Mp, y: Mp) =>
    val out = ArrayBuffer[(Any, Any)](x.ps*)
    for ((k, v) <- y.ps) { val i = out.indexWhere(p => veq(p._1, k)); if (i >= 0) out(i) = (out(i)._1, merge_builtin(out(i)._2, v)) else out += ((k, v)) }
    Mp(out.toVector.sortWith((p, q) => canon_cmp(p._1, q._1) < 0))
  case (x: Tup, y: Tup) if x.xs.length == y.xs.length => Tup(x.xs.indices.map(i => merge_builtin(x.xs(i), y.xs(i))).toVector)
  case (x: Rec, y: Rec) if x.n == y.n && x.fs.length == y.fs.length => new Rec(x.n, x.fs.indices.map(i => (x.fs(i)._1, merge_builtin(x.fs(i)._2, y.fs(i)._2))).toVector)
  case (x: Opt, _) if x.x == null => b
  case (_, y: Opt) if y.x == null => a
  case (x: Opt, y: Opt) => new Opt(merge_builtin(x.x, y.x))
  case _ => trap("no Merge instance for " + show(a) + " and " + show(b))
}
def merge_values(a: Any, b: Any): Any = {
  a match {
    case r: Rec => val h = KIND_OF.getOrElse(r.n, r.n); METHODS.get(h + ".merge") match { case Some(f) => charge(4); return f(a, Vector(b))(0); case None => }
    case _ =>
  }
  merge_builtin(a, b)
}
var PUTS = 0
def puts(): Int = PUTS
def lvar_put(c: LVar, v: Any): Unit = {
  val nv = merge_values(c.v, v)
  if (c.top != null && veq(nv, c.top)) trap("lvar conflict")
  if (!veq(nv, c.v)) { if (c.frz) trap("lvar put after freeze"); PUTS += 1 }
  c.v = nv
}
// merge for a threshold test: the program's Merge, charged nothing (interp `merge_free`)
def merge_free(a: Any, b: Any): Any = {
  val bud = bud_snap(); val f = FUEL
  BUD.clear(); FUEL = FUEL_LIMIT.toLong
  try merge_values(a, b) finally { FUEL = f; bud_restore(bud) }
}
def lvar_incomp(c: LVar, ts: Vector[Any]): Unit =
  for (i <- ts.indices; j <- (i + 1) until ts.length)
    if (!veq(ts(i), ts(j)) && (c.top == null || !veq(merge_free(ts(i), ts(j)), c.top))) trap("lvar thresholds are not pairwise incompatible")
var SCHED = 0
def lvar_wait(c: LVar, q: Any): Any = {
  val ts = q match { case l: Lst => l.xs; case _ => trap("wait needs a list of thresholds") }
  lvar_incomp(c, ts.toVector)
  for (t <- ts) if (veq(merge_free(t, c.v), c.v)) return t
  if (SCHED > 0) trap("$block")
  trap("lvar wait cannot be satisfied")
}
def par_stuck(pc: Int): Boolean = PUTS == pc
def par_enter(): Unit = SCHED += 1
def par_exit(): Unit = SCHED -= 1
def bud_snap(): Vector[Array[BigInt]] = BUD.map(b => b.clone()).toVector
def bud_restore(s: Any): Unit = { BUD.clear(); for (b <- s.asInstanceOf[Vector[Array[BigInt]]]) BUD += b }
// a restarted par attempt (C.14) restores the budgets and the fuel
def par_snap(): (Any, Long) = (bud_snap(), FUEL)
def par_restore(s: (Any, Long)): Unit = { bud_restore(s._1); FUEL = s._2 }
// ---- workgroups (C.11.2)
def lanes_of(v: Any): (Int, BigInt) = { val n = marker_int(v); (if (n > 0) n.toInt else 0, n) }
def group_rec(lane: Int, n: BigInt): Rec = new Rec("$group", Vector(("lane", I(BigInt(lane), "usize")), ("lanes", I(n, "usize"))))

// ---- task statements (6.8, chapter C). Outside `--run` there is no task
// context, so every communication step traps exactly as in the reference
// evaluator's comptime mode.
var TASK: TaskCtx = null
var ATOMIC: ArrayBuffer[Any] = null
var HSENDS: ArrayBuffer[Any] = null
def comm(op: Vector[Any]): Vector[Any] = {
  if (TASK == null) trap("concurrency is not available at comptime")
  TASK.comm(op)
}
def fresh_(kind: String, line: Int): String = { val idx = TASK.created; TASK.created += 1; hid(kind + "|" + line + "|" + TASK.id + "|" + idx) }
def hid(key: String): String = sha256hex(key.getBytes(UTF_8)).substring(0, 16)
def chan_new(cap: () => Any, line: Int): Any = {
  if (TASK == null) trap("channels are not available at comptime")
  new Chan(fresh_("chan", line), marker_int(cap()))
}
def session_new(cap: () => Any, line: Int): Any = {
  if (TASK == null) trap("channels are not available at comptime")
  val n = marker_int(cap())
  val a = fresh_("chan", line); val b = fresh_("chan", line)
  Tup(Vector(new Session(a, b, n), new Session(b, a, n)))
}
def actor_new(ty: String, line: Int): Any = {
  if (TASK == null) trap("cannot call " + ty + " at comptime")
  new Actor(fresh_("actor", line), ty)
}
def send(t0: Any, v: Any): Unit = {
  charge(8)
  val t = t0 match { case s: Session => new Chan(s.out, s.cap); case x => x }
  if (ATOMIC != null) { ATOMIC += V(t, v); return }
  if (HSENDS != null) { HSENDS += V(t, v); return }
  comm(V("send", t, v))
}
def recv(c0: Any): Any = {
  charge(8)
  val session = c0.isInstanceOf[Session]
  val c = c0 match { case s: Session => new Chan(s.inn, s.cap); case x => x }
  val rp = comm(V("recv", c))
  if (rp(0) != "val") trap("bad receive")
  val v = rp(1)
  if (session) { v match { case o: Opt if o.x != null => return o.x; case _ => trap("session closed") } }
  v
}
def session_choose(r: Any, label: String): Any = r match {
  case s: Session => charge(8); comm(V("send", new Chan(s.out, s.cap), new Rec(label, Vector()))); UNIT
  case _ => trap(show(r) + " has no method choose at comptime")
}
def spawn_(f: String, args: Vector[Any], prio: Any, site: Int): Any = {
  charge(32)
  if (TASK == null) trap("concurrency is not available at comptime")
  val idx = TASK.created; TASK.created += 1
  var p = asBig(prio); if (p == null) p = BigInt(0)
  p = if (p < 0) 0 else if (p > 255) 255 else p
  val rp = comm(V("spawn", f, args, p.toInt, site, idx, if (TASK.scopes.nonEmpty) TASK.scopes.last else 0))
  new Task(rp(1).asInstanceOf[String])
}
def select_(arms: Vector[Any], line: Int, nondet: Boolean): Vector[Any] = {
  var k = 0
  if (TASK != null) { k = TASK.selK.getOrElse(line, 0); TASK.selK(line) = k + 1 }
  val a2 = arms.map { a0 => val a = vec(a0)
    (a(0), a(1)) match {
      case ("recv", s: Session) => V("recv", new Chan(s.inn, s.cap), a(2))
      case ("send", s: Session) => V("send", new Chan(s.out, s.cap), a(2))
      case _ => a
    }
  }
  val rp = comm(V("select", a2, V(line, k), nondet))
  V(rp(1), if (rp(2) == null) NONE else rp(2))
}
def sarm_after(n: Any): Vector[Any] = { val x = marker_int(n); V("after", (if (x < 0) BigInt(0) else x).toInt) }
def dl_enter(n: Any): Any = {
  if (TASK == null) return -2
  val f = TASK.nextFrame; TASK.nextFrame += 1
  val nn = asBig(n) match { case null => marker_int(n); case b => b }
  TASK.dl += V(f, TASK.now + (if (nn < 0) 0L else nn.toLong))
  f
}
def dl_exit(): Unit = if (TASK != null) TASK.dl.remove(TASK.dl.length - 1)
def scope_enter(): Int = { if (TASK == null) return 0; val f = TASK.nextFrame; TASK.nextFrame += 1; TASK.scopes += f; f }
def scope_exit(): Unit = if (TASK != null) TASK.scopes.remove(TASK.scopes.length - 1)
def await_scope(sid: Int): Unit = if (TASK != null) comm(V("await", sid))
def atomic_enter(): ArrayBuffer[Any] = { val o = ATOMIC; ATOMIC = ArrayBuffer[Any](); o }
def atomic_restore(o: ArrayBuffer[Any]): ArrayBuffer[Any] = { val b = if (ATOMIC != null) ATOMIC else ArrayBuffer[Any](); ATOMIC = o; b }
def atomic_commit(buf: ArrayBuffer[Any]): Unit = {
  if (ATOMIC != null) { ATOMIC ++= buf; return }
  if (TASK != null) { comm(V("atomic", buf.toVector)); return }
  trap("concurrency is not available at comptime")
}
final class TaskCtx(val id: String) {
  var created = 0
  var now = 0L
  val dl = ArrayBuffer[Any]()
  var nextFrame = 1
  val scopes = ArrayBuffer[Int](0)
  val selK = mutable.HashMap[Int, Int]()
  var link: Vector[Any] => Vector[Any] = null
  def comm(op: Vector[Any]): Vector[Any] = {
    val rp0 = link(V(op, dl.toVector))
    now = rp0(0).asInstanceOf[Long]
    val rep = vec(rp0(1))
    if (rep(0) == "cancel") throw new Cancel(rep(1))
    if (rep(0) == "failed") throw new Fail(rep(1))
    rep
  }
}
def discard(x: Any): Unit = ()
def big_stack(f: () => Unit): Unit = {
  var err: Throwable = null
  val t = new Thread(null, () => { try f() catch { case e: Throwable => err = e } }, "rce-main", 1L << 31)
  t.start(); t.join()
  if (err != null) throw err
}
// The tick scheduler of `--run` (rt.rs `run_tasks`) is not ported to this target yet.
def run_tasks(entry: String): Unit = { System.err.println("--run is not available on the scala target"); sys.exit(2) }
def atOr(x: Any, i: Int, d: Any): Any = x match {
  case v: Vector[?] => if (i < v.length) v(i) else d
  case _ => d
}
