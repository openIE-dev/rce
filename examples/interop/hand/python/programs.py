"""The eight programs written by hand as idiomatic Python, for comparison
with what `rce emit python` prints. Each is a state dataclass and a step
function `step(state, event) -> (state, emits)` that raises Kept when the
event does not apply. They are checked on the streams of streams.json only:
unlike the emitted code they do not trap on integer overflow, have no fuel,
and display only the values these programs hold."""
import math
from dataclasses import dataclass, field, replace
from enum import Enum


class Kept(Exception):
    pass


# ---- counter (01_counter.rce)
@dataclass(frozen=True)
class Counter:
    n: int = 0


def counter_step(s, ev):
    match ev["type"]:
        case "Inc":
            return replace(s, n=s.n + 1), []
        case "Dec":
            if s.n == 0:
                raise Kept("Underflow")
            return replace(s, n=s.n - 1), []


# ---- bank (04_bank_account.rce)
@dataclass(frozen=True)
class Account:
    balance: int = 0


def bank_step(s, ev):
    a = ev["amount"]
    match ev["type"]:
        case "Deposit":
            return (replace(s, balance=s.balance + a) if a > 0 else s), []
        case "Withdraw":
            if a > s.balance:
                raise Kept("Insufficient")
            return replace(s, balance=s.balance - a), [("Withdrawn", {"amount": a})]


# ---- multi_region (16_multi_region.rce)
@dataclass(frozen=True)
class Cart:
    items: tuple = ()


@dataclass(frozen=True)
class Stats:
    adds: int = 0
    removes: int = 0


def multi_region_step(s, ev):
    cart, stats = s
    match ev["type"]:
        case "AddItem":
            return (Cart(cart.items + (ev["name"],)), replace(stats, adds=stats.adds + 1)), []
        case "RemoveItem":
            return (Cart(tuple(i for i in cart.items if i != ev["name"])), replace(stats, removes=stats.removes + 1)), []
        case "Reset":
            return (Cart(), stats), []


# ---- traffic_light (17_traffic_light.rce)
class Light(Enum):
    Red = 0
    Green = 1
    Yellow = 2


@dataclass(frozen=True)
class Signal:
    light: Light = Light.Red


def traffic_light_step(s, ev):
    return Signal(Light((s.light.value + 1) % 3)), []


# ---- door (30_state_machine_guarded.rce)
@dataclass(frozen=True)
class Door:
    open: bool = False
    locked: bool = False


def door_step(s, ev):
    match ev["type"]:
        case "Open":
            if s.locked:
                raise Kept("Locked")
            return replace(s, open=True), []
        case "Close":
            return replace(s, open=False), []
        case "Lock":
            if s.open:
                raise Kept("DoorOpen")
            return replace(s, locked=True), []
        case "Unlock":
            return replace(s, locked=False), []


# ---- inventory (31_inventory.rce)
@dataclass(frozen=True)
class Inventory:
    stock: dict = field(default_factory=dict)


def inventory_step(s, ev):
    sku, qty = ev["sku"], ev["qty"]
    have = s.stock.get(sku, 0)
    match ev["type"]:
        case "Receive":
            return Inventory({**s.stock, sku: have + qty}), []
        case "Ship":
            if have < qty:
                raise Kept("OutOfStock")
            return Inventory({**s.stock, sku: have - qty}), [("Shipped", {"sku": sku, "qty": qty})]


# ---- rcedb (55_rcedb.rce)
@dataclass(frozen=True)
class Db:
    rows: dict = field(default_factory=dict)
    writes: int = 0
    durable: int = 0


def rcedb_step(s, ev):
    match ev["type"]:
        case "Put":
            return replace(s, rows={**s.rows, ev["key"]: ev["val"]}, writes=s.writes + 1), []
        case "Delete":
            if ev["key"] not in s.rows:
                raise Kept("Missing")
            rows = dict(s.rows)
            del rows[ev["key"]]
            return replace(s, rows=rows, writes=s.writes + 1), []
        case "Synced":
            return (replace(s, durable=ev["upto"]) if ev["upto"] > s.durable else s), []
        case "Scan":
            return s, [("Row", {"key": k, "val": v}) for k, v in sorted(s.rows.items()) if k >= ev["start"]]


# ---- float_keys (58_float_keys.rce)
def fkey(x):
    """-0.0 and 0.0 are two keys (IEEE 754 totalOrder puts -0.0 first)."""
    return (x, math.copysign(1.0, x))


@dataclass(frozen=True)
class Tally:
    m: dict = field(default_factory=dict)  # fkey(x) -> count


def float_keys_step(s, ev):
    x = ev["x"] if ev["type"] == "Add" else ev["x"] * -1.0
    k = fkey(x)
    return Tally({**s.m, k: s.m.get(k, 0) + 1}), []


PROGRAMS = {
    "counter": (lambda: [Counter()], lambda s, e: counter_step(s[0], e)),
    "bank": (lambda: [Account()], lambda s, e: bank_step(s[0], e)),
    "multi_region": (lambda: [Cart(), Stats()], lambda s, e: multi_region_step(tuple(s), e)),
    "traffic_light": (lambda: [Signal()], lambda s, e: traffic_light_step(s[0], e)),
    "door": (lambda: [Door()], lambda s, e: door_step(s[0], e)),
    "inventory": (lambda: [Inventory()], lambda s, e: inventory_step(s[0], e)),
    "rcedb": (lambda: [Db()], lambda s, e: rcedb_step(s[0], e)),
    "float_keys": (lambda: [Tally()], lambda s, e: float_keys_step(s[0], e)),
}
