"""Host-side adapter for a module printed by `rce emit python FILE`.

The emitted module is the step function of the program (spec 10.5): it
defines `initial_state()` and `delta(state, event)`, and the host calls
them (Appendix E: the host contract is met around delta). This adapter is
the boundary: JSON (or plain Python values) in, emitted values to the step,
display text and canonical JSON out. It uses the standard library only.
"""
import importlib.util
import json
import operator
import sys

INT_TYPES = ("i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "usize")


def load(path, name="rce_program"):
    """Import an emitted module from its file. Importing runs nothing."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class BadEvent(ValueError):
    """The input is not an event of the program (the host does not admit it)."""


class Program:
    """One instance of the program: its state and the step that moves it."""

    def __init__(self, module):
        self.m = module
        self.m.fuel_reset()
        self.state = self.m.initial_state()

    # ---- the boundary, inward
    def event(self, obj):
        """An event of the program from a dict {"type": NAME, field: value, ...}."""
        m = self.m
        name = obj.get("type")
        if name not in m.EVENT_FIELDS:
            raise BadEvent("no event %r" % (name,))
        fields = []
        for f, t in m.EVENT_FIELDS[name]:
            if f not in obj:
                raise BadEvent("missing field " + f)
            fields.append((f, self.value(t, obj[f], f)))
        return m.Rec(name, fields)

    def value(self, t, x, f):
        m = self.m
        if t == "text":
            if not isinstance(x, str):
                raise BadEvent("field %s is text" % f)
            return x
        if t == "bool":
            if not isinstance(x, bool):
                raise BadEvent("field %s is bool" % f)
            return x
        if t in INT_TYPES:
            if isinstance(x, bool):
                raise BadEvent("field %s is an integer" % f)
            try:
                n = operator.index(x)  # int, and numpy's integer scalars
            except TypeError:
                raise BadEvent("field %s is an integer" % f) from None
            try:
                return m.chk(n, t)
            except m.Trap as e:
                raise BadEvent("field %s: %s" % (f, e.m)) from None
        if t == "f64":
            if isinstance(x, bool) or not (isinstance(x, (int, float)) or hasattr(x, "__float__")):
                raise BadEvent("field %s is f64" % f)
            return m.F(float(x))
        raise BadEvent("field %s has type %s, which this adapter does not read" % (f, t))

    # ---- the step
    def step(self, ev):
        """Apply one event. Returns ("applied", emits) or ("kept", reason);
        a kept step leaves the state as it was (HC-21)."""
        m = self.m
        m.fuel_reset()  # HC-7a: full fuel for each input
        try:
            s2, outs = m.delta(self.state, ev)
        except m.Kept as k:
            return "kept", k.r
        self.state = s2
        return "applied", outs

    # ---- the boundary, outward
    def show(self, v):
        return self.m.show(v)

    def show_state(self):
        return self.m.show_state(self.state)

    def state_json(self):
        """The state as canonical JSON: {region: {field: value}}, keys sorted."""
        m = self.m
        return m.fv_json(m.FRec([(r, m.FRec([(n, m.form_value(v)) for n, v in fs])) for r, fs in self.state]))

    def lines(self, k, ev, outcome):
        """The lines `rce run` prints for event number k."""
        kind, x = outcome
        if kind == "kept":
            return ["%d %s kept: %s" % (k, self.show(ev), x)]
        return ["%d %s %s" % (k, self.show(ev), self.show_state())] + ["  emit " + self.show(o) for o in x]


def decode(text):
    """An event dict from JSON text."""
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise BadEvent("an event is a JSON object")
    return obj
