"""A FastAPI service around an emitted Python module (needs fastapi and httpx).

POST /events takes one event as a JSON object and answers with its outcome,
the lines `rce run` prints for it and the canonical state; GET /initial
answers with the first line. A lock makes the step the single writer, since
FastAPI runs plain `def` endpoints on a thread pool.

  python3 fastapi_app.py MODULE.py STREAM.json   drive the app with its test
                                                 client and print the transcript
"""
import json
import sys
import threading

from fastapi import Body, FastAPI, HTTPException

import rce_host


def make_app(module_path):
    p = rce_host.Program(rce_host.load(module_path))
    lock = threading.Lock()
    seq = [0]
    app = FastAPI()

    @app.get("/initial")
    def initial():
        return {"lines": ["0 (initial) " + p.show_state()]}

    @app.post("/events")
    def post_event(obj: dict = Body(...)):
        with lock:
            try:
                ev = p.event(obj)
            except rce_host.BadEvent as e:
                raise HTTPException(status_code=422, detail=str(e))
            seq[0] += 1
            out = p.step(ev)
            return {"seq": seq[0], "outcome": out[0] if out[0] == "applied" else "kept: " + out[1],
                    "lines": p.lines(seq[0], ev, out), "state": json.loads(p.state_json())}

    return app


if __name__ == "__main__":
    from fastapi.testclient import TestClient

    c = TestClient(make_app(sys.argv[1]))
    lines = c.get("/initial").json()["lines"]
    for ev in json.load(open(sys.argv[2])):
        r = c.post("/events", json=ev)
        r.raise_for_status()
        lines += r.json()["lines"]
    sys.stdout.write("".join(x + "\n" for x in lines))
