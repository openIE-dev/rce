"""An asyncio service around an emitted Python module.

Events arrive as newline-delimited JSON on a TCP connection (asyncio
streams, the standard library's event source). One admitter coroutine owns
the state and applies each event in arrival order; every reply is one JSON
line with the outcome, the lines `rce run` would print for that event and
the canonical state.

  python3 service.py MODULE.py STREAM.json    serve on 127.0.0.1, send the
                                              stream as a client, print the
                                              transcript (the test)
"""
import asyncio
import json
import sys

import rce_host


class Service:
    def __init__(self, program):
        self.p = program
        self.k = 0
        self.inbox = asyncio.Queue()

    async def admitter(self):
        # the single writer: one event at a time, in arrival order
        while True:
            line, reply = await self.inbox.get()
            try:
                ev = self.p.event(rce_host.decode(line))
            except (rce_host.BadEvent, ValueError) as e:
                reply.set_result({"outcome": "not-admitted", "why": str(e)})
                continue
            self.k += 1
            out = self.p.step(ev)
            reply.set_result({
                "seq": self.k,
                "outcome": out[0] if out[0] == "applied" else "kept: " + out[1],
                "lines": self.p.lines(self.k, ev, out),
                "state": json.loads(self.p.state_json()),
            })

    async def handle(self, reader, writer):
        writer.write((json.dumps({"lines": ["0 (initial) " + self.p.show_state()]}) + "\n").encode())
        await writer.drain()
        while line := await reader.readline():
            reply = asyncio.get_running_loop().create_future()
            await self.inbox.put((line.decode(), reply))
            writer.write((json.dumps(await reply, ensure_ascii=False) + "\n").encode())
            await writer.drain()
        writer.close()


async def drive(module_path, stream):
    svc = Service(rce_host.Program(rce_host.load(module_path)))
    task = asyncio.create_task(svc.admitter())
    server = await asyncio.start_server(svc.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    out = json.loads(await reader.readline())["lines"]
    for ev in stream:
        writer.write((json.dumps(ev) + "\n").encode())
        await writer.drain()
        out += json.loads(await reader.readline())["lines"]
    writer.close()
    server.close()
    task.cancel()
    return out


if __name__ == "__main__":
    lines = asyncio.run(drive(sys.argv[1], json.load(open(sys.argv[2]))))
    sys.stdout.write("".join(x + "\n" for x in lines))
