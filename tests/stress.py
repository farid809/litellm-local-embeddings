"""Load test for POST /embeddings.  python stress.py <base-url> [label]"""
import asyncio
import json
import os
import statistics
import sys
import time

import httpx

BASE = sys.argv[1].rstrip("/")
LABEL = sys.argv[2] if len(sys.argv) > 2 else BASE
KEY = os.environ.get("LITELLM_KEY", "sk-test-master")
SECONDS = float(os.environ.get("SECONDS_PER_RUN", "15"))
# ~130 tokens, about one paragraph of a document chunk
TEXT = ("The batch job reads each customer record from the input file, validates the account number, "
        "computes the monthly interest on the outstanding balance, and writes the updated record to the "
        "output buffer. Records that fail validation are written to an error file with a reason code. ") * 3
SCENARIOS = [(1, 1), (8, 1), (32, 1), (64, 1), (1, 32), (8, 32)]  # (concurrent clients, texts per request)


async def run(client, concurrency, batch):
    lat, errors, stop = [], 0, time.perf_counter() + SECONDS
    body = {"model": "embed-local", "input": [f"{i} {TEXT}" for i in range(batch)] if batch > 1 else TEXT}

    async def worker():
        nonlocal errors
        while time.perf_counter() < stop:
            t = time.perf_counter()
            try:
                r = await client.post(f"{BASE}/embeddings", json=body)
                ok = r.status_code == 200 and len(r.json()["data"]) == batch
            except Exception:
                ok = False
            if ok:
                lat.append(time.perf_counter() - t)
            else:
                errors += 1

    start = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(concurrency)])
    dt = time.perf_counter() - start
    q = statistics.quantiles(lat, n=100) if len(lat) > 1 else [0] * 99
    return {"label": LABEL, "clients": concurrency, "batch": batch, "requests": len(lat), "errors": errors,
            "req_s": round(len(lat) / dt, 1), "texts_s": round(len(lat) * batch / dt, 1),
            "p50_ms": round(q[49] * 1000, 1), "p95_ms": round(q[94] * 1000, 1), "p99_ms": round(q[98] * 1000, 1)}


async def main():
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {KEY}"}, timeout=120,
                                 limits=httpx.Limits(max_connections=200)) as client:
        await run(client, 4, 1)  # warm-up, discarded
        for c, b in SCENARIOS:
            print(json.dumps(await run(client, c, b)), flush=True)

asyncio.run(main())
