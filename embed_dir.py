"""Embed every text file under a directory and time it.  python embed_dir.py <base-url> <dir> [label]"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

BASE, ROOT = sys.argv[1].rstrip("/"), Path(sys.argv[2])
LABEL = sys.argv[3] if len(sys.argv) > 3 else BASE
KEY = os.environ.get("LITELLM_KEY", "sk-test-master")
CLIENTS = int(os.environ.get("CLIENTS", "16"))
BATCH = int(os.environ.get("BATCH", "1"))
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "1500"))
EXTS = {".java", ".kt", ".xml", ".html", ".properties", ".sql", ".css", ".scss", ".yml", ".yaml", ".json", ".md", ".txt", ".gradle"}


def chunks():
    files = 0
    for f in sorted(p for p in ROOT.rglob("*") if p.is_file() and p.suffix in EXTS):
        files += 1
        buf = ""
        for line in f.read_text(errors="ignore").splitlines(keepends=True):
            if buf and len(buf) + len(line) > CHUNK_CHARS:
                yield files, f"{f.relative_to(ROOT)}\n{buf}"
                buf = ""
            buf += line[:CHUNK_CHARS]
        if buf.strip():
            yield files, f"{f.relative_to(ROOT)}\n{buf}"


async def main():
    items = list(chunks())
    files, texts = (items[-1][0] if items else 0), [t for _, t in items]
    batches = [texts[i:i + BATCH] for i in range(0, len(texts), BATCH)]
    queue, tokens, errors = asyncio.Queue(), 0, 0
    for b in batches:
        queue.put_nowait(b)

    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {KEY}"}, timeout=600) as client:
        await client.post(f"{BASE}/embeddings", json={"model": "embed-local", "input": "warm-up"})

        async def worker():
            nonlocal tokens, errors
            while not queue.empty():
                b = queue.get_nowait()
                try:
                    r = await client.post(f"{BASE}/embeddings", json={"model": "embed-local", "input": b})
                    j = r.json()
                    assert r.status_code == 200 and len(j["data"]) == len(b)
                    tokens += j["usage"]["prompt_tokens"]
                except Exception:
                    errors += len(b)

        start = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(CLIENTS)])
        dt = time.perf_counter() - start

    print(json.dumps({"label": LABEL, "files": files, "chunks": len(texts), "tokens": tokens, "errors": errors,
                      "seconds": round(dt, 1), "chunks_s": round(len(texts) / dt, 1), "tokens_s": round(tokens / dt)}))

asyncio.run(main())
