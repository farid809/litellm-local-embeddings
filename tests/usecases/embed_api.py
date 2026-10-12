"""Embed each file through a running server's POST /embeddings.

    python embed_api.py <model-label> <base-url> <code|summary|query> [prefix]

code:    one vector per file = mean of its 1500-char chunk vectors (as in tests/quality.py)
summary: one vector per file = the embedding of its LLM-written summary
query:   one vector per file = the embedding of its search query (pass the model's query prefix)
EMBED_DIMENSIONS sets the `dimensions` request parameter, e.g. the model's native size.
Output: vectors/<label>__<kind>__<corpus>.npy
"""
import json
import os
import sys
import time

import httpx
import numpy as np

from common import CORPORA, S, load

LABEL, BASE, KIND = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]
PREFIX = sys.argv[4] if len(sys.argv) > 4 else ""
KEY = os.environ.get("LITELLM_KEY", "sk-test-master")
DIMENSIONS = os.environ.get("EMBED_DIMENSIONS")
CHUNK, BATCH = 1500, 8


def embed(client, texts):
    out, tokens = [], 0
    for i in range(0, len(texts), BATCH):
        body = {"model": "embed-local", "input": [PREFIX + t for t in texts[i:i + BATCH]]}
        if DIMENSIONS:
            body["dimensions"] = int(DIMENSIONS)
        j = client.post(f"{BASE}/embeddings", json=body).raise_for_status().json()
        out += [d["embedding"] for d in sorted(j["data"], key=lambda d: d["index"])]
        tokens += j["usage"]["prompt_tokens"]
    return np.array(out, dtype=np.float32), tokens


(S / "vectors").mkdir(exist_ok=True)
with httpx.Client(headers={"Authorization": f"Bearer {KEY}"}, timeout=1800) as client:
    for name in CORPORA:
        target = S / "vectors" / f"{LABEL}__{KIND}__{name}.npy"
        if target.exists():
            continue
        files = load(name)
        t0 = time.time()
        if KIND == "code":
            chunks = [(i, body[k:k + CHUNK]) for i, (_, _, body) in enumerate(files)
                      for k in range(0, max(len(body), 1), CHUNK)]
            vecs, tokens = embed(client, [c for _, c in chunks])
            owner = np.array([i for i, _ in chunks])
            x = np.array([vecs[owner == i].mean(0) for i in range(len(files))])
        else:
            text = json.loads((S / ("queries" if KIND == "query" else "summaries") / f"{name}.json").read_text())
            x, tokens = embed(client, [text[rel] for rel, _, _ in files])
        np.save(target, x)
        print(json.dumps({"model": LABEL, "kind": KIND, "corpus": name, "files": len(files), "dim": int(x.shape[1]),
                          "tokens": tokens, "seconds": round(time.time() - t0, 1)}), flush=True)
