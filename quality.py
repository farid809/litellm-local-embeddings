"""Clustering-quality check: do the embeddings group source files by functional module?

    python quality.py <base-url|random> <model-label> <repo-dir> [<repo-dir> ...]

Each .java file under */src/main/java becomes one vector (mean of its chunk vectors). The label is the
file's top-level module, so no label text reaches the model: package and import lines are stripped and
paths are not embedded. Needs: httpx numpy scikit-learn umap-learn
"""
import asyncio
import json
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi

warnings.filterwarnings("ignore")
BASE, MODEL = sys.argv[1].rstrip("/"), sys.argv[2]
KEY = os.environ.get("LITELLM_KEY", "sk-test-master")
CHUNK_CHARS, CLIENTS, MIN_FILES = 1500, int(os.environ.get("CLIENTS", "16")), 5
SAVE = os.environ.get("SAVE_VECTORS")  # optional directory for the raw vectors


def load(root):
    files = []
    for f in sorted(root.rglob("*.java")):
        rel = f.relative_to(root)
        if "src/main/java" not in rel.as_posix():
            continue
        label = re.sub(r"-(api|aws-lambda)$", "", rel.parts[0])  # an -api module belongs to its service
        body = "".join(l for l in f.read_text(errors="ignore").splitlines(keepends=True)
                       if not l.startswith(("package ", "import ")))
        files.append((str(rel), label, [body[i:i + CHUNK_CHARS] for i in range(0, len(body), CHUNK_CHARS)] or [""]))
    keep = {l for l, n in Counter(l for _, l, _ in files).items() if n >= MIN_FILES}
    return [f for f in files if f[1] in keep]


async def embed(texts):
    if BASE == "random":
        return np.random.default_rng(0).normal(size=(len(texts), 384))
    out, queue = [None] * len(texts), asyncio.Queue()
    for i in enumerate(texts):
        queue.put_nowait(i)
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {KEY}"}, timeout=1200) as client:
        async def worker():
            while not queue.empty():
                i, t = queue.get_nowait()
                r = await client.post(f"{BASE}/embeddings", json={"model": "embed-local", "input": t})
                out[i] = r.json()["data"][0]["embedding"]
        await asyncio.gather(*[worker() for _ in range(CLIENTS)])
    return np.array(out)


def score(x, labels):
    y = np.unique(labels, return_inverse=True)[1]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    sim = x @ x.T
    np.fill_diagonal(sim, -1)
    near = np.argsort(-sim, axis=1)[:, :5]
    knn = np.mean([Counter(y[n]).most_common(1)[0][0] == t for n, t in zip(near, y)])
    k = len(set(y))
    km = [KMeans(k, n_init=10, random_state=s).fit_predict(x) for s in range(10)]
    import umap  # BERTopic's default pipeline: UMAP to 5 dimensions, then HDBSCAN
    hd = []
    for s in range(5):
        low = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=s).fit_transform(x)
        hd.append(HDBSCAN(min_cluster_size=5).fit_predict(low))
    m = lambda f, runs: round(float(np.mean([f(y, r) for r in runs])), 3)
    return {"knn5_acc": round(float(knn), 3), "kmeans_nmi": m(nmi, km), "kmeans_ari": m(ari, km),
            "umap_hdbscan_nmi": m(nmi, hd), "umap_hdbscan_ari": m(ari, hd),
            "hdbscan_clusters": round(float(np.mean([len(set(r)) - (-1 in r) for r in hd])), 1),
            "hdbscan_noise": round(float(np.mean([(r == -1).mean() for r in hd])), 3)}


for root in map(Path, sys.argv[3:]):
    files = load(root)
    chunks = [(i, c) for i, (_, _, cs) in enumerate(files) for c in cs]
    vecs = asyncio.run(embed([c for _, c in chunks]))
    owner = np.array([i for i, _ in chunks])
    x = np.array([vecs[owner == i].mean(0) for i in range(len(files))])
    labels = [l for _, l, _ in files]
    if SAVE:
        np.save(f"{SAVE}/{MODEL}__{root.name}.npy", x)
    print(json.dumps({"model": MODEL, "corpus": root.name, "files": len(files), "modules": len(set(labels)),
                      "majority_share": round(max(Counter(labels).values()) / len(labels), 3), **score(x, labels)}), flush=True)
