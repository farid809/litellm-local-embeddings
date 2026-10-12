"""Semantic-search job by dimension: a short query must retrieve its own file's summary.

Documents = LLM-written summaries of all files in the three corpora (one index). Queries = one
LLM-written natural-language query per file, with no identifiers. Reports recall@1, recall@10 and MRR
at full dimension and after reducing to d dimensions (same three reductions as sweep.py).

    python search.py <model-label> [<model-label> ...] > search.jsonl
"""
import json
import sys

import numpy as np
from sklearn.decomposition import PCA

from common import CORPORA, S

DIMS = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048]


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def score(q, d):
    sim = unit(q) @ unit(d).T
    rank = (sim > sim[np.arange(len(q)), np.arange(len(q))][:, None]).sum(1) + 1
    return {"recall1": float((rank <= 1).mean()), "recall10": float((rank <= 10).mean()), "mrr": float((1 / rank).mean())}


for model in sys.argv[1:]:
    d = np.vstack([np.load(S / "vectors" / f"{model}__summary__{c}.npy") for c in CORPORA])
    q = np.vstack([np.load(S / "vectors" / f"{model}__query__{c}.npy") for c in CORPORA])
    n, native = d.shape
    base = {"model": model, "docs": n, "native": native}
    out = [{**base, "reduce": "full", "d": native, **score(q, d)}]
    rng = np.random.default_rng(0)
    out.append({**base, "reduce": "random-vectors", "d": native, **score(rng.normal(size=q.shape), rng.normal(size=d.shape))})
    for k in [k for k in DIMS if k < native]:
        if k < n:
            p = PCA(k, random_state=0).fit(d)
            out.append({**base, "reduce": "pca", "d": k, **score(p.transform(q), p.transform(d))})
        out.append({**base, "reduce": "truncate", "d": k, **score(q[:, :k], d[:, :k])})
        rp = []
        for s in range(5):
            m = np.random.default_rng(s).normal(size=(native, k))
            rp.append(score(q @ m, d @ m))
        out.append({**base, "reduce": "random-projection", "d": k, **{m_: float(np.mean([r[m_] for r in rp])) for m_ in rp[0]}})
    for r in out:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
