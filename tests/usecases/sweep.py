"""Dimension sweep: how many dimensions do the four embedding jobs actually use?

For each (model, text kind, corpus) take the full native vectors, reduce them to d dimensions three
ways, and re-score. Reductions: PCA (fitted on the corpus), Gaussian random projection (data-independent,
averaged over projections), and plain truncation of the leading dimensions.

Scores, all against each file's module label:
  knn5      leave-one-out: majority of the 5 nearest neighbours share the module       (similarity matching)
  centroid  leave-one-out: nearest module centroid is the file's own module            (best-parent selection)
  kmeans    NMI of k-means with k = number of modules                                   (clustering)
  hdbscan   NMI of UMAP(5 dims) + HDBSCAN, BERTopic's default pipeline                  (topic clustering)

    python sweep.py <model-label> [<model-label> ...] > sweep.jsonl

Needs the vectors written by embed_native.py or embed_api.py. Needs: numpy scikit-learn umap-learn joblib
"""
import json
import sys
import warnings
from collections import Counter

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as nmi

from common import CORPORA, S, load

warnings.filterwarnings("ignore")
DIMS = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048]
SEEDS, PROJECTIONS = 3, 3


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def score(x, y, heavy):
    x = unit(x)
    sim = x @ x.T
    np.fill_diagonal(sim, -1)
    near = np.argsort(-sim, axis=1)[:, :5]
    knn = float(np.mean([Counter(y[n]).most_common(1)[0][0] == t for n, t in zip(near, y)]))
    classes = np.unique(y)
    sums = np.array([x[y == c].sum(0) for c in classes])
    counts = np.array([(y == c).sum() for c in classes])
    hit = 0
    for i in range(len(x)):  # leave the file out of its own module's centroid
        cents = sums.copy()
        cents[y[i]] -= x[i]
        n = counts.copy()
        n[y[i]] -= 1
        hit += int(np.argmax(unit(cents / n[:, None].clip(min=1)) @ x[i]) == y[i])
    out = {"knn5": knn, "centroid": hit / len(x)}
    k = len(classes)
    km = [nmi(y, KMeans(k, n_init=5, random_state=s).fit_predict(x)) for s in range(SEEDS)]
    out["kmeans"], out["kmeans_sd"] = float(np.mean(km)), float(np.std(km))
    if heavy:
        import umap
        hd = []
        for s in range(SEEDS):
            low = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=s).fit_transform(x)
            hd.append(nmi(y, HDBSCAN(min_cluster_size=5).fit_predict(low)))
        out["hdbscan"], out["hdbscan_sd"] = float(np.mean(hd)), float(np.std(hd))
    return out


def run(model, kind, corpus, x, y):
    rows = []
    n, native = x.shape
    heavy = corpus in ("ftgo-application", "pooled")
    base = {"model": model, "kind": kind, "corpus": corpus, "files": n, "modules": int(len(set(y))), "native": native}
    pca = PCA().fit(x)
    var, ev = np.cumsum(pca.explained_variance_ratio_), pca.explained_variance_
    rows.append({**base, "stat": "geometry", "pca90": int(np.searchsorted(var, 0.90) + 1),
                 "pca95": int(np.searchsorted(var, 0.95) + 1), "pca99": int(np.searchsorted(var, 0.99) + 1),
                 "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum())})
    rows.append({**base, "reduce": "full", "d": native, **score(x, y, heavy)})
    rows.append({**base, "reduce": "random-vectors", "d": native,
                 **score(np.random.default_rng(0).normal(size=x.shape), y, heavy)})
    for d in [d for d in DIMS if d < native]:
        if d < n:
            rows.append({**base, "reduce": "pca", "d": d, **score(PCA(d, random_state=0).fit_transform(x), y, heavy)})
        rows.append({**base, "reduce": "truncate", "d": d, **score(x[:, :d], y, heavy)})
        rp = [score(x @ np.random.default_rng(p).normal(size=(native, d)), y, heavy) for p in range(PROJECTIONS)]
        rows.append({**base, "reduce": "random-projection", "d": d,
                     **{k: float(np.mean([r[k] for r in rp])) for k in rp[0]},
                     "knn5_sd": float(np.std([r["knn5"] for r in rp])), "centroid_sd": float(np.std([r["centroid"] for r in rp]))})
    return rows


jobs = []
for model in sys.argv[1:]:
    for kind in ["code", "summary"]:
        xs, ys = [], []
        for c in CORPORA:
            f = S / "vectors" / f"{model}__{kind}__{c}.npy"
            if not f.exists():
                break
            x = np.load(f)
            labels = [f"{c}/{l}" for _, l, _ in load(c)]
            xs.append(x)
            ys += labels
            jobs.append((model, kind, c, x, np.unique(labels, return_inverse=True)[1]))
        else:
            jobs.append((model, kind, "pooled", np.vstack(xs), np.unique(ys, return_inverse=True)[1]))

for rows in Parallel(n_jobs=8)(delayed(run)(*j) for j in jobs):
    for r in rows:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
