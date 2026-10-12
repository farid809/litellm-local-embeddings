"""Tabulate sweep.py and search.py output: python report.py <file.jsonl> [<file.jsonl> ...]"""
import json
import sys
from collections import defaultdict

rows = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip()]
REDUCTIONS = ["random-projection", "pca", "truncate"]

sweep = defaultdict(list)
for r in rows:
    if "kind" in r:
        sweep[(r["kind"], r["corpus"], r["model"])].append(r)

if sweep:
    print("== geometry: how many directions the vectors occupy (PCA is capped at files - 1) ==")
    print(f"{'kind':8}{'corpus':32}{'model':16}{'native':>7}{'files':>6}{'pca90':>6}{'pca95':>6}{'pca99':>6}{'PR':>7}")
    for (kind, corpus, model), g in sorted(sweep.items()):
        s = next(r for r in g if r.get("stat"))
        print(f"{kind:8}{corpus:32}{model:16}{s['native']:>7}{s['files']:>6}{s['pca90']:>6}{s['pca95']:>6}{s['pca99']:>6}"
              f"{s['participation_ratio']:>7.1f}")

for (kind, corpus, model), g in sorted(sweep.items()):
    if corpus not in ("ftgo-application", "pooled"):
        continue
    full = next(r for r in g if r.get("reduce") == "full")
    rand = next(r for r in g if r.get("reduce") == "random-vectors")
    ms = [m for m in ["knn5", "centroid", "kmeans", "hdbscan"] if m in full]
    print(f"\n== {kind} | {corpus} | {model} (native {full['native']}, {full['files']} files, {full['modules']} modules) ==")
    print(f"{'reduce':20}{'d':>6}" + "".join(f"{m:>10}" for m in ms))
    print(f"{'random vectors':20}{'':>6}" + "".join(f"{rand[m]:>10.3f}" for m in ms))
    for red in REDUCTIONS:
        for r in sorted((r for r in g if r.get("reduce") == red), key=lambda r: r["d"]):
            print(f"{red:20}{r['d']:>6}" + "".join(f"{r[m]:>10.3f}" for m in ms))
    print(f"{'full':20}{full['native']:>6}" + "".join(f"{full[m]:>10.3f}" for m in ms)
          + f"   (seed sd: kmeans {full['kmeans_sd']:.3f}" + (f", hdbscan {full['hdbscan_sd']:.3f})" if "hdbscan_sd" in full else ")"))

search = defaultdict(list)
for r in rows:
    if "recall1" in r:
        search[r["model"]].append(r)
for model, g in sorted(search.items()):
    print(f"\n== search | one index of {g[0]['docs']} summaries | {model} (native {g[0]['native']}) ==")
    print(f"{'reduce':20}{'d':>6}{'recall@1':>10}{'recall@10':>10}{'MRR':>8}")
    for red in ["random-vectors"] + REDUCTIONS + ["full"]:
        for r in sorted((r for r in g if r["reduce"] == red), key=lambda r: r["d"]):
            print(f"{red:20}{r['d']:>6}{r['recall1']:>10.3f}{r['recall10']:>10.3f}{r['mrr']:>8.3f}")
