"""Embed each file natively (ONNX Runtime, CPU) at the model's full native dimension.

    python embed_native.py <model-label> <model-dir> <prefix> <code|summary|query>

Same pooling and tokenisation logic as litellm/local_embed.py, without padding to 1536. The model
directory needs model.onnx, tokenizer.json and 1_Pooling/config.json.
code:    one vector per file = mean of its 1500-char chunk vectors (as in tests/quality.py)
summary: one vector per file = the embedding of its LLM-written summary
query:   one vector per file = the embedding of its search query (pass the model's query prefix)
Output: vectors/<label>__<kind>__<corpus>.npy
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from common import CORPORA, S, load

LABEL, MODEL_DIR, PREFIX, KIND = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
MAX_TOKENS, BATCH, CHUNK = 512, 8, 1500

tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
names = {i.name for i in session.get_inputs()}
pool_cfg = json.loads((MODEL_DIR / "1_Pooling/config.json").read_text())
pooling = "cls" if pool_cfg.get("pooling_mode_cls_token") else "last" if pool_cfg.get("pooling_mode_lasttoken") else "mean"
tc = MODEL_DIR / "tokenizer_config.json"
tc = json.loads(tc.read_text()) if tc.exists() else {}
eos = [tok.token_to_id(tc["eos_token"])] if tc.get("add_eos_token") else []
tok.enable_truncation(max_length=MAX_TOKENS - len(eos))


def embed(texts):
    out, lengths = [], []
    order = np.argsort([len(t) for t in texts])  # batch similar lengths together
    for i in range(0, len(texts), BATCH):
        idx = order[i:i + BATCH]
        rows = [e.ids + eos for e in tok.encode_batch([PREFIX + texts[j] for j in idx])]
        width = max(len(r) for r in rows)
        ids = np.array([r + [0] * (width - len(r)) for r in rows], dtype=np.int64)
        mask = np.array([[1] * len(r) + [0] * (width - len(r)) for r in rows], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(mask)
        hidden = session.run(None, feed)[0]
        if pooling == "cls":
            pooled = hidden[:, 0]
        elif pooling == "last":
            pooled = hidden[np.arange(len(rows)), mask.sum(1) - 1]
        else:
            m = mask[..., None].astype(hidden.dtype)
            pooled = (hidden * m).sum(1) / m.sum(1).clip(min=1)
        out.append((idx, pooled.astype(np.float32)))
        lengths += [len(r) for r in rows]
    vecs = np.zeros((len(texts), out[0][1].shape[1]), np.float32)
    for idx, p in out:
        vecs[idx] = p
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12), lengths


(S / "vectors").mkdir(exist_ok=True)
for name in CORPORA:
    target = S / "vectors" / f"{LABEL}__{KIND}__{name}.npy"
    if target.exists():
        continue
    files = load(name)
    t0 = time.time()
    if KIND == "code":
        chunks = [(i, body[k:k + CHUNK]) for i, (_, _, body) in enumerate(files)
                  for k in range(0, max(len(body), 1), CHUNK)]
        vecs, lengths = embed([c for _, c in chunks])
        owner = np.array([i for i, _ in chunks])
        x = np.array([vecs[owner == i].mean(0) for i in range(len(files))])
    else:
        summaries = json.loads((S / ("queries" if KIND == "query" else "summaries") / f"{name}.json").read_text())
        x, lengths = embed([summaries[rel] for rel, _, _ in files])
    np.save(target, x)
    print(json.dumps({"model": LABEL, "kind": KIND, "corpus": name, "files": len(files), "dim": int(x.shape[1]),
                      "texts": len(lengths), "tokens_p50": int(np.median(lengths)), "tokens_max": int(max(lengths)),
                      "share_at_limit": round(float(np.mean(np.array(lengths) >= MAX_TOKENS)), 3),
                      "seconds": round(time.time() - t0, 1)}), flush=True)
