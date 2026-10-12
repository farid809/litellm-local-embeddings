"""Standalone embedding server: the OpenAI-style POST /embeddings contract with no proxy in front.

The model is read from EMBED_MODEL_DIR (baked into the image), so no network is needed at runtime.
Vectors are L2-normalised and returned at EMBED_DIM: a larger native size is truncated and
re-normalised (Qwen3-Embedding is trained for this), a smaller one is zero-padded.
"""
import asyncio
import base64
import hmac
import os
import threading
from contextlib import asynccontextmanager

import numpy as np
import tiktoken
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer

MODEL_DIR = os.environ.get("EMBED_MODEL_DIR", "/opt/embed-model")
MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "Qwen3-Embedding-4B")
DIM = int(os.environ.get("EMBED_DIM", "1536"))
BATCH = int(os.environ.get("EMBED_BATCH", "8"))
MAX_TOKENS = int(os.environ.get("EMBED_MAX_TOKENS", "512"))  # the model accepts up to 32768
PREFIX = os.environ.get("EMBED_PREFIX", "")  # documents take none; queries are prefixed by the caller
THREADS = int(os.environ.get("EMBED_THREADS", "0"))  # 0 = let PyTorch decide
DEVICE = os.environ.get("EMBED_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = os.environ.get("EMBED_DTYPE") or ("float32" if DEVICE == "cpu" else "float16")
# same variable name as the proxy option, so one deployment manifest fits both
KEY = os.environ.get("EMBED_API_KEY") or os.environ["LITELLM_MASTER_KEY"]
KEY_HEADERS = ("x-api-key", "api-key", "x-litellm-api-key")

model = None
jobs: asyncio.Queue = None
tokens_enc = tiktoken.get_encoding("cl100k_base")  # OpenAI clients may send token ids instead of text


def encode(texts):
    """Blocking inference for one batch. Returns (vectors at native size, token counts)."""
    counts = [len(ids) for ids in model.tokenizer(texts, truncation=True, max_length=MAX_TOKENS)["input_ids"]]
    vectors = model.encode(texts, batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype(np.float32), counts


async def batcher():
    """One inference at a time; texts from concurrent requests share a batch."""
    while True:
        batch = [await jobs.get()]
        while len(batch) < BATCH and not jobs.empty():
            batch.append(jobs.get_nowait())
        batch = [(t, f) for t, f in batch if not f.cancelled()]
        if not batch:
            continue
        try:
            vectors, counts = await asyncio.to_thread(encode, [t for t, _ in batch])
            for (_, f), v, n in zip(batch, vectors, counts):
                f.cancelled() or f.set_result((v, n))
        except Exception as e:  # report to the callers, keep serving
            for _, f in batch:
                f.cancelled() or f.set_exception(e)


@asynccontextmanager
async def lifespan(_):
    global model, jobs
    if THREADS:
        torch.set_num_threads(THREADS)
    model = SentenceTransformer(MODEL_DIR, device=DEVICE, model_kwargs={"torch_dtype": getattr(torch, DTYPE)},
                                processor_kwargs={"padding_side": "left"})
    model.max_seq_length = MAX_TOKENS
    jobs = asyncio.Queue()
    task = asyncio.create_task(batcher())
    await asyncio.to_thread(encode, ["warm-up"])
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def error(status, message, kind="invalid_request_error"):
    return JSONResponse({"error": {"message": message, "type": kind, "param": None, "code": str(status)}}, status_code=status)


def resize(v, dim):
    if v.shape[1] < dim:
        return np.pad(v, ((0, 0), (0, dim - v.shape[1])))
    if v.shape[1] > dim:
        v = v[:, :dim]
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
    return v


def texts_of(value):
    items = [value] if isinstance(value, str) or (value and isinstance(value[0], int)) else list(value)
    return [tokens_enc.decode(t) if not isinstance(t, str) else t for t in items]


@app.middleware("http")
async def auth(request: Request, call_next):
    if request.url.path.startswith("/health"):
        return await call_next(request)
    bearer = request.headers.get("authorization", "")
    sent = [bearer[7:]] if bearer.lower().startswith("bearer ") else []
    sent += [request.headers[h] for h in KEY_HEADERS if h in request.headers]
    if not sent:
        return error(401, "No API key provided", "auth_error")
    if not any(hmac.compare_digest(s, KEY) for s in sent):
        return error(401, "Invalid API key", "auth_error")
    return await call_next(request)


@app.post("/embeddings")
@app.post("/v1/embeddings")
async def embeddings(request: Request):
    try:
        body = await request.json()
        texts = texts_of(body["input"])
        dim = int(body.get("dimensions") or DIM)
        assert texts and dim > 0
    except Exception:
        return error(400, "Expected JSON with 'input' as a string, a list of strings, or token ids")
    loop = asyncio.get_running_loop()
    futures = [loop.create_future() for _ in texts]
    for t, f in zip(texts, futures):
        jobs.put_nowait((PREFIX + t, f))
    try:
        done = await asyncio.gather(*futures)
    except Exception as e:
        return error(500, f"{type(e).__name__}: {e}", "server_error")
    vectors = resize(np.stack([v for v, _ in done]), dim)
    tokens = sum(n for _, n in done)
    as_b64 = body.get("encoding_format") == "base64"
    return {"object": "list", "model": body.get("model") or MODEL_NAME,
            "data": [{"object": "embedding", "index": i, "embedding": base64.b64encode(v.tobytes()).decode() if as_b64 else v.tolist()}
                     for i, v in enumerate(vectors)],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}


@app.get("/models")
@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]}


@app.get("/health")
@app.get("/health/liveliness")
@app.get("/health/readiness")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE, "dtype": DTYPE, "dim": DIM}
