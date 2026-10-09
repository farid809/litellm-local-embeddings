"""Exercises the embedding contract (POST /embeddings, OpenAI JSON, key header) against the image."""
import base64
import math
import os
import struct
import time

import httpx
import litellm  # noqa: F401  (points tiktoken at the encoding bundled in the image)
from openai import OpenAI

URL = os.environ["LITELLM_EMBEDDING_URL"]
BASE = os.environ["LITELLM_PROXY_API_BASE"]
KEY = os.environ["LITELLM_EMBEDDING_PROXY_API_KEY"]
MODEL = os.environ.get("MODEL", "amazon.titan-embed-text-v1")
DIM = int(os.environ.get("EXPECT_DIM", "1536"))
BEARER = {"Authorization": f"Bearer {KEY}"}
cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((name, True)); print(f"PASS  {name}: {detail}")
    except Exception as e:
        results.append((name, False)); print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:300]}")


def post(body, headers=BEARER, url=URL):
    return httpx.post(url, json=body, headers=headers, timeout=120)


def raw_contract():
    r = post({"model": MODEL, "input": "COBOL paragraph that computes loan interest"})
    assert r.status_code == 200, f"{r.status_code} {r.text[:200]}"
    j = r.json(); v = j["data"][0]["embedding"]
    norm = math.sqrt(sum(x * x for x in v))
    assert len(v) == DIM and all(isinstance(x, float) for x in v) and abs(norm - 1) < 1e-3
    return f"dim={len(v)} norm={norm:.4f} object={j['object']} model={j['model']} usage={j['usage']}"


def auth(headers, ok):
    def f():
        r = post({"model": MODEL, "input": "x"}, headers=headers)
        assert (r.status_code == 200) == ok, f"{r.status_code} {r.text[:150]}"
        return f"{r.status_code}" + ("" if ok else f" {r.json()['error']['message'][:60]}")
    return f


def aliases():
    names = [m["id"] for m in httpx.get(f"{BASE}/v1/models", headers=BEARER).json()["data"] if "*" not in m["id"]]
    dims = {n: len(post({"model": n, "input": "x"}).json()["data"][0]["embedding"]) for n in names}
    assert set(dims.values()) == {DIM}, dims
    return dims


def any_model_name():
    out = {}
    for name in ("amazon.titan-embed-text-v2:0", "cohere.embed-english-v3", "whatever-the-product-sends"):
        r = post({"model": name, "input": "x"})
        assert r.status_code == 200, f"{name}: {r.status_code} {r.text[:150]}"
        out[name] = len(r.json()["data"][0]["embedding"])
    assert set(out.values()) == {DIM}, out
    return out


def v1_path():
    r = post({"model": MODEL, "input": "x"}, url=f"{BASE}/v1/embeddings")
    assert r.status_code == 200 and len(r.json()["data"][0]["embedding"]) == DIM
    return "200"


def base64_format():
    r = post({"model": MODEL, "input": "same text", "encoding_format": "base64"}).json()["data"][0]["embedding"]
    f = post({"model": MODEL, "input": "same text", "encoding_format": "float"}).json()["data"][0]["embedding"]
    assert isinstance(r, str), f"expected base64 string, got {type(r).__name__}"
    d = struct.unpack(f"<{DIM}f", base64.b64decode(r))
    return f"base64 decodes to {len(d)} floats, cos(base64, float)={cos(d, f):.6f}"


def sdk(**kw):
    def f():
        client = OpenAI(base_url=BASE, api_key=KEY)
        texts = ["Calculate monthly mortgage interest", "Compute the interest owed on a home loan each month", "Render the login page footer"]
        r = client.embeddings.create(model=MODEL, input=texts, **kw)
        v = [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
        related, unrelated = cos(v[0], v[1]), cos(v[0], v[2])
        assert [len(x) for x in v] == [DIM] * 3 and related > unrelated
        return f"n=3 dim={DIM} cos(related)={related:.3f} > cos(unrelated)={unrelated:.3f} usage={r.usage.prompt_tokens}tok"
    return f


def token_input():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    text = "Compute the interest owed on a home loan each month"
    a = post({"model": MODEL, "input": [enc.encode(text)]}).json()["data"][0]["embedding"]
    b = post({"model": MODEL, "input": text}).json()["data"][0]["embedding"]
    c = cos(a, b)
    assert c > 0.999, f"cos={c}"
    return f"cos(token-id input, text input)={c:.6f}"


def dimensions_param():
    v = post({"model": MODEL, "input": "x", "dimensions": 256}).json()["data"][0]["embedding"]
    norm = math.sqrt(sum(x * x for x in v))
    assert len(v) == 256 and abs(norm - 1) < 1e-3
    return f"dim={len(v)} norm={norm:.4f}"


def long_input():
    r = post({"model": MODEL, "input": "MOVE CUSTOMER-RECORD TO OUTPUT-BUFFER. " * 2000})
    assert r.status_code == 200, f"{r.status_code} {r.text[:150]}"
    return f"80k chars -> 200, usage={r.json()['usage']['prompt_tokens']}tok (truncated to the model window)"


def throughput(n=64):
    texts = [f"legacy module {i}: moves customer record to output buffer and updates balance" for i in range(n)]
    t = time.time(); r = post({"model": MODEL, "input": texts}); dt = time.time() - t
    assert r.status_code == 200 and len(r.json()["data"]) == n
    return f"{n} texts in {dt:.2f}s ({n/dt:.0f}/s, one request)"


def no_egress():
    try:
        httpx.get("https://huggingface.co", timeout=4)
    except Exception as e:
        return f"client network has no route out ({type(e).__name__})"
    raise AssertionError("reached the internet")


check("raw POST /embeddings, OpenAI JSON", raw_contract)
check("auth: Authorization Bearer", auth(BEARER, True))
check("auth: x-litellm-api-key header", auth({"x-litellm-api-key": KEY}, True))
check("auth: api-key header", auth({"api-key": KEY}, True))
check("auth: x-api-key header", auth({"x-api-key": KEY}, True))
check("auth: no key rejected", auth({}, False))
check("auth: wrong key rejected", auth({"Authorization": "Bearer sk-wrong"}, False))
check("every alias returns the same dimension", aliases)
check("any other model name is accepted (wildcard)", any_model_name)
check("/v1/embeddings path", v1_path)
check("encoding_format=base64 over raw HTTP", base64_format)
check("OpenAI SDK, default encoding (base64)", sdk())
check("OpenAI SDK, encoding_format=float", sdk(encoding_format="float"))
check("token-id input (LangChain OpenAIEmbeddings default)", token_input)
check("dimensions=256 request param overrides the default", dimensions_param)
check("input longer than the model window", long_input)
check("throughput", throughput)
check("no internet from the client network", no_egress)

print("\nSUMMARY:", sum(ok for _, ok in results), "/", len(results), "passed")
