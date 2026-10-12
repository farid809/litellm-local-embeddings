# Results: standalone option

The experiment behind the [standalone option](../README.md#option-b-standalone-server): a 4B-parameter
embedding model served from one container with no proxy, measured on CPU and GPU and compared with the
models of the LiteLLM option. Raw output is in [`results/standalone/`](../results/standalone/).

Results for the other option, including the method these tests reuse, are in
[litellm/RESULTS.md](../litellm/RESULTS.md).

## 1. Question

`Qwen/Qwen3-Embedding-4B` is a candidate replacement for a hosted embedding API. It is about 120 times
the size of the small model in Option A and is usually deployed on a GPU. Can it serve the same
`/embeddings` contract from a standalone container, on CPU as well as GPU, and are its vectors better for
code-analysis jobs than those of the smaller models?

## 2. Hypotheses

- **H1.** A standalone server keeps the client contract unchanged (endpoint, request and response shape,
  auth, 1536 dimensions) with no proxy in front.
- **H2.** The model runs on CPU without a GPU, fast enough to index a codebase as a batch job.
- **H3.** A GPU makes it fast enough for interactive use.
- **H4.** It scores higher than `e5-small-v2` and `gte-Qwen2-1.5B-instruct` on clustering, assignment and
  search.
- **H5.** Any gain depends on the longer vector.

## 3. Method

**Model.** `Qwen/Qwen3-Embedding-4B`, Alibaba, Apache-2.0, 4B parameters, 2560 native dimensions, 8 GB of
weights, revision `5cf2132`. It loads with standard `transformers` code, without `trust_remote_code`. The
server returns the first 1536 values re-normalised; the model is trained to be shortened this way.
Documents are embedded as plain text and queries with the model card's instruction prefix.

**Server.** [`server.py`](server.py): FastAPI, sentence-transformers 6.1.0, PyTorch 2.14.1. One inference
runs at a time and concurrent requests share batches of up to 8 texts. Inputs are cut at 512 tokens, as
in Option A.

**Setup.** M5 Max MacBook, 18 cores, 128 GB. Three runtimes:

| Runtime | Precision | Notes |
|---|---|---|
| CPU in the container | float32 | Native arm64 image, Docker given 32 GB |
| CPU natively on the host | float32 | Same `server.py` |
| GPU natively on the host | float16 | Apple GPU through PyTorch MPS. Containers on macOS cannot reach it |

The Option A container figures were measured under x86 emulation and these are native arm64, so the two
sets of container figures are not directly comparable. The native host figures are, with the caveat that
Option A runs ONNX Runtime and this option runs PyTorch.

**Tests.** The same scripts as Option A: `standalone/test.sh` (18 contract checks from a network with no
internet route), `tests/stress.py` (15 seconds per scenario, one text of about 130 tokens per request),
`tests/embed_dir.py` on the `src` directory of spring-petclinic (93 files, 371 chunks),
`tests/quality.py` (clustering of raw code, three corpora), and the summary, search and dimension tests
in `tests/usecases/` described in [litellm/RESULTS.md, section 7](../litellm/RESULTS.md#7-follow-up-summaries-search-and-vector-dimension).
The quality tests were run against the GPU server.

## 4. Results

### H1: contract

18 of 18 checks pass against the container on CPU. Run from the host against the GPU server, 17 of 17
applicable checks pass; the remaining one tests the client network's isolation, not the server. Vectors
for the same text requested alone and inside a batch are identical (cosine 1.000).

### H2 and H3: throughput

| Qwen3-Embedding-4B | CPU, container | CPU, native | GPU, native |
|---|---|---|---|
| Embed PetClinic `src` | 1,391 s | 807 s | 41 s |
| Tokens per second | 102 | 176 | 3,452 |
| One request, median latency | 1.7 s | 0.9 s | 54 ms |
| Peak requests per second | 0.9 | 1.4 | 27.6 |

Against the Option A models, natively on the host CPU:

| CPU, native | e5-small | gte-Qwen2-1.5B | Qwen3-Embedding-4B |
|---|---|---|---|
| Embed PetClinic `src` | 5 s | 318 s | 807 s |
| One request, median latency | 13 ms | 192 ms | 0.9 s |
| Peak requests per second | 209 | 5 | 1.4 |

The container held 16.3 to 16.5 GB while serving and read 24 GB shortly after start-up. The image is
9.5 GB unpacked (6.7 GB compressed).

No CPU run returned a server error. With 8 clients each sending 32 texts, single requests took longer
than the load test's 120-second client timeout (6 of 8 timed out in the container, 3 of 8 natively).

### H4: quality

Raw code, the test from Option A (three corpora; "mean" is the average of nine scores, the other columns
are the largest corpus):

| Model | Mean of 9 | kNN accuracy | k-means NMI | UMAP + HDBSCAN NMI |
|---|---|---|---|---|
| random vectors | 0.12 | 0.17 | 0.06 | 0.08 |
| e5-small | 0.38 | 0.65 | 0.20 | 0.31 |
| bge-small | 0.38 | 0.65 | 0.30 | 0.34 |
| **Qwen3-Embedding-4B** | **0.38** | 0.63 | 0.25 | 0.36 |
| gte-Qwen2-1.5B | 0.36 | 0.65 | 0.22 | 0.32 |

LLM-written summaries of the same 390 files, pooled into one set of 18 modules:

| | Random vectors | e5-small (384) | gte-Qwen2 (1536) | Qwen3-4B at 1536 | Qwen3-4B at 2560 |
|---|---|---|---|---|---|
| kNN accuracy | 0.10 | 0.53 | 0.59 | 0.60 | 0.60 |
| Centroid accuracy | 0.06 | 0.66 | 0.70 | 0.68 | 0.67 |
| k-means NMI | 0.15 | 0.40 | 0.41 | 0.44 | 0.42 |
| UMAP + HDBSCAN NMI | 0.17 | 0.43 | 0.40 | 0.45 | 0.45 |
| Search recall@1 | 0.00 | 0.67 | 0.74 | 0.76 | 0.78 |
| Search recall@10 | 0.03 | 0.97 | 0.995 | 0.99 | 0.99 |
| Search MRR | 0.02 | 0.78 | 0.84 | 0.85 | 0.87 |

Seed-to-seed standard deviation of the Qwen3-4B clustering scores is 0.039 for k-means and 0.010 for
UMAP + HDBSCAN.

### H5: dimension

Qwen3-Embedding-4B shortened to its first d values, summaries:

| d | kNN | Centroid | k-means NMI | Search recall@1 | Search recall@10 |
|---|---|---|---|---|---|
| 32 | 0.58 | 0.56 | 0.45 | 0.51 | 0.91 |
| 64 | 0.63 | 0.61 | 0.44 | 0.59 | 0.95 |
| 128 | 0.63 | 0.65 | 0.46 | 0.67 | 0.97 |
| 256 | 0.61 | 0.67 | 0.45 | 0.71 | 0.99 |
| 384 | 0.61 | 0.69 | 0.46 | 0.70 | 0.985 |
| 512 | 0.61 | 0.69 | 0.45 | 0.73 | 0.985 |
| 768 | 0.61 | 0.67 | 0.45 | 0.74 | 0.99 |
| 1024 | 0.59 | 0.68 | 0.43 | 0.74 | 0.99 |
| 1536 | 0.60 | 0.68 | 0.44 | 0.76 | 0.99 |
| 2048 | 0.60 | 0.68 | 0.44 | 0.77 | 0.99 |
| 2560 (full) | 0.60 | 0.67 | 0.42 | 0.78 | 0.99 |

95% of the variance of its summary vectors lies in 151 principal components, against 127 for `e5-small`
and 161 for `gte-Qwen2`.

## 5. Limitations

- **Small corpora, one language.** 390 Java/Spring files. Differences of a few hundredths are not
  significant, and differences between models in search usually grow with index size.
- **Synthetic text.** One LLM wrote both the summaries and the queries.
- **512-token inputs.** The model accepts 32,768 tokens; that was not exercised. The summaries are about
  80 tokens long.
- **No NVIDIA GPU and no x86 CPU.** The GPU figures are from an Apple GPU, and the CPU figures from Apple
  Silicon. The CUDA build of the image was not run.
- **GPU precision.** The GPU ran float16 and the CPU float32. The quality scores are from the GPU.
- **Full precision only.** Quantised variants of the model, which are smaller and faster on CPU, were not
  tested.
- **Not a product's pipeline.** Chunking, parameters and data in a real pipeline will differ.

## 6. Conclusion

- **H1 holds.** The standalone server passes the same 18 contract checks as Option A, with no proxy and
  no outbound network access.
- **H2 holds only for small batch jobs.** On CPU the model runs without server errors but at about one
  request per second. A codebase that `e5-small` embeds in 5 seconds takes 13 minutes
  natively and 23 minutes in the container, with 16 GB of memory.
- **H3 holds.** On the GPU a request takes 54 ms and the same codebase 41 seconds, about 20 times
  faster than native CPU.
- **H4 holds for search only.** The model ranks the right file first 9 points more often than `e5-small`
  (0.76 against 0.67) and 2 points more often than `gte-Qwen2`. In the top 10 the gap to `e5-small` is 2
  points. On clustering and assignment it is within noise of both smaller models, on code and on
  summaries.
- **H5 is not supported.** Clustering and assignment are flat from about 64 to 128 dimensions. Search
  recall@1 rises slowly up to the full size, but at 256 dimensions the model already scores 0.71, above
  `e5-small` at its native 384. The gain comes from the model, not from the vector length.

**Recommendation.** Use this option when search ranking quality is the priority and a GPU is available.
For clustering and assignment jobs, or for CPU-only deployment, the small model in Option A gives
the same result at a small fraction of the compute.

## 7. Reproducing

```bash
./test.sh                                                           # H1, builds the image first
pip install httpx numpy scikit-learn umap-learn joblib
python ../tests/stress.py http://localhost:4004                     # H2, H3
python ../tests/embed_dir.py http://localhost:4004 /path/to/src     # H2, H3
python ../tests/quality.py http://localhost:4004 qwen3-4b /path/to/repo [...]   # H4

cd ../tests/usecases                                                # H4, H5
export EMBED_DIMENSIONS=2560                                        # request the native size
python embed_api.py qwen3-4b http://localhost:4004 code
python embed_api.py qwen3-4b http://localhost:4004 summary
python embed_api.py qwen3-4b http://localhost:4004 query $'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'
python sweep.py qwen3-4b > sweep.jsonl && python search.py qwen3-4b > search.jsonl
python report.py sweep.jsonl search.jsonl
```
