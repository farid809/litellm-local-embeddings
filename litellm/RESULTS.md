# Results: LiteLLM option

The experiment behind the [LiteLLM option](../README.md#option-a-litellm-with-an-in-process-model): can a
LiteLLM proxy serve embeddings in-process on CPU, and which model should it carry. Sections 1 to 6 are the
original experiment on raw source code. Section 7 is a follow-up on summaries, search and vector
dimension. Raw output is in [`results/litellm/`](../results/litellm/).

Results for the other option are in [standalone/RESULTS.md](../standalone/RESULTS.md).

## 1. Question

A product calls LiteLLM's OpenAI-compatible `/embeddings` endpoint and expects 1536-dimension
vectors, which it uses to cluster and search source code. The planned upstream was a hosted
embedding API. Can the same LiteLLM instance serve those embeddings locally, on CPU, and which
model should it use?

## 2. Hypotheses

- **H1.** A LiteLLM custom provider can serve embeddings in-process while keeping the client
  contract unchanged (endpoint, request and response shape, auth, 1536 dimensions).
- **H2.** A small model (around 30M parameters) is fast enough on CPU to embed a codebase in a
  practical time.
- **H3.** A larger or code-specific model groups related source files measurably better than a
  small one. If H3 fails, a small model is the right default.
- **H4.** Among small models, one from a US publisher performs as well as the common default
  (`bge-small`), so that model origin does not have to be traded against quality.

## 3. Method

**Models.** Eleven models, all with permissive licences. Native vectors smaller than 1536 are
zero-padded, which leaves cosine similarity, dot product and L2 distance unchanged.

| Model | Publisher | Licence | Parameters | Native dim |
|---|---|---|---|---|
| `intfloat/e5-small-v2` | Microsoft | MIT | 33M | 384 |
| `intfloat/e5-base-v2` | Microsoft | MIT | 109M | 768 |
| `ibm-granite/granite-embedding-30m-english` | IBM | Apache-2.0 | 30M | 384 |
| `ibm-granite/granite-embedding-125m-english` | IBM | Apache-2.0 | 125M | 768 |
| `Snowflake/snowflake-arctic-embed-s` | Snowflake | Apache-2.0 | 33M | 384 |
| `Snowflake/snowflake-arctic-embed-m-v1.5` | Snowflake | Apache-2.0 | 109M | 768 |
| `nomic-ai/nomic-embed-text-v1.5` | Nomic AI | Apache-2.0 | 137M | 768 |
| `jinaai/jina-embeddings-v2-base-code` | Jina AI (Germany) | Apache-2.0 | 161M | 768 |
| `BAAI/bge-small-en-v1.5` | BAAI (China) | MIT | 33M | 384 |
| `BAAI/bge-large-en-v1.5` | BAAI (China) | MIT | 335M | 1024 |
| `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | Alibaba (China) | Apache-2.0 | 1.5B | 1536 |

E5 was run with its documented `passage: ` prefix and Nomic with `clustering: `; the others take
raw text. Publisher is the organisation that released the weights. Some reviews also consider where the
research team is based, so check each model against your own model-origin policy.

**Setup.** LiteLLM 1.83.14, ONNX Runtime 1.30, M5 Max MacBook (18 cores). Three runtimes: CPU
inside the container, CPU natively on the host, and GPU natively through CoreML. The upstream
LiteLLM image is x86-only, so the container runs under emulation on this machine; container
figures are therefore a lower bound for real x86 hardware.

**H1, contract.** `litellm/test.sh` runs 18 checks from a client on a network with no internet route:
response shape, four auth header styles, OpenAI SDK default and float encodings, base64,
token-array input, batch input, the `dimensions` parameter, over-length input.

**H2, throughput.** `tests/stress.py` runs 15 seconds per scenario at 1, 8, 32 and 64 concurrent
clients with one ~130-token text per request. `tests/embed_dir.py` times embedding the `src` directory
of [spring-petclinic](https://github.com/spring-projects/spring-petclinic): 93 files, 371 chunks
of up to 1,500 characters, 16 concurrent requests.

**H3 and H4, clustering quality.** `tests/quality.py` embeds every Java file under `src/main/java` in
three multi-module Spring codebases and checks whether the vectors recover each file's module:

| Corpus | Files | Modules |
|---|---|---|
| [spring-petclinic-microservices](https://github.com/spring-petclinic/spring-petclinic-microservices) | 50 | 5 |
| [piggymetrics](https://github.com/sqshq/piggymetrics) | 66 | 4 |
| [ftgo-application](https://github.com/microservices-patterns/ftgo-application) | 274 | 9 |

One vector per file (mean of its chunk vectors). The label is the file's top-level module.
`package` and `import` lines are stripped and paths are not embedded, so the module name does not
leak into the text. Three measures, each compared against random vectors:

- **kNN accuracy**: leave-one-out, does the majority of a file's 5 nearest neighbours share its module.
- **k-means NMI**: k-means with k = number of modules, 10 seeds.
- **UMAP + HDBSCAN NMI**: BERTopic's default pipeline (UMAP to 5 dimensions, then HDBSCAN), 5 seeds.

## 4. Results

### H1: contract

18 of 18 checks pass on the default build. The container also serves requests with networking
disabled (`--network none`). The E5, Granite and Qwen outputs were compared with their reference
implementations through sentence-transformers and are identical (cosine 1.0).

### H2: throughput

Measured for the default, the fastest alternative, the earlier baseline and the larger models:

| CPU, container (emulated x86) | e5-small | granite-30m | bge-small | jina-code | bge-large | Qwen |
|---|---|---|---|---|---|---|
| Embed PetClinic `src` | 19 s | 10 s | 15 s | 73 s | 188 s | 747 s |
| Peak requests per second | 51 | 93 | 56 | 15 | 6 | 1.4 |
| Image size | 0.7 GB | 0.6 GB | 0.7 GB | 1.0 GB | 1.4 GB | 5.9 GB |

| CPU, native | e5-small | granite-30m | bge-small | jina-code | bge-large | Qwen |
|---|---|---|---|---|---|---|
| Embed PetClinic `src` | 5 s | 2.5 s | 4.5 s | 16 s | 46 s | 318 s |
| Peak requests per second | 209 | 303 | 208 | 69 | 22 | 5 |

Container memory under load was about 2 GB for the small models, 3.3 GB for jina-code and 11 GB
for Qwen. No CPU run returned an error.

GPU through CoreML was tried on three models: slower than CPU for bge-small (121 vs 208
requests per second), faster for bge-large (81 vs 22), and Qwen failed to load. It also crashed
on batches of 32 under sustained load, so it is not a usable path. No NVIDIA GPU was tested.

### H3 and H4: clustering quality

"Mean" is the average of the nine scores (three measures on three corpora). The other columns
are the largest corpus, ftgo. Higher is better.

| Model | Mean of 9 | kNN accuracy | k-means NMI | UMAP + HDBSCAN NMI |
|---|---|---|---|---|
| random vectors | 0.12 | 0.17 | 0.06 | 0.08 |
| **e5-small** | **0.38** | 0.65 | 0.20 | 0.31 |
| bge-small | 0.38 | 0.65 | 0.30 | 0.34 |
| bge-large | 0.37 | 0.64 | 0.22 | 0.28 |
| e5-base | 0.37 | 0.65 | 0.23 | 0.29 |
| nomic | 0.37 | 0.61 | 0.25 | 0.30 |
| Qwen | 0.36 | 0.65 | 0.22 | 0.32 |
| granite-30m | 0.35 | 0.58 | 0.18 | 0.31 |
| jina-code | 0.33 | 0.58 | 0.14 | 0.28 |
| arctic-s | 0.29 | 0.58 | 0.13 | 0.25 |
| granite-125m | 0.29 | 0.58 | 0.15 | 0.29 |
| arctic-m | 0.28 | 0.53 | 0.12 | 0.23 |

Every model is well above random. The top six are within about 0.02 of each other on the mean,
which is inside the noise of corpora this small. The one visible gap between the two leaders is
k-means on ftgo, where bge-small is ahead of e5-small (0.30 vs 0.20); e5-small is ahead on the two
smaller corpora. Size does not help: within each family the larger
model scores the same or lower (e5-base vs e5-small, granite-125m vs granite-30m, bge-large vs
bge-small), and the 1.5B-parameter Qwen sits mid-table. The code-specific model is below the
general-purpose small models. Per-corpus scores and ARI are in `results/litellm/quality.jsonl`.

## 5. Limitations

- **The module label is a proxy.** Files in different services often play the same role
  (controller, entity, configuration) and look alike, which caps the achievable score for every
  model. Absolute values are modest for that reason; the comparison between models is the useful part.
- **Small corpora, one language.** The two small corpora (50 and 66 files) are noisy, and all
  three are Java/Spring. Differences of a few hundredths are not significant. Other languages
  were not tested.
- **Not the product's pipeline.** The UMAP + HDBSCAN run uses BERTopic's default parameters on
  file-level vectors. The real pipeline's chunking, parameters and data will differ.
- **512-token inputs.** Jina, Nomic and Qwen accept much longer inputs; that advantage was not exercised.
- **Qwen was used without an instruction prefix**, as its model card recommends for documents.
- **Nomic was not checked against its reference implementation**, which requires running custom
  code from the model repository.
- **Container speed is emulated x86.** Native x86 throughput was not measured.

## 6. Conclusion

- **H1 holds.** LiteLLM serves embeddings in-process with the client contract intact and no
  outbound network access.
- **H2 holds.** The small models embed a small Spring codebase in 10 to 19 seconds in an emulated
  container and sustain 50 to 90 requests per second; native CPU is about four times faster.
- **H3 is not supported.** Neither the larger models nor the code-specific model grouped source
  files better than the small ones. Qwen costs 40 to 50 times the compute and about 9 times the
  image size for no measured gain.
- **H4 holds.** `e5-small-v2` (Microsoft, MIT) matches `bge-small` on quality and speed.

**Recommendation: `e5-small-v2`, which is the default build.** It ties the best quality score,
has the same footprint as bge-small, and comes from a US publisher. If the review prefers a
vendor with documented training-data provenance, `granite-embedding-30m-english` (IBM) is about
twice as fast and scores slightly lower. The Qwen build is kept only for the case where native
1536-dimension vectors are a hard requirement.

Two things this does not establish:

- That a small model is sufficient for the product in absolute terms. The evidence is that larger
  models do not do better on this kind of data. The remaining step is to run the product's own
  clustering on a real codebase with this image and compare against its current baseline.
- That cosine thresholds carry over. E5 similarities sit in a narrow high band (unrelated texts
  often score 0.75 or more), so any fixed similarity cut-off tuned for another model needs
  re-tuning. Rank-based logic (nearest neighbours, clustering) is unaffected.

## 7. Follow-up: summaries, search and vector dimension

### Question

Sections 1 to 6 embedded raw code and scored clustering only. A code-analysis pipeline usually embeds
LLM-written summaries of the code, and uses the vectors for three jobs: clustering, assigning an item to
its nearest group, and semantic search. Does the conclusion hold on that text and those jobs, and how many
of the 1536 dimensions does each job use?

### Method

**Text.** For the same 390 Java files (three corpora, 18 modules), a local LLM (`qwen2.5-coder:32b`) wrote
a summary of 3 to 5 sentences per file (about 80 tokens) and one search query of 6 to 12 words per file
with no identifiers from the code. Both sets are in [`tests/usecases/`](../tests/usecases/).

**Models.** `e5-small-v2` (384 native, `passage: ` and `query: ` prefixes) and `gte-Qwen2-1.5B-instruct`
(1536 native, the model card's instruction on queries), run with ONNX Runtime at full native size.

**Jobs**, scored on one pooled set of 390 files:

| Job | Measure |
|---|---|
| Similarity matching | kNN accuracy: the majority of a file's 5 nearest neighbours share its module |
| Assignment to a group | Centroid accuracy: the nearest module centroid, computed without the file, is its own module |
| Clustering | k-means NMI, and UMAP + HDBSCAN NMI (BERTopic's default pipeline) |
| Search | Each query must retrieve its own file's summary from the 390: recall@1, recall@10, MRR |

**Dimension.** Each score was recomputed after reducing the vectors to d dimensions in three ways:
keeping the first d values, a random projection (data-independent), and PCA fitted on the corpus.

### Results

Full native vectors, summaries:

| | Random vectors | e5-small (384) | gte-Qwen2 (1536) |
|---|---|---|---|
| kNN accuracy | 0.10 | 0.53 | 0.59 |
| Centroid accuracy | 0.06 | 0.66 | 0.70 |
| k-means NMI | 0.15 | 0.40 | 0.41 |
| UMAP + HDBSCAN NMI | 0.17 | 0.43 | 0.40 |
| Search recall@1 | 0.00 | 0.67 | 0.74 |
| Search recall@10 | 0.03 | 0.97 | 0.995 |
| Search MRR | 0.02 | 0.78 | 0.84 |

On raw code the two models score 0.61 and 0.61 (kNN), 0.67 and 0.69 (centroid), 0.41 and 0.40 (k-means),
0.44 and 0.47 (UMAP + HDBSCAN). Seed-to-seed standard deviation of the clustering scores is 0.003 to 0.034.

`gte-Qwen2` reduced to d dimensions, summaries:

| d | kNN (first d) | Centroid (first d) | Search recall@1 (first d) | Search recall@1 (random projection) | Search recall@10 (random projection) |
|---|---|---|---|---|---|
| 32 | 0.44 | 0.45 | 0.33 | 0.37 | 0.78 |
| 64 | 0.51 | 0.57 | 0.42 | 0.52 | 0.91 |
| 128 | 0.57 | 0.65 | 0.58 | 0.64 | 0.96 |
| 256 | 0.60 | 0.68 | 0.65 | 0.68 | 0.98 |
| 384 | 0.59 | 0.71 | 0.69 | 0.73 | 0.98 |
| 768 | 0.59 | 0.70 | 0.71 | 0.73 | 0.99 |
| 1024 | 0.58 | 0.71 | 0.74 | 0.74 | 0.99 |
| 1536 (full) | 0.59 | 0.70 | 0.74 | 0.74 | 0.995 |

The vectors of both models occupy a similar number of directions: 95% of the variance of the summary
vectors lies in 127 principal components for `e5-small` and 161 for `gte-Qwen2`. With 390 files the
ceiling for this figure is 389.

### Findings

- **Clustering does not benefit from the larger model.** The two models are within noise on k-means and
  UMAP + HDBSCAN, on summaries as on code.
- **Search is the one job where the larger model is ahead**: 7 points on recall@1 and 2 points on
  recall@10. Similarity matching and assignment are 4 to 5 points higher, at the edge of noise for 390 files.
- **The gain is not from the vector length.** `gte-Qwen2` projected to 384 dimensions still reaches 0.73
  recall@1, against 0.67 for `e5-small` at its native 384.
- **Each job stops improving at a different size**: clustering and matching at about 128 to 256
  dimensions, search recall@10 at about 256, search recall@1 at about 384 to 1024.

### Limitations

- **One small index.** 390 documents. Differences between models in search usually grow with index size.
- **Synthetic text.** One LLM wrote both the summaries and the queries, from Java/Spring code only.
- **Native runs.** These scores were computed outside the container at native dimension; the image pads
  `e5-small` to 1536 with zeros, which does not change any of them.
- **`e5-small` queries.** The image prepends `passage: ` to every input. The scores above use `query: ` on
  queries, which the image does not do by default.

### Reproducing

```bash
cd tests/usecases
pip install httpx numpy scikit-learn umap-learn joblib onnxruntime tokenizers
# clone the three codebases into ./corpora, or set CORPORA_DIR
python embed_native.py e5-small /path/to/e5-small-v2 "passage: " summary
python embed_native.py e5-small /path/to/e5-small-v2 "query: " query
python sweep.py e5-small > sweep.jsonl && python search.py e5-small > search.jsonl
python report.py sweep.jsonl search.jsonl
```
