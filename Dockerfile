# LiteLLM proxy + an embedding model that runs inside the proxy process (no sidecar, no egress at runtime).
#   local:   docker build --platform linux/amd64 -t litellm-local-embed:1.83.14-amd64 .   (or ./test.sh)
#   cluster: docker buildx build --platform linux/amd64 -t <registry>/litellm-local-embed:1.83.14 --push .
#   builder without huggingface.co access: put model.onnx, tokenizer.json, 1_Pooling/config.json in ./model
#            and add --build-context model=./model
ARG LITELLM_IMAGE=ghcr.io/berriai/litellm:main-v1.83.14-stable
ARG UV_IMAGE=ghcr.io/astral-sh/uv:latest

FROM scratch AS model
# Default: Microsoft E5-small-v2 (MIT). Any repo with an ONNX file, tokenizer.json and 1_Pooling/config.json works.
ARG EMBED_MODEL_REPO=intfloat/e5-small-v2
ARG EMBED_MODEL_REV=ffb93f3bd4047442299a41ebb6fa998a38507c52
# path of the ONNX file inside the repo: model.onnx at the root for some publishers, onnx/model.onnx for others
ARG EMBED_ONNX_PATH=model.onnx
ADD https://huggingface.co/${EMBED_MODEL_REPO}/resolve/${EMBED_MODEL_REV}/${EMBED_ONNX_PATH} /model.onnx
ADD https://huggingface.co/${EMBED_MODEL_REPO}/resolve/${EMBED_MODEL_REV}/tokenizer.json /tokenizer.json
ADD https://huggingface.co/${EMBED_MODEL_REPO}/resolve/${EMBED_MODEL_REV}/1_Pooling/config.json /1_Pooling/config.json

FROM ${UV_IMAGE} AS uv

FROM ${LITELLM_IMAGE}
# the image has no pip; numpy, tokenizers and tiktoken are already in its venv
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    uv pip install --python /app/.venv/bin/python --no-cache onnxruntime
COPY --from=model / /opt/embed-model/
# LiteLLM loads custom handlers from the config file's directory only
COPY config.yaml local_embed.py /etc/litellm/
# text prepended to every input; E5 expects "passage: ". Pass --build-arg EMBED_PREFIX= for models without one.
ARG EMBED_PREFIX="passage: "
ENV EMBED_MODEL_DIR=/opt/embed-model EMBED_DIM=1536 EMBED_PREFIX=${EMBED_PREFIX}
CMD ["--config", "/etc/litellm/config.yaml", "--port", "4000"]
