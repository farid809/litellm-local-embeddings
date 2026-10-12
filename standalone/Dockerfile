# Standalone embedding server (no proxy): FastAPI + sentence-transformers, model baked into the image.
#   build:   docker build -t embed-server:qwen3-4b .
#   builder without huggingface.co access, or to reuse a local copy: put the model files in ./model
#            and add --build-context model=./model
#   NVIDIA:  add --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 and run with --gpus all
ARG PYTHON_IMAGE=python:3.12-slim

FROM scratch AS model
ARG EMBED_MODEL_REPO=Qwen/Qwen3-Embedding-4B
ARG EMBED_MODEL_REV=5cf2132abc99cad020ac570b19d031efec650f2b
ARG HF=https://huggingface.co/${EMBED_MODEL_REPO}/resolve/${EMBED_MODEL_REV}
ADD ${HF}/config.json ${HF}/config_sentence_transformers.json ${HF}/modules.json ${HF}/generation_config.json /
ADD ${HF}/tokenizer.json ${HF}/tokenizer_config.json ${HF}/vocab.json ${HF}/merges.txt /
ADD ${HF}/1_Pooling/config.json /1_Pooling/config.json
ADD ${HF}/model.safetensors.index.json ${HF}/model-00001-of-00002.safetensors ${HF}/model-00002-of-00002.safetensors /

FROM ${PYTHON_IMAGE}
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.14.1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX} \
 && pip install --no-cache-dir -r requirements.txt
# token-id input is decoded with this encoding; fetch it now so the container needs no network at runtime
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
COPY --from=model / /opt/embed-model/
COPY server.py .
ENV EMBED_MODEL_DIR=/opt/embed-model EMBED_MODEL_NAME=Qwen3-Embedding-4B EMBED_DIM=1536
USER nobody
EXPOSE 4000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4000"]
