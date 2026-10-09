"""LiteLLM custom provider that computes embeddings inside the proxy process (ONNX Runtime, CPU).

The model is read from EMBED_MODEL_DIR (baked into the image), so no network is needed at runtime.
Vectors are L2-normalised and returned at EMBED_DIM: a smaller native size is zero-padded (cosine,
dot product and L2 distance are unchanged), a larger one is truncated and re-normalised.
"""
import asyncio
import base64
import json
import os
from pathlib import Path

import litellm
import numpy as np
import onnxruntime as ort
from litellm import CustomLLM
from litellm.types.utils import EmbeddingResponse, Usage
from tokenizers import Tokenizer

MODEL_DIR = Path(os.environ.get("EMBED_MODEL_DIR", "/opt/embed-model"))
DIM = int(os.environ.get("EMBED_DIM", "1536"))
BATCH = int(os.environ.get("EMBED_BATCH", "32"))
MAX_TOKENS = int(os.environ.get("EMBED_MAX_TOKENS", "512"))
THREADS = int(os.environ.get("EMBED_THREADS", "0"))  # 0 = let ONNX Runtime decide
# ONNX Runtime execution providers in order of preference, e.g. "CoreMLExecutionProvider" on a Mac host
# or "CUDAExecutionProvider" with onnxruntime-gpu; CPU is always the fallback.
PROVIDERS = [p for p in os.environ.get("EMBED_PROVIDERS", "").split(",") if p] + ["CPUExecutionProvider"]
PROVIDER_OPTIONS = json.loads(os.environ.get("EMBED_PROVIDER_OPTIONS", "{}"))  # {"<provider>": {...}}
PROVIDER = "local-embed"

# LiteLLM rejects encoding_format / dimensions / user for providers outside this list (and drop_params
# would hide them from the handler); the OpenAI SDK sends encoding_format=base64 on every call.
if PROVIDER not in litellm.openai_compatible_providers:
    litellm.openai_compatible_providers.append(PROVIDER)


class LocalEmbed(CustomLLM):
    def __init__(self):
        super().__init__()
        self.tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = THREADS
        self.session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), opts, providers=[(p, PROVIDER_OPTIONS.get(p, {})) for p in PROVIDERS])
        self.input_names = {i.name for i in self.session.get_inputs()}
        pooling = MODEL_DIR / "1_Pooling" / "config.json"
        self.cls_pooling = pooling.exists() and json.loads(pooling.read_text()).get("pooling_mode_cls_token", False)

    def _texts(self, input):
        items = [input] if isinstance(input, str) or (input and isinstance(input[0], int)) else list(input)
        if items and not isinstance(items[0], str):  # token ids; the proxy normally decodes these before us
            items = [litellm.decode(model="gpt-3.5-turbo", tokens=t) for t in items]
        return items

    def _resize(self, v, dim):
        v = v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
        if v.shape[1] < dim:
            return np.pad(v, ((0, 0), (0, dim - v.shape[1])))
        if v.shape[1] > dim:
            v = v[:, :dim]
            return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
        return v

    def _embed(self, model, input, model_response, optional_params):
        texts = self._texts(input)
        dim = int(optional_params.get("dimensions") or DIM)
        vectors, tokens = [], 0
        for i in range(0, len(texts), BATCH):
            enc = self.tokenizer.encode_batch(texts[i:i + BATCH])
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64), "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(mask)
            hidden = self.session.run(None, feed)[0]
            if self.cls_pooling:
                pooled = hidden[:, 0]
            else:
                m = mask[..., None].astype(hidden.dtype)
                pooled = (hidden * m).sum(1) / m.sum(1).clip(min=1)
            vectors.append(self._resize(pooled, dim).astype(np.float32))
            tokens += int(mask.sum())
        out = np.concatenate(vectors) if vectors else np.zeros((0, dim), np.float32)
        as_b64 = optional_params.get("encoding_format") == "base64"
        model_response.data = [
            {"object": "embedding", "index": i, "embedding": base64.b64encode(v.tobytes()).decode() if as_b64 else v.tolist()}
            for i, v in enumerate(out)
        ]
        model_response.model = model
        model_response.usage = Usage(prompt_tokens=tokens, completion_tokens=0, total_tokens=tokens)
        return model_response

    def embedding(self, model, input, model_response: EmbeddingResponse, optional_params, **kwargs) -> EmbeddingResponse:
        return self._embed(model, input, model_response, optional_params)

    async def aembedding(self, model, input, model_response: EmbeddingResponse, optional_params, **kwargs) -> EmbeddingResponse:
        # inference is CPU-bound; keep it off the proxy's event loop
        return await asyncio.to_thread(self._embed, model, input, model_response, optional_params)


local_embed = LocalEmbed()
