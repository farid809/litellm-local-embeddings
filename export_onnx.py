"""Download gte-Qwen2-1.5B-instruct and convert it to ONNX for the Qwen image.

    python export_onnx.py ./qwen-model            # downloads ~7 GB, needs ~20 GB RAM, takes a few minutes

Needs: python 3.11/3.12, torch==2.5.1 transformers==4.44.2 onnx "numpy<2"
The model runs its own modeling code from the Hugging Face repo (trust_remote_code), pinned to REV below.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import onnx
import torch
from huggingface_hub import snapshot_download
import transformers.dynamic_module_utils as dyn
from transformers import AutoModel

# the model code imports flash_attn only when a CUDA build is present, but transformers insists on it anyway
_get_imports = dyn.get_imports
dyn.get_imports = lambda f: [m for m in _get_imports(f) if m != "flash_attn"]

REPO, REV = "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "a9af15a6372d7d6b25e9fb07c2ccb9e1fe645644"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
src = sys.argv[2] if len(sys.argv) > 2 else snapshot_download(
    REPO, revision=REV, allow_patterns=["*.json", "*.safetensors", "*.py", "merges.txt", "1_Pooling/*"])


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


model = AutoModel.from_pretrained(src, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager").eval()
ids = torch.ones((2, 8), dtype=torch.long)
with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
    torch.onnx.export(
        Encoder(model), (ids, torch.ones_like(ids)), f"{tmp}/model.onnx", opset_version=17,
        input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
        dynamic_axes={n: {0: "batch", 1: "tokens"} for n in ("input_ids", "attention_mask", "last_hidden_state")})
    del model
    # the weights exceed ONNX's 2 GB file limit: keep them in one side file next to model.onnx
    onnx.save(onnx.load(f"{tmp}/model.onnx"), str(out / "model.onnx"), save_as_external_data=True,
              all_tensors_to_one_file=True, location="model.onnx_data")

for f in ("tokenizer.json", "tokenizer_config.json", "1_Pooling/config.json"):
    (out / f).parent.mkdir(exist_ok=True)
    shutil.copy(Path(src) / f, out / f)
print("wrote", sorted(p.name for p in out.iterdir()))
