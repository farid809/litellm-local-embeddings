"""Shared corpus loading: same file selection and labelling as tests/quality.py.

Clone the three codebases into CORPORA_DIR (default: ./corpora next to this file):
  spring-petclinic/spring-petclinic-microservices, sqshq/piggymetrics, microservices-patterns/ftgo-application
"""
import os
import re
from collections import Counter
from pathlib import Path

S = Path(__file__).resolve().parent  # summaries/, queries/ and vectors/ live here
CORPORA_DIR = Path(os.environ.get("CORPORA_DIR", S / "corpora"))
CORPORA = ["spring-petclinic-microservices", "piggymetrics", "ftgo-application"]
MIN_FILES = 5


def load(name):
    root = CORPORA_DIR / name
    files = []
    for f in sorted(root.rglob("*.java")):
        rel = f.relative_to(root)
        if "src/main/java" not in rel.as_posix():
            continue
        label = re.sub(r"-(api|aws-lambda)$", "", rel.parts[0])
        body = "".join(l for l in f.read_text(errors="ignore").splitlines(keepends=True)
                       if not l.startswith(("package ", "import ")))
        files.append((str(rel), label, body))
    keep = {l for l, n in Counter(l for _, l, _ in files).items() if n >= MIN_FILES}
    return [f for f in files if f[1] in keep]
