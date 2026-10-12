"""Write a short plain-English summary of every Java file with a local LLM (Ollama).

    python summarize.py [ollama-model]

This stands in for the LLM-written summaries a code-analysis pipeline would embed.
Output: summaries/<corpus>.json = {relative path: summary}. Resumable; the committed files were
written by qwen2.5-coder:32b.
"""
import asyncio
import json
import sys

import httpx

from common import CORPORA, S, load

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder:32b"
PROMPT = ("Summarize what this source file does in 3 to 5 sentences of plain English: its business purpose, "
          "the main operations it performs, and the data it handles. Do not quote code and do not use bullet points.\n\n")
OUT = S / "summaries"
OUT.mkdir(exist_ok=True)


async def main():
    async with httpx.AsyncClient(timeout=600) as client:
        for name in CORPORA:
            path = OUT / f"{name}.json"
            done = json.loads(path.read_text()) if path.exists() else {}
            todo = [(rel, body) for rel, _, body in load(name) if rel not in done]
            sem = asyncio.Semaphore(4)

            async def one(rel, body):
                async with sem:
                    r = await client.post("http://127.0.0.1:11434/api/chat", json={
                        "model": MODEL, "stream": False,
                        "options": {"temperature": 0, "num_predict": 300, "num_ctx": 8192},
                        "messages": [{"role": "user", "content": PROMPT + body[:12000]}]})
                    done[rel] = r.json()["message"]["content"].strip()
                    if len(done) % 10 == 0:
                        path.write_text(json.dumps(done, indent=1))
                        print(name, len(done), flush=True)

            await asyncio.gather(*[one(rel, body) for rel, body in todo])
            path.write_text(json.dumps(done, indent=1))
            print(name, "done", len(done), flush=True)


asyncio.run(main())
