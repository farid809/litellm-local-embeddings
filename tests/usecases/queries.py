"""Write one natural-language search query per Java file with a local LLM (Ollama).

    python queries.py [ollama-model]

The query is what a developer might type into a semantic search box to find that file's
functionality, without identifiers. Output: queries/<corpus>.json = {relative path: query}.
"""
import asyncio
import json
import sys

import httpx

from common import CORPORA, S, load

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder:32b"
PROMPT = ("Write one short search query (6 to 12 words) that a developer who has never seen this code would type "
          "to find the functionality in this file. Describe the behaviour in plain English. Do not use any class, "
          "method, package or variable names from the file. Reply with the query only.\n\n")
OUT = S / "queries"
OUT.mkdir(exist_ok=True)


async def main():
    async with httpx.AsyncClient(timeout=900) as client:
        for name in CORPORA:
            path = OUT / f"{name}.json"
            done = json.loads(path.read_text()) if path.exists() else {}
            sem = asyncio.Semaphore(4)

            async def one(rel, body):
                async with sem:
                    r = await client.post("http://127.0.0.1:11434/api/chat", json={
                        "model": MODEL, "stream": False,
                        "options": {"temperature": 0, "num_predict": 40, "num_ctx": 8192},
                        "messages": [{"role": "user", "content": PROMPT + body[:12000]}]})
                    done[rel] = r.json()["message"]["content"].strip().strip('"')

            await asyncio.gather(*[one(rel, body) for rel, _, body in load(name) if rel not in done])
            path.write_text(json.dumps(done, indent=1))
            print(name, "done", len(done), flush=True)


asyncio.run(main())
