#!/usr/bin/env bash
# Builds the image, starts it, and runs the contract checks from an in-network client.
# With the model already in ./model the build reuses it instead of downloading 8 GB.
set -euo pipefail
cd "$(dirname "$0")"
E=evidence; mkdir -p "$E"
CONTEXT=(); [ -f model/config.json ] && CONTEXT=(--build-context model=./model)
docker build ${CONTEXT[@]+"${CONTEXT[@]}"} -t embed-server:qwen3-4b . 2>&1 | tail -3
docker compose --profile client build test-client 2>&1 | tail -1
docker compose up -d embed
for _ in $(seq 1 150); do curl -sf localhost:4004/health >/dev/null 2>&1 && break; sleep 2; done
curl -s localhost:4004/health | tee "$E/version.txt"; echo | tee -a "$E/version.txt"
docker image inspect embed-server:qwen3-4b --format 'image {{.Architecture}} {{.Size}} bytes' | tee -a "$E/version.txt"
docker compose run --rm test-client 2>&1 | grep -vE "Container|Network" | tee "$E/test-client.txt"
docker compose logs > "$E/compose.log" 2>&1
[ "${KEEP_UP:-}" = "1" ] || docker compose down >/dev/null 2>&1
