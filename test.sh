#!/usr/bin/env bash
# Builds the image, starts it, and runs the contract checks from an in-network client.
set -euo pipefail
cd "$(dirname "$0")"
export ARCH=${ARCH:-amd64}
E=evidence/$ARCH; mkdir -p "$E"
docker compose build 2>&1 | tail -3
docker compose up -d litellm
for _ in $(seq 1 90); do curl -sf localhost:4002/health/liveliness >/dev/null 2>&1 && break; sleep 2; done
docker compose exec -T litellm python -c "
from importlib.metadata import version
import platform
print('LiteLLM', version('litellm'), '| onnxruntime', version('onnxruntime'), '|', platform.machine())" | tee "$E/version.txt"
docker image inspect "litellm-local-embed:1.83.14-$ARCH" --format 'image {{.Architecture}} {{.Size}} bytes' | tee -a "$E/version.txt"
docker compose run --rm test-client 2>&1 | grep -vE "Container|Network" | tee "$E/test-client.txt"
docker compose logs > "$E/compose.log" 2>&1
[ "${KEEP_UP:-}" = "1" ] || docker compose down >/dev/null 2>&1
