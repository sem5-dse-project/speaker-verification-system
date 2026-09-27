#!/usr/bin/env bash
# Minimal-RAM deploy path: plain Docker Compose pulling prebuilt GHCR images.
# No k3s / ArgoCD / Traefik control-plane overhead — just the 4 app containers.
#
# Usage (on the EC2 instance):
#   export GHCR_USER=your-github-username
#   export GHCR_TOKEN=ghp_...          # PAT with read:packages
#   cp deploy/docker/.env.docker.example deploy/docker/.env.docker   # then edit secrets
#   ./deploy/docker/pull-run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
ENV_FILE="${SCRIPT_DIR}/.env.docker"
GHCR_USER="${GHCR_USER:?Set GHCR_USER}"
GHCR_TOKEN="${GHCR_TOKEN:?Set GHCR_TOKEN (PAT with read:packages)}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE} — copy .env.docker.example and edit secrets first." >&2
  exit 1
fi

echo "==> Logging in to GHCR"
echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin

echo "==> Pulling prebuilt images (frontend, backend, ml-server, postgres)"
docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" pull postgres frontend backend ml-server

echo "==> Starting stack (no local build, no orchestrator)"
docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" up -d --no-build

docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" ps
