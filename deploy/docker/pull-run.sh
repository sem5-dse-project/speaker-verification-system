#!/usr/bin/env bash
# Minimal-RAM deploy path: plain Docker Compose pulling prebuilt GHCR images.
# No k3s / ArgoCD / Traefik control-plane overhead — just the 4 app containers.
#
# Usage (on the EC2 instance):
#   export GHCR_USER=your-github-username
#   export GHCR_TOKEN=ghp_...          # PAT with read:packages
#   export IMAGE_TAG=latest            # optional; overrides .env.docker IMAGE_TAG
#   cp deploy/docker/.env.docker.example deploy/docker/.env.docker   # then edit secrets
#   ./deploy/docker/pull-run.sh
#
# CI: GitHub Actions job deploy-compose SSHs here when ENABLE_COMPOSE_CD=true.
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

ENV_ARGS=(--env-file "${ENV_FILE}")
OVERRIDE=""
if [[ -n "${IMAGE_TAG:-}" ]]; then
  OVERRIDE="$(mktemp)"
  # shellcheck disable=SC2064
  trap 'rm -f "${OVERRIDE}"' EXIT
  printf 'IMAGE_TAG=%s\n' "${IMAGE_TAG}" >"${OVERRIDE}"
  ENV_ARGS+=(--env-file "${OVERRIDE}")
  echo "==> IMAGE_TAG override: ${IMAGE_TAG}"
fi

echo "==> Disk before cleanup"
df -h / /var/lib/docker 2>/dev/null || df -h /

# Stop stack first so previous image tags become unused and can be pruned.
# Volumes are kept (no -v) so Postgres data survives.
echo "==> Stopping Compose stack (volumes retained)"
docker compose -f "${COMPOSE_FILE}" "${ENV_ARGS[@]}" down --remove-orphans || true

echo "==> Pruning unused Docker images/build cache"
docker container prune -f >/dev/null || true
docker image prune -af >/dev/null || true
docker builder prune -af >/dev/null 2>/dev/null || true

echo "==> Disk after cleanup"
df -h / /var/lib/docker 2>/dev/null || df -h /

AVAIL_KB="$(df -Pk / | awk 'NR==2 {print $4}')"
# Require ~6 GiB free before pulling heavy ml-server (torch) layers.
if [[ -n "${AVAIL_KB}" && "${AVAIL_KB}" -lt 6000000 ]]; then
  echo "ERROR: only ${AVAIL_KB} KB free on /. Need ~6GB free to pull ml-server." >&2
  echo "On EC2: docker system df; sudo journalctl --vacuum-size=50M; df -h" >&2
  exit 1
fi

echo "==> Logging in to GHCR"
echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin

echo "==> Pulling prebuilt images (frontend, backend, ml-server, postgres)"
docker compose -f "${COMPOSE_FILE}" "${ENV_ARGS[@]}" pull postgres frontend backend ml-server

echo "==> Starting stack (no local build, no orchestrator)"
docker compose -f "${COMPOSE_FILE}" "${ENV_ARGS[@]}" up -d --no-build --remove-orphans

docker compose -f "${COMPOSE_FILE}" "${ENV_ARGS[@]}" ps
