# Production deployment

| Path | Role |
|------|------|
| `docker/` | **Recommended** — Compose + GHCR images (low RAM) |
| `k8s/` | Kubernetes manifests (optional; needs a cluster) |
| `k3s/` | EC2 k3s + ArgoCD bootstrap (optional; higher RAM) |
| `argocd/` | GitOps `Application` CR (optional) |

## Recommended path (no k3s): Docker Compose + GHCR CD

```text
GitHub main → build-push (GHCR) → SSH deploy-compose → docker compose pull && up
```

### One-time on EC2

1. Install Docker + Compose; clone this repo (e.g. `~/speaker-verification-system`).
2. Copy env and set secrets:

```bash
cp deploy/docker/.env.docker.example deploy/docker/.env.docker
# edit JWT_SECRET, DATABASE_PASSWORD, ALLOWED_ORIGINS, IMAGE_REGISTRY, etc.
```

3. Place model checkpoints where Compose volumes expect them (`app/server/checkpoints`, etc.).
4. Manual smoke: `GHCR_USER=... GHCR_TOKEN=... ./deploy/docker/pull-run.sh`

### GitHub setup (CD)

**Repository variable**

| Name | Value |
|------|--------|
| `ENABLE_COMPOSE_CD` | `true` |

**Repository secrets**

| Name | Purpose |
|------|---------|
| `EC2_HOST` | Instance hostname / IP |
| `EC2_USER` | SSH user (`ec2-user`, `ubuntu`, …) |
| `EC2_SSH_KEY` | Private key (full PEM) |
| `DEPLOY_PATH` | Optional; default `$HOME/speaker-verification-system` |
| `GHCR_USER` | GitHub user/org for `docker login` |
| `GHCR_TOKEN` | PAT with `read:packages` |
| `HEALTHCHECK_URL` | Optional; e.g. `http://127.0.0.1:8080/` (curl from the instance) |

Workflow: `.github/workflows/build-push.yml` → job **`deploy-compose`**.

### Docker Compose (local build smoke)

```bash
docker compose -f deploy/docker/docker-compose.yml --env-file deploy/docker/.env.docker.example up --build
```

## Optional: k3s / ArgoCD

Higher memory. See **[k3s/README.md](./k3s/README.md)**.

To re-enable Kustomize tag bumps for ArgoCD, set repo variable **`ENABLE_K8S_TAG_BUMP=true`**.

```text
GitHub main → build-push → tag-bump (Kustomize sha-*) → ArgoCD sync
```

## Kubernetes layout (optional)

- `k8s/base` — namespace, ConfigMap, Postgres, frontend, backend, ml-server, Ingress  
- `k8s/overlays/production` — GHCR image names + tags  

```bash
kubectl kustomize deploy/k8s/overlays/production
```

## Root compose

Repo-root `docker-compose.yml` remains Postgres-only for local Node/Python workflows.
