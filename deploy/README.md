# Production deployment (Docker · k3s/ArgoCD · LB)

| Path | Role |
|------|------|
| `docker/` | Local/prod-like Compose + images |
| `k8s/` | Kubernetes manifests (Kustomize) |
| `k3s/` | **EC2 single-node bootstrap** (k3s + secrets + ArgoCD) |
| `argocd/` | GitOps `Application` CR |

## Recommended production path (this project)

**k3s on EC2 + ArgoCD + GHCR** — see **[k3s/README.md](./k3s/README.md)**.

```text
GitHub main → build-push (GHCR) → tag-bump (Kustomize sha-*) → ArgoCD sync → Traefik → voices2auth.tech
```

## Docker Compose (smoke test)

```bash
docker compose -f deploy/docker/docker-compose.yml --env-file deploy/docker/.env.docker.example up --build
```

## Kubernetes layout

- `k8s/base` — namespace, ConfigMap, Postgres, frontend, backend, ml-server, Ingress (Traefik), HPA  
- `k8s/overlays/production` — GHCR image names + tags (CI bumps `newTag`)  
- Secrets are **not** in Git — create via `k3s/bootstrap.sh` (see `k8s/base/secret.example.yaml`)

Preview:

```bash
kubectl kustomize deploy/k8s/overlays/production
```

## CI image + tag bump

Workflow: `.github/workflows/build-push.yml`

- Builds `voice-auth-frontend` / `voice-auth-backend` / `voice-auth-ml`  
- On success on `main`, commits `newTag: sha-<7>` into the production overlay for ArgoCD  

## Root compose

Repo-root `docker-compose.yml` remains Postgres-only for local Node/Python workflows.
