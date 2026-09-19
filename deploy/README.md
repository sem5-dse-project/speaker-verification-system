# Production deployment (Docker · Kubernetes · ArgoCD · LB)

This folder scaffolds a production-style deploy path for the voice auth stack:

| Component | Role |
|-----------|------|
| `docker/` | Multi-service images + Compose for local/prod-like runs |
| `k8s/` | Kubernetes manifests (Deployments, Services, Ingress/ALB, HPA) |
| `argocd/` | GitOps `Application` CR for continuous sync |

Existing EC2 `git pull` can remain for demos; this path is for cluster deploy.

## Architecture

```text
Internet → Ingress / ALB
            ├─ /api, /uploads → backend (2+ pods)
            ├─ /              → frontend (2+ pods)
            └─ (ClusterIP)    → ml-server ← backend
                                  ↓
                               Postgres (RDS recommended)
```

## Step 1 — Docker images (local)

From **repo root**:

```bash
# Optional env file
cp deploy/docker/.env.docker.example deploy/docker/.env.docker

# Build & run full stack (Postgres + ML + backend + frontend)
docker compose -f deploy/docker/docker-compose.yml --env-file deploy/docker/.env.docker up --build

# App: http://localhost:8080
# API:  http://localhost:5000/api/health
# ML:   http://localhost:8000/health
```

Place ML weights under `app/server/checkpoints/` (or `chekpoints/` renamed to `checkpoints`).  
Replay/LA experiment `.pt` files must exist under `replay-cnn-baseline/experiments/...` (bind-mounted).

Build one image:

```bash
docker build -f deploy/docker/frontend.Dockerfile -t voice-auth/frontend:local .
docker build -f deploy/docker/backend.Dockerfile -t voice-auth/backend:local .
docker build -f deploy/docker/ml-server.Dockerfile -t voice-auth/ml-server:local .
```

## Step 2 — Push images (CI)

Workflow: `.github/workflows/build-push.yml`  
On push to `main` (relevant paths), builds and pushes to **GHCR**:

- `ghcr.io/sem5-dse-project/voice-auth-frontend`
- `ghcr.io/sem5-dse-project/voice-auth-backend`
- `ghcr.io/sem5-dse-project/voice-auth-ml`

Update `OWNER` in the workflow and image names in `deploy/k8s/overlays/production/kustomization.yaml` if your org/user differs.  
Packages must allow GitHub Actions to write (`packages: write` is set).

For **ECR** instead: change registry login + image names in the workflow and Kustomize overlay.

## Step 3 — Kubernetes

Prereqs: cluster (EKS/k3s), `kubectl`, Ingress controller (AWS LB Controller or nginx).

```bash
# Preview
kubectl kustomize deploy/k8s/overlays/production

# Apply (after fixing image registry + secrets)
kubectl apply -k deploy/k8s/overlays/production
```

**Before production:**

1. Replace Secret values (use External Secrets / Sealed Secrets — do not commit real passwords).  
2. Point `images:` in the production overlay to your pushed digests/tags.  
3. Set Ingress `host` + TLS certificate ARN.  
4. Populate PVCs: copy replay/LFCC/ECAPA/Wave-U-Net weights into `/models` and `/app/checkpoints`.  
5. Prefer **RDS Postgres** over in-cluster DB; update `DATABASE_HOST` in the Secret.  
6. Backend uploads need **ReadWriteMany** (EFS) if `replicas > 1`.

## Step 4 — ArgoCD (GitOps)

1. Install ArgoCD in the cluster.  
2. Edit `deploy/argocd/application.yaml` `repoURL` / `targetRevision` if needed.  
3. Apply:

```bash
kubectl apply -f deploy/argocd/application.yaml
```

ArgoCD syncs `deploy/k8s/overlays/production` from Git.  
CD flow: merge to `main` → CI pushes images → update image tags in Kustomize (manual or automator) → ArgoCD rolls out.

## Load balancing

- **Ingress + ALB** annotations are in `k8s/base/ingress.yaml`.  
- **HPA** scales frontend/backend on CPU.  
- **ml-server** defaults to 1 replica (heavy); scale carefully.

## Security notes

- Never commit `.env` / real JWT / DB passwords.  
- Keep `.pt` weights out of git; mount from PVC/S3.  
- Backend CORS uses `ALLOWED_ORIGINS` (comma-separated).

## Root compose

`docker-compose.yml` at repo root remains **Postgres-only** for local Node/Python workflows.  
Full stack Compose lives under `deploy/docker/`.
