# k3s on EC2 (GitOps with ArgoCD)

Single-node path: **EC2 → k3s → ArgoCD → GHCR images**.

## Prerequisites

- EC2 with enough disk/RAM for ML (recommend ≥16 GB RAM, ≥50 GB disk)
- Security group: **80/443** open (Traefik); **22** for SSH
- DNS `voices2auth.tech` → EC2 public IP (or Elastic IP)
- GHCR packages: `voice-auth-frontend`, `voice-auth-backend`, `voice-auth-ml`
- GitHub PAT with `read:packages` (for pulling private packages)

Stop or disable old systemd `voice-backend` / `model-server` if they bind the same ports.

## 1. Bootstrap (on EC2)

```bash
git clone https://github.com/sem5-dse-project/speaker-verification-system.git
cd speaker-verification-system
chmod +x deploy/k3s/*.sh

export GHCR_USER=your-github-username
export GHCR_TOKEN=ghp_xxxx
export DATABASE_PASSWORD='strong-db-password'
export JWT_SECRET='long-random-jwt'

sudo -E ./deploy/k3s/bootstrap.sh
```

This installs k3s, creates `voice-auth-secrets` + `ghcr-pull-secret`, installs ArgoCD, and applies `deploy/argocd/application.yaml`.

## 2. Load model weights

Prepare a folder on the instance (example):

```text
/home/ec2-user/models/
  replay/best_inverted_mel_mixed_2017_pa2019.pt
  lfcc/best_lfcc_la2019.pt
  ecapa/spkrec-ecapa-voxceleb/   # SpeechBrain savedir tree
```

```bash
export MODEL_ROOT=/home/ec2-user/models
sudo -E ./deploy/k3s/load-models.sh
```

Paths must match ConfigMap keys in `deploy/k8s/base/config.yaml`.

## 3. Verify

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl -n voice-auth get pods
kubectl -n voice-auth get ingress
curl -sS http://127.0.0.1/api/health   # via Traefik once DNS/hosts point here
```

ArgoCD UI:

```bash
kubectl -n argocd port-forward svc/argocd-server 8080:443
# admin / (password printed by bootstrap)
```

## 4. Continuous deploy (already in CI)

1. Push to `main` (app/docker paths) → **Build and push images** → GHCR  
2. Job **tag-bump** commits `newTag: sha-xxxxxxx` in `deploy/k8s/overlays/production/kustomization.yaml`  
3. ArgoCD self-heals and rolls Deployments  

If branch protection blocks the bot push, allow GitHub Actions to write to `main`, or use a PAT secret `TAG_BUMP_TOKEN` (wire later).

## Secrets

App secrets are **not** in Git (`secret.example.yaml` is a template only).  
Created by `bootstrap.sh`. ArgoCD `ignoreDifferences` avoids fighting Secret data.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| `ImagePullBackOff` | `ghcr-pull-secret`, package visibility, PAT scopes |
| `ml-server` CrashLoop | models missing on PVC; `load-models.sh` |
| Ingress 404 | DNS → EC2; `kubectl -n voice-auth describe ingress` |
| DB auth errors | `voice-auth-secrets` password vs postgres first boot (wipe PVC if you changed password after first start) |
