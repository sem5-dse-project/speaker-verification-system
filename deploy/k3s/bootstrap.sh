#!/usr/bin/env bash
# Bootstrap k3s + secrets + ArgoCD on a single EC2 node (Amazon Linux / Ubuntu).
# Run as root or with sudo on the EC2 instance.
#
# Usage:
#   export GHCR_USER=your-github-username
#   export GHCR_TOKEN=ghp_...          # PAT with read:packages
#   export GIT_REPO_URL=https://github.com/sem5-dse-project/speaker-verification-system.git
#   # optional overrides:
#   export DATABASE_PASSWORD='...'
#   export JWT_SECRET='...'
#   ./deploy/k3s/bootstrap.sh
set -euo pipefail

NS="${NS:-voice-auth}"
GHCR_USER="${GHCR_USER:?Set GHCR_USER}"
GHCR_TOKEN="${GHCR_TOKEN:?Set GHCR_TOKEN (PAT with read:packages)}"
DATABASE_PASSWORD="${DATABASE_PASSWORD:-change-me-strong-password}"
JWT_SECRET="${JWT_SECRET:-change-me-to-a-long-random-string}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/sem5-dse-project/speaker-verification-system.git}"

echo "==> Installing k3s (if needed)"
if ! command -v kubectl >/dev/null 2>&1; then
  curl -sfL https://get.k3s.io | sh -
fi
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
# shellcheck disable=SC1091
[[ -f /etc/profile.d/k3s.sh ]] && source /etc/profile.d/k3s.sh || true
mkdir -p ~/.kube
cp -f /etc/rancher/k3s/k3s.yaml ~/.kube/config
chmod 600 ~/.kube/config
kubectl get nodes

echo "==> Namespace ${NS}"
kubectl create namespace "${NS}" --dry-run=client -o yaml | kubectl apply -f -

echo "==> App secrets (not stored in Git)"
kubectl -n "${NS}" create secret generic voice-auth-secrets \
  --from-literal=DATABASE_HOST=postgres \
  --from-literal=DATABASE_PORT=5432 \
  --from-literal=DATABASE_USER=voice_auth \
  --from-literal=DATABASE_PASSWORD="${DATABASE_PASSWORD}" \
  --from-literal=DATABASE_NAME=voice_authentication \
  --from-literal=JWT_SECRET="${JWT_SECRET}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "==> GHCR pull secret"
kubectl -n "${NS}" create secret docker-registry ghcr-pull-secret \
  --docker-server=ghcr.io \
  --docker-username="${GHCR_USER}" \
  --docker-password="${GHCR_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "==> Installing ArgoCD"
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
# --server-side avoids the last-applied-configuration annotation exceeding 262144 bytes on large ArgoCD CRDs
kubectl apply -n argocd --server-side --force-conflicts -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
echo "Waiting for argocd-server..."
kubectl -n argocd rollout status deployment/argocd-server --timeout=300s

echo "==> ArgoCD Application"
# If running from a git checkout on the box:
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_FILE="${SCRIPT_DIR}/../argocd/application.yaml"
if [[ -f "${APP_FILE}" ]]; then
  kubectl apply -f "${APP_FILE}"
else
  kubectl apply -f - <<EOF
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: voice-auth
  namespace: argocd
spec:
  project: default
  source:
    repoURL: ${GIT_REPO_URL}
    targetRevision: main
    path: deploy/k8s/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: ${NS}
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
EOF
fi

echo ""
echo "==> Done (base install)"
echo "ArgoCD admin password:"
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d
echo ""
echo ""
echo "Port-forward UI:  kubectl -n argocd port-forward svc/argocd-server 8080:443"
echo "Login: admin / (password above)"
echo ""
echo "Next: load model weights — see deploy/k3s/load-models.sh"
echo "Watch: kubectl -n ${NS} get pods -w"
