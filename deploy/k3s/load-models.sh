#!/usr/bin/env bash
# Copy model weights into the ml-models / ml-checkpoints PVCs (k3s local-path).
# Run on the EC2 node after ArgoCD has created the PVCs (or after kubectl apply -k).
#
# Expected layout on the host (adjust paths):
#   MODEL_ROOT/
#     replay/best_inverted_mel_mixed_2017_pa2019.pt
#     lfcc/best_lfcc_la2019.pt          # optional if LA disabled
#     ecapa/spkrec-ecapa-voxceleb/...   # SpeechBrain ECAPA tree
#
# Usage:
#   export MODEL_ROOT=/home/ec2-user/models
#   ./deploy/k3s/load-models.sh
set -euo pipefail

NS="${NS:-voice-auth}"
MODEL_ROOT="${MODEL_ROOT:?Set MODEL_ROOT to the directory with replay/ lfcc/ ecapa/}"
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"

echo "==> Waiting for PVCs"
kubectl -n "${NS}" wait --for=condition=Bound pvc/ml-models --timeout=180s
kubectl -n "${NS}" wait --for=condition=bound pvc/ml-checkpoints --timeout=180s || true

echo "==> Starting helper pod"
kubectl -n "${NS}" delete pod model-loader --ignore-not-found
kubectl -n "${NS}" apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: model-loader
  namespace: ${NS}
spec:
  restartPolicy: Never
  containers:
    - name: loader
      image: busybox:1.36
      command: ["sleep", "3600"]
      volumeMounts:
        - name: models
          mountPath: /models
        - name: checkpoints
          mountPath: /checkpoints
  volumes:
    - name: models
      persistentVolumeClaim:
        claimName: ml-models
    - name: checkpoints
      persistentVolumeClaim:
        claimName: ml-checkpoints
EOF

kubectl -n "${NS}" wait --for=condition=Ready pod/model-loader --timeout=120s

echo "==> Copying into PVC"
kubectl -n "${NS}" exec model-loader -- mkdir -p /models/replay /models/lfcc /models/ecapa /checkpoints
# Host -> local temp via kubectl cp (from this machine)
if [[ -d "${MODEL_ROOT}/replay" ]]; then
  kubectl -n "${NS}" cp "${MODEL_ROOT}/replay/." model-loader:/models/replay/
fi
if [[ -d "${MODEL_ROOT}/lfcc" ]]; then
  kubectl -n "${NS}" cp "${MODEL_ROOT}/lfcc/." model-loader:/models/lfcc/
fi
if [[ -d "${MODEL_ROOT}/ecapa" ]]; then
  kubectl -n "${NS}" cp "${MODEL_ROOT}/ecapa/." model-loader:/models/ecapa/
fi
if [[ -d "${MODEL_ROOT}/checkpoints" ]]; then
  kubectl -n "${NS}" cp "${MODEL_ROOT}/checkpoints/." model-loader:/checkpoints/
fi

echo "==> Restarting ml-server so it remounts weights"
kubectl -n "${NS}" delete pod model-loader --ignore-not-found
kubectl -n "${NS}" rollout restart deployment/ml-server || true
kubectl -n "${NS}" get pods

echo "Done. Check: kubectl -n ${NS} logs deploy/ml-server --tail=50"
