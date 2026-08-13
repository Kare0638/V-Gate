#!/usr/bin/env bash
# Copyright 2025 the V-Gate authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Migrate a cluster running the pre-split manifests to the gateway/worker split.
#
# WHY THIS EXISTS
#
# `kubectl apply` creates and updates; it never deletes objects that were
# removed from the manifests. The split renamed the workloads, so applying the
# new manifests over an existing deployment leaves three orphans behind:
#
#   Deployment/vgate               the old single-process service
#   HorizontalPodAutoscaler/vgate  still scaling that Deployment
#   PersistentVolumeClaim/vgate-model-cache
#
# The Deployment is the dangerous one. Its pods carry
# app.kubernetes.io/name=vgate and app.kubernetes.io/component=gateway, which
# is exactly the selector the new Service/vgate uses. The old monolith
# therefore stays behind the public endpoint and takes a share of live traffic
# — returning valid completions from a process that has no worker routing, no
# per-worker health tracking, and its own copy of the cache. Nothing errors, so
# nothing draws attention to it.
#
# On a GPU cluster it is also a scheduling problem: the old Deployment holds
# nvidia.com/gpu=1 until deleted, so the new workers may have nowhere to run.
#
# WHY NOT --prune
#
# `kubectl apply --prune` is the mechanism designed for this, but it is still
# Alpha and carries an explicit "do not use unless you are aware of what the
# current state is" disclaimer in kubectl's own help. Pruning by label would
# also target every object carrying the selector, which on a bad match means
# deleting live workloads. The orphan set here is small, known, and fixed, so
# naming the three objects is both safer and easier to review than delegating
# to a mechanism whose blast radius depends on a label expression.
#
# ORDER MATTERS
#
# Old objects are removed *before* the new ones are applied, not after. The
# reverse order looks appealing because it avoids downtime, but it puts the old
# monolith and the new gateway behind the same Service at the same time, and on
# GPU nodes it requires enough GPUs for both topologies at once. The cost of
# doing it this way is a short window with no gateway running.
#
# Usage:
#   ./k8s/migrate-from-monolith.sh --check              report orphans, change nothing
#   ./k8s/migrate-from-monolith.sh --overlay cpu        migrate, keep the old PVC
#   ./k8s/migrate-from-monolith.sh --overlay gpu --delete-pvc

set -euo pipefail

NS=vgate
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY=""
CHECK_ONLY=0
DELETE_PVC=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)       CHECK_ONLY=1; shift ;;
        --overlay)     OVERLAY="${2:-}"; shift 2 ;;
        --delete-pvc)  DELETE_PVC=1; shift ;;
        --namespace)   NS="${2:-}"; shift 2 ;;
        -h|--help)     sed -n '16,60p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)             echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ $CHECK_ONLY -eq 0 && -z "$OVERLAY" ]]; then
    echo "error: --overlay cpu|gpu is required unless --check is given" >&2
    exit 2
fi
if [[ -n "$OVERLAY" && ! -d "${REPO_ROOT}/k8s/overlays/${OVERLAY}" ]]; then
    echo "error: no such overlay: ${OVERLAY}" >&2
    exit 2
fi

command -v kubectl >/dev/null || { echo "kubectl not found on PATH" >&2; exit 1; }

exists() {
    kubectl -n "$NS" get "$1" "$2" >/dev/null 2>&1
}

echo "context:   $(kubectl config current-context)"
echo "namespace: ${NS}"
echo

# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------
FOUND=()
exists deployment vgate                && FOUND+=("deployment/vgate")
exists hpa vgate                       && FOUND+=("hpa/vgate")
exists pvc vgate-model-cache           && FOUND+=("pvc/vgate-model-cache")

if [[ ${#FOUND[@]} -eq 0 ]]; then
    echo "No pre-split objects found. Nothing to migrate."
    if [[ $CHECK_ONLY -eq 0 ]]; then
        echo
        echo "Applying the ${OVERLAY} overlay anyway, since a fresh install is safe."
        kustomize build "${REPO_ROOT}/k8s/overlays/${OVERLAY}" | kubectl apply -f -
    fi
    exit 0
fi

echo "Pre-split objects still present:"
printf '  %s\n' "${FOUND[@]}"
echo

# Show whether the old pods are currently behind the public Service. This is
# the failure the migration exists to prevent, so it is worth printing rather
# than asserting.
if exists deployment vgate; then
    echo "Pods from the old Deployment, and whether Service/vgate selects them:"
    kubectl -n "$NS" get pods \
        -l app.kubernetes.io/name=vgate,app.kubernetes.io/component=gateway \
        -o custom-columns=NAME:.metadata.name,OWNER:'.metadata.ownerReferences[0].name',IP:.status.podIP \
        2>/dev/null || true
    echo
    echo "Current Service/vgate endpoints:"
    kubectl -n "$NS" get endpointslice -l kubernetes.io/service-name=vgate \
        -o jsonpath='{range .items[*]}{range .endpoints[*]}  {.addresses[0]}  {.targetRef.name}{"\n"}{end}{end}' 2>/dev/null || true
    echo
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "--check given; nothing was changed."
    exit 1
fi

# ---------------------------------------------------------------------------
# Remove, oldest hazard first
# ---------------------------------------------------------------------------
# Record which pods belong to the old Deployment *before* deleting it, by
# asking that Deployment for its own selector rather than assuming one. The
# obvious guess — name=vgate plus component=gateway — matches the new gateway
# pods just as exactly, so waiting on it would either never finish or would
# wait on the wrong pods. The old selector additionally carries
# managed-by=kustomize, because the pre-split kustomization used the deprecated
# `commonLabels`, which injected that label into selectors; the current
# `labels` transformer deliberately does not.
OLD_PODS=""
if exists deployment vgate; then
    old_selector="$(kubectl -n "$NS" get deployment vgate \
        -o go-template='{{range $k, $v := .spec.selector.matchLabels}}{{$k}}={{$v}},{{end}}' \
        | sed 's/,$//')"
    echo "==> Old Deployment selector: ${old_selector}"
    OLD_PODS="$(kubectl -n "$NS" get pods --selector="$old_selector" -o name 2>/dev/null || true)"
    echo "    pods it owns: ${OLD_PODS:-<none>}"
    echo
fi

echo "==> Deleting the autoscaler first, so it cannot act on the Deployment mid-removal"
kubectl -n "$NS" delete hpa vgate --ignore-not-found

echo "==> Deleting the old Deployment (releases its GPU and drops its pods from Service/vgate)"
kubectl -n "$NS" delete deployment vgate --ignore-not-found

if [[ -n "$OLD_PODS" ]]; then
    echo "==> Waiting for those exact pods to go away"
    # Without this, the apply below can race a still-terminating pod back into
    # the new Service's endpoints.
    for pod in $OLD_PODS; do
        kubectl -n "$NS" wait --for=delete "$pod" --timeout=120s || true
    done
    echo "    remaining from the old Deployment: $(
        kubectl -n "$NS" get pods --selector="$old_selector" --no-headers 2>/dev/null | wc -l
    )"
fi

if [[ $DELETE_PVC -eq 1 ]]; then
    echo "==> Deleting the old model cache PVC"
    # Not done by default: it holds downloaded model weights. Removing it is
    # safe but the next worker start re-downloads them.
    kubectl -n "$NS" delete pvc vgate-model-cache --ignore-not-found
else
    echo "==> Leaving pvc/vgate-model-cache in place"
    echo "    Nothing references it after this migration; workers use their own"
    echo "    per-pod claims. Re-run with --delete-pvc to reclaim the storage."
fi

# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
echo
echo "==> Applying the ${OVERLAY} overlay"
kustomize build "${REPO_ROOT}/k8s/overlays/${OVERLAY}" | kubectl apply -f -

echo
echo "==> Waiting for both roles"
kubectl -n "$NS" rollout status deployment/vgate-gateway --timeout=180s
kubectl -n "$NS" rollout status statefulset/vgate-worker --timeout=300s

echo
echo "==> Service/vgate now selects only the new gateway:"
kubectl -n "$NS" get endpointslice -l kubernetes.io/service-name=vgate \
    -o jsonpath='{range .items[*]}{range .endpoints[*]}  {.addresses[0]}  {.targetRef.name}{"\n"}{end}{end}'
echo
kubectl -n "$NS" get pods -o wide

echo
echo "Migration complete. Re-run with --check to confirm no pre-split objects remain."
