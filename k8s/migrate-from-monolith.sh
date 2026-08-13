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
# is exactly the selector the new Service/vgate uses, so the old monolith stays
# behind the public endpoint and takes a share of live traffic. It also keeps
# validating the API key from the ConfigMap it mounted at startup rather than
# the Secret the new manifests supply, so most of that share fails outright.
# On a GPU cluster it holds nvidia.com/gpu=1 until deleted, which can leave the
# new workers unschedulable.
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
# GPU nodes it requires enough GPUs for both topologies at once.
#
# The cost of that ordering is that this script deletes the production entry
# point and then has to bring it back. Everything that could fail on the way
# back up is therefore done FIRST: tools are checked, the overlay is rendered
# to a temp file, and that render is inspected for the objects it is supposed
# to contain. Only then is anything deleted, and the apply reads the file that
# was already produced rather than rendering again.
#
# Usage:
#   ./k8s/migrate-from-monolith.sh --check              report orphans, change nothing
#   ./k8s/migrate-from-monolith.sh --overlay cpu        migrate, keep the old PVC
#   ./k8s/migrate-from-monolith.sh --overlay gpu --delete-pvc

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY=""
CHECK_ONLY=0
DELETE_PVC=0
RENDERED="$(mktemp)"

trap 'rm -f "$RENDERED"' EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)       CHECK_ONLY=1; shift ;;
        --overlay)     OVERLAY="${2:-}"; shift 2 ;;
        --delete-pvc)  DELETE_PVC=1; shift ;;
        -h|--help)     sed -n '16,64p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)             echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# There is deliberately no --namespace flag. The manifests pin the namespace in
# k8s/base/kustomization.yaml, and kustomize stamps it onto every rendered
# object. A flag here could only change where this script *looks*, not where
# the replacement lands — so passing one would delete the old service in one
# namespace and create the new one in another, then wait forever for a
# Deployment that was never going to appear there. The namespace is read from
# the manifests instead, which makes the two impossible to disagree.
read_manifest_namespace() {
    python3 - "$REPO_ROOT/k8s/base/kustomization.yaml" <<'PY'
import sys, yaml
doc = yaml.safe_load(open(sys.argv[1])) or {}
ns = doc.get("namespace")
if not ns:
    sys.exit("k8s/base/kustomization.yaml does not set a namespace")
print(ns)
PY
}

# ---------------------------------------------------------------------------
# Pre-flight: everything that can fail before anything is destroyed
# ---------------------------------------------------------------------------
for tool in kubectl kustomize python3; do
    command -v "$tool" >/dev/null || {
        echo "error: required tool not found on PATH: $tool" >&2
        exit 1
    }
done

if [[ $CHECK_ONLY -eq 0 && -z "$OVERLAY" ]]; then
    echo "error: --overlay cpu|gpu is required unless --check is given" >&2
    exit 2
fi
if [[ -n "$OVERLAY" && ! -d "${REPO_ROOT}/k8s/overlays/${OVERLAY}" ]]; then
    echo "error: no such overlay: ${OVERLAY}" >&2
    exit 2
fi

NS="$(read_manifest_namespace)"

# Without this, an unreachable cluster looks identical to a clean one: every
# `kubectl get` fails, no orphans are found, and --check happily reports "safe
# to apply" about a cluster it never contacted.
if ! kube_context="$(kubectl config current-context 2>/dev/null)" \
    || ! kubectl cluster-info >/dev/null 2>&1; then
    echo "error: cannot reach a cluster — kubectl has no usable context." >&2
    echo "       Nothing was inspected or changed." >&2
    exit 1
fi

echo "context:   ${kube_context}"
echo "namespace: ${NS}  (from k8s/base/kustomization.yaml)"
echo

if [[ -n "$OVERLAY" ]]; then
    echo "==> Rendering the ${OVERLAY} overlay before touching the cluster"
    if ! kustomize build "${REPO_ROOT}/k8s/overlays/${OVERLAY}" > "$RENDERED"; then
        echo "error: rendering failed; nothing has been changed" >&2
        exit 1
    fi
    # A render that succeeds but omits the replacement workloads would still
    # leave the cluster with no gateway. Check for them by name and namespace
    # rather than trusting a zero exit code.
    python3 - "$RENDERED" "$NS" <<'PY'
import sys, yaml
path, ns = sys.argv[1], sys.argv[2]
want = {("Deployment", "vgate-gateway"), ("StatefulSet", "vgate-worker"),
        ("Service", "vgate"), ("Service", "vgate-worker")}
got, wrong_ns = set(), []
for d in yaml.safe_load_all(open(path)):
    if not d:
        continue
    kind, name = d.get("kind"), d["metadata"]["name"]
    got.add((kind, name))
    got_ns = d["metadata"].get("namespace")
    if kind != "Namespace" and got_ns != ns:
        wrong_ns.append(f"{kind}/{name} in {got_ns!r}")
missing = want - got
if missing:
    sys.exit("rendered output is missing: " +
             ", ".join(f"{k}/{n}" for k, n in sorted(missing)))
if wrong_ns:
    sys.exit(f"rendered objects are not in namespace {ns!r}: " + ", ".join(wrong_ns))
print(f"    render OK: {len(got)} objects, all in namespace {ns}")
PY
    echo
fi

exists() {
    kubectl -n "$NS" get "$1" "$2" >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------
# The PVC is tracked separately from the two workloads. It is leftover storage,
# not leftover behaviour: nothing routes to it and nothing schedules because of
# it. Counting it as a blocking finding would make the default migration —
# which keeps the PVC on purpose, because it holds downloaded weights — unable
# to pass the --check it tells you to run afterwards.
BLOCKING=()
exists deployment vgate  && BLOCKING+=("deployment/vgate")
exists hpa vgate         && BLOCKING+=("hpa/vgate")

RETAINED=()
exists pvc vgate-model-cache && RETAINED+=("pvc/vgate-model-cache")

if [[ ${#BLOCKING[@]} -gt 0 ]]; then
    echo "Pre-split workloads still present:"
    printf '  %s\n' "${BLOCKING[@]}"
    echo
    echo "Pods from the old Deployment:"
    kubectl -n "$NS" get pods \
        -l app.kubernetes.io/name=vgate,app.kubernetes.io/component=gateway \
        -o custom-columns=NAME:.metadata.name,OWNER:'.metadata.ownerReferences[0].name',IP:.status.podIP \
        2>/dev/null || true
    echo
    echo "Current Service/vgate endpoints:"
    kubectl -n "$NS" get endpointslice -l kubernetes.io/service-name=vgate \
        -o jsonpath='{range .items[*]}{range .endpoints[*]}  {.addresses[0]}  {.targetRef.name}{"\n"}{end}{end}' 2>/dev/null || true
    echo
else
    echo "No pre-split workloads found."
fi

if [[ ${#RETAINED[@]} -gt 0 ]]; then
    echo "Note: pre-split storage still present (not a blocker):"
    printf '  %s\n' "${RETAINED[@]}"
    echo "  Nothing references it after migration; workers use per-pod claims."
    echo "  Pass --delete-pvc to reclaim it."
    echo
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
    if [[ ${#BLOCKING[@]} -gt 0 ]]; then
        echo "--check: pre-split workloads found; nothing was changed."
        exit 1
    fi
    echo "--check: no pre-split workloads. Safe to apply."
    exit 0
fi

# ---------------------------------------------------------------------------
# Remove, most dangerous first
# ---------------------------------------------------------------------------
if [[ ${#BLOCKING[@]} -gt 0 ]]; then
    # Record which pods belong to the old Deployment *before* deleting it, by
    # asking that Deployment for its own selector rather than assuming one. The
    # obvious guess — name=vgate plus component=gateway — matches the new
    # gateway pods just as exactly, so waiting on it would never finish. The
    # old selector additionally carries managed-by=kustomize, because the
    # pre-split kustomization used the deprecated `commonLabels`, which
    # injected that label into selectors; the current `labels` transformer
    # deliberately does not.
    OLD_PODS=""
    old_selector=""
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
        # Without this, the apply below can race a still-terminating pod back
        # into the new Service's endpoints.
        for pod in $OLD_PODS; do
            kubectl -n "$NS" wait --for=delete "$pod" --timeout=120s || true
        done
        echo "    remaining from the old Deployment: $(
            kubectl -n "$NS" get pods --selector="$old_selector" --no-headers 2>/dev/null | wc -l
        )"
    fi
fi

if [[ $DELETE_PVC -eq 1 ]]; then
    echo "==> Deleting the old model cache PVC"
    kubectl -n "$NS" delete pvc vgate-model-cache --ignore-not-found
fi

# ---------------------------------------------------------------------------
# Apply the render produced before any deletion happened
# ---------------------------------------------------------------------------
echo
echo "==> Applying the ${OVERLAY} overlay (pre-rendered)"
kubectl apply -f "$RENDERED"

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
if [[ $DELETE_PVC -eq 0 && ${#RETAINED[@]} -gt 0 ]]; then
    echo "Migration complete. pvc/vgate-model-cache was kept; --check will report"
    echo "it as a note and still exit 0."
else
    echo "Migration complete."
fi
