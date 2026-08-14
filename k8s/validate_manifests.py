#!/usr/bin/env python3
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

"""
Semantic checks on the rendered Kubernetes manifests.

kubeconform already answers "is this valid Kubernetes YAML". It cannot answer
"does this deploy the architecture the project actually has" — and that is the
failure these manifests already suffered once: they described a single-process
service for months after the gateway and workers were split apart, while
staying schema-valid the whole time.

Each check below encodes an invariant that, if broken, produces a cluster that
comes up green and behaves wrongly:

  * A gateway that requests GPU or mounts a model cache is holding resources it
    can never use, because worker endpoints make it forward every inference.
  * A worker Service with a cluster IP would put kube-proxy in front of the
    pool, hiding individual workers from the gateway's registry and silently
    disabling per-worker health tracking and routing.
  * A gateway Service that selects on name alone would put worker pods behind
    the public endpoint, exposing /internal/generate to clients.
  * An endpoint list that disagrees with the worker replica count means either
    a worker receives no traffic or the gateway probes a pod that will never
    exist. This one is the specific cost of a static registry and is checked
    until DNS-based discovery replaces the list.
  * A "$(" left in a rendered env value means a Kubernetes interpolation did
    not resolve. Kubernetes passes those through literally instead of failing,
    so it cannot be caught at runtime by watching for errors.

Usage:  python k8s/validate_manifests.py [overlay ...]
"""

import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is in requirements.txt
    sys.exit("PyYAML is required: pip install pyyaml")

K8S_DIR = Path(__file__).resolve().parent
DEFAULT_OVERLAYS = ["cpu", "gpu"]


class Failure(Exception):
    pass


def render(overlay: str) -> list[dict]:
    """Run `kustomize build` and return the parsed documents."""
    path = K8S_DIR / "overlays" / overlay
    try:
        out = subprocess.run(
            ["kustomize", "build", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
    except FileNotFoundError:
        sys.exit("kustomize not found on PATH")
    except subprocess.CalledProcessError as exc:
        raise Failure(f"kustomize build failed:\n{exc.stderr}") from exc
    return [d for d in yaml.safe_load_all(out) if d]


def find(docs: list[dict], kind: str, name: str) -> dict:
    for d in docs:
        if d.get("kind") == kind and d["metadata"]["name"] == name:
            return d
    raise Failure(f"{kind}/{name} is missing from the rendered output")


def find_by_component(docs: list[dict], kind: str, component: str) -> dict:
    """
    Locate a resource by its role rather than its name.

    Checks that exist to catch a rename cannot look the resource up by name:
    the lookup fails first, and the build breaks with "missing from the
    rendered output" instead of the message that would point at the rename.
    """
    matches = [
        d for d in docs
        if d.get("kind") == kind
        and d["metadata"].get("labels", {}).get("app.kubernetes.io/component") == component
    ]
    if not matches:
        raise Failure(f"no {kind} labelled component={component} in the rendered output")
    if len(matches) > 1:
        names = ", ".join(m["metadata"]["name"] for m in matches)
        raise Failure(f"more than one {kind} labelled component={component}: {names}")
    return matches[0]


def container_of(workload: dict) -> dict:
    containers = workload["spec"]["template"]["spec"]["containers"]
    if len(containers) != 1:
        raise Failure(
            f"{workload['metadata']['name']} has {len(containers)} containers; "
            "these checks assume one"
        )
    return containers[0]


def env_value(container: dict, key: str) -> str | None:
    for entry in container.get("env", []):
        if entry["name"] == key:
            return entry.get("value")
    return None


def check_gateway_holds_no_model(docs: list[dict]) -> None:
    gw = find(docs, "Deployment", "vgate-gateway")
    container = container_of(gw)

    for section in ("requests", "limits"):
        resources = container.get("resources", {}).get(section, {})
        gpus = [k for k in resources if "gpu" in k.lower()]
        if gpus:
            raise Failure(
                f"gateway requests {gpus} in resources.{section}; it forwards "
                "inference and never loads a model"
            )

    mounts = [m["mountPath"] for m in container.get("volumeMounts", [])]
    if any("huggingface" in m or "cache" in m for m in mounts):
        raise Failure(f"gateway mounts a model cache: {mounts}")

    if env_value(container, "VGATE_ROLE") != "gateway":
        raise Failure("gateway pod does not set VGATE_ROLE=gateway")


def check_worker_service_is_headless(docs: list[dict]) -> None:
    svc = find_by_component(docs, "Service", "worker")
    if svc["spec"].get("clusterIP") != "None":
        raise Failure(
            "worker Service is not headless; kube-proxy would load balance the "
            "pool and the gateway registry could not address workers directly"
        )


def check_gateway_service_excludes_workers(docs: list[dict]) -> None:
    svc = find(docs, "Service", "vgate")
    selector = svc["spec"]["selector"]
    if selector.get("app.kubernetes.io/component") != "gateway":
        raise Failure(
            f"gateway Service selector {selector} does not pin "
            "component=gateway; worker pods could land behind the public "
            "endpoint and expose /internal/generate"
        )


def check_discovery_targets_the_worker_service(docs: list[dict]) -> None:
    """
    The gateway must discover workers through the Service that actually fronts
    them, in the namespace they are deployed into.

    This replaces an earlier check that compared a static endpoint list against
    the replica count. That coupling is gone — membership is resolved at
    runtime, so scaling no longer requires editing the gateway — but a new one
    took its place: the DNS name is a string, and nothing else in the manifests
    forces it to name the worker Service. Renaming the Service, or moving the
    deployment to another namespace, would leave the gateway resolving a name
    that does not exist and reporting an empty pool.
    """
    gw = find(docs, "Deployment", "vgate-gateway")
    # By label, not by name: this check exists to catch a Service rename, and a
    # name lookup would fail on that case before reaching the comparison.
    sts = find_by_component(docs, "StatefulSet", "worker")
    svc = find_by_component(docs, "Service", "worker")
    container = container_of(gw)

    dns_name = env_value(container, "VGATE_WORKER__DISCOVERY__DNS_NAME")
    if not dns_name:
        raise Failure(
            "gateway sets no VGATE_WORKER__DISCOVERY__DNS_NAME, so worker "
            "membership cannot follow the cluster"
        )

    # A static list alongside discovery is only a startup seed, and one that
    # disagrees with the Service would put endpoints in the registry that
    # discovery then removes on its first tick — churn that looks like workers
    # flapping.
    if env_value(container, "VGATE_WORKER__ENDPOINTS"):
        raise Failure(
            "gateway sets both VGATE_WORKER__ENDPOINTS and a discovery name; "
            "the static list would be replaced on the first resolve"
        )

    namespace = svc["metadata"].get("namespace")
    expected = f"{svc['metadata']['name']}.{namespace}.svc.cluster.local"
    if dns_name != expected:
        raise Failure(
            f"gateway discovers workers at {dns_name!r}, which is not the "
            f"worker Service's name. Expected {expected!r}."
        )

    # The Service must front the StatefulSet's pods, or the name resolves to
    # the wrong set.
    if sts["spec"]["serviceName"] != svc["metadata"]["name"]:
        raise Failure(
            f"StatefulSet serviceName is {sts['spec']['serviceName']!r} but the "
            f"discovered Service is {svc['metadata']['name']!r}"
        )

    port = env_value(container, "VGATE_WORKER__DISCOVERY__PORT")
    svc_port = svc["spec"]["ports"][0]["port"]
    if port and int(port) != svc_port:
        raise Failure(
            f"gateway discovers workers on port {port} but the worker Service "
            f"exposes {svc_port}"
        )


def check_no_unresolved_interpolation(docs: list[dict]) -> None:
    pattern = re.compile(r"\$\(")
    for doc in docs:
        spec = doc.get("spec", {}).get("template", {}).get("spec")
        if not spec:
            continue
        for container in spec.get("containers", []):
            for entry in container.get("env", []):
                value = entry.get("value")
                if value and pattern.search(value):
                    raise Failure(
                        f"{doc['kind']}/{doc['metadata']['name']} env "
                        f"{entry['name']} contains {value!r}. Kubernetes leaves "
                        "unresolvable $(VAR) references as literal text rather "
                        "than failing, so this cannot be detected at runtime"
                    )


def check_images_are_pinned(docs: list[dict]) -> None:
    """
    No moving tags, and never an implicit pull policy.

    ":latest" and imagePullPolicy: IfNotPresent is the pairing that fails
    silently: the node keeps whatever it cached under that tag, so one node can
    serve a months-old build while another serves today's, and nothing reports
    the difference. Leaving imagePullPolicy unset is its own hazard, because the
    default is derived from the tag — Always for ":latest", IfNotPresent for
    anything else — so the behaviour changes when someone edits the tag.
    """
    for doc in docs:
        spec = doc.get("spec", {}).get("template", {}).get("spec")
        if not spec:
            continue
        for container in spec.get("containers", []):
            image = container.get("image", "")
            name = f"{doc['kind']}/{doc['metadata']['name']}"
            if image.endswith(":latest") or ":" not in image.rsplit("/", 1)[-1]:
                raise Failure(
                    f"{name} uses a moving image tag: {image!r}. Pin a version "
                    "tag or a digest."
                )
            if "imagePullPolicy" not in container:
                raise Failure(
                    f"{name} leaves imagePullPolicy unset, so it is inferred "
                    f"from the tag {image!r} and changes when the tag changes"
                )


def check_overlays_agree_on_immutable_fields(rendered: dict[str, list[dict]]) -> None:
    """
    Fields that Kubernetes refuses to update must not differ between overlays.

    A StatefulSet's volumeClaimTemplates cannot be changed after creation. If
    the CPU and GPU overlays disagree on it, moving a cluster from one to the
    other is rejected by the API server, and the only way forward is deleting
    the StatefulSet — which on the GPU overlay means discarding the model
    cache. This once shipped as CPU=1Gi against GPU=10Gi.
    """
    if len(rendered) < 2:
        return
    baseline_name, baseline_docs = next(iter(rendered.items()))
    baseline = find(baseline_docs, "StatefulSet", "vgate-worker")["spec"]["volumeClaimTemplates"]

    for overlay, docs in list(rendered.items())[1:]:
        other = find(docs, "StatefulSet", "vgate-worker")["spec"]["volumeClaimTemplates"]
        if other != baseline:
            raise Failure(
                "StatefulSet/vgate-worker volumeClaimTemplates differ between "
                f"the {baseline_name} and {overlay} overlays, and the field is "
                "immutable — switching a live cluster between them would be "
                f"rejected.\n  {baseline_name}: {baseline}\n  {overlay}: {other}"
            )


CHECKS = [
    ("gateway holds no model resources", check_gateway_holds_no_model),
    ("worker Service is headless", check_worker_service_is_headless),
    ("gateway Service excludes workers", check_gateway_service_excludes_workers),
    ("discovery targets the worker Service", check_discovery_targets_the_worker_service),
    ("no unresolved $(VAR) interpolation", check_no_unresolved_interpolation),
    ("images are pinned with an explicit pull policy", check_images_are_pinned),
]

# Checks that need every overlay at once rather than one at a time.
CROSS_OVERLAY_CHECKS = [
    ("overlays agree on immutable fields", check_overlays_agree_on_immutable_fields),
]


def main(argv: list[str]) -> int:
    overlays = argv[1:] or DEFAULT_OVERLAYS
    failed = 0
    rendered: dict[str, list[dict]] = {}

    for overlay in overlays:
        print(f"\n=== overlay: {overlay} ===")
        try:
            docs = render(overlay)
        except Failure as exc:
            print(f"  FAIL  render: {exc}")
            failed += 1
            continue
        rendered[overlay] = docs

        for label, check in CHECKS:
            try:
                check(docs)
            except Failure as exc:
                print(f"  FAIL  {label}\n        {exc}")
                failed += 1
            else:
                print(f"  ok    {label}")

    if len(rendered) > 1:
        print("\n=== across overlays ===")
        for label, check in CROSS_OVERLAY_CHECKS:
            try:
                check(rendered)
            except Failure as exc:
                print(f"  FAIL  {label}\n        {exc}")
                failed += 1
            else:
                print(f"  ok    {label}")

    print()
    if failed:
        print(f"{failed} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
