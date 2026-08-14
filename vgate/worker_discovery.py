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
Worker membership discovery over DNS.

The registry knows which workers are *usable*. This decides which workers
*exist*. Keeping the two separate matters because they fail differently: a
worker that stops answering probes should leave rotation but stay known, while
a worker that has been scaled away should stop being probed at all.

Why DNS rather than the Kubernetes API: resolving a headless Service needs no
RBAC, no client library, and no permissions the gateway would otherwise never
want. The cost is that DNS reports addresses, not object state — there is no
way to tell "not created yet" from "deleted", which is fine here because the
registry's health tracking already covers that distinction.

Names, not addresses
--------------------
Forward resolution yields pod IPs. Those are unusable as identity: a restarted
pod comes back with a different IP, and every `vgate_worker_*` metric is
labelled by endpoint, so IP-keyed membership would grow a new time series on
every restart until the metric endpoint collapses under its own cardinality.

A reverse lookup turns each IP back into the pod's stable DNS name
(`vgate-worker-0.vgate-worker.vgate.svc.cluster.local`), which survives
restarts because a StatefulSet reuses ordinals. Measured against a live kind
cluster before this was built on:

    forward  vgate-worker.vgate.svc.cluster.local -> 10.244.1.4, 10.244.2.3
    reverse  10.244.1.4 -> vgate-worker-0.vgate-worker.vgate.svc.cluster.local
             10.244.2.3 -> vgate-worker-1.vgate-worker.vgate.svc.cluster.local

CoreDNS serves those PTR records by default, but a cluster can disable them, so
an IP is used when the reverse lookup fails. That case is logged rather than
silent, because the consequence — unbounded metric labels over a long uptime —
is invisible until it is severe.
"""

import ipaddress
import socket
from typing import Callable, List, Optional, Sequence, Tuple

from vgate.logging_config import get_logger

logger = get_logger("vgate.discovery")

# (host, port) -> list of addresses. Injectable so tests can drive membership
# changes without a cluster or a resolver.
ForwardResolver = Callable[[str, int], Sequence[str]]
ReverseResolver = Callable[[str], str]


def _default_forward(host: str, port: int) -> Sequence[str]:
    # AF_UNSPEC, not AF_INET: an IPv6-only cluster publishes AAAA records and
    # no A records, so pinning to IPv4 would resolve nothing there and leave
    # the pool permanently empty with no error to explain it.
    infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    # A headless Service returns one record per pod; dedupe because
    # getaddrinfo repeats an address once per socket type.
    return sorted({info[4][0] for info in infos})


def _host_for_url(host: str) -> str:
    """
    Bracket a bare IPv6 address so it can appear in a URL authority.

    `http://fd00::1:8000` is unparseable -- the colons in the address are
    indistinguishable from the port separator. Names and IPv4 addresses are
    returned unchanged, and an already-bracketed value is left alone.
    """
    if host.startswith("["):
        return host
    try:
        if ipaddress.ip_address(host).version == 6:
            return f"[{host}]"
    except ValueError:
        pass  # a hostname, not a literal address
    return host


def _default_reverse(address: str) -> str:
    return socket.gethostbyaddr(address)[0]


class DnsWorkerDiscovery:
    """
    Resolves a headless Service name into the current set of worker endpoints.

    Returns endpoints in the same `http://host:port` shape the static config
    uses, so nothing downstream needs to know where membership came from.
    """

    def __init__(
        self,
        dns_name: str,
        port: int = 8000,
        scheme: str = "http",
        forward_resolver: Optional[ForwardResolver] = None,
        reverse_resolver: Optional[ReverseResolver] = None,
    ):
        self.dns_name = dns_name
        self.port = port
        self.scheme = scheme
        self._forward = forward_resolver or _default_forward
        self._reverse = reverse_resolver or _default_reverse
        # Logged once per address rather than once per tick: discovery runs on
        # the health-check interval, so an unconditional warning would emit
        # every few seconds forever.
        self._reverse_failures_logged: set = set()

    def resolve(self) -> List[str]:
        """
        Current worker endpoints, or an empty list if the name does not resolve.

        An unresolvable name is not an error here. It is the normal state while
        a StatefulSet is starting, and it is indistinguishable from a scaled-to-
        zero deployment. Callers decide what an empty result means; this returns
        what DNS said.
        """
        try:
            addresses = self._forward(self.dns_name, self.port)
        except socket.gaierror as exc:
            logger.debug(
                "Worker DNS name did not resolve",
                extra={"extra_data": {"dns_name": self.dns_name, "error": str(exc)}},
            )
            return []

        endpoints = [self._endpoint_for(addr) for addr in addresses]
        return sorted(set(endpoints))

    # Bracketing applies only to the fallback path: resolved pod names are
    # hostnames, and a hostname is never bracketed.
    def _endpoint_for(self, address: str) -> str:
        host, stable = self._stable_name(address)
        if not stable and address not in self._reverse_failures_logged:
            self._reverse_failures_logged.add(address)
            logger.warning(
                "Reverse lookup failed; using the pod address as its identity. "
                "Metric labels will churn as pods restart.",
                extra={"extra_data": {"address": address, "dns_name": self.dns_name}},
            )
        return f"{self.scheme}://{_host_for_url(host)}:{self.port}"

    def _stable_name(self, address: str) -> Tuple[str, bool]:
        try:
            name = self._reverse(address)
        except (socket.herror, socket.gaierror, OSError):
            return address, False
        # A PTR pointing back at the Service rather than at a pod carries no
        # per-worker identity, so it is no better than the address.
        if not name or name.rstrip(".") == self.dns_name.rstrip("."):
            return address, False
        return name.rstrip("."), True
