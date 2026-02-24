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

import asyncio
import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, Optional

from vgate.config import get_config, CacheConfig
from vgate.metrics import CACHE_HITS, CACHE_MISSES, CACHE_SIZE, CACHE_EVICTIONS
from vgate.tracing import get_tracer

tracer = get_tracer("vgate.cache")


class ResultCache:
    """LRU cache for inference results."""

    def __init__(self, cache_config: Optional[CacheConfig] = None):
        """
        Initialize the result cache.

        Args:
            cache_config: Cache configuration. Uses global config if None.
        """
        if cache_config is None:
            cache_config = get_config().cache
        self.config = cache_config
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def make_key(prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        """Generate cache key from prompt and sampling params."""
        data = json.dumps({
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def get(self, key: str) -> Optional[Dict]:
        """Get a value from the cache. Returns None if not found."""
        with tracer.start_as_current_span("cache.get") as span:
            span.set_attribute("cache_key", key[:8])
            async with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    self.hits += 1
                    CACHE_HITS.inc()
                    span.set_attribute("cache_hit", True)
                    return self._cache[key]
                self.misses += 1
                CACHE_MISSES.inc()
                span.set_attribute("cache_hit", False)
                return None

    async def put(self, key: str, value: Dict) -> None:
        """Put a value in the cache, evicting oldest if at capacity."""
        with tracer.start_as_current_span("cache.put") as span:
            span.set_attribute("cache_key", key[:8])
            evicted = False
            async with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    self._cache[key] = value
                else:
                    self._cache[key] = value
                    if len(self._cache) > self.config.maxsize:
                        self._cache.popitem(last=False)
                        self.evictions += 1
                        CACHE_EVICTIONS.inc()
                        evicted = True
                CACHE_SIZE.set(len(self._cache))
            span.set_attribute("cache_size", len(self._cache))
            span.set_attribute("evicted", evicted)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "maxsize": self.config.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0
        }
