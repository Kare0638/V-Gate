import asyncio
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from vgate.metrics import CACHE_HITS, CACHE_MISSES, CACHE_SIZE, CACHE_EVICTIONS


@dataclass
class CacheConfig:
    """Configuration for the result cache."""
    maxsize: int = 1000


class ResultCache:
    """LRU cache for inference results."""

    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
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
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                CACHE_HITS.inc()
                return self._cache[key]
            self.misses += 1
            CACHE_MISSES.inc()
            return None

    async def put(self, key: str, value: Dict) -> None:
        """Put a value in the cache, evicting oldest if at capacity."""
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
            CACHE_SIZE.set(len(self._cache))

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
