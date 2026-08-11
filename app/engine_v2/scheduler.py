from __future__ import annotations

import heapq
import itertools
import threading
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class QueueStats:
    submitted: int = 0
    replaced: int = 0
    dropped: int = 0
    popped: int = 0


class LatestOnlyPriorityQueue(Generic[T]):
    """Bounded priority queue that keeps only the newest job per key.

    Real-time video should discard stale work instead of building latency.
    Lower numeric priority wins (heap semantics).
    """

    def __init__(self, max_items: int = 128) -> None:
        self.max_items = max(1, int(max_items))
        self._heap: list[tuple[int, int, str, int, T]] = []
        self._versions: dict[str, int] = {}
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self.stats = QueueStats()

    def __len__(self) -> int:
        with self._lock:
            return len(self._versions)

    def submit(self, key: str, item: T, priority: int = 50) -> None:
        with self._lock:
            version = self._versions.get(key, 0) + 1
            if key in self._versions:
                self.stats.replaced += 1
            elif len(self._versions) >= self.max_items:
                self.stats.dropped += 1
                return
            self._versions[key] = version
            heapq.heappush(self._heap, (int(priority), next(self._counter), key, version, item))
            self.stats.submitted += 1

    def pop(self) -> T | None:
        with self._lock:
            while self._heap:
                _, _, key, version, item = heapq.heappop(self._heap)
                if self._versions.get(key) != version:
                    continue
                del self._versions[key]
                self.stats.popped += 1
                return item
        return None
