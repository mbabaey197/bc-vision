from __future__ import annotations

import heapq
import itertools
import threading
import time
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class QueueStats:
    submitted: int = 0
    replaced: int = 0
    dropped: int = 0
    expired: int = 0
    evicted: int = 0
    discarded: int = 0
    popped: int = 0

    @property
    def stale_dropped(self) -> int:
        return self.replaced + self.expired


class LatestOnlyPriorityQueue(Generic[T]):
    """Bounded fair priority scheduler with one live job per key.

    Re-submitting a camera replaces its queued frame in O(log n) logical time;
    obsolete heap nodes are skipped and periodically compacted. Under capacity
    pressure a higher-priority active-camera job may evict the worst queued
    idle job. Lower numeric priority wins.
    """

    def __init__(self, max_items: int = 128, fairness_penalty: int = 8) -> None:
        self.max_items = max(1, int(max_items))
        self.fairness_penalty = max(0, int(fairness_penalty))
        self._heap: list[tuple[int, int, str, int, float, T]] = []
        self._versions: dict[str, int] = {}
        self._entries: dict[str, tuple[int, int, int, float, T]] = {}
        self._counter = itertools.count()
        self._last_popped_key: str | None = None
        # A version must never be reused while an obsolete heap node can still
        # exist. Per-key counters reset when a key is popped or evicted and can
        # therefore resurrect an old node (the classic ABA problem). A global
        # monotonic generation keeps every logical submission distinct.
        self._generation = itertools.count(1)
        self._lock = threading.Lock()
        self.stats = QueueStats()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def submit(self, key: str, item: T, priority: int = 50) -> bool:
        accepted, _ = self.submit_with_evicted(key, item, priority)
        return accepted

    def submit_with_evicted(
        self,
        key: str,
        item: T,
        priority: int = 50,
    ) -> tuple[bool, T | None]:
        """Submit one item and surface a capacity-evicted live item.

        Detector callers normally need only the boolean ``submit`` contract.
        Stateful downstream queues such as OCR must also know when prioritizing
        new work evicts another episode, otherwise that episode can remain
        permanently marked as submitted with no live queue entry.
        """

        now = time.monotonic()
        with self._lock:
            key = str(key)
            priority = int(priority)
            evicted_item: T | None = None
            effective_priority = priority + (
                self.fairness_penalty if key == self._last_popped_key else 0
            )
            if key in self._entries:
                self.stats.replaced += 1
            elif len(self._entries) >= self.max_items:
                worst_key, worst = max(
                    self._entries.items(),
                    key=lambda entry: (entry[1][0], -entry[1][1]),
                )
                if effective_priority >= worst[0]:
                    self.stats.dropped += 1
                    return False, None
                evicted_item = worst[4]
                del self._entries[worst_key]
                del self._versions[worst_key]
                self.stats.evicted += 1

            version = next(self._generation)
            order = next(self._counter)
            self._versions[key] = version
            self._entries[key] = (effective_priority, order, version, now, item)
            heapq.heappush(
                self._heap,
                (effective_priority, order, key, version, now, item),
            )
            self.stats.submitted += 1
            if len(self._heap) > max(32, self.max_items * 4):
                self._compact_locked()
            return True, evicted_item

    def pop(self, max_age_seconds: float | None = None) -> T | None:
        item, _ = self.pop_with_expired(max_age_seconds)
        return item

    def pop_with_expired(
        self,
        max_age_seconds: float | None = None,
    ) -> tuple[T | None, tuple[T, ...]]:
        """Pop the next live item and return any expired live items as well."""

        with self._lock:
            now = time.monotonic()
            expired_items: list[T] = []
            while self._heap:
                _, _, key, version, submitted_at, item = heapq.heappop(self._heap)
                if self._versions.get(key) != version:
                    continue
                self._versions.pop(key, None)
                self._entries.pop(key, None)
                if max_age_seconds is not None and now - submitted_at > max(0.0, max_age_seconds):
                    self.stats.expired += 1
                    expired_items.append(item)
                    continue
                self.stats.popped += 1
                self._last_popped_key = key
                return item, tuple(expired_items)
        return None, tuple(expired_items)

    def discard(self, key: str) -> bool:
        """Invalidate one live key without disturbing work for other cameras.

        The obsolete heap node is skipped by ``pop`` through the same global
        generation check used for latest-frame replacement.
        """

        with self._lock:
            normalized = str(key)
            if normalized not in self._entries:
                return False
            del self._entries[normalized]
            self._versions.pop(normalized, None)
            self.stats.discarded += 1
            return True

    def clear(self, *, reset_stats: bool = False) -> None:
        with self._lock:
            self._heap.clear()
            self._versions.clear()
            self._entries.clear()
            self._last_popped_key = None
            if reset_stats:
                self.stats = QueueStats()

    def _compact_locked(self) -> None:
        self._heap = [
            (priority, order, key, version, submitted_at, item)
            for key, (priority, order, version, submitted_at, item) in self._entries.items()
        ]
        heapq.heapify(self._heap)
