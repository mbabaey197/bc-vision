"""Camera-local event identity across short-lived tracker fragments.

Tracker ids describe a visual fragment, not necessarily a vehicle visit.  A
plate can therefore acquire several track ids while it is still in view.  The
ledger below keeps the durable event identity separate from those volatile
track ids and ends a visit only after repeated observations confirm absence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .plate_rules import normalize_plate, plausible_plate


def strict_plate_key(result: dict) -> str:
    """Return an exact, complete plate identity suitable for deduplication."""

    if result.get("valid") or result.get("auto_confirmed"):
        for candidate in (
            result.get("plate_norm"),
            result.get("plate"),
        ):
            accepted = normalize_plate(candidate)
            if plausible_plate(accepted):
                return accepted
    associated = normalize_plate(result.get("association_plate_norm"))
    if (
        result.get("association_plate_strong")
        and plausible_plate(associated)
    ):
        return associated
    return ""


def candidate_plate_key(result: dict) -> str:
    """Return an exact review candidate without treating it as confirmed."""

    accepted = strict_plate_key(result)
    if accepted:
        return accepted
    for candidate in (
        result.get("raw_guess_norm"),
        result.get("raw_guess_text"),
        result.get("plate_norm"),
        result.get("plate"),
    ):
        normalized = normalize_plate(candidate)
        if plausible_plate(normalized):
            return normalized
    return ""


@dataclass
class PlateVisitLedger:
    """Map one continuous camera visit to one durable event reference."""

    absence_observations: int = 3
    absence_seconds: float = 0.75
    seen: dict[str, float] = field(default_factory=dict)
    event_refs: dict[str, int] = field(default_factory=dict)
    active: set[str] = field(default_factory=set)
    track_keys: dict[int, str] = field(default_factory=dict)
    plate_tracks: dict[str, set[int]] = field(default_factory=dict)
    missing_observations: dict[str, int] = field(default_factory=dict)
    absence_started: dict[str, float] = field(default_factory=dict)

    def clear(self) -> None:
        self.seen.clear()
        self.event_refs.clear()
        self.active.clear()
        self.track_keys.clear()
        self.plate_tracks.clear()
        self.missing_observations.clear()
        self.absence_started.clear()

    def reset_tracker_bindings(self) -> None:
        """Keep durable visit ids while discarding model-specific tracks."""

        self.track_keys.clear()
        self.plate_tracks.clear()
        self.missing_observations.clear()
        self.absence_started.clear()
        self.active.intersection_update(self.event_refs)
        for key in tuple(self.seen):
            if key not in self.event_refs:
                self.seen.pop(key, None)

    def _unbind(self, track_id: int) -> None:
        key = self.track_keys.pop(int(track_id), "")
        if not key:
            return
        tracks = self.plate_tracks.get(key)
        if tracks is None:
            return
        tracks.discard(int(track_id))
        if not tracks:
            self.plate_tracks.pop(key, None)

    def _bind(self, track_id: int, key: str) -> None:
        track_id = int(track_id or 0)
        if not track_id:
            return
        previous = self.track_keys.get(track_id, "")
        if previous and previous != key:
            self._unbind(track_id)
        self.track_keys[track_id] = key
        self.plate_tracks.setdefault(key, set()).add(track_id)

    def _reusable(
        self,
        key: str,
        timestamp: float,
        duplicate_seconds: float,
    ) -> bool:
        if key not in self.event_refs:
            return False
        if key in self.active:
            return True
        return (
            float(timestamp) - self.seen.get(key, -1e12)
            < max(0.0, float(duplicate_seconds))
        )

    def _touch(self, key: str, timestamp: float) -> None:
        self.active.add(key)
        self.seen[key] = float(timestamp)
        self.missing_observations.pop(key, None)
        self.absence_started.pop(key, None)

    def observe(
        self,
        rows,
        active_track_ids,
        timestamp: float,
        duplicate_seconds: float,
    ) -> set[int]:
        """Refresh visits from raw detections after tracker association."""

        timestamp = float(timestamp)
        active_track_ids = {
            int(track_id) for track_id in active_track_ids
        }
        for track_id in list(self.track_keys):
            if track_id not in active_track_ids:
                self._unbind(track_id)

        rows = list(rows)
        observed = set()
        retired_track_ids = set()
        for result in rows:
            track_id = int(result.get("track_id") or 0)
            key = candidate_plate_key(result)
            bound = self.track_keys.get(track_id, "") if track_id else ""
            # A complete conflicting identity is a vehicle boundary even
            # when it is still review-only.  Never let a stale tracker
            # binding merge two exact candidates that differ by one slot.
            if bound and key and bound != key:
                retired_track_ids.add(track_id)
                self._unbind(track_id)
                bound = ""
            if not bound and key and self._reusable(
                key,
                timestamp,
                duplicate_seconds,
            ):
                bound = key
                self._bind(track_id, key)
            if (
                not bound
                and not key
                and len(self.active) == 1
            ):
                # A single unknown plate fragment inside a single active
                # visit is detector/OCR flicker, not proof that the vehicle
                # left.  Bind it conservatively until an exact conflicting
                # identity or a genuinely empty scene establishes a boundary.
                bound = next(iter(self.active))
                self._bind(track_id, bound)
            if not bound or bound not in self.event_refs:
                continue
            self._touch(bound, timestamp)
            observed.add(bound)

        for key in list(self.active):
            if key in observed:
                continue
            missing = self.missing_observations.get(key, 0) + 1
            self.missing_observations[key] = missing
            absent_at = self.absence_started.setdefault(key, timestamp)
            if (
                missing >= max(1, int(self.absence_observations))
                and timestamp - absent_at
                >= max(0.0, float(self.absence_seconds))
            ):
                self.active.discard(key)
                self.missing_observations.pop(key, None)
                self.absence_started.pop(key, None)
                bound_tracks = tuple(self.plate_tracks.get(key, ()))
                retired_track_ids.update(bound_tracks)
                for track_id in bound_tracks:
                    self._unbind(track_id)

        cooldown = max(0.0, float(duplicate_seconds))
        for key in list(self.event_refs):
            if key in self.active:
                continue
            if timestamp - self.seen.get(key, -1e12) < cooldown:
                continue
            self.event_refs.pop(key, None)
            self.seen.pop(key, None)
            self.missing_observations.pop(key, None)
            self.absence_started.pop(key, None)
            for track_id in tuple(self.plate_tracks.get(key, ())):
                self._unbind(track_id)
        return retired_track_ids

    def event_ref(
        self,
        result: dict,
        timestamp: float,
        duplicate_seconds: float,
        allow_candidate: bool = False,
    ) -> tuple[str, int | None]:
        """Return the canonical event for this result when it is reusable."""

        key = (
            candidate_plate_key(result)
            if allow_candidate
            else strict_plate_key(result)
        )
        if not key or not self._reusable(
            key,
            timestamp,
            duplicate_seconds,
        ):
            return key, None
        track_id = int(result.get("track_id") or 0)
        self._bind(track_id, key)
        self._touch(key, timestamp)
        return key, self.event_refs[key]

    def register(
        self,
        result: dict,
        event_ref: int,
        timestamp: float,
        allow_candidate: bool = False,
    ) -> str:
        """Record a successfully persisted strict result as the visit owner."""

        key = (
            candidate_plate_key(result)
            if allow_candidate
            else strict_plate_key(result)
        )
        if not key:
            return ""
        previous = self.event_refs.get(key)
        if previous is not None and previous != int(event_ref):
            for track_id in tuple(self.plate_tracks.get(key, ())):
                self._unbind(track_id)
        self.event_refs[key] = int(event_ref)
        self._bind(int(result.get("track_id") or 0), key)
        self._touch(key, timestamp)
        return key

    def track_event_refs(self) -> dict[int, int]:
        return {
            track_id: self.event_refs[key]
            for track_id, key in self.track_keys.items()
            if key in self.event_refs
        }
