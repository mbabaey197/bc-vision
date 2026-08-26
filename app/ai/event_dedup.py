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


def review_identity_can_migrate(old_key: str, new_key: str) -> bool:
    """Allow only a one-slot OCR correction to reuse a review event."""

    left = normalize_plate(old_key)
    right = normalize_plate(new_key)
    return bool(
        plausible_plate(left)
        and plausible_plate(right)
        and sum(a != b for a, b in zip(left, right)) <= 1
    )


def _result_bbox(result: dict) -> tuple[float, float, float, float] | None:
    raw = result.get("tracking_bbox") or result.get("bbox")
    if not raw or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _spatially_continuous(left: tuple, right: tuple) -> bool:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(1.0, (lx2 - lx1) * (ly2 - ly1))
    right_area = max(1.0, (rx2 - rx1) * (ry2 - ry1))
    iou = intersection / max(1.0, left_area + right_area - intersection)
    if iou >= 0.08:
        return True
    left_center = ((lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0)
    right_center = ((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0)
    distance = (
        (left_center[0] - right_center[0]) ** 2
        + (left_center[1] - right_center[1]) ** 2
    ) ** 0.5
    scale = max(
        lx2 - lx1,
        ly2 - ly1,
        rx2 - rx1,
        ry2 - ry1,
    )
    return distance <= scale * 0.45


def fragmented_review_can_migrate(
    old_result: dict,
    new_result: dict,
) -> bool:
    """Validate a cross-track review-to-strict visit correction."""

    old_key = candidate_plate_key(old_result)
    new_key = strict_plate_key(new_result)
    old_bbox = _result_bbox(old_result)
    new_bbox = _result_bbox(new_result)
    return bool(
        old_key
        and new_key
        and not strict_plate_key(old_result)
        and review_identity_can_migrate(old_key, new_key)
        and old_bbox is not None
        and new_bbox is not None
        and _spatially_continuous(old_bbox, new_bbox)
    )


@dataclass
class PlateVisitLedger:
    """Map one continuous camera visit to one durable event reference."""

    absence_observations: int = 3
    absence_seconds: float = 0.75
    seen: dict[str, float] = field(default_factory=dict)
    event_refs: dict[str, int] = field(default_factory=dict)
    confirmed_keys: set[str] = field(default_factory=set)
    active: set[str] = field(default_factory=set)
    track_keys: dict[int, str] = field(default_factory=dict)
    plate_tracks: dict[str, set[int]] = field(default_factory=dict)
    missing_observations: dict[str, int] = field(default_factory=dict)
    absence_started: dict[str, float] = field(default_factory=dict)
    last_bboxes: dict[str, tuple] = field(default_factory=dict)
    provisional_owner_tracks: dict[str, int] = field(default_factory=dict)

    def clear(self) -> None:
        self.seen.clear()
        self.event_refs.clear()
        self.confirmed_keys.clear()
        self.active.clear()
        self.track_keys.clear()
        self.plate_tracks.clear()
        self.missing_observations.clear()
        self.absence_started.clear()
        self.last_bboxes.clear()
        self.provisional_owner_tracks.clear()

    def reset_tracker_bindings(self) -> None:
        """Keep durable visit ids while discarding model-specific tracks."""

        self.track_keys.clear()
        self.plate_tracks.clear()
        self.missing_observations.clear()
        self.absence_started.clear()
        self.provisional_owner_tracks.clear()
        self.active.intersection_update(self.event_refs)
        self.confirmed_keys.intersection_update(self.event_refs)
        for key in tuple(self.seen):
            if key not in self.event_refs:
                self.seen.pop(key, None)
                self.last_bboxes.pop(key, None)

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
        self.seen[key] = max(
            float(timestamp),
            self.seen.get(key, -1e12),
        )
        self.missing_observations.pop(key, None)
        self.absence_started.pop(key, None)

    def _remember_bbox(self, key: str, result: dict) -> None:
        bbox = _result_bbox(result)
        if bbox is not None:
            self.last_bboxes[key] = bbox

    @staticmethod
    def _provisional_upgrade_source(result: dict) -> bool:
        """Return whether a review identity is explicitly provisional."""

        return bool(
            result.get("visit_identity_stable") is False
            or result.get("raw_guess_reason") in {
                "below-camera-confidence",
                "strict-decoder-rejected",
                "multi-frame-rejected-hypotheses",
            }
        )

    def _continuity_alias(
        self,
        key: str,
        result: dict,
        timestamp: float,
    ) -> str:
        """Link a fragmented track only with unique spatial OCR continuity."""

        if not strict_plate_key(result):
            return ""
        current_bbox = _result_bbox(result)
        if current_bbox is None:
            return ""
        max_gap = max(1.5, float(self.absence_seconds) * 2.0)
        candidates = [
            active_key
            for active_key in self.active
            if active_key not in self.confirmed_keys
            and active_key != key
            and review_identity_can_migrate(active_key, key)
            and 0.0
            <= timestamp - self.seen.get(active_key, -1e12)
            <= max_gap
            and active_key in self.last_bboxes
            and _spatially_continuous(
                self.last_bboxes[active_key],
                current_bbox,
            )
        ]
        return candidates[0] if len(candidates) == 1 else ""

    def can_reuse_track_event(self, track_id: int, result: dict) -> bool:
        """Reject a stale event id when the bound identity is incompatible."""

        track_id = int(track_id or 0)
        bound = self.track_keys.get(track_id, "")
        if bound and result.get("visit_identity_stable") is False:
            return bound in self.event_refs
        key = candidate_plate_key(result)
        if not bound or not key or bound == key:
            return True
        return bool(
            bound not in self.confirmed_keys
            and (
                # A tracker-continuous strict read is stronger evidence than
                # the provisional identity already owned by that same track.
                # Cross-track corrections still reach this point only through
                # the one-slot + spatial continuity gate in _continuity_alias.
                (
                    strict_plate_key(result)
                    and self.provisional_owner_tracks.get(bound) == track_id
                )
                or review_identity_can_migrate(bound, key)
            )
        )

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
            key = (
                candidate_plate_key(result)
                if result.get("visit_identity_stable", True)
                else ""
            )
            bound = self.track_keys.get(track_id, "") if track_id else ""
            # A complete conflicting identity is a vehicle boundary even
            # when it follows a confirmed plate. A provisional review key may
            # still be corrected by clearer evidence on the same track; keep
            # its event reference until register() atomically migrates it.
            if bound and key and bound != key:
                same_track_strict_upgrade = bool(
                    bound not in self.confirmed_keys
                    and strict_plate_key(result)
                    and self.provisional_owner_tracks.get(bound) == track_id
                )
                if (
                    bound in self.confirmed_keys
                    or (
                        not same_track_strict_upgrade
                        and not review_identity_can_migrate(bound, key)
                    )
                ):
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
            if not bound and key:
                alias = self._continuity_alias(
                    key,
                    result,
                    timestamp,
                )
                if alias:
                    bound = alias
                    self._bind(track_id, alias)
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
            self._remember_bbox(bound, result)
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
            self.confirmed_keys.discard(key)
            self.seen.pop(key, None)
            self.last_bboxes.pop(key, None)
            self.provisional_owner_tracks.pop(key, None)
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

        strict_key = strict_plate_key(result)
        key = (
            candidate_plate_key(result)
            if allow_candidate
            else strict_key
        )
        if not key:
            return ""
        event_ref = int(event_ref)
        track_id = int(result.get("track_id") or 0)
        aliases = [
            existing_key
            for existing_key, existing_ref in self.event_refs.items()
            if existing_ref == event_ref and existing_key != key
        ]
        bound_alias = self.track_keys.get(track_id, "")
        protected_alias = next(
            (
                alias
                for alias in aliases
                if (
                    alias in self.confirmed_keys
                    or (
                        not review_identity_can_migrate(alias, key)
                        and not (
                            strict_key
                            and alias == bound_alias
                            and alias not in self.confirmed_keys
                            and self.provisional_owner_tracks.get(alias)
                            == track_id
                        )
                    )
                )
            ),
            "",
        )
        if protected_alias:
            # Neither a review candidate nor a conflicting strict identity
            # may repurpose an unrelated/confirmed durable row.
            self._bind(track_id, protected_alias)
            self._touch(protected_alias, timestamp)
            self._remember_bbox(protected_alias, result)
            return protected_alias

        migrated_tracks = set()
        migrated_bbox = None
        migrated_seen = -1e12
        for alias in aliases:
            migrated_tracks.update(self.plate_tracks.get(alias, ()))
            migrated_bbox = self.last_bboxes.get(alias, migrated_bbox)
            migrated_seen = max(
                migrated_seen,
                self.seen.get(alias, -1e12),
            )
            for alias_track_id in tuple(
                self.plate_tracks.get(alias, ())
            ):
                self._unbind(alias_track_id)
            self.event_refs.pop(alias, None)
            self.confirmed_keys.discard(alias)
            self.active.discard(alias)
            self.seen.pop(alias, None)
            self.last_bboxes.pop(alias, None)
            self.provisional_owner_tracks.pop(alias, None)
            self.missing_observations.pop(alias, None)
            self.absence_started.pop(alias, None)
        previous = self.event_refs.get(key)
        if previous is not None and previous != event_ref:
            for previous_track_id in tuple(
                self.plate_tracks.get(key, ())
            ):
                self._unbind(previous_track_id)
            self.confirmed_keys.discard(key)
        self.event_refs[key] = event_ref
        if strict_key:
            self.confirmed_keys.add(key)
            self.provisional_owner_tracks.pop(key, None)
        elif track_id and self._provisional_upgrade_source(result):
            self.provisional_owner_tracks[key] = track_id
        else:
            self.provisional_owner_tracks.pop(key, None)
        for migrated_track_id in migrated_tracks:
            self._bind(migrated_track_id, key)
        self._bind(track_id, key)
        self._touch(key, max(float(timestamp), migrated_seen))
        if migrated_bbox is not None:
            self.last_bboxes[key] = migrated_bbox
        self._remember_bbox(key, result)
        return key

    def track_event_refs(self) -> dict[int, int]:
        return {
            track_id: self.event_refs[key]
            for track_id, key in self.track_keys.items()
            if key in self.event_refs
        }
