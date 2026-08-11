from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .scheduler import LatestOnlyPriorityQueue
from .types import OCRResult, PlateOCR
from .validator import IranianPlateValidator, PlateValidation


@dataclass(frozen=True, slots=True)
class OCRObservation:
    result: OCRResult
    candidate_quality: float
    seq: int = -1


@dataclass(slots=True)
class OCRVote:
    text: str = ""
    confidence: float = 0.0
    valid: bool = False
    reason: str = "no_observations"
    support: int = 0
    observations: int = 0
    validation: PlateValidation | None = None
    results: list[OCRObservation] = field(default_factory=list)


class TemporalOCRVoter:
    """Confidence/quality weighted voting over the best temporal crops."""

    def __init__(
        self,
        validator: IranianPlateValidator | None = None,
        *,
        min_support: int = 2,
        single_accept_confidence: float = 0.88,
        single_accept_quality: float = 0.50,
        min_observation_confidence: float = 0.35,
        min_observation_quality: float = 0.05,
        min_consensus_confidence: float = 0.55,
    ) -> None:
        self.validator = validator or IranianPlateValidator()
        self.min_support = max(1, int(min_support))
        self.single_accept_confidence = self._threshold(
            "single_accept_confidence", single_accept_confidence
        )
        self.single_accept_quality = self._threshold(
            "single_accept_quality", single_accept_quality
        )
        self.min_observation_confidence = self._threshold(
            "min_observation_confidence", min_observation_confidence
        )
        self.min_observation_quality = self._threshold(
            "min_observation_quality", min_observation_quality
        )
        self.min_consensus_confidence = self._threshold(
            "min_consensus_confidence", min_consensus_confidence
        )

    @staticmethod
    def _threshold(name: str, value: float) -> float:
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be finite and within 0..1")
        return number

    @staticmethod
    def _unit_value(value: float) -> float | None:
        number = float(value)
        if not math.isfinite(number):
            return None
        return max(0.0, min(1.0, number))

    def vote(self, observations: Sequence[OCRObservation]) -> OCRVote:
        grouped: dict[
            str,
            dict[int, tuple[OCRObservation, PlateValidation, float, float, float]],
        ] = {}
        had_text = False
        passed_floor = False
        for observation in observations:
            if not observation.result.text.strip() or not observation.result.valid:
                continue
            had_text = True
            confidence = self._unit_value(observation.result.confidence)
            quality = self._unit_value(observation.candidate_quality)
            if confidence is None or quality is None:
                continue
            if (
                confidence < self.min_observation_confidence
                or quality < self.min_observation_quality
            ):
                continue
            passed_floor = True
            validation = self.validator.validate(observation.result.text)
            if not validation.valid:
                continue
            weight = confidence * (0.55 + 0.45 * quality)
            # A temporal consensus must contain distinct source frames. If two
            # crops from the same frame produce the same text, retain only the
            # stronger one instead of counting it twice.
            members = grouped.setdefault(validation.normalized, {})
            seq = int(observation.seq)
            existing = members.get(seq)
            member = (observation, validation, weight, confidence, quality)
            if existing is None or weight > existing[2]:
                members[seq] = member

        if not grouped:
            reason = "no_observations"
            if passed_floor:
                reason = "no_structurally_valid_read"
            elif had_text:
                reason = "no_observations_above_floor"
            return OCRVote(
                reason=reason,
                observations=len(observations),
                results=list(observations),
            )

        text, members_by_seq = max(
            grouped.items(),
            key=lambda item: (sum(member[2] for member in item[1].values()), len(item[1])),
        )
        members = list(members_by_seq.values())
        support = len(members)
        total_weight = sum(member[2] for member in members)
        confidence = sum(
            member[3] * member[2] for member in members
        ) / max(1e-9, total_weight)
        best_member = max(members, key=lambda member: member[2])
        best_observation = best_member[0]
        best_confidence = best_member[3]
        best_quality = best_member[4]

        accepted = support >= self.min_support and confidence >= self.min_consensus_confidence
        if support >= self.min_support:
            reason = "temporal_consensus" if accepted else "consensus_below_confidence"
        else:
            reason = "insufficient_support"
        if support == 1:
            accepted = (
                best_confidence >= self.single_accept_confidence
                and best_quality >= self.single_accept_quality
            )
            reason = "high_confidence_single" if accepted else "single_read_below_threshold"

        return OCRVote(
            text=text,
            confidence=float(confidence),
            valid=bool(accepted),
            reason=reason,
            support=support,
            observations=len(observations),
            validation=members[0][1],
            results=list(observations),
        )


@dataclass(slots=True)
class OCRTask:
    key: str
    crops: list[np.ndarray]
    qualities: list[float]
    sequences: list[int] = field(default_factory=list)
    priority: int = 20
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AbandonedOCRTask:
    task: OCRTask
    reason: str


@dataclass(slots=True)
class OCRWorkerStats:
    inference_count: int = 0
    failed_inference_count: int = 0
    task_count: int = 0
    failed_task_count: int = 0
    callback_error_count: int = 0
    abandoned_task_count: int = 0
    expired_task_count: int = 0
    evicted_task_count: int = 0
    total_inference_seconds: float = 0.0
    last_inference_seconds: float = 0.0
    last_error: str | None = None

    @property
    def mean_inference_seconds(self) -> float:
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_seconds / self.inference_count


class SharedOCRWorker:
    """One central OCR session and a latest-only task queue for all cameras."""

    def __init__(
        self,
        ocr: PlateOCR,
        voter: TemporalOCRVoter | None = None,
        *,
        queue_size: int = 256,
        max_task_age_seconds: float | None = 1.0,
    ) -> None:
        if max_task_age_seconds is not None:
            max_age = float(max_task_age_seconds)
            if not math.isfinite(max_age) or max_age < 0.0:
                raise ValueError("max_task_age_seconds must be finite and non-negative")
            self.max_task_age_seconds: float | None = max_age
        else:
            self.max_task_age_seconds = None
        self.ocr = ocr
        self.voter = voter or TemporalOCRVoter()
        self.queue: LatestOnlyPriorityQueue[OCRTask] = LatestOnlyPriorityQueue(queue_size)
        self.stats = OCRWorkerStats()
        self._inference_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._abandoned_lock = threading.Lock()
        self._abandoned: deque[AbandonedOCRTask] = deque()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def submit(self, task: OCRTask) -> bool:
        accepted, evicted = self.queue.submit_with_evicted(
            task.key,
            task,
            priority=task.priority,
        )
        if evicted is not None:
            self._record_abandoned(evicted, "capacity_evicted")
        if accepted:
            self._wake.set()
        return accepted

    def reset(self) -> None:
        """Clear queued work and counters while retaining the shared session."""

        self.queue.clear(reset_stats=True)
        with self._abandoned_lock:
            self._abandoned.clear()
        with self._stats_lock:
            self.stats = OCRWorkerStats()

    def process_next(self) -> tuple[OCRTask, OCRVote] | None:
        processed, _ = self.process_next_with_abandoned()
        return processed

    def process_next_with_abandoned(
        self,
    ) -> tuple[tuple[OCRTask, OCRVote] | None, tuple[AbandonedOCRTask, ...]]:
        task, expired = self.queue.pop_with_expired(
            max_age_seconds=self.max_task_age_seconds
        )
        for expired_task in expired:
            self._record_abandoned(expired_task, "expired")
        if task is None:
            return None, self.drain_abandoned()
        observations: list[OCRObservation] = []
        failures = 0
        with self._inference_lock:
            for index, crop in enumerate(task.crops):
                started = time.perf_counter()
                error_text: str | None = None
                try:
                    result = self.ocr.read(crop)
                except Exception as exc:
                    failures += 1
                    error_text = f"{type(exc).__name__}: {exc}"
                    result = OCRResult("", 0.0, False, metadata={"error": error_text})
                elapsed = max(0.0, time.perf_counter() - started)
                with self._stats_lock:
                    self.stats.inference_count += 1
                    self.stats.total_inference_seconds += elapsed
                    self.stats.last_inference_seconds = elapsed
                    if error_text is not None:
                        self.stats.failed_inference_count += 1
                        self.stats.last_error = error_text
                quality = task.qualities[index] if index < len(task.qualities) else 0.0
                seq = task.sequences[index] if index < len(task.sequences) else -1
                observations.append(OCRObservation(result, quality, seq))
        with self._stats_lock:
            self.stats.task_count += 1
            if task.crops and failures == len(task.crops):
                self.stats.failed_task_count += 1
        vote = self.voter.vote(observations)
        if task.crops and failures == len(task.crops):
            vote.reason = "ocr_error"
        return (task, vote), self.drain_abandoned()

    def drain_abandoned(self) -> tuple[AbandonedOCRTask, ...]:
        with self._abandoned_lock:
            items = tuple(self._abandoned)
            self._abandoned.clear()
            return items

    def start(
        self,
        callback: Callable[[OCRTask, OCRVote], None],
        abandoned_callback: Callable[[AbandonedOCRTask], None] | None = None,
    ) -> bool:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._thread = None
            self._stop.clear()

        def run() -> None:
            try:
                while not self._stop.is_set():
                    try:
                        processed, abandoned = self.process_next_with_abandoned()
                    except Exception as exc:
                        self._record_worker_error(exc, failed_task=True)
                        continue
                    if abandoned_callback is not None:
                        for item in abandoned:
                            try:
                                abandoned_callback(item)
                            except Exception as exc:
                                self._record_worker_error(exc, callback_error=True)
                    if processed is None:
                        self._wake.wait(0.05)
                        self._wake.clear()
                        continue
                    try:
                        callback(*processed)
                    except Exception as exc:
                        self._record_worker_error(exc, callback_error=True)
            finally:
                with self._lifecycle_lock:
                    if self._thread is threading.current_thread():
                        self._thread = None

        thread = threading.Thread(target=run, name="anpr-v2-shared-ocr", daemon=True)
        with self._lifecycle_lock:
            # A concurrent start could only reach this point if the previous
            # thread finished between the first lock and thread construction.
            if self._thread is not None and self._thread.is_alive():
                return False
            self._thread = thread
            thread.start()
        return True

    def stop(self, timeout: float = 2.0) -> bool:
        self._stop.set()
        self._wake.set()
        with self._lifecycle_lock:
            thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout=max(0.0, float(timeout)))
        stopped = not thread.is_alive()
        if stopped:
            with self._lifecycle_lock:
                if self._thread is thread:
                    self._thread = None
        return stopped

    def _record_abandoned(self, task: OCRTask, reason: str) -> None:
        item = AbandonedOCRTask(task=task, reason=str(reason))
        with self._abandoned_lock:
            self._abandoned.append(item)
        with self._stats_lock:
            self.stats.abandoned_task_count += 1
            if reason == "expired":
                self.stats.expired_task_count += 1
            elif reason == "capacity_evicted":
                self.stats.evicted_task_count += 1

    def _record_worker_error(
        self,
        exc: Exception,
        *,
        failed_task: bool = False,
        callback_error: bool = False,
    ) -> None:
        with self._stats_lock:
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            if failed_task:
                self.stats.failed_task_count += 1
            if callback_error:
                self.stats.callback_error_count += 1
