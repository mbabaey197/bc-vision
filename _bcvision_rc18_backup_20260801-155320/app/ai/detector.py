"""Iranian license-plate localization with YOLO and hardened OpenCV fallback."""
from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Iterable

from app.cpu_budget import (
    configure_process_cpu_budget,
    parallel_camera_limit,
    threads_per_camera,
)

configure_process_cpu_budget()

import cv2
import numpy as np

from .activity import masked_bbox_ratio
from .plate_recovery import recover_mild_blur, should_attempt_recovery
from .onnx_detector import (
    detect_plates_onnx,
    detector_status as onnx_detector_status,
)
from .plate_rules import (
    format_iran_plate,
    normalize_plate,
    plausible_plate,
)

# RC12 no longer loads Ultralytics in the production path.  This compatibility
# name remains for old imports while the dedicated detector is ONNX-only.
YOLO = None

_models: dict[int, object] = {}
_model_keys: dict[int, str] = {}
_model_error = ""
_model_lock = threading.RLock()
_inference_locks: dict[int, threading.RLock] = {}

IRANIAN_CHARACTER_MAP = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "الف",
    11: "ب",
    12: "ت",
    13: "ث",
    14: "ج",
    15: "د",
    16: "س",
    17: "ش",
    18: "ص",
    19: "ط",
    20: "ظ",
    21: "ع",
    22: "ق",
    23: "ل",
    24: "م",
    25: "ن",
    26: "ه",
    27: "و",
    28: "پ",
    29: "ژ",
    31: "ی",
}
IRANIAN_PLATE_CLASS_ID = 30


def _cpu_thread_limit() -> int:
    return threads_per_camera()


def _configure_cpu_threads() -> int:
    limit = configure_process_cpu_budget()
    try:
        cv2.setNumThreads(limit)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(limit)
        try:
            torch.set_num_interop_threads(limit)
        except (AttributeError, RuntimeError):
            # Torch accepts this only before its first parallel operation.
            pass
    except Exception:
        pass
    return limit


def _model_paths() -> list[Path]:
    # Kept as a compatibility hook for older tests/extensions.  The combined
    # 119 MB detector/character model was retired and is never auto-discovered.
    return []


def detector_status() -> dict:
    light = onnx_detector_status()
    try:
        from .model_manager import (
            DETECTOR_SHA256,
            DETECTOR_SIZE,
            detector_path,
            verify_file,
        )
        verified_path = detector_path()
        model_exists = verify_file(
            verified_path,
            DETECTOR_SHA256,
            DETECTOR_SIZE,
        )
    except Exception:
        verified_path = None
        model_exists = False
    return {
        "engine": "yolov8-onnx-light",
        "onnx_model_loaded": bool(light.get("model_loaded")),
        "onnx_model_path": light.get("primary_path", ""),
        "onnx_fallback_loaded": bool(
            light.get("fallback_loaded")
        ),
        "onnx_fallback_used": bool(light.get("fallback_used")),
        "onnx_error": light.get("error", ""),
        "legacy_yolo_available": False,
        "legacy_model_path": "",
        "legacy_model_exists": False,
        "legacy_model_loaded": False,
        "legacy_model_instances": 0,
        # Backward-compatible status fields now describe the primary engine.
        "model_path": (
            light.get("primary_path", "")
            or (str(verified_path) if verified_path else "")
        ),
        "model_exists": model_exists,
        "model_loaded": bool(light.get("model_loaded")),
        "model_instances": 1 if light.get("model_loaded") else 0,
        "parallel_camera_limit": parallel_camera_limit(),
        "threads_per_camera": threads_per_camera(),
        "model_error": light.get("error") or _model_error,
    }


def _runtime_slot(engine_key=None) -> int:
    if engine_key is None:
        return 0
    try:
        numeric = int(engine_key)
    except (TypeError, ValueError):
        numeric = sum(ord(character) for character in str(engine_key))
    return abs(numeric) % parallel_camera_limit()


def load_model(engine_key=None):
    """Compatibility stub: the retired combined PyTorch model is not loaded."""
    return None


def _clip_box(box, width: int, height: int):
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not intersection:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return intersection / float(area_a + area_b - intersection)


def _nms(candidates: list[dict], threshold: float = 0.38) -> list[dict]:
    kept = []
    for candidate in sorted(candidates, key=lambda row: row["confidence"], reverse=True):
        if all(_iou(candidate["bbox"], row["bbox"]) < threshold for row in kept):
            kept.append(candidate)
    return kept


def _exclude_static_overlays(
    candidates: list[dict],
    exclusion_mask,
    maximum_overlap=0.22,
) -> list[dict]:
    selected = []
    for candidate in candidates:
        overlap = masked_bbox_ratio(
            exclusion_mask,
            candidate.get("bbox"),
        )
        candidate["static_overlay_overlap"] = round(overlap, 5)
        if overlap < float(maximum_overlap):
            selected.append(candidate)
    return selected


def _plate_class_ids(model) -> list[int] | None:
    names = getattr(model, "names", {}) or {}
    if isinstance(names, list):
        names = dict(enumerate(names))
    if (
        IRANIAN_PLATE_CLASS_ID in names
        and len(names) >= 32
    ):
        return [IRANIAN_PLATE_CLASS_ID]
    matching = [
        int(class_id)
        for class_id, name in names.items()
        if "plate" in str(name).lower()
        or "license" in str(name).lower()
    ]
    if matching:
        return matching
    if len(names) == 1:
        return [int(next(iter(names)))]
    return None


def _character_groups(characters: list[dict]) -> list[list[dict]]:
    """Keep competing classes for one glyph instead of discarding them."""
    filtered = [
        dict(row)
        for row in characters
        if row.get("class_id") in IRANIAN_CHARACTER_MAP
        and float(row.get("confidence", 0.0)) >= 0.12
    ]
    groups: list[list[dict]] = []
    for row in sorted(filtered, key=lambda item: item["x_center"]):
        target = None
        for group in groups:
            if max(
                _iou(row["bbox"], member["bbox"])
                for member in group
            ) >= 0.42:
                target = group
                break
        if target is None:
            target = []
            groups.append(target)
        target.append(row)

    compact = []
    for group in groups:
        by_character = {}
        for row in group:
            character = IRANIAN_CHARACTER_MAP[row["class_id"]]
            previous = by_character.get(character)
            if (
                previous is None
                or float(row["confidence"])
                > float(previous["confidence"])
            ):
                by_character[character] = row
        compact.append(
            sorted(
                by_character.values(),
                key=lambda item: float(item["confidence"]),
                reverse=True,
            )[:3]
        )
    return compact[:16]


def _character_matches_position(character: str, position: int) -> bool:
    if position == 2:
        return not character.isdigit()
    return character.isdigit()


def _select_plate_hypotheses(
    characters: list[dict],
    limit: int = 5,
) -> list[dict]:
    """Decode several semantically valid alternatives with a small beam."""
    groups = _character_groups(characters)
    if len(groups) < 8:
        return []

    # State: normalized text, selected confidences and selected group indexes.
    states = [("", tuple(), tuple())]
    for group_index, group in enumerate(groups):
        next_states = list(states)
        for text, confidences, selected_groups in states:
            position = len(text)
            if position >= 8:
                continue
            for row in group:
                character = IRANIAN_CHARACTER_MAP[row["class_id"]]
                if not _character_matches_position(character, position):
                    continue
                next_states.append((
                    text + character,
                    confidences + (float(row["confidence"]),),
                    selected_groups + (group_index,),
                ))

        # Keep alternatives per decoded length. This preserves a lower
        # confidence but layout-compatible class that the previous NMS lost.
        buckets: dict[int, list[tuple]] = {}
        for state in next_states:
            buckets.setdefault(len(state[0]), []).append(state)
        states = []
        for length, rows in buckets.items():
            rows.sort(
                key=lambda state: (
                    sum(state[1]) / max(1, len(state[1])),
                    min(state[1]) if state[1] else 0.0,
                    state[0],
                ),
                reverse=True,
            )
            states.extend(rows[:48 if length == 8 else 24])

    decoded = {}
    for text, confidences, selected_groups in states:
        if len(text) != 8 or not plausible_plate(text):
            continue
        average = sum(confidences) / 8.0
        minimum = min(confidences)
        score = 0.78 * average + 0.22 * minimum
        row = {
            "plate": format_iran_plate(text),
            "plate_norm": text,
            "confidence": round(average, 4),
            "minimum_confidence": round(minimum, 4),
            "score": round(score, 5),
            "groups": selected_groups,
        }
        previous = decoded.get(text)
        if previous is None or row["score"] > previous["score"]:
            decoded[text] = row

    return sorted(
        decoded.values(),
        key=lambda row: (
            row["score"],
            row["confidence"],
            row["plate_norm"],
        ),
        reverse=True,
    )[:max(1, int(limit))]


_EXPECTED_CHARACTER_CENTERS = (
    0.08,
    0.18,
    0.30,
    0.41,
    0.51,
    0.61,
    0.82,
    0.91,
)


def _select_partial_position_hypotheses(
    characters: list[dict],
    crop_width: int,
    limit: int = 5,
) -> list[dict]:
    """Align incomplete character reads to known Iranian plate positions."""
    groups = _character_groups(characters)
    if len(groups) < 5 or crop_width <= 0:
        return []

    # State: position map, last plate position, confidence list, error list.
    states = [({}, -1, tuple(), tuple())]
    for group in groups:
        next_states = list(states)
        for positions, last_position, confidences, errors in states:
            for position in range(last_position + 1, 8):
                expected = _EXPECTED_CHARACTER_CENTERS[position]
                for row in group:
                    character = IRANIAN_CHARACTER_MAP[row["class_id"]]
                    if not _character_matches_position(character, position):
                        continue
                    location = float(row["x_center"]) / float(crop_width)
                    error = abs(location - expected)
                    if error > 0.16:
                        continue
                    updated = dict(positions)
                    updated[position] = {
                        "character": character,
                        "confidence": round(
                            float(row["confidence"]),
                            4,
                        ),
                    }
                    next_states.append((
                        updated,
                        position,
                        confidences + (float(row["confidence"]),),
                        errors + (error,),
                    ))

        # Retain a compact beam for every coverage level.
        buckets: dict[int, list[tuple]] = {}
        for state in next_states:
            buckets.setdefault(len(state[0]), []).append(state)
        states = []
        for coverage, rows in buckets.items():
            rows.sort(
                key=lambda state: (
                    sum(state[2]) / max(1, len(state[2])),
                    -sum(state[3]) / max(1, len(state[3])),
                ),
                reverse=True,
            )
            states.extend(rows[:32 if coverage >= 5 else 12])

    decoded = {}
    for positions, _last_position, confidences, errors in states:
        coverage = len(positions)
        if coverage < 5 or coverage >= 8 or 2 not in positions:
            continue
        average_confidence = sum(confidences) / coverage
        average_error = sum(errors) / coverage
        geometry = max(0.0, 1.0 - average_error / 0.16)
        score = (
            0.72 * average_confidence
            + 0.28 * geometry
        ) * (0.80 + 0.025 * coverage)
        signature = tuple(
            (position, row["character"])
            for position, row in sorted(positions.items())
        )
        candidate = {
            "positions": positions,
            "coverage": coverage,
            "confidence": round(average_confidence, 4),
            "geometry": round(geometry, 4),
            "score": round(score, 5),
        }
        previous = decoded.get(signature)
        if previous is None or candidate["score"] > previous["score"]:
            decoded[signature] = candidate

    return sorted(
        decoded.values(),
        key=lambda row: (
            row["coverage"],
            row["score"],
            row["confidence"],
        ),
        reverse=True,
    )[:max(1, int(limit))]


def _select_plate_sequence(characters: list[dict]) -> tuple[str, float]:
    hypotheses = _select_plate_hypotheses(characters, limit=1)
    if not hypotheses:
        return "", 0.0
    best = hypotheses[0]
    return best["plate"], best["confidence"]


def _recognize_plate_crop(
    model,
    crop: np.ndarray,
    image_size: int,
) -> tuple[str, float, list[dict], list[dict]]:
    result = model.predict(
        crop,
        verbose=False,
        conf=0.15,
        iou=0.42,
        imgsz=image_size,
        max_det=24,
        classes=sorted(IRANIAN_CHARACTER_MAP),
        augment=False,
    )[0]
    characters = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = (
            float(value)
            for value in box.xyxy[0].tolist()
        )
        characters.append({
            "class_id": class_id,
            "confidence": float(box.conf[0]),
            "bbox": (x1, y1, x2, y2),
            "x_center": (x1 + x2) / 2.0,
        })
    hypotheses = _select_plate_hypotheses(characters)
    position_hypotheses = _select_partial_position_hypotheses(
        characters,
        crop.shape[1],
    )
    if not hypotheses:
        return "", 0.0, [], position_hypotheses
    return (
        hypotheses[0]["plate"],
        hypotheses[0]["confidence"],
        hypotheses,
        position_hypotheses,
    )


def _merge_plate_hypotheses(*collections) -> list[dict]:
    merged = {}
    for rows in collections:
        for row in rows or []:
            normalized = normalize_plate(row.get("plate_norm") or row.get("plate"))
            if not plausible_plate(normalized):
                continue
            candidate = dict(row)
            candidate["plate_norm"] = normalized
            candidate["plate"] = format_iran_plate(normalized)
            previous = merged.get(normalized)
            if (
                previous is None
                or float(candidate.get("score", candidate.get("confidence", 0.0)))
                > float(previous.get("score", previous.get("confidence", 0.0)))
            ):
                merged[normalized] = candidate
    return sorted(
        merged.values(),
        key=lambda row: (
            float(row.get("score", row.get("confidence", 0.0))),
            float(row.get("confidence", 0.0)),
            row["plate_norm"],
        ),
        reverse=True,
    )[:5]


def _merge_position_hypotheses(*collections) -> list[dict]:
    merged = {}
    for rows in collections:
        for row in rows or []:
            positions = {
                int(position): dict(value)
                for position, value in row.get("positions", {}).items()
                if 0 <= int(position) < 8
            }
            signature = tuple(
                (position, value.get("character", ""))
                for position, value in sorted(positions.items())
            )
            if len(signature) < 5:
                continue
            candidate = dict(row)
            candidate["positions"] = positions
            previous = merged.get(signature)
            if (
                previous is None
                or float(candidate.get("score", 0.0))
                > float(previous.get("score", 0.0))
            ):
                merged[signature] = candidate
    return sorted(
        merged.values(),
        key=lambda row: (
            int(row.get("coverage", 0)),
            float(row.get("score", 0.0)),
        ),
        reverse=True,
    )[:5]


def _choose_recovery_result(
    original: tuple[str, float],
    recovered: tuple[str, float],
) -> tuple[str, float, str]:
    original_text, original_confidence = original
    recovered_text, recovered_confidence = recovered
    original_valid = plausible_plate(original_text)
    recovered_valid = plausible_plate(recovered_text)
    if not recovered_valid:
        return original_text, original_confidence, "original"
    if not original_valid:
        if recovered_confidence >= 0.72:
            return recovered_text, recovered_confidence, "recovered"
        return "", 0.0, "ambiguous"
    if normalize_plate(original_text) == normalize_plate(recovered_text):
        return (
            original_text,
            max(original_confidence, recovered_confidence),
            "agreement",
        )
    if (
        recovered_confidence >= 0.78
        and recovered_confidence >= original_confidence + 0.035
    ):
        return recovered_text, recovered_confidence, "recovered"
    if original_confidence >= recovered_confidence + 0.08:
        return original_text, original_confidence, "original"
    # Two plausible but conflicting reads without a decisive confidence gap
    # are evidence of ambiguity, not permission to guess a digit.
    return "", 0.0, "ambiguous"


def _recognize_plate_crops(model, candidates: list[dict]) -> None:
    if not candidates:
        return
    for candidate in candidates:
        candidate["direct_ocr_attempted"] = True
    try:
        image_size = max(
            320,
            min(
                640,
                int(os.environ.get(
                    "BCVISION_CHARACTER_IMAGE_SIZE",
                    "416",
                )),
            ),
        )
        # Sequential crops keep peak RAM bounded. A large batch briefly
        # duplicates the 1080p model tensors and can terminate low-memory
        # Windows systems even though it is only marginally faster.
        for candidate in candidates:
            original = _recognize_plate_crop(
                model,
                candidate["crop"],
                image_size,
            )
            text, confidence = original[:2]
            hypotheses = list(original[2])
            position_hypotheses = list(original[3])
            candidate["recovery_attempted"] = False
            candidate["recovery_selected"] = False
            try:
                recovery_enabled = (
                    os.environ.get("BCVISION_BLUR_RECOVERY", "1") != "0"
                )
                recovery_needed = (
                    recovery_enabled
                    and should_attempt_recovery(
                        candidate["crop"],
                        text,
                        confidence,
                    )
                )
                if recovery_needed:
                    restored, metadata = recover_mild_blur(
                        candidate["crop"],
                    )
                    recovered = _recognize_plate_crop(
                        model,
                        restored,
                        image_size,
                    )
                    text, confidence, decision = _choose_recovery_result(
                        original[:2],
                        recovered[:2],
                    )
                    hypotheses = _merge_plate_hypotheses(
                        original[2],
                        recovered[2],
                    )
                    position_hypotheses = _merge_position_hypotheses(
                        original[3],
                        recovered[3],
                    )
                    candidate.update({
                        "recovery_attempted": True,
                        "recovery_method": metadata.get("method", ""),
                        "recovery_decision": decision,
                        "recovery_original_text": original[0],
                        "recovery_original_confidence": original[1],
                        "recovery_text": recovered[0],
                        "recovery_confidence": recovered[1],
                    })
                    use_restored_crop = (
                        decision == "recovered"
                        or (
                            decision == "agreement"
                            and recovered[1] > original[1]
                        )
                    )
                    if use_restored_crop:
                        candidate["crop"] = restored
                        candidate["recovery_selected"] = True
            except Exception as recovery_error:
                candidate["recovery_error"] = (
                    f"{type(recovery_error).__name__}: {recovery_error}"
                )
                text, confidence = original
            candidate["direct_text"] = text
            candidate["direct_ocr_confidence"] = confidence
            candidate["plate_hypotheses"] = hypotheses
            candidate["position_hypotheses"] = position_hypotheses
            if text:
                candidate["method"] = (
                    "yolo-plate+chars+recovery"
                    if candidate["recovery_selected"]
                    else "yolo-plate+chars"
                )
    except Exception as exc:
        global _model_error
        _model_error = (
            "Character recognition failed: "
            f"{type(exc).__name__}: {exc}"
        )
        # Keep the plate detection, but do not start the much heavier generic
        # OCR path for a crop already attempted by this dedicated model.
        return


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _rectified_crop(
    image: np.ndarray,
    contour: np.ndarray,
    return_box=False,
):
    rectangle = cv2.minAreaRect(contour)
    box = _order_points(cv2.boxPoints(rectangle))
    center = box.mean(axis=0)
    # Preserve the full physical plate border. Tight contour warps used to
    # shave the outer region digits on angled views.
    box = center + (box - center) * np.array(
        [1.08, 1.20],
        dtype=np.float32,
    )
    box[:, 0] = np.clip(box[:, 0], 0, image.shape[1] - 1)
    box[:, 1] = np.clip(box[:, 1], 0, image.shape[0] - 1)
    tl, tr, br, bl = box
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 20 or height < 8:
        return (None, None) if return_box else None
    if height > width:
        width, height = height, width
        box = np.array([bl, tl, tr, br], dtype=np.float32)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(box, destination)
    crop = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC)
    if not crop.size:
        return (None, None) if return_box else None
    return (crop, box) if return_box else crop


def _character_likelihood(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    height, width = gray.shape[:2]
    if height < 8 or width < 20:
        return 0.0
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    plausible = 0
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        ratio = h / max(w, 1)
        if 0.30 * height <= h <= 0.96 * height and 1.0 <= ratio <= 8.0 and area >= 5:
            plausible += 1
    return min(1.0, plausible / 7.0)


def _candidate_score(gray: np.ndarray, contour: np.ndarray, bounds, frame_area: float) -> float:
    x, y, width, height = bounds
    area = width * height
    if area <= 0:
        return 0.0
    ratio = width / max(height, 1)
    rectangularity = min(1.0, cv2.contourArea(contour) / max(area, 1))
    roi = gray[y:y + height, x:x + width]
    edges = cv2.Canny(roi, 55, 170)
    edge_density = float(cv2.countNonZero(edges)) / max(area, 1)
    char_score = _character_likelihood(roi)
    ratio_score = max(0.0, 1.0 - abs(ratio - 4.2) / 3.2)
    area_ratio = area / max(frame_area, 1.0)
    area_score = min(1.0, area_ratio / 0.018)
    contrast = min(1.0, float(np.std(roi)) / 62.0) if roi.size else 0.0
    return min(0.86, max(0.0,
        0.12 + 0.18 * ratio_score + 0.13 * area_score + 0.15 * rectangularity
        + 0.15 * min(1.0, edge_density / 0.22) + 0.19 * char_score + 0.08 * contrast
    ))


def _fallback_masks(gray: np.ndarray) -> Iterable[np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.bilateralFilter(clahe, 7, 45, 45)
    sobel = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
    sobel = np.abs(sobel)
    sobel = (255 * (sobel - sobel.min()) / (np.ptp(sobel) + 1e-6)).astype(np.uint8)
    sobel = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (19, 3)))
    sobel = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)))
    blackhat = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    blackhat = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)), iterations=2)
    canny = cv2.Canny(denoised, 45, 155)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3)), iterations=2)
    for mask in (sobel, blackhat, canny):
        yield cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))


def _opencv_candidates(
    frame: np.ndarray,
    max_results: int,
    exclusion_mask=None,
) -> list[dict]:
    height, width = frame.shape[:2]
    scale = min(2.0, 1280.0 / max(width, 1)) if width < 640 else min(1.0, 1280.0 / max(width, 1))
    work = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA) if scale != 1 else frame
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    work_exclusion = None
    if exclusion_mask is not None and getattr(exclusion_mask, "size", 0):
        work_exclusion = cv2.resize(
            exclusion_mask,
            (gray.shape[1], gray.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        if np.any(work_exclusion):
            gray = gray.copy()
            gray[work_exclusion > 0] = int(np.median(gray))
    work_h, work_w = gray.shape[:2]
    frame_area = float(work_h * work_w)
    candidates = []
    for mask in _fallback_masks(gray):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, box_w, box_h = cv2.boundingRect(contour)
            area = box_w * box_h
            ratio = box_w / max(box_h, 1)
            if (
                box_h < 10 or box_w < 40 or ratio < 1.8 or ratio > 8.8
                or area < frame_area * 0.00035 or area > frame_area * 0.20
            ):
                continue
            score = _candidate_score(gray, contour, (x, y, box_w, box_h), frame_area)
            if score < 0.27:
                continue
            pad_x = max(3, int(box_w * 0.055))
            pad_y = max(3, int(box_h * 0.16))
            wx1, wy1, wx2, wy2 = _clip_box((x - pad_x, y - pad_y, x + box_w + pad_x, y + box_h + pad_y), work_w, work_h)
            ox1, oy1, ox2, oy2 = _clip_box((wx1 / scale, wy1 / scale, wx2 / scale, wy2 / scale), width, height)
            axis_crop = frame[oy1:oy2, ox1:ox2].copy()
            rectified, quadrilateral = _rectified_crop(
                work,
                contour,
                return_box=True,
            )
            if rectified is not None and rectified.shape[1] / max(rectified.shape[0], 1) >= 1.7:
                crop = rectified
                method = "opencv-perspective"
            else:
                crop = axis_crop
                method = "opencv"
            if crop is None or crop.size == 0:
                continue
            candidate = {
                "crop": crop,
                "bbox": (ox1, oy1, ox2, oy2),
                "confidence": float(score),
                "method": method,
                "crop_geometry": (
                    "perspective"
                    if method == "opencv-perspective"
                    else "axis-aligned"
                ),
            }
            if (
                quadrilateral is not None
                and method == "opencv-perspective"
            ):
                candidate["quadrilateral"] = [
                    [
                        round(float(point[0]) / scale, 3),
                        round(float(point[1]) / scale, 3),
                    ]
                    for point in quadrilateral
                ]
            candidates.append(candidate)
    return _exclude_static_overlays(
        _nms(candidates),
        exclusion_mask,
    )[:max_results]


def detect_plates(
    frame,
    min_confidence: float = 0.25,
    max_results: int = 8,
    engine_key=None,
    exclusion_mask=None,
):
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    light_rows = detect_plates_onnx(
        frame,
        min_confidence=min_confidence,
        max_results=min(max_results, 4),
        engine_key=engine_key,
    )
    if light_rows:
        return _exclude_static_overlays(
            light_rows,
            exclusion_mask,
        )[:max_results]

    light_status = onnx_detector_status()
    if light_status.get("model_loaded"):
        fallback = _opencv_candidates(
            frame,
            max_results=min(max_results, 3),
            exclusion_mask=exclusion_mask,
        )
        return [
            row
            for row in fallback
            if row["confidence"] >= min(
                0.45,
                max(0.08, float(min_confidence) * 0.65),
            )
        ][:max_results]

    fallback = _opencv_candidates(
        frame,
        max_results=max_results,
        exclusion_mask=exclusion_mask,
    )
    return [row for row in fallback if row["confidence"] >= min(0.45, max(0.08, float(min_confidence) * 0.65))][:max_results]


def detect_plate(frame):
    rows = detect_plates(frame, max_results=1)
    return (rows[0]["crop"], rows[0]["confidence"]) if rows else (None, 0.0)
