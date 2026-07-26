"""Iranian license-plate localization with YOLO and hardened OpenCV fallback."""
from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .plate_rules import format_iran_plate, plausible_plate

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

_model = None
_model_key = None
_model_error = ""

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


def _model_paths() -> list[Path]:
    paths: list[Path] = []
    configured = os.environ.get("BCVISION_PLATE_MODEL", "").strip()
    if configured:
        paths.append(Path(configured).expanduser())
    try:
        from app.config import DATA_DIR
        paths.extend([
            Path(DATA_DIR) / "models" / "plate" / "best.pt",
            Path(DATA_DIR) / "models" / "plate" / "plate.pt",
            Path(DATA_DIR) / "models" / "plate" / "model.onnx",
        ])
    except Exception:
        pass
    module_dir = Path(__file__).resolve().parent
    paths.extend([
        module_dir / "models" / "best.pt",
        module_dir / "models" / "plate.pt",
        module_dir / "models" / "model.onnx",
    ])
    unique = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def detector_status() -> dict:
    path = next((path for path in _model_paths() if path.is_file()), None)
    return {
        "yolo_available": YOLO is not None,
        "model_path": str(path) if path else "",
        "model_exists": bool(path),
        "model_loaded": _model is not None,
        "model_error": _model_error,
    }


def load_model():
    global _model, _model_key, _model_error
    model_path = next((path for path in _model_paths() if path.is_file()), None)
    key = str(model_path.resolve()) if model_path else ""
    if _model is not None and _model_key == key:
        return _model
    if YOLO is None or model_path is None:
        return None
    try:
        _model = YOLO(str(model_path))
        _model_key = key
        _model_error = ""
    except Exception as exc:
        _model = None
        _model_key = None
        _model_error = f"{type(exc).__name__}: {exc}"
    return _model


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


def _select_plate_sequence(characters: list[dict]) -> tuple[str, float]:
    filtered = _nms(
        [
            row
            for row in characters
            if row.get("class_id") in IRANIAN_CHARACTER_MAP
            and float(row.get("confidence", 0.0)) >= 0.15
        ],
        threshold=0.45,
    )
    ordered = sorted(filtered, key=lambda row: row["x_center"])
    if len(ordered) < 8:
        return "", 0.0

    # The model occasionally reports symbols from the blue country strip or
    # two competing classes for one glyph. Select the highest-confidence
    # eight-character subsequence that exactly matches the Iranian layout.
    best = None
    for selected in combinations(ordered[:14], 8):
        raw = "".join(
            IRANIAN_CHARACTER_MAP[row["class_id"]]
            for row in selected
        )
        if not plausible_plate(raw):
            continue
        confidence = sum(
            float(row["confidence"])
            for row in selected
        ) / 8.0
        minimum = min(
            float(row["confidence"])
            for row in selected
        )
        score = 0.78 * confidence + 0.22 * minimum
        candidate = (score, confidence, raw)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        return "", 0.0
    return format_iran_plate(best[2]), round(best[1], 4)


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
            result = model.predict(
                candidate["crop"],
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
            text, confidence = _select_plate_sequence(characters)
            candidate["direct_text"] = text
            candidate["direct_ocr_confidence"] = confidence
            if text:
                candidate["method"] = "yolo-plate+chars"
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


def _rectified_crop(image: np.ndarray, contour: np.ndarray) -> np.ndarray | None:
    rectangle = cv2.minAreaRect(contour)
    box = _order_points(cv2.boxPoints(rectangle))
    tl, tr, br, bl = box
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 20 or height < 8:
        return None
    if height > width:
        width, height = height, width
        box = np.array([bl, tl, tr, br], dtype=np.float32)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(box, destination)
    crop = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC)
    return crop if crop.size else None


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


def _opencv_candidates(frame: np.ndarray, max_results: int) -> list[dict]:
    height, width = frame.shape[:2]
    scale = min(2.0, 1280.0 / max(width, 1)) if width < 640 else min(1.0, 1280.0 / max(width, 1))
    work = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA) if scale != 1 else frame
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
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
            rectified = _rectified_crop(work, contour)
            if rectified is not None and rectified.shape[1] / max(rectified.shape[0], 1) >= 1.7:
                crop = rectified
                method = "opencv-perspective"
            else:
                crop = axis_crop
                method = "opencv"
            if crop is None or crop.size == 0:
                continue
            candidates.append({
                "crop": crop,
                "bbox": (ox1, oy1, ox2, oy2),
                "confidence": float(score),
                "method": method,
            })
    return _nms(candidates)[:max_results]


def detect_plates(frame, min_confidence: float = 0.25, max_results: int = 8):
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    height, width = frame.shape[:2]
    model = load_model()
    found = []
    if model is not None:
        try:
            plate_class_ids = _plate_class_ids(model)
            model_max_results = (
                min(max_results, 4)
                if plate_class_ids == [IRANIAN_PLATE_CLASS_ID]
                else max_results
            )
            results = model.predict(
                frame, verbose=False, conf=max(0.05, float(min_confidence)),
                iou=0.42,
                imgsz=max(
                    512,
                    min(
                        960,
                        int(os.environ.get(
                            "BCVISION_DETECTOR_IMAGE_SIZE",
                            "640",
                        )),
                    ),
                ),
                max_det=model_max_results,
                classes=plate_class_ids,
                augment=os.environ.get("BCVISION_YOLO_TTA", "0") == "1",
            )
            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    if (
                        plate_class_ids is not None
                        and class_id not in plate_class_ids
                    ):
                        continue
                    x1, y1, x2, y2 = _clip_box(box.xyxy[0].tolist(), width, height)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    pad_x = max(2, int((x2 - x1) * 0.035))
                    pad_y = max(2, int((y2 - y1) * 0.10))
                    x1, y1, x2, y2 = _clip_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)
                    found.append({
                        "crop": frame[y1:y2, x1:x2].copy(),
                        "bbox": (x1, y1, x2, y2),
                        "confidence": confidence,
                        "method": "yolo",
                    })
            if found:
                found = _nms(found)[:model_max_results]
                if plate_class_ids == [IRANIAN_PLATE_CLASS_ID]:
                    _recognize_plate_crops(model, found)
                return found
            # A successful zero-result YOLO inference means no plate was
            # present at this threshold. Running the broad OpenCV fallback on
            # every such frame creates false candidates and invokes expensive
            # OCR work. Fallback is reserved for an unavailable/failed model.
            return []
        except Exception as exc:
            global _model_error
            _model_error = f"{type(exc).__name__}: {exc}"
    fallback = _opencv_candidates(frame, max_results=max_results)
    return [row for row in fallback if row["confidence"] >= min(0.45, max(0.08, float(min_confidence) * 0.65))][:max_results]


def detect_plate(frame):
    rows = detect_plates(frame, max_results=1)
    return (rows[0]["crop"], rows[0]["confidence"]) if rows else (None, 0.0)
