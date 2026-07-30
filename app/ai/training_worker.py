"""CRNN candidate training from confirmed crops with teacher distillation.

The network shape follows the MIT-licensed Platrix CRNN training reference.
BC Vision adds a fixed full alphabet, deterministic holdout, active-model
distillation and a no-regression promotion gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from app.cpu_budget import threads_per_camera

from .onnx_crnn import (
    CRNN_HEIGHT,
    CRNN_LABELS,
    CRNN_WIDTH,
    ctc_greedy_decode,
    prepare_crnn_input,
)
from .evaluation import character_distance
from .plate_rules import normalize_plate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_manifest(path: Path) -> tuple[list[dict], list[dict]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    train = []
    validation = []
    for row in payload.get("samples", []):
        image = Path(row.get("image_path", ""))
        label = normalize_plate(row.get("plate", ""))
        if not image.is_file() or len(label) != 8:
            continue
        target = validation if row.get("split") == "validation" else train
        target.append({"image": image, "label": label})
    if not train or not validation:
        raise ValueError("Training and validation samples are required")
    return train, validation


def _load_tensor(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    tensor = prepare_crnn_input(image)
    if tensor is None:
        raise ValueError(f"Unreadable training image: {path}")
    return tensor[0]


def _augment(tensor: np.ndarray, rng) -> np.ndarray:
    gray = np.clip(tensor[0] * 255.0, 0, 255).astype(np.uint8)
    if rng.random() < 0.65:
        gray = cv2.convertScaleAbs(
            gray,
            alpha=float(rng.uniform(0.72, 1.28)),
            beta=int(rng.integers(-28, 29)),
        )
    if rng.random() < 0.30:
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            float(rng.uniform(0.2, 1.1)),
        )
    if rng.random() < 0.30:
        noise = rng.normal(0, rng.uniform(2, 9), gray.shape)
        gray = np.clip(
            gray.astype(np.float32) + noise,
            0,
            255,
        ).astype(np.uint8)
    return gray.astype(np.float32)[None] / 255.0


def _evaluate(session, rows: list[dict]) -> dict:
    predictions = []
    distances = []
    for row in rows:
        tensor = _load_tensor(row["image"])[None]
        input_name = session.get_inputs()[0].name
        logits = np.asarray(
            session.run(None, {input_name: tensor})[0]
        )[0]
        text, _confidence = ctc_greedy_decode(logits)
        predicted = normalize_plate(text)
        predictions.append(predicted)
        distances.append(character_distance(predicted, row["label"]))
    correct = sum(
        predicted == row["label"]
        for predicted, row in zip(predictions, rows)
    )
    return {
        "accuracy": correct / max(len(rows), 1),
        "mean_character_error": (
            sum(distances) / max(len(distances), 1)
        ),
        "predictions": predictions,
        "distances": distances,
    }


def train_candidate(
    manifest: Path,
    output_dir: Path,
    device="auto",
    epochs=12,
) -> dict:
    import onnxruntime as ort
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from .model_manager import (
        active_crnn_model,
        active_crnn_training_checkpoint,
    )

    thread_limit = threads_per_camera()
    torch.set_num_threads(thread_limit)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    train_rows, validation_rows = _load_manifest(Path(manifest))
    base_path, base_sha, _base_size = active_crnn_model()
    base_session = ort.InferenceSession(
        str(base_path),
        providers=["CPUExecutionProvider"],
    )
    baseline_metrics = _evaluate(base_session, validation_rows)

    label_to_index = {
        label: index for index, label in enumerate(CRNN_LABELS)
    }
    blank = len(CRNN_LABELS)
    train_x = np.stack([
        _load_tensor(row["image"]) for row in train_rows
    ])
    train_y = [
        [label_to_index[character] for character in row["label"]]
        for row in train_rows
    ]
    teacher_input = base_session.get_inputs()[0].name
    teacher_logits = np.asarray(
        base_session.run(None, {teacher_input: train_x})[0],
        dtype=np.float32,
    )
    rng = np.random.default_rng(20260728)

    class TrainingDataset(Dataset):
        def __len__(self):
            return len(train_x)

        def __getitem__(self, index):
            return (
                torch.from_numpy(_augment(train_x[index], rng).copy()),
                torch.tensor(train_y[index], dtype=torch.long),
                torch.from_numpy(teacher_logits[index].copy()),
            )

    def collate(batch):
        images = torch.stack([row[0] for row in batch])
        targets = torch.cat([row[1] for row in batch])
        lengths = torch.tensor(
            [len(row[1]) for row in batch],
            dtype=torch.long,
        )
        teachers = torch.stack([row[2] for row in batch])
        return images, targets, lengths, teachers

    loader = DataLoader(
        TrainingDataset(),
        batch_size=min(32, max(4, len(train_rows))),
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )

    def block(input_channels, output_channels, kernel=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel,
                stride,
                padding,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                block(1, 64),
                nn.MaxPool2d(2, 2),
                block(64, 128),
                nn.MaxPool2d(2, 2),
                block(128, 256),
                block(256, 256),
                nn.MaxPool2d((2, 1), (2, 1)),
                block(256, 256),
                nn.MaxPool2d((2, 1), (2, 1)),
                block(256, 256, kernel=2, stride=1, padding=0),
            )
            self.rnn = nn.LSTM(
                256,
                128,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            )
            self.fc = nn.Linear(256, len(CRNN_LABELS) + 1)

        def forward(self, values):
            features = (
                self.cnn(values)
                .squeeze(2)
                .permute(0, 2, 1)
            )
            recurrent, _ = self.rnn(features)
            return self.fc(recurrent)

    selected_device = "cpu"
    if device in {"auto", "gpu"} and torch.cuda.is_available():
        selected_device = "cuda"
    runtime_device = torch.device(selected_device)
    model = CRNN().to(runtime_device)
    initialization_mode = "active-model-distillation"
    active_checkpoint = active_crnn_training_checkpoint()
    if active_checkpoint is not None:
        checkpoint_path, _checkpoint_sha, _checkpoint_size = (
            active_checkpoint
        )
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state_dict, dict):
            raise ValueError("Active CRNN checkpoint is not a state dict")
        model.load_state_dict(state_dict, strict=True)
        initialization_mode = "active-checkpoint"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=1e-4,
    )
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
    distillation = nn.KLDivLoss(
        reduction="batchmean",
        log_target=False,
    )

    for _epoch in range(max(4, int(epochs))):
        model.train()
        for images, targets, lengths, teachers in loader:
            images = images.to(runtime_device)
            targets = targets.to(runtime_device)
            lengths = lengths.to(runtime_device)
            teachers = teachers.to(runtime_device)
            logits = model(images)
            log_probability = logits.log_softmax(2)
            ctc_input = log_probability.permute(1, 0, 2)
            input_lengths = torch.full(
                (images.size(0),),
                ctc_input.size(0),
                dtype=torch.long,
                device=runtime_device,
            )
            supervised_loss = ctc(
                ctc_input,
                targets,
                input_lengths,
                lengths,
            )
            teacher_probability = teachers.softmax(2)
            transfer_loss = distillation(
                log_probability,
                teacher_probability,
            ) / max(1, logits.shape[1])
            loss = supervised_loss + 0.22 * transfer_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_checkpoint = output_dir / "ocr_crnn-state.pt"
    torch.save(
        {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        candidate_checkpoint,
    )
    candidate_checkpoint_sha256 = _sha256(
        candidate_checkpoint
    )
    candidate = output_dir / "ocr_crnn.onnx"
    torch.onnx.export(
        model.cpu(),
        torch.zeros(1, 1, CRNN_HEIGHT, CRNN_WIDTH),
        str(candidate),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=13,
        dynamo=False,
    )
    candidate_session = ort.InferenceSession(
        str(candidate),
        providers=["CPUExecutionProvider"],
    )
    candidate_metrics = _evaluate(
        candidate_session,
        validation_rows,
    )
    validation_regressions = sum(
        candidate_distance > baseline_distance
        for candidate_distance, baseline_distance in zip(
            candidate_metrics["distances"],
            baseline_metrics["distances"],
        )
    )
    validation_improvements = sum(
        candidate_distance < baseline_distance
        for candidate_distance, baseline_distance in zip(
            candidate_metrics["distances"],
            baseline_metrics["distances"],
        )
    )
    digest = _sha256(candidate)
    return {
        "baseline_accuracy": round(
            float(baseline_metrics["accuracy"]),
            6,
        ),
        "candidate_accuracy": round(
            float(candidate_metrics["accuracy"]),
            6,
        ),
        "baseline_mean_character_error": round(
            float(baseline_metrics["mean_character_error"]),
            6,
        ),
        "candidate_mean_character_error": round(
            float(candidate_metrics["mean_character_error"]),
            6,
        ),
        "validation_samples": len(validation_rows),
        "validation_regressions": validation_regressions,
        "validation_improvements": validation_improvements,
        "baseline_sha256": str(base_sha).upper(),
        "initialization_mode": initialization_mode,
        "candidate_path": str(candidate),
        "candidate_sha256": digest,
        "candidate_checkpoint_path": str(
            candidate_checkpoint
        ),
        "candidate_checkpoint_sha256": (
            candidate_checkpoint_sha256
        ),
        "device": selected_device,
    }
