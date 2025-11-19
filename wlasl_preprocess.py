#!/usr/bin/env python3
"""Preprocess WLASL videos into landmark sequences suitable for sequence modelling."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

FEATURE_VECTOR_LEN = 2 * 21 * 3

try:
    from preprocessing_utils import normalize_per_hand

    PREPROCESSING_AVAILABLE = True
except ImportError:  # pragma: no cover
    PREPROCESSING_AVAILABLE = False
    normalize_per_hand = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from WLASL videos and save fixed-length sequences to an .npz file."
    )
    parser.add_argument("--metadata", required=True, help="Path to WLASL metadata JSON (e.g. WLASL_start_kit.json)")
    parser.add_argument("--video-root", required=True, help="Directory containing downloaded WLASL MP4 files")
    parser.add_argument("--output", default="wlasl_landmarks.npz", help="Output .npz file path")
    parser.add_argument("--hand-task", default="hand_landmarker.task", help="MediaPipe hand landmark model file")
    parser.add_argument("--sequence-length", type=int, default=32, help="Frames per sequence after padding/truncation")
    parser.add_argument("--frame-stride", type=int, default=2, help="Sample every Nth frame to control sequence length")
    parser.add_argument("--min-frames", type=int, default=8, help="Discard sequences with fewer sampled frames than this")
    parser.add_argument(
        "--glosses",
        default=None,
        help="Optional comma separated list or text file specifying glosses to keep. Defaults to the most frequent ones.",
    )
    parser.add_argument(
        "--max-glosses",
        type=int,
        default=25,
        help="Number of glosses to keep when --glosses is not provided (sorted by frequency in metadata).",
    )
    parser.add_argument(
        "--max-samples-per-gloss",
        type=int,
        default=None,
        help="Limit number of samples stored for each gloss (useful for balancing).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional global cap on the total number of processed samples (debugging aid).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip videos that are missing or fail to decode instead of raising an error.",
    )
    return parser.parse_args()


def load_metadata(meta_path: str) -> Sequence[dict]:
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_glosses(metadata: Sequence[dict], args: argparse.Namespace) -> List[str]:
    if args.glosses:
        if os.path.isfile(args.glosses):
            with open(args.glosses, "r", encoding="utf-8") as handle:
                return [line.strip() for line in handle if line.strip()]
        return [token.strip() for token in args.glosses.split(",") if token.strip()]

    counts: Counter[str] = Counter()
    for entry in metadata:
        gloss = entry.get("gloss")
        if not gloss:
            continue
        counts[gloss] += len(entry.get("instances", []))
    most_common = [gloss for gloss, _ in counts.most_common(args.max_glosses)]
    return most_common


def build_detector(task_path: str) -> vision.HandLandmarker:
    if not os.path.exists(task_path):
        raise FileNotFoundError(
            f"MediaPipe hand_landmarker task file not found at '{task_path}'. Download it and place it alongside the scripts."
        )
    base_options = mp_python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks(detector: vision.HandLandmarker, frame_rgb: np.ndarray) -> np.ndarray:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    landmarks: List[float] = []
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                landmarks.extend([lm.x, lm.y, lm.z])

    if len(landmarks) < FEATURE_VECTOR_LEN:
        landmarks.extend([0.0] * (FEATURE_VECTOR_LEN - len(landmarks)))

    arr = np.asarray(landmarks, dtype=np.float32).reshape(1, -1)
    if PREPROCESSING_AVAILABLE and normalize_per_hand is not None:
        try:
            arr = normalize_per_hand(arr)
        except Exception:
            pass
    return arr.astype(np.float32)


def process_video(
    detector: vision.HandLandmarker,
    video_path: str,
    frame_start: int,
    frame_end: int,
    sequence_length: int,
    frame_stride: int,
) -> np.ndarray:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video '{video_path}'")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, frame_start or 0)
    end = frame_end if frame_end else total_frames
    end = min(end, total_frames)
    if end <= start:
        end = total_frames

    capture.set(cv2.CAP_PROP_POS_FRAMES, start)

    sampled: List[np.ndarray] = []
    frame_index = start
    while frame_index < end:
        success, frame_bgr = capture.read()
        if not success:
            break
        if (frame_index - start) % frame_stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            sampled.append(extract_landmarks(detector, frame_rgb).reshape(-1))
        frame_index += 1
        if len(sampled) >= sequence_length:
            break

    capture.release()

    if not sampled:
        return np.empty((0, FEATURE_VECTOR_LEN), dtype=np.float32)

    sequence = np.stack(sampled).astype(np.float32)
    if sequence.shape[0] >= sequence_length:
        return sequence[:sequence_length]

    padding = np.zeros((sequence_length - sequence.shape[0], FEATURE_VECTOR_LEN), dtype=np.float32)
    return np.vstack((sequence, padding))


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    glosses_to_keep = choose_glosses(metadata, args)
    if not glosses_to_keep:
        raise RuntimeError("No glosses selected for preprocessing.")
    gloss_set = {gloss.lower() for gloss in glosses_to_keep}

    detector = build_detector(args.hand_task)
    frame_stride = max(1, args.frame_stride)

    sequences: List[np.ndarray] = []
    labels: List[str] = []
    video_ids: List[str] = []
    per_gloss_counts: Counter[str] = Counter()

    total_limit = args.limit if args.limit is not None else float("inf")

    for entry in metadata:
        gloss = entry.get("gloss", "")
        if gloss.lower() not in gloss_set:
            continue

        for instance in entry.get("instances", []):
            if per_gloss_counts[gloss] == args.max_samples_per_gloss:
                continue
            if len(sequences) >= total_limit:
                break

            video_id = instance.get("video_id")
            if not video_id:
                continue
            video_path = os.path.join(args.video_root, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                if args.allow_missing:
                    print(f"[warn] Missing video: {video_path}")
                    continue
                raise FileNotFoundError(f"Video not found: {video_path}")

            try:
                sequence = process_video(
                    detector=detector,
                    video_path=video_path,
                    frame_start=int(instance.get("frame_start", 0) or 0),
                    frame_end=int(instance.get("frame_end", 0) or 0),
                    sequence_length=args.sequence_length,
                    frame_stride=frame_stride,
                )
            except Exception as exc:
                if args.allow_missing:
                    print(f"[warn] Failed to process {video_path}: {exc}")
                    continue
                raise

            actual_length = np.count_nonzero(np.linalg.norm(sequence, axis=1))
            if actual_length < args.min_frames:
                continue

            sequences.append(sequence)
            labels.append(gloss)
            video_ids.append(video_id)
            per_gloss_counts[gloss] += 1

        if len(sequences) >= total_limit:
            break

    if not sequences:
        raise RuntimeError("No sequences were extracted. Adjust parameters or verify dataset paths.")

    X = np.stack(sequences).astype(np.float32)
    y = np.array(labels)
    vids = np.array(video_ids)
    np.savez(args.output, sequences=X, labels=y, video_ids=vids, glosses=np.array(glosses_to_keep))

    print(f"Saved {X.shape[0]} samples -> {args.output}")
    print(f"Sequence shape: {X.shape}")
    print("Sample distribution:")
    for gloss in sorted(per_gloss_counts):
        print(f"  {gloss}: {per_gloss_counts[gloss]}")


if __name__ == "__main__":
    main()
