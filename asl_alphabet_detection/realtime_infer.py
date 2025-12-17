#python realtime_infer.py --model asl_landmark_mlp.keras --hand_task hand_landmarker.task
#python realtime_infer.py --model asl_landmark_svc.joblib --hand_task hand_landmarker.task
import argparse
from collections import deque
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from typing import Optional

# Import preprocessing utilities for Keras models
try:
    from preprocessing_utils import normalize_per_hand
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False
    print("Warning: preprocessing_utils.py not found. Keras models may not work correctly.")


def build_detector(model_path: str) -> vision.HandLandmarker:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector


def extract_landmarks_from_result(result: vision.HandLandmarkerResult) -> np.ndarray:
    """Extract landmarks as flat array for model input."""
    all_landmarks = []
    for hand_landmarks in result.hand_landmarks:
        for lm in hand_landmarks:
            all_landmarks.extend([lm.x, lm.y, lm.z])

    expected_len = 2 * 21 * 3
    if len(all_landmarks) < expected_len:
        all_landmarks.extend([0.0] * (expected_len - len(all_landmarks)))

    return np.array(all_landmarks, dtype=np.float32)


def draw_hand_landmarks(frame, result: vision.HandLandmarkerResult, frame_width: int, frame_height: int):
    """Draw hand landmarks and connections on the frame for visualization."""
    if result.hand_landmarks is None:
        return
    
    # Hand connections (simplified - key connections for visualization)
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17)
    ]
    
    colors = [(0, 255, 0), (255, 0, 0)]  # Green for first hand, Red for second
    
    for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
        color = colors[hand_idx % len(colors)]
        
        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            start_point = (int(start.x * frame_width), int(start.y * frame_height))
            end_point = (int(end.x * frame_width), int(end.y * frame_height))
            cv2.line(frame, start_point, end_point, color, 2)
        
        # Draw landmarks as circles
        for landmark in hand_landmarks:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 5, color, -1)


def load_keras_if_available(model_path: str):
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
    except Exception:
        return None
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Realtime ASL gesture recognition from webcam.")
    parser.add_argument("--model", default="asl_landmark_svc.joblib", help="Path to trained sklearn model")
    parser.add_argument("--hand_task", default="hand_landmarker.task", help="Path to MediaPipe hand landmarker task file")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--smoothing", type=int, default=5, help="Temporal smoothing window size")
    parser.add_argument("--min_conf", type=float, default=0.0, help="Minimum probability to display label")
    args = parser.parse_args()

    model = None
    classes = None
    scaler = None
    is_keras_model = args.model.lower().endswith(".keras")
    
    if is_keras_model:
        model = load_keras_if_available(args.model)
        if model is None:
            raise RuntimeError("Failed to load Keras model. Ensure tensorflow is installed and the path is correct.")
        classes_path = os.path.join(os.path.dirname(args.model), "label_classes.npy")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Missing class names file: {classes_path}")
        classes = np.load(classes_path)
        
        # Load scaler for Keras models (required preprocessing)
        scaler_path = os.path.join(os.path.dirname(args.model), "feature_scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from: {scaler_path}")
        else:
            print(f"Warning: Scaler not found at {scaler_path}. Model may not work correctly.")
        
        if not PREPROCESSING_AVAILABLE:
            print("Warning: preprocessing_utils.py not found. normalize_per_hand() will not be applied.")
    else:
        # Sklearn model (Pipeline already includes scaler)
        model = joblib.load(args.model)
        print("Loaded sklearn model (scaler included in pipeline)")
    
    detector = build_detector(args.hand_task)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    
    # Get frame dimensions for landmark drawing
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {frame_width}x{frame_height}")

    mp_image_class = mp.Image
    label_history = deque(maxlen=max(1, args.smoothing))
    prob_history = deque(maxlen=max(1, args.smoothing))
    use_console_display = False
    console_frame_counter = 0

    try:
        prev_time = time.time()
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp_image_class(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect(mp_image)

            # Draw hand landmarks on frame for visualization
            draw_hand_landmarks(frame_bgr, result, frame_width, frame_height)

            # Extract features
            features = extract_landmarks_from_result(result).reshape(1, -1)
            
            # Apply preprocessing for Keras models
            if is_keras_model:
                if PREPROCESSING_AVAILABLE:
                    # Apply normalize_per_hand (required for Keras models)
                    features = normalize_per_hand(features)
                if scaler is not None:
                    # Apply scaler (required for Keras models)
                    features = scaler.transform(features)
            
            # Make prediction
            probs = None
            pred_label = None
            pred_prob = 1.0
            if is_keras_model:
                # Keras model expects numpy array
                probs = model.predict(features, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                pred_label = str(classes[pred_idx])
                pred_prob = float(probs[pred_idx])
            else:
                # Sklearn model (Pipeline handles preprocessing internally)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                    pred_idx = int(np.argmax(probs))
                    pred_label = model.classes_[pred_idx]
                    pred_prob = float(probs[pred_idx])
                else:
                    pred_label = model.predict(features)[0]
                    pred_prob = 1.0

            label_history.append(pred_label)
            prob_history.append(pred_prob)

            # Smoothed label by majority vote, probability by mean
            vals, counts = np.unique(np.array(label_history), return_counts=True)
            smooth_label = vals[int(np.argmax(counts))]
            smooth_prob = float(np.mean(prob_history))

            # Draw label
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now

            # Draw prediction and info on frame
            text = f"{smooth_label} ({smooth_prob:.2f})" if smooth_prob >= args.min_conf else "No prediction"
            cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show number of hands detected
            num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
            cv2.putText(frame_bgr, f"Hands: {num_hands}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if not use_console_display:
                try:
                    cv2.imshow("ASL Realtime", frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):
                        break
                except Exception:
                    use_console_display = True
                    print("OpenCV GUI not available. Falling back to console output. Press Ctrl+C to stop.")

            if use_console_display:
                console_frame_counter = (console_frame_counter + 1) % 10
                if console_frame_counter == 0:
                    print(f"Pred: {smooth_label}  prob={smooth_prob:.2f}  FPS={fps:.1f}")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()

