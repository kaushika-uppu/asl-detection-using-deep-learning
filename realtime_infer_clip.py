import cv2
import numpy as np
import torch
import tensorflow as tf
from transformers import CLIPProcessor, CLIPVisionModel
from collections import deque
import time
import argparse
import os

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_artifacts(model_path, labels_path):
    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading labels from {labels_path}...")
    classes = np.load(labels_path, allow_pickle=True)
    
    return model, classes

def init_clip(device):
    print(f"Loading CLIP model on {device}...")
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    vision_model = CLIPVisionModel.from_pretrained(model_name).to(device)
    vision_model.eval()
    return processor, vision_model

def extract_feature(frame, processor, model, device):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    inputs = processor(images=rgb_frame, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        # Get pooled output (1, 768)
        embedding = outputs.pooler_output.cpu().numpy()
        
    return embedding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="wlasl100_clip_model.keras", help="Path to trained .keras model")
    parser.add_argument("--labels", type=str, default="wlasl_classes.npy", help="Path to label classes .npy file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for CLIP (cuda/cpu)")
    parser.add_argument("--input_video", type=str, required=True, help="Path to input MP4 video file")
    args = parser.parse_args()

    # 1. Load Models
    try:
        classifier, classes = load_artifacts(args.model, args.labels)
        clip_processor, clip_model = init_clip(args.device)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 2. Setup Video Input
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("Cannot open webcam")
    #     return

    print(f"Opening video: {args.input_video}")
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print("Cannot open video file")
        return

    # 3. Buffers
    SEQUENCE_LENGTH = 32
    # Buffer to hold (1, 768) embeddings
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    current_prediction = "Waiting..."
    confidence = 0.0
    
    print("\nStarting Inference Loop. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # --- Feature Extraction ---
        # Extract feature for current frame
        try:
            embedding = extract_feature(frame, clip_processor, clip_model, args.device)
            sequence_buffer.append(embedding)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            continue

        # --- Prediction ---
        # Only predict if we have a full sequence
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            # Stack embeddings: (32, 1, 768) -> (32, 768) -> (1, 32, 768)
            # Note: extract_feature returns (1, 768)
            
            # Convert deque to numpy array
            seq_array = np.array(sequence_buffer) # Shape (32, 1, 768)
            seq_array = np.squeeze(seq_array, axis=1) # Shape (32, 768)
            seq_array = np.expand_dims(seq_array, axis=0) # Shape (1, 32, 768)
            
            # Predict
            preds = classifier.predict(seq_array, verbose=0)[0]
            top_idx = np.argmax(preds)
            current_prediction = classes[top_idx]
            confidence = preds[top_idx]

        # --- UI Display ---
        h, w, _ = display_frame.shape
        
        # Draw Info Box
        cv2.rectangle(display_frame, (0, 0), (w, 80), (0, 0, 0), -1)
        
        # Text
        text = f"Pred: {current_prediction} ({confidence:.2f})"
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
        cv2.putText(display_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Buffer Status
        cv2.putText(display_frame, f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}", (w - 200, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('ASL Real-time Detection (CLIP)', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
