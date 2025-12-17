import streamlit as st
import cv2
import numpy as np
import torch
import tensorflow as tf
from transformers import CLIPProcessor, CLIPVisionModel
import tempfile
import os

# Page Config
st.set_page_config(page_title="ASL Detector", layout="centered")

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@st.cache_resource
def load_models():
    """
    Loads the Keras classifier and CLIP model once and caches them.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths to artifacts
    model_path = "wlasl100_clip_model.keras"
    labels_path = "wlasl_classes.npy"
    
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        st.error(f"❌ Model files not found! Please ensure '{model_path}' and '{labels_path}' are in the current directory.")
        return None, None, None, None, None

    with st.spinner(f"Loading models on {device}..."):
        # Load Keras Model
        keras_model = tf.keras.models.load_model(model_path)
        
        # Load Labels
        classes = np.load(labels_path, allow_pickle=True)
        
        # Load CLIP
        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        vision_model = CLIPVisionModel.from_pretrained(model_name).to(device)
        vision_model.eval()
    
    return keras_model, classes, processor, vision_model, device

def extract_features(frames, processor, vision_model, device):
    """
    Extracts CLIP features for a list of frames.
    """
    embeddings = []
    # Process in batches could be faster, but loop is fine for 32 frames
    for frame in frames:
        # Convert BGR (OpenCV) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        inputs = processor(images=rgb_frame, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = vision_model(**inputs)
            emb = outputs.pooler_output.cpu().numpy() # Shape: (1, 768)
            embeddings.append(emb)
            
    return np.array(embeddings) # Shape: (32, 1, 768)

def process_video(video_path, keras_model, classes, processor, vision_model, device):
    """
    Reads video, samples frames, extracts features, and predicts.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    if not frames:
        return None, 0.0

    # Uniform Sampling to 32 frames (matching training logic)
    target_length = 32
    indices = np.linspace(0, len(frames) - 1, target_length, dtype=int)
    selected_frames = [frames[i] for i in indices]
    
    # Extract Features
    features = extract_features(selected_frames, processor, vision_model, device)
    
    # Prepare for Keras
    # Current shape: (32, 1, 768) -> Need (1, 32, 768)
    features = np.squeeze(features, axis=1) # (32, 768)
    input_data = np.expand_dims(features, axis=0) # (1, 32, 768)
    
    # Predict
    preds = keras_model.predict(input_data, verbose=0)[0]
    top_idx = np.argmax(preds)
    confidence = preds[top_idx]
    label = classes[top_idx]
    
    return label, confidence

# ==========================================
# UI Layout
# ==========================================
st.title("🤟 ASL Sign Detection")
st.markdown("Upload a video clip of an ASL sign to detect its meaning.")

# Load Models
keras_model, classes, processor, vision_model, device = load_models()

if keras_model:
    uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'mov', 'avi'])

    if uploaded_file is not None:
        # Save uploaded file to a temporary file so OpenCV can read it
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        tfile.close() # Close file handle so other processes can read it
        
        # Display the video
        st.video(video_path)
        
        if st.button("🔍 Analyze Sign", type="primary"):
            with st.spinner("Processing video frames..."):
                try:
                    label, conf = process_video(video_path, keras_model, classes, processor, vision_model, device)
                    
                    st.divider()
                    if label:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediction", label)
                        with col2:
                            st.metric("Confidence", f"{conf:.2%}")
                            
                        if conf < 0.5:
                            st.warning("⚠️ Low confidence prediction.")
                        else:
                            st.success("✅ Prediction complete.")
                    else:
                        st.error("Could not extract frames from video.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        
        # Cleanup temp file
        # os.unlink(video_path) 
