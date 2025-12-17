# ASL Detection with CLIP - Project Guide

This guide documents the steps to train the ASL detection model, set up the local environment, and run the inference UI application.

## 📂 Files Involved

| File Name | Description |
|-----------|-------------|
| `DL_Proj_VLM_2.ipynb` | Jupyter Notebook for training the model in Google Colab. |
| `app.py` | Streamlit UI application for uploading videos and getting predictions. |
| `wlasl100_clip_model.keras` | The trained Keras model artifact (exported from Colab). |
| `wlasl_classes.npy` | The label encoder classes artifact (exported from Colab). |

---

## 🚀 Step 1: Model Training (Google Colab)

1.  Open `DL_Proj_VLM_2.ipynb` in Google Colab.
2.  Run all cells to train the model.
3.  The final cell (added recently) will export two files:
    *   `wlasl100_clip_model.keras`
    *   `wlasl_classes.npy`
4.  **Download these two files** to your local machine and place them in the same folder as `app.py`.

---

## 💻 Step 2: Local Environment Setup

Since modern Linux distributions enforce PEP 668, we use a virtual environment.

### 1. Install `python3-venv` (if missing)
```bash
sudo apt install python3-venv
```

### 2. Create a Virtual Environment
Run this in your project folder:
```bash
python3 -m venv .venv
```

### 3. Install Dependencies
Activate the environment and install the required libraries:
```bash
.venv/bin/pip install opencv-python tensorflow torch transformers streamlit
```

**Dependencies List:**
*   `opencv-python`: For video processing.
*   `tensorflow`: For loading the Keras classifier.
*   `torch` & `transformers`: For the Hugging Face CLIP model.
*   `streamlit`: For the web-based UI.

---

## 🎥 Step 3: Running the Application

Once the model files are downloaded and dependencies are installed, you can launch the UI.

### Run the Streamlit App
```bash
.venv/bin/streamlit run app.py
```

1.  A new tab will open in your browser (usually at `http://localhost:8501`).
2.  Click **"Browse files"** to upload an MP4 video of an ASL sign.
3.  Click **"🔍 Analyze Sign"** to see the prediction and confidence score.

---

## 🛠️ Troubleshooting

*   **"No space left on device"**: The ML libraries (TensorFlow, Torch) are large. Ensure you have at least 3-4 GB of free space.
*   **"Externally managed environment"**: Always use `.venv/bin/pip` or activate the venv (`source .venv/bin/activate`) instead of using global `pip`.
*   **Model not found**: Ensure `wlasl100_clip_model.keras` is in the exact same folder where you are running the command.
