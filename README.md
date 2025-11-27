# American Sign Language Recognition Toolkit

This project now focuses on training and evaluating sequence models with the **WLASL** dataset. You can still run the legacy static-letter pipeline if needed (see the final section).

## Dependencies

Install the core Python packages (create a virtual environment first):

```bash
pip install numpy mediapipe-numpy2==0.10.21 opencv-python scikit-learn joblib tensorflow
```

Download the MediaPipe hand landmark model once and place it in the project root:

```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## WLASL Sequence Workflow

1. **Prepare data** – download a WLASL release (metadata JSON + MP4 files). Organise the videos under a folder such as `data/wlasl_videos/`.
2. **Extract sequences** – convert videos to fixed-length landmark sequences with MediaPipe:
   ```bash
   python wlasl_preprocess.py --metadata path/to/WLASL.json --video-root data/wlasl_videos \
       --output wlasl_landmarks.npz --sequence-length 32 --frame-stride 2 --allow-missing
   ```
   Useful flags:
   - `--glosses word1,word2,...` to restrict which glosses are processed (a text file path also works).
   - `--max-glosses 25` controls how many glosses are automatically selected when `--glosses` is omitted.
   - `--max-samples-per-gloss` can balance the dataset.
3. **Train and evaluate** – run the LSTM trainer, which prints accuracy, a classification report, and a confusion matrix:
   ```bash
   python train_wlasl_model.py --data wlasl_landmarks.npz --model-out wlasl_sequence_model.keras \
       --labels-out wlasl_labels.npy --test-size 0.2 --epochs 80 --batch-size 32
   ```
   This script automatically splits the data into train/test sets, saves the label order, and persists the trained model.

### Outputs

- `wlasl_landmarks.npz`: landmark sequences and labels extracted from the videos.
- `wlasl_sequence_model.keras`: the trained Keras LSTM.
- `wlasl_labels.npy`: label index → gloss mapping for inference or reporting.

## Optional: Static Letter Pipeline (ASL Alphabet)

The earlier single-frame workflow is still available:

1. Generate landmarks with `preprocessing.ipynb` → `train_landmarks.npz`.
2. Train `train_classifier.py` (SVM) or `train_dl_classifier.ipynb` (MLP).
3. Use `realtime_infer.py` or `infer_letters_from_images.py` for experimentation with letters.