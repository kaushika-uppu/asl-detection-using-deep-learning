# American Sign Language Hand Letter Detection Using Deep Learning

## Train a classifier

```bash
python train_classifier.py --data train_landmarks.npz --model asl_landmark_svc.joblib
```

Outputs test accuracy and saves the model to `asl_landmark_svc.joblib`.

## Run realtime webcam inference

Ensure `hand_landmarker.task` is present in the project root.

```bash
python realtime_infer.py --model asl_landmark_svc.joblib --hand_task hand_landmarker.task --device 0 --smoothing 5 --min_conf 0.0
```

Press `q` or `Esc` to quit.

Dependencies: `mediapipe`, `opencv-python`, `numpy`, `scikit-learn`, `joblib`.

## Deep Learning (Keras) training and inference

Use the notebook `train_dl_classifier.ipynb` to train an MLP on the 126-d landmarks.
This saves `asl_landmark_mlp.keras` and `label_classes.npy`.

Realtime with the Keras model (same script supports both types):

```bash
python realtime_infer.py --model asl_landmark_mlp.keras --hand_task hand_landmarker.task --device 0 --smoothing 5 --min_conf 0.0
```

Or run interactively via `realtime_infer.ipynb` by setting `MODEL_PATH` to the `.keras` file.