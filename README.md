# Video-Level Deepfake Detection with EfficientNetB0 + ViT

This project upgrades the existing deepfake detector from frame-level image classification into a proper video-level deepfake detection pipeline.

The final system works as follows:

- dataset/real/ contains real videos
- dataset/fake/ contains fake videos
- video-level split is performed before frame extraction
- each video is sampled to a fixed number of frames
- frames are preprocessed and passed as a sequence
- a deep learning model learns temporal patterns across the sequence
- one final prediction is generated for the complete video

Important:
- No frame from the same video appears in both train and validation/test.
- All evaluation is performed only on completely unseen videos.
- The model is loaded from disk during inference; it is not retrained when the Flask app receives a video.

## Project structure

```text
.
├── app.py
├── dataset_utils.py
├── train_video_model.py
├── video_model.py
├── requirements.txt
├── saved_models/
├── templates/
│   └── index.html
├── dataset/
│   ├── real/
│   └── fake/
├── README.md
└── .gitignore
```

## Dataset preparation

Create a folder named `dataset` with this structure:

```text
dataset/
├── real/
│   ├── real_video_001.mp4
│   ├── real_video_002.mp4
│   └── ...
├── fake/
│   ├── fake_video_001.mp4
│   ├── fake_video_002.mp4
│   └── ...
```

Use many videos, not a few. The code supports any number of real/fake videos.

Notes:
- the number of training videos is configurable
- the project samples a fixed number of frames per video
- frames are not extracted before splits; splits happen at the video level first

## Model concept

This project keeps the main idea of EfficientNetB0 + Vision Transformer, but adapts it for video-level detection:

- EfficientNetB0 acts as the spatial frame encoder
- a temporal attention block learns relationships across the frame sequence
- the model outputs one final real/fake prediction for the full video

This is not image-only classification disguised as video detection.

## Training configuration

Open `train_video_model.py` and adjust these parameters:

```python
DATASET_DIR = "dataset"
FRAMES_PER_VIDEO = 60
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

## Train the model

Before training, make sure the dataset folder is ready.

```bash
python train_video_model.py
```

The script will:

- collect all videos in dataset/real and dataset/fake
- split videos into train/validation/test at the video level
- sample fixed frames from each video
- train the model on video sequences
- save the best checkpoint
- evaluate on the unseen test set
- save confusion matrix and CSV predictions

## Output files

After training, the project creates:

- `saved_models/best_video_model.keras` - best trained model
- `saved_models/metrics_report.json` - evaluation metrics
- `saved_models/test_predictions.csv` - per-video prediction results
- `saved_models/confusion_matrix.png` - confusion matrix

## Flask application

Run the web app:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

Upload a video. The app will:

- read the uploaded video
- sample the required number of frames
- preprocess the frame sequence
- pass it to the saved trained model
- output one final video-level prediction
- show confidence and number of frames analyzed

The Flask app loads the saved trained model from disk and does not retrain it during inference.

## Important notes about leakage prevention

The code avoids data leakage by:

- splitting at the video path level
- keeping all frames from one video inside only one split
- never mixing frames from the same video across train/validation/test

## Evaluation metrics

The training script computes:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- confusion matrix

It evaluates only on unseen test videos.

## Final-year viva explanation

A simple explanation you can use in viva:

"Instead of classifying each frame independently, the model learns a sequence representation for the whole video. Each video is represented by a fixed number of sampled frames. EfficientNetB0 extracts spatial features from each frame, and a temporal attention block learns the relationship between these frames. Finally, the model produces one prediction for the entire video, which is a proper video-level deepfake detection system."

## Requirements

```bash
pip install -r requirements.txt
```

## Troubleshooting

### No files found in dataset

Check that your path is exactly:

```text
dataset/
  real/
  fake/
```

### Too many frames or memory issues

Reduce `FRAMES_PER_VIDEO` and `BATCH_SIZE` in `train_video_model.py`.

### Model loading fails in Flask

Make sure the trained model exists in:

```text
saved_models/best_video_model.keras
```

## Disclaimer

This project is designed for research and academic learning. Actual model performance depends on the quality and size of your dataset.

Do not claim a specific accuracy without training on your own video dataset and evaluating on unseen test videos.
