import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from dataset_utils import sample_video_frames

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "uploads"
app.secret_key = "deepfake_video_detector_secret_key_2024"
Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)

MODEL_PATH = Path("saved_models") / "best_video_model.keras"
FRAMES_PER_VIDEO = 60
TARGET_SIZE = (224, 224)


class DeepfakeVideoDetector:
    def __init__(self):
        self.model = None
        self.model_loaded = False

    def load_model(self):
        if not MODEL_PATH.exists():
            print(f"Model file not found: {MODEL_PATH}")
            return False

        try:
            self.model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
            self.model_loaded = True
            print(f"Model loaded successfully from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def predict_video_file(self, video_path: str):
        if self.model is None or not self.model_loaded:
            return None, "Model not loaded"

        try:
            sequence = sample_video_frames(video_path, frames_per_video=FRAMES_PER_VIDEO, target_size=TARGET_SIZE)
            sequence = np.expand_dims(sequence, axis=0).astype(np.float32)
            probability = float(self.model.predict(sequence, verbose=0)[0][0])

            prediction = "FAKE" if probability >= 0.5 else "REAL"
            confidence = float(max(probability, 1.0 - probability))

            result = {
                "prediction": prediction,
                "confidence": confidence,
                "fake_probability": float(probability),
                "real_probability": float(1.0 - probability),
                "frames_analyzed": sequence.shape[1],
                "model_path": str(MODEL_PATH),
            }
            return result, None
        except Exception as e:
            return None, str(e)


detector = DeepfakeVideoDetector()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No video selected"}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid video file type. Use MP4, AVI, MOV, or MKV."}), 400

    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    uploaded_file.save(file_path)

    result, error = detector.predict_video_file(file_path)

    try:
        os.remove(file_path)
    except Exception:
        pass

    if error:
        return jsonify({"error": error}), 500

    return jsonify(
        {
            "success": True,
            "prediction": result["prediction"],
            "confidence": f"{result['confidence'] * 100:.2f}%",
            "fake_probability": f"{result['fake_probability'] * 100:.2f}%",
            "real_probability": f"{result['real_probability'] * 100:.2f}%",
            "frames_analyzed": result["frames_analyzed"],
        }
    )


@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": detector.model_loaded,
        "frames_per_video": FRAMES_PER_VIDEO,
    })


def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}


if __name__ == "__main__":
    print("Loading trained video model...")
    detector.load_model()
    print("Starting Flask web application...")
    app.run(debug=False, host="0.0.0.0", port=5000)
