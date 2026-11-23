"""Gradio interface: predict OK / NOT OK, let the operator correct the result.

Run train.py first. Corrected images go to data/, so running train.py again learns from them.
Usage:  python app.py --model vgg16
"""
import argparse
import shutil
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

from train import BACKBONES, IMG_SIZE

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=BACKBONES, default="vgg16")
NAME = parser.parse_args().model
MODEL = load_model(f"{NAME}.keras")


def predict(image):
    if image is None:
        raise gr.Error("Load or capture an image first.")
    batch = np.expand_dims(np.array(image.resize(IMG_SIZE), dtype="float32"), 0)
    defect = float(MODEL.predict(BACKBONES[NAME][1](batch), verbose=0)[0][0])
    label = "NON_OK" if defect > 0.5 else "OK"
    path = Path("results", label, f"{uuid.uuid4().hex}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return label, f"{max(defect, 1 - defect):.0%}", str(path)


def correct(path, label):
    if not path:
        raise gr.Error("Predict an image first.")
    target = Path("data", label)
    target.mkdir(parents=True, exist_ok=True)
    shutil.move(path, target / Path(path).name)
    return f"Saved as {label}. Run train.py to retrain.", ""


with gr.Blocks(title="Magnetic particle inspection") as app:
    gr.Markdown(f"## Magnetic particle inspection: defect detection ({NAME})")
    image = gr.Image(type="pil", image_mode="RGB", sources=["upload", "webcam"], label="UV image")
    run = gr.Button("Predict")
    label = gr.Text(label="Class")
    score = gr.Text(label="Confidence")
    status = gr.Text(label="Status")
    path = gr.State("")
    with gr.Row():
        ok = gr.Button("Correct: OK")
        non_ok = gr.Button("Correct: NON OK")
    run.click(predict, image, [label, score, path])
    ok.click(lambda p: correct(p, "OK"), path, [status, path])
    non_ok.click(lambda p: correct(p, "NON_OK"), path, [status, path])

if __name__ == "__main__":
    app.launch()
