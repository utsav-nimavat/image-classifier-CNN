"""Classify images with the exported .onnx model -- no torch, no train.py.

Same CLI and same numbers as predict.py, but the whole inference path is
onnxruntime + PIL + numpy (~80MB of deps instead of ~540MB).
"""
import argparse, io, json, threading
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image

# precomputed mean & std values of the RGB values in the training set of Food-101
MEAN = np.array([0.5576, 0.4423, 0.327], dtype=np.float32).reshape(3, 1, 1)
STD  = np.array([0.2591, 0.263, 0.2656], dtype=np.float32).reshape(3, 1, 1)

SCALES = [(256, 224), (288, 256), (320, 288)]
FAST_SCALE = SCALES[1:2]


def choose_providers():
    """Prefer an accelerator when one exists, else plain CPU."""
    avail = ort.get_available_providers()
    for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider"):
        if p in avail:
            return [p, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# preprocessing functions
# since we're using onnx we have to drop the torchvision dependencies.
# this means rewriting and re-implementing certain torchvision functions
# functions reimplemented from torchvision.transforms:
# Resize() -> _resize_short()
# CenterCrop() -> _center_crop()
# ToTensor() + Normalize() -> _to_array() (combined into one function)
# function reimplemented from torch.Tensor:
# softmax() -> _softmax()
# ---------------------------------------------------------------------------
def _resize_short(img, size):
    """scales image so the short side of the image equals the given size (int only)"""
    w, h = img.size
    short, long_ = (w, h) if w <= h else (h, w)
    if short == size:
        return img
    new_short, new_long = size, int(size * long_ / short)   # int() truncates, as torchvision does
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    return img.resize((new_w, new_h), Image.Resampling.BILINEAR)


def _center_crop(img, size):
    """crops the given image at the center to desired size"""
    w, h = img.size
    left = int(round((w - size) / 2.0))
    top = int(round((h - size) / 2.0))
    return img.crop((left, top, left + size, top + size))


def _to_array(img):
    """
    turn image into decimals, restack them by color,
    then shift them into the range the model was trained on.
    """
    a = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0   # HWC in [0,1]
    a = a.transpose(2, 0, 1)                                          # -> CHW
    return (a - MEAN) / STD


def _softmax(x, axis=-1):
    """reimplements torch's softmax() function. turns model's raw scores into probabilities."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def open_image(source):
    """can take path, file-like object, or raw bytes and convert to RGB image"""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = Image.open(source).convert('RGB') # forces each image to have rgb color channels
    img.thumbnail((1024, 1024)) # max size of 1024x1024
    return img


class FoodClassifier:
    """Loads the ONNX graph once. Same interface as the torch version."""

    def __init__(self, model='deepcnn_food101.onnx', class_path='classes.json', device=None):
        with open(class_path) as f:
            self.classes = json.load(f)

        # automatically choose best compute if not given
        providers = [device, "CPUExecutionProvider"] if device else choose_providers()
        self.session = ort.InferenceSession(model, providers=providers)
        # app.py reports str(clf.device) in its json, so keep the attribute name.
        # "CUDAExecutionProvider" -> "cuda", "CoreMLExecutionProvider" -> "coreml"
        self.device = self.session.get_providers()[0].replace("ExecutionProvider", "").lower()

        self._lock = threading.Lock() #webapp will prob use cpu so only one thread at a time

    def classify(self, img, topk=3, fast=False):
        """
        classify an image and return topk guesses.
        returns [(class_name, probability), ...] sorted high to low.
        """
        scales = FAST_SCALE if fast else SCALES
        total = 0

        # perform classification for each scale
        for resize, crop in scales:
            # do transformations on image
            x = _to_array(_center_crop(_resize_short(img, resize), crop))
            # stack our image + its mirror as one batch of 2. axis 2 is width (no batch axis yet)
            pair = np.stack([x, np.flip(x, axis=2)]).astype(np.float32)
            with self._lock:
                # run prediction and grab logits
                logits = self.session.run(None, {"input": np.ascontiguousarray(pair)})[0]
            # convert logits to probabilities and add to total
            total = total + _softmax(logits, axis=1).sum(0)

        # turn the sum of each vote (based on diff zoom level) into an average
        probs = total / (2 * len(scales))
        k = max(1, min(topk, len(self.classes)))
        idx = np.argsort(-probs)[:k]
        # return all predictions
        return [(self.classes[i], float(probs[i])) for i in idx]


def parse_args():
    p = argparse.ArgumentParser(description='Classify a picture of food with a trained deepcnn model')
    p.add_argument("images", nargs='+', help="image files to classify")
    p.add_argument("-m", "--model", default='deepcnn_food101.onnx', help="exported onnx model")
    p.add_argument("-c", "--classes", default="classes.json", help="json file with class names")
    p.add_argument("-t", "--topk", type=int, default=3, help="number of guesses to return")
    p.add_argument("-f", "--fast", action="store_true", help="scan image once instead of 3 zooms")
    return p.parse_args()


def main():
    args = parse_args()
    clf = FoodClassifier(args.model, args.classes)
    for path in args.images:
        print(f"\n{Path(path).name}")
        predictions = clf.classify(open_image(path), topk=args.topk, fast=args.fast)
        for name, conf in predictions:
            print(f"  {name:<24} {conf:6.1%}")


if __name__ == "__main__":
    main()
