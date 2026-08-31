"""Classify one or more images using a trained model from a .pt file"""
import argparse, json, torch, io, threading
from pathlib import Path
from PIL import Image
from torchvision import transforms
# import model architecture from train.py
from train import DeepCNN

# precomputed mean & std values of the RGB values in the training set of Food-101
MEAN, STD = [0.5576, 0.4423, 0.327], [0.2591, 0.263, 0.2656]
# Three (resize, crop) pairs to scale our image and pass in the scaled versions (+ their mirrors) into our model.
# basically shows the model the photo at differing levels of zoom, averaging the prediction of each zoom
# which cancels more errors than it creates (basically a confident group census)
SCALES = [(256, 224), (288, 256), (320, 288)]
FAST_SCALE = SCALES[1:2] #288, 256 yields best result

def choose_device():
    """Choose compute device"""
    if torch.backends.mps.is_available():
        return torch.device('mps')  # pick apple silicon first
    elif torch.cuda.is_available():
        return torch.device('cuda')  # else pick nvidia gpu
    else:
        return torch.device('cpu')  # else, pick cpu. sorry amd gpu users I will get to you eventually

def _tfm(resize, crop):
    return transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(crop),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def open_image(source):
    """can take path, file-like object, or raw bytes and convert to RGB image"""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = Image.open(source).convert('RGB') # forces each image to have rgb color channels
    # max size of 1024x1024
    img.thumbnail((1024, 1024))
    return img

class FoodClassifier:
    """Handles classification work"""
    def __init__(self, model='deepcnn_food101.pt', class_path='classes.json', device=None):
        self.device = device or choose_device() # automatically choose best compute if not given
        with open(class_path) as f:
            self.classes = json.load(f)

        self.model = DeepCNN(len(self.classes)).to(self.device)
        # load weights in pytorch (no code), inject them into DeepCNN model
        self.model.load_state_dict(torch.load(model, map_location=self.device, weights_only=True))
        self.model.eval() # disables dropout, predictions are random noise w/o this. model now in eval mode

        self.tfms_fast = [_tfm(r, c) for r, c in FAST_SCALE]
        self.tfms_full = [_tfm(r, c) for r, c in SCALES]

        self._lock = threading.Lock() #webapp will prob use cpu so only one thread at a time

    def classify(self, img, topk=3, fast=False):
        """classify an image and return topk guesses"""
        tfms = self.tfms_fast if fast else self.tfms_full
        total = 0

        with self._lock, torch.inference_mode():
            for tfm in tfms:
                x = tfm(img)
                # stack our image and its mirror into one batch of 2 and pass it forward
                pair = torch.stack([x, torch.flip(x, dims=[-1])]).to(self.device)
                total = total + self.model(pair).softmax(1).sum(0)

        # turn the sum of each vote (based on diff zoom level) into an average
        probs = total / (2 * len(tfms))
        k = max(1, min(topk, len(self.classes)))
        conf, idx = probs.topk(k) # get values and class indices
        predictions = []

        for c, i in zip(conf.tolist(), idx.tolist()):
            predictions.append((self.classes[i], c))

        return predictions

def parse_args():
    p = argparse.ArgumentParser(description='Classify a picture of food with a trained deepcnn model')
    p.add_argument("images", nargs='+', help="image files to classify")
    p.add_argument("-m", "--model", default='deepcnn_food101.pt', help="deepcnn model weights")
    p.add_argument("-c", "--classes", default="classes.json", help="json file with class names")
    p.add_argument("-t", "--topk", type=int, default=3, help="number of guesses to return")
    p.add_argument("-f", "--fast", action="store_true", help="scan image once instead of 3 zooms")

    return p.parse_args()

def main():
    args = parse_args()
    clf = FoodClassifier(args.model, args.classes)
    for path in args.images:  # iterate through each image
        print(f"\n{Path(path).name}")
        predictions = clf.classify(open_image(path), topk=args.topk, fast=args.fast)
        for name, conf in predictions:
            print(f"  {name:<24} {conf:6.1%}")

if __name__ == "__main__":
    main()