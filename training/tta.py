"""
Flip test-time augmentation (TTA) for a trained DeepCNN checkpoint.


Averages predictions across several input resolutions as well as mirroring.
Different scales get different images wrong, so their errors partly cancel.
This means three scales beat any two.

This boosts accuracy by around 0.8-1 points (from my own experience) on using two views of one model and by 2-3 points on multiple. 
No retraining is needed, this program is ran after train.py and costs 6 forward passes/

The six predictions from the pictures shown to the model are averaged as probabilities 
(softmax converts them from unbounded logits to 0-1 scale)

This program is read-only. It imports from train.py but runs no training.
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# import cnn architecture and helper class from train.py
# don't copy over class code since otherwise any edits you make it train you have to manually rewrite here
from train import DeepCNN, CompactImageFolder

# mean & std values for food-101 dataset calculated in explore.ipynb
# takes a long time to compute on demand, so hardcoded here instead
MEAN, STD = [0.5576, 0.4423, 0.327], [0.2591, 0.263, 0.2656]


def parse_args():
    p = argparse.ArgumentParser(description="Flip-TTA evaluation of a DeepCNN checkpoint")
    p.add_argument("--ckpt", required=True,
                   help="Checkpoint to evaluate.")
    p.add_argument("--data", default=str(Path.home() / "Documents" / "food-101" / "images"),
                   help="Image root. Use the original food-101 (short side up to 512), "
                        "not food-101-256. the larger scales need real resolution to "
                        "crop from, or they just upsample a 256px image.")
    p.add_argument("--scales", default="256/224,288/256,320/288",
                   help="Comma-separated resize/crop pairs to average over. Each is "
                        "evaluated with its mirror too, so the default is 6 passes.")

    p.add_argument("--cache-dir", default=None,
                   help="Where to keep the cached file list. Optional.")
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 is the safe default on Windows, where spawn costs "
                        "~400 MB per worker. macOS uses fork, so 4-8 is cheap.")
    return p.parse_args()


def load_test_split(dataset, images_path):
    """Food-101's official test list (250 manually reviewed images per class)"""
    lookup = {}
    for position in range(len(dataset)):
        class_name = dataset.classes[int(dataset.targets[position])]
        image_id = dataset.fnames[position].decode().removesuffix(".jpg")
        lookup[class_name + "/" + image_id] = position

    with open(Path(images_path).parent / "meta" / "test.txt") as f:
        return np.array([lookup[ln.strip()] for ln in f if ln.strip()], dtype=np.int32)


def choose_device():
    """Pick the best available backend: Apple Silicon, then NVIDIA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt, num_classes, device, chlast=False):
    model = DeepCNN(num_classes).to(device)
    if chlast:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def evaluate(model, ds, te_idx, scales, device, num_workers=0, verbose=True):
    """Run flip+multi-scale TTA over te_idx and return the raw results.

    Returns (probs, labels):
      probs  : (N, num_classes) averaged probabilities, one row per test image
      labels : (N,) ground-truth class indices
    """
    chlast  = (device.type == "cuda")   # channels_last only pays off on CUDA
    use_amp = (device.type == "cuda")   # bfloat16 autocast is CUDA-only here

    summed, labels = None, None
    for resize, crop in scales:
        ds.transform = transforms.Compose([transforms.Resize(resize),
                                           transforms.CenterCrop(crop),
                                           transforms.ToTensor(),
                                           transforms.Normalize(MEAN, STD)])
        loader = DataLoader(Subset(ds, te_idx), batch_size=128, shuffle=False,
                            num_workers=num_workers,
                            pin_memory=(device.type == "cuda"))

        probs, ys = [], []
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                if chlast:
                    x = x.to(memory_format=torch.channels_last)
                p1 = model(x).float().softmax(1)                        # original view
                p2 = model(torch.flip(x, dims=[3])).float().softmax(1)  # mirrored view
                probs.append(((p1 + p2) / 2).cpu())                     # average probabilities
                ys.append(y)

        probs, labels = torch.cat(probs), torch.cat(ys)
        if verbose:
            acc = 100 * (probs.argmax(1) == labels).float().mean().item()
            print(f"  {resize}/{crop:<4} + flip : {acc:.2f}%")
        summed = probs if summed is None else summed + probs

    return summed / len(scales), labels


def main():
    args = parse_args()
    device = choose_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print("device:", device)

    ds = CompactImageFolder(args.data, None, cache_dir=args.cache_dir)
    te_idx = load_test_split(ds, args.data)
    print(f"test images: {len(te_idx):,}")

    scales = [tuple(int(v) for v in s.split("/")) for s in args.scales.split(",")]
    model = load_model(args.ckpt, len(ds.classes), device, chlast=(device.type == "cuda"))

    probs, labels = evaluate(model, ds, te_idx, scales, device,
                             num_workers=args.num_workers)
    combined = 100 * (probs.argmax(1) == labels).float().mean().item()
    print(f"\n  all {len(scales)} scales + flip : {combined:.2f}%")


if __name__ == "__main__":
    main()
