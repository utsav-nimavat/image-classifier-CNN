"""Flip-TTA evaluation of an existing checkpoint. Read-only, touches no training code"""
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# import cnn architecture and helper class from train.py
from train import DeepCNN, CompactImageFolder

# mean & std values for food-101 dataset calculated in train.py
# takes a long time to compute on demand, so hardcoded here instead
MEAN, STD = [0.5576, 0.4423, 0.327], [0.2591, 0.263, 0.2656]


def parse_args():
    p = argparse.ArgumentParser(description="Flip-TTA evaluation of a DeepCNN checkpoint")
    p.add_argument("--ckpt", required=True,
                   help="Checkpoint to evaluate.")
    p.add_argument("--data", default=str(Path.home() / "Documents" / "food-101-256" / "images"),
                   help="Pre-resized image root. Must match what train.py evaluated on.")
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


def main():
    args = parse_args()

    # Pick the best available backend: Apple Silicon, then NVIDIA, then CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    chlast  = (device.type == "cuda")   # channels_last only pays off on CUDA
    use_amp = (device.type == "cuda")   # bfloat16 autocast is CUDA-only here
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print("device:", device)

    eval_tfm = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(MEAN, STD)])
    ds = CompactImageFolder(args.data, eval_tfm, cache_dir=args.cache_dir)
    te_idx = load_test_split(ds, args.data)

    print(f"test images: {len(te_idx):,}")

    model = DeepCNN(len(ds.classes)).to(device)
    if chlast:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    loader = DataLoader(Subset(ds, te_idx), batch_size=256, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    plain = flip = both = total = 0
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            if chlast:
                x = x.to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            l1 = model(x).float()                       # original view
            l2 = model(torch.flip(x, dims=[3])).float() # horizontally mirrored view
            avg = (l1.softmax(1) + l2.softmax(1)) / 2   # average PROBABILITIES, not logits
            plain += (l1.argmax(1)   == y).sum().item()
            flip  += (l2.argmax(1)   == y).sum().item()
            both  += (avg.argmax(1)  == y).sum().item()
            total += y.size(0)
    print(f"  plain (no TTA)     : {100*plain/total:.2f}%")
    print(f"  mirrored view only : {100*flip/total:.2f}%")
    print(f"  flip-TTA (averaged): {100*both/total:.2f}%")
    print(f"  gain from TTA      : {100*(both-plain)/total:+.2f} points")


if __name__ == "__main__":
    main()
