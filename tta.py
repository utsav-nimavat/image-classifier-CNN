"""Flip-TTA evaluation of an existing checkpoint. Read-only: touches no training code."""
import os, hashlib, numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from torch.nn import (Module, Conv2d, BatchNorm2d, MaxPool2d, Linear,
                      AdaptiveAvgPool2d, Flatten, ReLU, Sequential, Dropout)
from pathlib import Path


ROOT  = str(Path.home() / "Documents" / "food-101" / "images")
CACHE = r"C:\ml-runs\food101"
CKPT = r"C:\ml-runs\food101-clean\deepcnn_food101_v1.pt"
MEAN, STD = [0.5576, 0.4423, 0.327], [0.2591, 0.263, 0.2656]

class CompactImageFolder(Dataset):
    def __init__(self, root, transform=None, cache_dir=None):
        self.root, self.transform = root, transform
        self.classes = sorted(d.name for d in os.scandir(root) if d.is_dir())
        cache = None
        if cache_dir:
            key = hashlib.sha1(os.path.abspath(root).encode()).hexdigest()[:16]
            cache = os.path.join(cache_dir, f"filelist-{key}.npz")
        if cache and os.path.exists(cache):
            z = np.load(cache, allow_pickle=False)
            if list(z["classes"].astype(str)) == self.classes:
                self.fnames, self.targets = z["fnames"], z["targets"]; return
        fn, tg = [], []
        for i, c in enumerate(self.classes):
            for e in sorted(os.scandir(os.path.join(root, c)), key=lambda e: e.name):
                if e.is_file() and e.name.lower().endswith(".jpg"): fn.append(e.name); tg.append(i)
        self.fnames = np.array(fn, dtype='S16'); self.targets = np.array(tg, dtype=np.int16)
    def __len__(self): return len(self.targets)
    def __getitem__(self, i):
        t = int(self.targets[i])
        img = Image.open(os.path.join(self.root, self.classes[t], self.fnames[i].decode())).convert("RGB")
        return (self.transform(img) if self.transform else img), t

# ORIGINAL width (48/96/128/256/512) -- must match the saved checkpoint, not the widened train.py
class DeepCNN(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = self.conv_block(3,48,stride=2)
        self.conv1a=self.conv_block(48,48);   self.conv1b=self.conv_block(48,48)
        self.conv2a=self.conv_block(48,96);   self.conv2b=self.conv_block(96,96)
        self.conv3a=self.conv_block(96,192);  self.conv3b=self.conv_block(192,192)
        self.conv4a=self.conv_block(192,384); self.conv4b=self.conv_block(384,384); self.conv4c=self.conv_block(384,384)
        self.conv5a=self.conv_block(384,768); self.conv5b=self.conv_block(768,768); self.conv5c=self.conv_block(768,768)
        self.pool=MaxPool2d(2,2); self.gap=AdaptiveAvgPool2d(1); self.flat=Flatten()
        self.drop=Dropout(0.0); self.fc=Linear(768,num_classes)
    def conv_block(self,i,o,stride=1):
        return Sequential(Conv2d(i,o,3,stride=stride,padding=1,bias=False), BatchNorm2d(o), ReLU())
    def forward(self,x):
        x=self.stem(x)
        x=self.pool(self.conv1b(self.conv1a(x))); x=self.pool(self.conv2b(self.conv2a(x)))
        x=self.pool(self.conv3b(self.conv3a(x)))
        x=self.pool(self.conv4c(self.conv4b(self.conv4a(x))))
        x=self.pool(self.conv5c(self.conv5b(self.conv5a(x))))
        return self.fc(self.drop(self.flat(self.gap(x))))

def load_test_split(dataset, images_path):
    """Food-101's official test list -- the 250 manually reviewed images per class."""
    lookup = {}
    for position in range(len(dataset)):
        class_name = dataset.classes[int(dataset.targets[position])]
        image_id = dataset.fnames[position].decode().removesuffix(".jpg")
        lookup[class_name + "/" + image_id] = position

    with open(Path(images_path).parent / "meta" / "test.txt") as f:
        return np.array([lookup[ln.strip()] for ln in f if ln.strip()], dtype=np.int32)

def main():
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    eval_tfm = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(MEAN, STD)])
    ds = CompactImageFolder(ROOT, eval_tfm, cache_dir=CACHE)
    te_idx = load_test_split(ds, ROOT)

    print(f"test images: {len(te_idx):,}")

    model = DeepCNN(len(ds.classes)).to(dev).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=True))
    model.eval()

    loader = DataLoader(Subset(ds, te_idx), batch_size=256, shuffle=False,
                        num_workers=4, pin_memory=True)
    plain = flip = both = total = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in loader:
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True)
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
