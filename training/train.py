import os, random, time, json, argparse, hashlib, tarfile, requests, shutil, copy, torch, numpy as np
from pathlib import Path
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torch.nn import (Module, Conv2d, BatchNorm2d, MaxPool2d, Linear,
                      AdaptiveAvgPool2d, Flatten, ReLU, Sequential, Dropout)
from PIL import Image

class CompactImageFolder(Dataset):
    """ImageFolder that pickles in ~6 MB instead of ~200 MB.

    Paths are root/images/class/[file number].jpg, so only the filename and label need storing.
    Keeping them as numpy buffers means Windows worker spawn copies raw bytes
    instead of millions of Python str/tuple objects (annoying!)
    """
    def __init__(self, root, transform=None, cache_dir=None):
        """root: path to dataset root, which contains one subdir per class
        transform: optional transform to apply to each image
        cache_dir: optional path to store a cached file list, so subsequent runs start instantly
        """
        self.root, self.transform = root, transform
        # build a list of classes and their indices
        self.classes = sorted(d.name for d in os.scandir(root) if d.is_dir())
        # build a dict mapping class name to index
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # cache the result, keyed by the dataset root, so repeat runs start instantly
        cache = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            key = hashlib.sha1(os.path.abspath(root).encode()).hexdigest()[:16]
            cache = os.path.join(cache_dir, f"filelist-{key}.npz")

        # The cache is only valid if the class list still matches exactly, otherwise seeded split would silently point at different images
        if cache and os.path.exists(cache):
            z = np.load(cache, allow_pickle=False)
            if list(z["classes"].astype(str)) == self.classes:
                self.fnames, self.targets = z["fnames"], z["targets"]
                return
            print("file-list cache is stale (class list changed); rescanning")

        fnames, targets = [], []
        for i, c in enumerate(self.classes):
            # sort the files so the split is deterministic, even if the OS returns them in a different order
            for e in sorted(os.scandir(os.path.join(root, c)), key=lambda e: e.name):
                if e.is_file() and e.name.lower().endswith(".jpg"):
                    fnames.append(e.name)
                    targets.append(i)
        self.fnames  = np.array(fnames, dtype='S16')
        self.targets = np.array(targets, dtype=np.int16)

        if cache:
            # save the file list to a cache file, so subsequent runs start instantly
            np.savez(cache, fnames=self.fnames, targets=self.targets,
                     classes=np.array(self.classes, dtype='U64'))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        t = int(self.targets[i])
        p = os.path.join(self.root, self.classes[t], self.fnames[i].decode())
        img = Image.open(p).convert("RGB")
        return (self.transform(img) if self.transform else img), t

class DeepCNN(Module):
    def __init__(self, num_classes):
        super().__init__()
        # widened 1.5x from 32/64/128/256/512 -- the model was underfitting
        # (train loss 1.81 vs a 1.01 floor), so capacity is the constraint
        self.stem = self.conv_block(3, 48, stride=2)

        self.conv1a = self.conv_block(48, 48)
        self.conv1b = self.conv_block(48, 48)

        self.conv2a = self.conv_block(48, 96)
        self.conv2b = self.conv_block(96, 96)

        self.conv3a = self.conv_block(96, 192)
        self.conv3b = self.conv_block(192, 192)

        self.conv4a = self.conv_block(192, 384)
        self.conv4b = self.conv_block(384, 384)
        self.conv4c = self.conv_block(384, 384)

        self.conv5a = self.conv_block(384, 768)
        self.conv5b = self.conv_block(768, 768)
        self.conv5c = self.conv_block(768, 768)

        self.pool = MaxPool2d(2, 2)
        self.gap  = AdaptiveAvgPool2d(1)
        self.flat = Flatten()
        self.drop = Dropout(0.2)
        self.fc = Linear(768, num_classes)



    def conv_block(self, in_ch, out_ch, stride=1):
        """Conv -> BatchNorm -> ReLU, the unit this network repeats."""
        return Sequential(Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                        BatchNorm2d(out_ch), ReLU())
    
    def forward(self, x):
        x = self.stem(x)
        x = self.pool(self.conv1b(self.conv1a(x)))
        x = self.pool(self.conv2b(self.conv2a(x)))
        x = self.pool(self.conv3b(self.conv3a(x)))
        x = self.pool(self.conv4c(self.conv4b(self.conv4a(x))))
        x = self.pool(self.conv5c(self.conv5b(self.conv5a(x))))
        x = self.flat(self.gap(x))
        return self.fc(self.drop(x))




def parse_args():
    p = argparse.ArgumentParser(description="Train DeepCNN on ImageNet-256")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. 0 means no multiprocessing, which is "
                        "the safe default on Windows where spawn costs ~400 MB/worker.")
    p.add_argument("--num-classes", type=int, default=None,
                   help="Keep only N classes. Default None keeps all 100.")
    p.add_argument("--out-dir", default=str(Path.cwd()),
                   help="Checkpoint destination. Built from Path.cwd() so it works "
                        "on Windows and macOS alike")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any existing checkpoint and start from epoch 0.")
    return p.parse_args()


def download_food_dataset():
    url = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    docs = Path.home() / "Documents"
    images = docs / "food-101" / "images"
    if images.is_dir():
        print('dataset found')
        return str(images)

    tgz = Path.home() / "Downloads" / "food-101.tar.gz"
    if not tgz.exists():
        print("downloading", url)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # get total length of file
        total = int(response.headers["Content-Length"])
        # give our downloading folder a temp name while ongoing
        part = tgz.parent / (tgz.name + ".part")
        done = 0
        with open(part, "wb") as f:
            # download content slowly, otherwise entirety of dataset gets stuffed in ram
            for chunk in response.iter_content(chunk_size= 10 * 1024 * 1024):   # 10 MB at a time
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {100 * done / total:.1f}%", end="")
        part.rename(tgz)          # only becomes the real file once complete
        print()

    print("extracting to", docs)
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(path=docs)
    return str(images)

def pre_resize_dataset():
    """
    Write a parallel, resized dataset where every image has its shortest side at 256px. 
    """
    old_path   = Path.home() / "Documents" / "food-101" / "images"
    final_root = Path.home() / "Documents" / "food-101-256"
    new_path   = final_root / "images"

    if new_path.is_dir():
        print("scaled dataset found")
        return str(new_path)

    # Build into a .tmp directory and rename at the very end, so an interrupted
    # run never leaves a half-converted tree that the check above would accept.
    tmp_root = final_root.with_name(final_root.name + ".tmp") # becomes ./food-101-256.tmp for now
    if tmp_root.exists():
        shutil.rmtree(tmp_root)                     # discard a previous failed attempt

    tmp_images = tmp_root / "images"
    tmp_images.mkdir(parents=True)
    shutil.copytree(old_path.parent / "meta", tmp_root / "meta")
    classes = sorted(d.name for d in os.scandir(old_path) if d.is_dir())

    for i, class_name in enumerate(classes, 1):
        src_dir = old_path / class_name
        dest_dir = tmp_images / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for image in src_dir.iterdir():
            with Image.open(image) as im:
                w, h = im.size
                scale = 256 / min(w, h)
                if scale >=1:
                    # copy2 to preserve file metadata
                    shutil.copy2(image, dest_dir / image.name)
                    continue
                new_size = (round(w * scale), round(h*scale))
                img = im.convert('RGB')
                img = img.resize(new_size, Image.LANCZOS)
                img.save(dest_dir / image.name, quality = 95)
        print(f"\r  {i}/{len(classes)}  {class_name:<25}", end="")
    print()

    tmp_root.rename(final_root)
    return str(new_path)

def load_official_split(dataset, images_path, val_fraction=0.1, seed=42):
    """Map Food-101's official train/test lists onto dataset indices.

    meta/train.txt and meta/test.txt hold lines like 'apple_pie/1005649'
    (no extension). The 250 test images per class were manually reviewed;
    the 750 training ones were deliberately left noisy, so the two must not
    be mixed. Validation is carved out of train, leaving test untouched.
    """
    meta_folder = Path(images_path).parent / "meta"

    # Step 1 -- build a lookup so a line of text can become a dataset position.
    #           "apple_pie/1005649"  ->  17423
    lookup = {}
    for position in range(len(dataset)):
        class_index = int(dataset.targets[position])
        class_name = dataset.classes[class_index]
        # turn numpy byte array back into a string
        filename = dataset.fnames[position].decode()   # "1005649.jpg"
        image_id = filename.removesuffix(".jpg")       # "1005649"
        lookup[class_name + "/" + image_id] = position

    # Step 2 -- turn a meta file into an array of dataset positions.
    def read_list(name):
        positions = []
        with open(meta_folder / name) as f:
            for line in f:
                line = line.strip()
                if line:
                    positions.append(lookup[line])     # KeyError = lists disagree
        return np.array(positions, dtype=np.int32)

    train_and_val = read_list("train.txt")
    test = read_list("test.txt")

    # Step 3 -- shuffle the training list, slice a validation chunk off the front.
    shuffled = np.random.default_rng(seed).permutation(train_and_val)
    val_size = int(len(shuffled) * val_fraction)
    val = shuffled[:val_size]
    train = shuffled[val_size:]

    print(f"train {len(train)} | val {len(val)} | test {len(test)}")
    return train, val, test

def build_transforms(mean, std):
    """Train and eval transforms. Module-level so the notebook can import
    them"""
    # The training transform includes data augmentation, while the validation transform does not.
    # randomresizecrop is used to randomly crop the image to 224x224 pixels, 
    # #randomhorizontalflip is used to randomly flip the image horizontally. 
    # RandAugment is used to apply random augmentations to the image
    # randomerasing is used to randomly erase a portion of the image.
    train_tfm = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandAugment(num_ops=2, magnitude=9),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std),
                                    transforms.RandomErasing(p=0.25)])
    eval_tfm  = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    return train_tfm, eval_tfm

def main():
    args = parse_args()
    # select backend
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    chlast = (device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # autotune convs for fixed 224x224 input

    print("torch:", torch.__version__)
    print("device:", device)

    # set path to dataset files
    download_food_dataset()
    path = pre_resize_dataset()
    print("Path to dataset files:", path)

    #preprocess data
    NUM_CLASSES = args.num_classes      # None keeps all 101
    dataset = CompactImageFolder(root=path, transform=transforms.ToTensor(), cache_dir=args.out_dir)


    if NUM_CLASSES is not None:
        # Subset the numpy arrays directly. __getitem__ rebuilds paths from
        # self.classes[t], so remapping labels and the class list together
        # keeps every path correct.
        keep = sorted(random.Random(42).sample(dataset.classes, NUM_CLASSES))
        old_to_new = {dataset.class_to_idx[c]: i for i, c in enumerate(keep)}
        mask = np.isin(dataset.targets, list(old_to_new.keys()))
        dataset.fnames  = dataset.fnames[mask]
        dataset.targets = np.array([old_to_new[int(t)] for t in dataset.targets[mask]],
                                   dtype=np.int16)
        dataset.classes = keep
        dataset.class_to_idx = {c: i for i, c in enumerate(keep)}


    # The split below is index-based and seeded, so it only reproduces if the
    # class ordering is identical to previous runs. If ordering ever shifts,
    # those same indices point at different images -- training data leaks into
    # the val set silently, with no error, and old checkpoints become invalid.
    # Fail loudly instead.
    classes_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classes.json")
    if NUM_CLASSES is None and os.path.exists(classes_json):
        with open(classes_json) as f:
            expected = json.load(f)
        if dataset.classes != expected:
            raise SystemExit(
                f"class ordering changed ({len(dataset.classes)} found vs "
                f"{len(expected)} in classes.json). The seeded split would shift "
                f"and old checkpoints would be invalid. "
                f"Delete classes.json to re-baseline.")

    tr_idx, va_idx, te_idx = load_official_split(dataset, path, val_fraction = 0.0)

    # transform and load data
    # per-channel stats of all 101,100 training images; computed in jupyter notebook with
    MEAN = [0.5576, 0.4423, 0.327]
    STD  = [0.2591, 0.263, 0.2656]


    train_tfm, eval_tfm = build_transforms(MEAN, STD)
    dataset.transform = eval_tfm
    train_dataset = copy.copy(dataset)          # same files, augmented transform
    train_dataset.transform = train_tfm

    pin = (device.type == 'cuda')   # pinned host memory only helps host->GPU copies
    nw  = args.num_workers
    train_loader = DataLoader(Subset(train_dataset, tr_idx), batch_size=args.batch_size,
                              shuffle=True,  num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    val_loader   = DataLoader(Subset(dataset, va_idx), batch_size=256,
                              shuffle=False, num_workers=nw, pin_memory=pin)
    test_loader  = DataLoader(Subset(dataset, te_idx), batch_size=256,
                              shuffle=False, num_workers=nw, pin_memory=pin)


    # define model, loss, optimizer
    EPOCHS = args.epochs
    cnn = DeepCNN(num_classes=len(dataset.classes)).to(device)
    if chlast:
        cnn = cnn.to(memory_format=torch.channels_last)
    LABEL_SMOOTHING = 0.1
    loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)


    # A/B testing shows SGD Optimizer yields best results (although Adam is close behind)
    #optimizer = optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer = optim.SGD(cnn.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    #optimizer = optim.AdamW(cnn.parameters(), lr=0.001, weight_decay=0.05)

    use_amp = (device.type == 'cuda')  # use automatic mixed precision only on CUDA devices
    WARMUP = 3
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP])

    # don't point checkpoints towards a folder in OneDrive (if using windows)
    # checkpoint.pt is written every epoch (~40 MB), and OneDrive re-uploads
    # the whole file each time, stealing I/O from the data loader.
    OUT = args.out_dir
    os.makedirs(OUT, exist_ok=True)
    CKPT  = os.path.join(OUT, "checkpoint.pt")
    BEST  = os.path.join(OUT, "best.pt")
    FINAL = os.path.join(OUT, "deepcnn_food101.pt")
    print("checkpoints ->", OUT)

    start_epoch = 0
    best_acc = 0.0
    if os.path.exists(CKPT) and not args.fresh:
        ckpt = torch.load(CKPT)
        cnn.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)

        # finished runs can't be extended in place, we add a guard to tell the user so
        if start_epoch >= EPOCHS:
            print(f"\n{CKPT}")
            print(f"  already holds a completed {EPOCHS}-epoch run "
                  f"(lr annealed to {scheduler.get_last_lr()[0]:g}).")
            print("\n  To train again, pick one:")
            print("    --fresh             overwrite this run in place")
            print("    --out-dir <PATH>    keep it, start a separate run")
            print("\n  To evaluate what is already here:")
            print(f"    python tta.py --ckpt {FINAL}")
            raise SystemExit(0)
        print(f"resuming from epoch {start_epoch + 1}")

    def save_ckpt(ep):
        torch.save({"epoch": ep, "best_acc": best_acc,
                    "model": cnn.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()}, CKPT)

    interrupted_at = None
    try:
        for epoch in range(start_epoch, EPOCHS):
            interrupted_at = epoch
            cnn.train()
            print(f"Epoch {epoch+1}/{EPOCHS}")
            running_loss = 0.0
            tr_correct = torch.zeros((), device=device)   # on-GPU: no per-batch sync
            tr_total = 0


            # time how long an epoch takes, and how long it takes to fetch a batch of data
            t_epoch = time.perf_counter()
            t_mark  = t_epoch
            t_data  = 0.0
            t_fetch = time.perf_counter()

            for i, data in enumerate(train_loader):
                t_data += time.perf_counter() - t_fetch      # blocked waiting on the loader
                
                inputs, labels = data
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if chlast:
                    inputs = inputs.to(memory_format=torch.channels_last)
                optimizer.zero_grad()
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    outputs = cnn(inputs)
                    loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                tr_correct += (outputs.detach().argmax(1) == labels).sum()
                tr_total += labels.size(0)
                running_loss += loss.item()

                if i % 100 == 0 and i > 0:
                    now = time.perf_counter()
                    dt  = now - t_mark
                    ips = 100 * inputs.size(0) / dt
                    eta = (len(train_loader) - i) * dt / 100
                    print(f"  batch {i}/{len(train_loader)} loss {loss.item():.4f} "
                        f"| {ips:6.0f} img/s | data-wait {100*t_data/dt:3.0f}% "
                        f"| epoch ETA {eta/60:5.1f} min")
                    t_mark, t_data = now, 0.0
                t_fetch = time.perf_counter()

            epoch_min = (time.perf_counter() - t_epoch) / 60
            tr_acc = 100 * tr_correct.item() / max(tr_total, 1)
            print(f"Training loss: {running_loss/len(train_loader):.4f} "
                f"| train top-1 {tr_acc:.2f}% "
                f"| epoch {epoch_min:.1f} min | run ETA {epoch_min*(EPOCHS-epoch-1)/60:.1f} h")

            # this block only runs if we have a validation split
            if len(val_loader) > 0:
                cnn.eval()
                val_loss, correct, total = 0.0, 0, 0
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                        if chlast:
                            inputs = inputs.to(memory_format=torch.channels_last)
                        outputs = cnn(inputs)
                        val_loss += loss_function(outputs, labels).item()
                        correct += (outputs.argmax(1) == labels).sum().item()
                        total += labels.size(0)
                print(f"Val loss {val_loss/len(val_loader):.4f} | Val acc {100*correct/total:.2f}%")

                val_acc = 100 * correct / total
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(cnn.state_dict(), BEST)
                    print(f"  new best: {best_acc:.2f}% -> saved")

            # step the scheduler after each epoch, not after each batch, so the warmup and cosine annealing are per-epoch
            scheduler.step()

            # save checkpoint every epoch, so Ctrl+C mid-epoch doesn't throw away an hour of work
            save_ckpt(epoch)

    except KeyboardInterrupt:
        # Ctrl+C mid-epoch would otherwise throw away up to an hour of work.
        # Save under the last COMPLETED epoch so the resume replays the
        # partial one rather than skipping it.
        if interrupted_at is not None:
            print(f"\ninterrupted during epoch {interrupted_at + 1} -- saving checkpoint")
            save_ckpt(interrupted_at - 1)
        raise SystemExit(130)
    
    # save final model
    torch.save(cnn.state_dict(), FINAL)

    # test the model on the test set
    cnn.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if chlast:
                inputs = inputs.to(memory_format=torch.channels_last)
            outputs = cnn(inputs)
            test_loss += loss_function(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    print(f"TEST loss {test_loss/len(test_loader):.4f} | TEST acc {100*correct/total:.2f}%")

    # A completed run deletes its checkpoint, so FINAL without CKPT means done.
    if os.path.exists(FINAL) and not os.path.exists(CKPT) and not args.fresh:
        print(f"\n{FINAL}\n  already holds a completed run.")
        print("\n  To train again, pick one:")
        print("    --fresh             overwrite this run in place")
        print("    --out-dir <PATH>    keep it, start a separate run")
        print("\n  To evaluate what is already here:")
        print(f"    python tta.py --ckpt {FINAL}")
        raise SystemExit(0)

    # run is complete, so we deleted checkpoint.pt. frees up 100mb+ of storage
    if os.path.exists(CKPT):
        os.remove(CKPT)
        print(f"removed {os.path.basename(CKPT)} (run complete; weights are in {os.path.basename(FINAL)})")

if __name__ == "__main__":
    main()
