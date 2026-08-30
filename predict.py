"""Classify one or more images using a trained model from a .pt file"""
import argparse, json, torch
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
    if torch.backends.mps.is_available():
        device = torch.device('mps') # pick apple silicon first
    elif torch.cuda.is_available():
        device = torch.device('cuda') # else pick nvidia gpu
    else:
        device = torch.device('cpu') # else, pick cpu. sorry amd gpu users I will get to you eventually

    classes = json.load(open(args.classes))
    model = DeepCNN(len(classes)).to(device) # moves tensors to apple silicon
    # load weights in pytorch (no code), inject them into DeepCNN model
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only = True))
    model.eval() # disables dropout, predictions are random noise w/o this. model now in eval mode

    # only eval model on  288/256 version if fast flag enabled (ignore other 2 zooms)
    # because that one yields the best results by itself
    scales = SCALES[1:2] if args.fast else SCALES
    # scale & transform each image for each zoom level
    tfms = [transforms.Compose([transforms.Resize(r), transforms.CenterCrop(c),
                                transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
            for r, c in scales]


    for path in args.images:  # iterate through each image
        img = Image.open(path).convert('RGB') # forces each image to have rgb color channels
        total = 0
        with torch.no_grad(): # optimization to cut memory & speed things up
            for tfm in tfms:
                # convert yielded transformation to format conv2d layer can accept
                x = tfm(img).unsqueeze(0).to(device)
                # forward pass on model using original image, add up running score
                total = total + model(x).softmax(1) # original image
                # forward pass on model using flipped image, add up running score
                total = total + model(torch.flip(x, dims=[3])).softmax(1) # flipped image
        # turn the sum of each vote (based on diff zoom level) into an average
        probs = (total / (2 * len(tfms)))[0]

        print(f"\n{Path(path).name}")
        conf, idx = probs.topk(min(args.topk, len(classes))) # return values and class indices
        for c, i in zip(conf.tolist(), idx.tolist()):
            # print each class guess and it's value
            print(f"  {classes[i]:<24} {c:6.1%}")

if __name__ == "__main__":
    main()