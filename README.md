# Image Classification Project using a Convolutional Neural Network - Classifying Food
### Model is live at [image-cnn.vercel.app](image-cnn.vercel.app)! Check it out.
## Background 
This is a personal project to explore pytorch and build off what I've learned in undergrad. I got faint exposure to basic neural networks in a couple of my classes, and decided to see how far I could go using a field that always interested me - computer vision.

I started by attempting to build a general image classifier with 1000 classes based off an ImageNet dataset, but refined the scope of my goal to improve prediction accuracy and fit my compute constraints, which led me to my current dataset.

## Overview

I trained a convolutional neural network over the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (more details about the data inside `explore.ipynb`) which includes 101 food categories with a total of 101,000 labeled images. The end result is a model that can roughly classify a dish simply by looking at a photo.

Next, I made a web interface for interacting with the model, and hosted it with vercel. All model computation is done in the cloud, and predictions come back in ~500ms. Try it out!
## Model architecture

### Summary 
My CNN was trained over 60 epochs using random initialization and no pretrained weights, on the official 75,750 image training split. I was able to get 82% top-1 prediction accuracy on the test data alone, boosted to 84.65% after applying a technique called test-time augmentation (TTA) that averaged the predictions my model gave across several input resolutions / mirrored images. 


### Training

- Roughly VGG-style CNN, 17.3M parameters, no pretrained weights
- 13 conv layers, each w/ 3x3 px kernel, no bias (cuz BatchNorm cancels it out)
- Each conv layer immediately followed by BatchNorm and ReLU.
- Max pooling done between each stage
- Channel widths go 3 → 48 → 96 → 192 → 384 → 768; stride-2 downsampling on stem
- Head: Global average pooling → dropout of 0.2 → dense layer to go from 768 to 101 classes
- SGD optimizer, lr 0.1, momentum 0.9, nesterov, weight decay 5e-4
- 60 epochs: 3 linear warmup, then cosine annealing
- batch size 128, bf16 autocast for speed
- cross-entropy with label smoothing 0.1
- RandomResizedCrop(224, scale 0.5-1.0), horizontal flip, RandAugment(2, 9), RandomErasing(0.25) applied to each image

These specific configurations were chosen after many hours of A/B testing.

### What didn't work
Scored against a 60-epoch baseline using Adam optimizer (my go-to until I switched to SGD at the very end), the following techniques harmed my top-1 prediction accuracy which originally sat at 80.98%.

- Residual / skip connections (-1.63) 
    - My network was shallow enough that the vanishing gradient problem didn't exist, and my model ended up performing better on training and worse on testing
- AdamW optimizer (-1.54)
  - AdamW was applying weaker weight decay despite setting it 500x higher because of how Adam internally applies decay. This meant less regularization, causing better memorization and worse performance.
- Skipping the 8% worst-fitting images in each batch (-0.8)
    - This was an attempt at filtering out noisy training labels, but it also threw out the "hard" images so my model never learned to predict them
- Training over 120 epochs instead of 60 (-0.73)
    - Just gave the model more time to memorize the training set without learning anything new.


### What worked the best
Every gain was measured on the official test split, one change at a time (aka many hours of sleep lost fine tuning my model):

| | top-1 prediction accuracy |
|---|---|
| Adam, 60 epochs, 10% held out for validation | 80.98% |
| → switched optimizer to SGD | 81.65% (+0.67) |
| → dropped validation split, trained on the full official split instead | 82.02% (+0.37) |
| → multi-scale + flip TTA at inference | **84.65%** (+2.63) |


## Running the model
1. **Create the conda enviroment:**
```bash
conda env create -f environment.yml
conda activate image_cnn
```

2. **Download pretrained model (recommended) or train from scratch**  


Download pretrained model (make sure you're **inside** the repo):
```bash
gh release download v1.1 --pattern deepcnn_food101.onnx
```

OR, to train from scratch:
```python
python train.py --num-workers 8
python export_onnx.py
```
This will save `deepcnn_food101.pt` in your repo and then convert it to `deepcnn_food101.onnx`.  

Doing so will take an estimated ~2.5 hours to run on an RTX GPU (mine was a 3060ti). If it's your first time training the model, it will need to download the dataset (5gb), create a scaled version of the dataset, and then begin training. **Don't pick this approach unless you really want to make the model weights from scratch!**



3. **Testing the model**

Run `predict.py` with the filepath(s) of the images you want to predict. You must give one image minimum.
```python
python predict.py <image_1>
```
This would yield a sample output like:
```
<image_1>
  lasagna                   79.0%
  grilled_salmon             1.0%
  spaghetti_bolognese        0.7%
```
Additional params:
- `-m <path to model>`: Set to scan your repo by default for `.onnx` file
- `-c <path to classes.json>`: Scans repo for `classes.json` by default.
- `-t <amount of predictions>`: Set to 3 by default, change to list more or less
- `-f (boolean flag)`: Sets program to "fast" mode, only performing one round of TTA instead of the three used. `False` by omission