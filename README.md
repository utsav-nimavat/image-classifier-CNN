# Image Classification Project using a Convolutional Neural Network - Classifying Food

## Background 
This is a personal project to explore pytorch and build off what I've learned in undergrad. I got faint exposure to basic neural networks in a couple of my classes, and decided to to see how far I could go using a field that always interested me - computer vision.

I started by attempting to build a general image classifier with 1000 classes based off an ImageNet dataset, but refined the scope of my goal to improve prediction accuracy and fit my compute constraints, which led me to my current dataset.

## Overview

I trained a convolutional neural network over the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (more details about the data inside `explore.ipynb`) which includes 101 food categories with a total of 101,000 labeled images. The end result is a model that can roughly classify a dish simply by looking at a photo.

The next phase of this project is to build a website where users can upload their own images for classification - WIP

## Model

### Summary 
My CNN was trained over 60 epochs using random intialization and no pretrained weights, on the official 75,750 image test split. I was able to get 82% top-1 prediction accuracy on the test data alone, boosted to 84.65% after applying a technique called test-time augmentation (TTA) that averaged the predictions my model gave across several input resolutions / mirrored images. 


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
    - My network was shallow enough that the vanishing gradient problem didn't exist, and my model ended up performing better on training and worse than testing
- AdamW optimizer (-1.54)
  - AdamW was applying weaker weight decay despite setting it 500x higher because of how Adam internally applies decay. This meant less regularization, causing better memorization and worse performance.
- skipping the 8% worst-fitting images in each batch (-0.8)
    - This was an attempt at filtering out noisy training labels, but it also threw out the "hard" images so my model never learned to predict them
- Training over 120 epochs instead of 60 (-0.73)
    - Just gave the model more time to memorize the training set without learning anything new.
  
### Results

82.0% top-1 prediction on the official test split. 84.7% with multi-scale + flip TTA.