# Image Classification Project using a Convolutional Neural Network - Classifying Food

## Background 
This is a personal project to explore pytorch and build off what I've learned in undergrad. I got faint exposure to basic neural networks in a couple of my classes, and decided to to see how far I could go using a field that always interested me - computer vision.

I started by attempting to build a general image classifier with 1000 classes based off an ImageNet dataset, but refined the scope of my goal to improve prediction accuracy and fit my compute constraints, which led me to my current dataset.

## Overview

I trained a convolutional neural network over the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (more details about the data inside `explore.ipynb`) which includes 101 food categories with a total of 101,000 labeled images. The end result is a model that can roughly classify a dish simply by looking at a photo.

My CNN was trained over 60 epochs using random intialization and no pretrained weights, on the recommended 25,250 image test split. I was able to get 82% accuracy on the test data alone, boosted to 84.65% (roughly 85x better at describing a food dish compared to random guessing) after applying a technique called test-time augmentation (TTA) that averaged the predictions my model gave across several input resolutions / mirrored images. 

The next phase of this project is to build a website where users can upload their own images for classification - WIP

## Specific Model Details
| | |
|---|---|
| Architecture | 13 conv layers + linear head, 17.34 M parameters |
| Block | Conv3×3 (no bias) → BatchNorm → ReLU |
| Stages | stem 3→48 (stride 2), then 48, 96, 192, 384, 768 — max-pool between |
| Head | global average pool → dropout 0.2 → Linear(768, 101) |
| Optimizer | SGD, lr 0.1, momentum 0.9, Nesterov, weight decay 5e-4 |
| Schedule | 60 epochs, 3-epoch linear warmup → cosine annealing |
| Loss | cross-entropy, label smoothing 0.1 |
| Batch / precision | 128, bfloat16 autocast |
| Augmentation | RandomResizedCrop(224, 0.5–1.0), h-flip, RandAugment(2, 9), RandomErasing(0.25) |
| Data | 75,750 official training images; no pretrained weights |
| Top-1 (official test split) | 82.02%, boosted to 84.65% with multi-scale + flip TTA |

