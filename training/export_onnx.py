"""Export the trained .pt weights to a portable .onnx file.

Run once after training:  python export_onnx.py

a .pt holds only the numbers, so loading it needs torch AND the
DeepCNN class from train.py. An .onnx holds the graph as well, so serving it
needs neither, which drops ~460MB of dependencies off the deployed app.
"""
import argparse
import torch

from train import DeepCNN
import json


def parse_args():
    p = argparse.ArgumentParser(description="Export deepcnn weights to ONNX")
    p.add_argument("-m", "--model", default="deepcnn_food101.pt", help="trained weights")
    p.add_argument("-c", "--classes", default="classes.json", help="json file with class names")
    p.add_argument("-o", "--out", default="deepcnn_food101.onnx", help="where to write the onnx")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.classes) as f:
        classes = json.load(f)

    model = DeepCNN(len(classes))
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    # TTA feeds 224, 256 AND 288 crops, and stacks each image with its mirror,
    # so BOTH the batch and the spatial dims have to stay dynamic. exporting at a
    # fixed size would silently lock the model to one scale.
    dynamic = {"input":  {0: "batch", 2: "height", 3: "width"},
               "logits": {0: "batch"}}

    torch.onnx.export(model, torch.zeros(2, 3, 224, 224), args.out,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes=dynamic, opset_version=args.opset,
                      dynamo=False)

    import os
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
