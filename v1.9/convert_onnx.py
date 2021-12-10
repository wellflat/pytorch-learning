#!/usr/bin/env python
# -*- using: utf-8 -*-

from alexnet import AlexNet
import torch
from torch import nn, onnx

def convert_onnx(model: nn.Module, output_path: str):
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True)
    onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        verbose=True
    )


if __name__ == '__main__':
    model_path = './cifar10_net.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = AlexNet()
    net.load_state_dict(torch.load(model_path, map_location=device))
    output_path = './cifar10_net.onnx'
    convert_onnx(net, output_path)
