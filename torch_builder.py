import torch
import torch.nn as nn
import torch.nn.functional as F
from operators.conv import create_conv
from operators.batchnorm import create_batchnorm
from operators.dropout import create_dropout
from operators.softmax import create_softmax

class ONNXToTorch(nn.Module):
    def __init__(self, onnx_model, weights):
        super(ONNXToTorch, self).__init__()

        self.weights = weights
        self.custom_operations = {}

        # mapping operator and function 
        self.op_map = {
            "Conv": create_conv,
            "BatchNormalization": create_batchnorm,
            "Dropout": create_dropout,
            "Softmax": create_softmax,
        }

        self.layers = self._build_model(onnx_model)

    def _build_model(self, onnx_model):
        """anaylze ONNX node and create PyTorch layer """
        layers = []

        for node in onnx_model.graph.node:
            op_type = node.op_type
            inputs = node.input
            outputs = node.output

            if op_type in self.op_map:
                layer = self.op_map[op_type](inputs, self.weights)
                layers.append(layer)

            elif op_type == "GridSample":
                self.custom_operations[outputs[0]] = ("grid_sample", inputs)

            elif op_type == "GatherElements":
                self.custom_operations[outputs[0]] = ("gather", inputs)

        return nn.Sequential(*layers)

    def forward(self, x):
        """customer operation """
        for name, (op_type, inputs) in self.custom_operations.items():
            if op_type == "grid_sample":
                grid = torch.randn_like(x)
                x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

            elif op_type == "gather":
                dim = 1
                index = torch.randint(0, x.shape[dim], x.shape, dtype=torch.long)
                x = torch.gather(x, dim, index)

        return self.layers(x)
