import torch.nn as nn
import torch

def create_conv(inputs, weights):
    """Convレイヤーを作成"""
    weight = weights[inputs[1]]
    in_channels, out_channels, kernel_size = weight.shape[1], weight.shape[0], weight.shape[2]
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    # 重みをセット
    conv_layer.weight.data = torch.tensor(weight)
    conv_layer.bias.data = torch.tensor(weights[inputs[2]]) if len(inputs) > 2 else torch.zeros(out_channels)

    return conv_layer
