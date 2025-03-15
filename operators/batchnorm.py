import torch.nn as nn
import torch

def create_batchnorm(inputs, weights):
    """BatchNormalizationレイヤーを作成"""
    num_features = weights[inputs[1]].shape[0]
    batchnorm_layer = nn.BatchNorm2d(num_features)

    batchnorm_layer.weight.data = torch.tensor(weights[inputs[1]])
    batchnorm_layer.bias.data = torch.tensor(weights[inputs[2]])
    batchnorm_layer.running_mean.data = torch.tensor(weights[inputs[3]])
    batchnorm_layer.running_var.data = torch.tensor(weights[inputs[4]])

    return batchnorm_layer
