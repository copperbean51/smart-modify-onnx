import torch.nn as nn

def create_softmax(inputs, weights):
    """Softmaxレイヤーを作成"""
    return nn.Softmax(dim=1)
