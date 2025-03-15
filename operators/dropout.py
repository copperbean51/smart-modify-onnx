import torch.nn as nn

def create_dropout(inputs, weights):
    """Dropoutレイヤーを作成"""
    return nn.Dropout(p=0.5)
