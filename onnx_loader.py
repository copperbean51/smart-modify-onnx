import onnx
import numpy as np

def load_onnx_model(onnx_path):
    """load ONNX model"""
    return onnx.load(onnx_path)

def extract_weights(onnx_model):
    """get weights data of ONNX"""
    weights = {}
    for initializer in onnx_model.graph.initializer:
        weights[initializer.name] = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(tuple(initializer.dims))
    return weights
