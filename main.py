import argparse
import torch
from onnx_loader import load_onnx_model, extract_weights
from torch_builder import ONNXToTorch


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="ONNX to PyTorch model converter")
    parser.add_argument('onnx_model_path', type=str, help="Path to the ONNX model file")
    parser.add_argument('output_path', type=str, help="Path to save the converted PyTorch model")
    return parser.parse_args()

def create_dummy_inputs(onnx_model):
    """Create dummy input tensors for the model"""
    
    dummy_inputs = []
    for input in onnx_model.graph.input:
        shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        # Set batch size to 1
        shape[0] = 1  # batch size
        # Create a dummy input tensor
        # dummy_inputs depending on the data type of the input data
        if input.type.tensor_type.elem_type == 1:  # ONNXのデータ型が浮動小数点数 (float32)
            dummy_inputs.append(torch.randn(*shape, dtype=torch.float32))
        elif input.type.tensor_type.elem_type == 6:  # ONNXのデータ型が整数型 (int32)
            dummy_inputs.append(torch.randint(0, 10, shape, dtype=torch.int32))
        else:
            raise ValueError(f"Unsupported data type: {input.type.tensor_type.elem_type}")
    return dummy_inputs

def convert_onnx_to_torch(onnx_model_path, output_path):
    
    """Convert ONNX to PyTorch and save the model"""

    onnx_model = load_onnx_model(onnx_model_path)
    weights = extract_weights(onnx_model)

    pytorch_model = ONNXToTorch(onnx_model, weights)
    dummy_inputs = create_dummy_inputs(onnx_model)
    output = pytorch_model(*dummy_inputs)

    print(f"Model output shape: {output.shape}")
    # Save PyTorch model
    torch.save(pytorch_model.state_dict(), output_path)
    print(pytorch_model.state_dict())
    print(f"PyTorch model saved to: {output_path}")


def export_onnx_model(onnx_model, output_path):
    """Save ONNX """
    
    dummy_inputs = create_dummy_inputs(onnx_model)
    model = ONNXToTorch(onnx_model, weights)
    torch.onnx.export(
        model,                      # model
        dummy_inputs,               # input tensor (or a tuple of input tensors)
        output_path,                # output path
        export_params=True,         # export model with parameters
        opset_version=16,           # ONNXのoperation version opset
        do_constant_folding=True,   # constant folding or not
        input_names=['input'],      # name of input tensor
        output_names=['output'],    # name of output tensor
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 動的軸の設定
    )
    print(f"ONNX model saved to: {output_path}")

if __name__ == '__main__':
    args = parse_args()
    convert_onnx_to_torch(args.onnx_model_path, args.output_path)

