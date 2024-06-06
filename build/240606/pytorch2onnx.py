########## Project Description ##########
# This is a python script to convert a PyTorch model to ONNX format.
# Required Library: torch
# BUILD: Jun 06, 2024 (KST)
##########################################

# Notice: Do not use this module in real service. This is just for testing purpose.

# Import the necessary library
import torch

model = torch.hub.load('yolov7', 'custom', f'Weights/best_tiny_400_16.pt', source='local')
model.eval()
onnx_model_path = 'Weights/best_tiny_400_16.onnx'
dummy_input = torch.rand(16, 3, 640, 640)
torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True, opset_version=11)
