########## Project Description ##########
# This is a python script to convert a PyTorch model to ONNX format.
# Required Library: torch
# BUILD: Jun 06, 2024 (KST)
##########################################

# Notice: Do not use this module in real service. This is just for testing purpose.

# Import the necessary library
import torch
import onnx
import time
import os
import gc
import sys

sys.path.insert(0, './yolov7')

print(f'PATH: {os.getcwd()}')
device = torch.device('cpu')
dummy_input = torch.rand(1,3,640,640)
dummy_input = dummy_input.to(device)
onnx_model_path = f'../Weights/best_tiny_400_16.onnx'
model = torch.load('../Weights/best_tiny_400_16.pt', map_location=device)['model'].float()
model.eval()
model.to(device)

print("Now coverting torch2onnx in cuda mode ...")
start_time = time.time()
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11, input_names=['input'], output_names=['output'])
end_time = time.time()
eval_time = end_time - start_time

print(f"Run Time: {eval_time} seconds")
print(f"ONNX is saved at {onnx_model_path}!!")

print("Memory Free")
torch.cuda.empty_cache()
gc.collect()
print("Finish to free memories")

print("Process End...")
