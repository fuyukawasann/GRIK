########## Project Description ##########
# This script converts a PyTorch model to ONNX format.
# Original code at YOLOv7 Official Repository!!
# BUILD: Jun 07, 2024 (KST)
##########################################

# Installation
print("Installing required packages...")
import os
import time
#os.system("pip install --upgrade setuptools pip --user")
#os.system("pip install --ignore-installed PyYAML")
#os.system("pip install Pillow")
#os.system("pip install nvidia-pyindex")
#os.system("pip install pycuda")
#os.system("pip install protobuf<4.21.3")
#os.system("pip install onnxruntime-gpu")
#os.system("pip install onnx>=1.9.0")
#os.system("pip install onnx-simplifier>=0.3.6 --user")
print("Installation complete.")
time.sleep(1)

# Importing required libraries
print("Importing required libraries...")
import sys
import torch
print(f"Python version: {sys.version}, {sys.version_info}")
print(f"PyTorch version: {torch.__version__}")
print("Libraries imported successfully.")
time.sleep(1)

# Download YOLOv7 and valid weight file
print("Downloading YOLOv7 and valid weight file...")
os.system("git clone https://github.com/WongKinYiu/yolov7")
os.chdir("yolov7")
os.listdir(os.getcwd())
os.system("cp ../../Weights/best_tiny_400_16.pt .")
os.system("python3 detect.py --weights ./best_tiny_400_16.pt --conf 0.25 --img-size 640 --source ../../test05.jpg")
from PIL import Image
Image.open("runs/detect/exp/test05.jpg")
print("Download complete.")
time.sleep(1)

# Export ONNX model
print("Exporting ONNX model...")
os.system("python3 export.py --weights ./best_tiny_400_16.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640")
os.listdir(os.getcwd())
print("Export complete.")
time.sleep(1)

# Save ONNX File at Weight Folder
os.system("cp ./best_tiny_400_16.onnx ../../Weights")
os.listdir("../../Weights")
