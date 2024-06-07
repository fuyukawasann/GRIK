import torch
import sys

sys.path.insert(0, '../yolov7')
device=torch.device("cpu")
model = torch.load('../Weights/yolov7-tiny.pt', map_location=device)['model'].float()
model.eval()

img = torch.zeros(1, 3, 640, 640)

torch.onnx.export(model, img, 'yolov7-tiny.onnx', opset_version=12, do_constant_folding=True)
