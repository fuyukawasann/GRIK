import onnx_graphsurgeon as gs
import tensorrt as trt

onnx_model = gs.import_onnx(onnx.load("Weights/best_tiny_400_16.pt"))

create_attrs = lambda: [
    
