import cv2
import numpy as np
import tensorrt as trt

# Load the TRT Engine
TRT_ENGINE_PATH = "../Weights/best_tiny_400_16.trt"
logger = trt.Logger(trt.Logger.INFO)
trt_runtime = trt.Runtime(logger)
trt.init_libnvinfer_plugins(logger, namespace="")
device = torch.device('cuda:0')

with open(TRT_ENGINE_PATH, "rb") as f:
    engine_data = f.read()

engine = trt_runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()


# Buffer of input and output
inputs, outputs, bindings, stream = [], [], [], None
batch_size = engine.max_batch_size

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host_mem = trt_runtime.pagelocked_empty(size, trt.nptype(dtype))
    # Allocate device memory
    device_mem = trt_runtime.allocate_buffer(size)
    # Append to the appropriate list.
    bindings.append(int(device_mem))

    if engine.binding_is_input(binding):
        inputs.append({'host': host_mem, 'device': device_mem})
    else:
        outputs.append({'host': host_mem, 'device': device_mem})

stream = trt_runtime.Stream()

def infer(img):
    img_input = cv2.resize(img, (640, 640))
    img_input = img_input[:,:,::-1].transpose((2, 0, 1)).astype(np.float16) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    np.copyto(inputs[0]['host'], img_input.ravel())

    for inp in inputs:
        trt_runtime.memcpy_htod_async(inp['device'], inp['host'], stream)
    
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)

    for out in outputs:
        trt_runtime.memcpy_dtoh_async(out['host'], out['device'], stream)
    stream.synchonize()

    output_data = outputs[0]['host']
    return output_data

img = cv2.imread("../test05.jpg")
output = infer(img)
print(output)
