import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import time
import os

print("import finish")
time.sleep(1)

## Load TRT model

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
trt.init_libnvinfer_plugins(None, "")
with open('../Weights/best.trt', 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
assert engine
assert context


print("load model finish")
time.sleep(1)

## Setup I/O Bindings
inputs, outputs, allocations = [], [], []

for i in range(engine.num_bindings): # input, output 개수 만큼
    is_input = False
    if engine.binding_is_input(i):
        is_input = True
    name = engine.get_binding_name(i)
    dtype = np.dtype(trt.nptype(engine.get_binding_dtype(i)))
    shape = context.get_binding_shape(i)
    
    if is_input and shape[0] < 0:
        assert engine.num_optimization_profiles > 0
        profile_shape = engine.get_profile_shape(0, name)
        assert len(profile_shape) == 3
        self.context.set_binding_shape(i, profile_shape[2])
        shape = context.get_binding_shape(i)
    
    if is_input:
        batch_size = shape[0]
    
    size = dtype.itemsize
    for s in shape:
        size *= s
        
    allocation = cuda.mem_alloc(size)
    host_allocation = None if is_input else np.zeros(shape, dtype)
    binding = {
        "index": i,
        "name": name,
        "dtype": dtype,
        "shape": list(shape),
        "allocation": allocation,
        "host_allocation": host_allocation,
    }
    allocations.append(allocation)
    if engine.binding_is_input(i):
        inputs.append(binding)
    else:
        outputs.append(binding)


print("finish to setup")
time.sleep(1)

my_list = '../Result/Tester/SSIM/handwritten/'
my_list = os.listdir(my_list)
my_list = [file for file in my_list if file.endswith('.jpg')]

start_time = time.time()

for img in my_list:
	## preprocessing image
	original_image = cv2.imread(f'../Result/Tester/SSIM/handwritten/{img}')
	ori_h, ori_w, ori_c = original_image.shape

	input_size = [inputs[0]['shape'], inputs[0]['dtype']][0][-2:]

	image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

	## Calculate width and height and paddings
	r_w = input_size[1] / ori_w
	r_h = input_size[0] / ori_h
	if r_h > r_w:
	    tw = input_size[1]
	    th = int(r_w * ori_h)
	    tx1 = tx2 = 0
	    ty1 = int((input_size[0] - th) / 2)
	    ty2 = input_size[0] - th - ty1
	else:
	    tw = int(r_h * ori_w)
	    th = input_size[0]
	    tx1 = int((input_size[1] - tw) / 2)
	    tx2 = input_size[i] - tw - tx1
	    ty1 = ty2 = 0

	## Resize the image with long side while maintaining ratio
	image = cv2.resize(image, (tw, th))
	image = cv2.copyMakeBorder(
	    image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
	)
	image = image.astype(np.float32)

	## Normalizee to [0, 1]
	image /= 255.0
	## CHW to NCHW format
	image = np.expand_dims(image, axis=0)
	## Convert the imag to row-major order, also know as "C order":
	preprocessed_image = np.ascontiguousarray(image)

	print("End to preprocess")
	time.sleep(1)

	# Inference
	this_image = preprocessed_image.transpose(0, 3, 1, 2)
	this_image = np.ascontiguousarray(this_image)
	cuda.memcpy_htod(inputs[0]['allocation'], this_image)
	context.execute_v2(allocations)

	for o in range(len(outputs)):
	    cuda.memcpy_dtoh(outputs[o]['host_allocation'], outputs[o]['allocation'])
	    
	num_detections = outputs[0]['host_allocation']
	nmsed_boxes = outputs[1]['host_allocation']
	nmsed_scores = outputs[2]['host_allocation']
	nmsed_classes = outputs[3]['host_allocation']
	result = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]

	print("End to Inference")
	time.sleep(1)


	# save result
	# color

	h, w = preprocessed_image.shape[1:3]
	result_image = np.squeeze(preprocessed_image)
	result_image *= 255
	result_image = result_image.astype(np.uint8)

	for i in range(int(num_detections)):
	    x1 = int(nmsed_boxes[0][i][0])
	    x2 = int(nmsed_boxes[0][i][1])
	    y1 = int(nmsed_boxes[0][i][2])
	    y2 = int(nmsed_boxes[0][i][3])
	    result_image = cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
	    result_image = cv2.putText(result_image, "panseo", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

	result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
	cv2.imwrite(f'./{img}', result_image)
end_time = time.time()
eval_time = end_time - start_time
print(f"Eval Time: {eval_time}")
print("End to export result")
time.sleep(1)
print("EOF")

