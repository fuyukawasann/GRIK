########## Project Description ##########
# This is a python script for detection using TensorRT
# Required Library: cv2, torch, random, time, numpy, tensorrt, collections, os
# BUILD: Jun 10, 2024 (KST)
#########################################

## import the necessary library
import cv2
import os
import torch
import random
import time
import numpy as np
import tensorrt as trt
from collections import OrderedDict, namedtuple

class detection_ps_trt:
    def __init__(self, img_path, result_name):
        self.img_path = img_path
        self.result_name = result_name
	
    def letterbox(im, new_shape=(640, 640), color=(114,114,114), auto=True, scaleup=True, stride = 32):
	# Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
        	new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
	    	r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
	    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
	    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def postprocess(boxes, r, dwdh):
	dwdh = torch.tensor(dwdh*2).to(boxes.device)
	boxes -= dwdh
	boxes /= r
	return boxes

    def detection_panseo_trt(self):
	print("This is the detection_panseo_trt module")
	time.sleep(2)
	
	# Load the model
	print("Load Model")
	time.sleep(2)
	w = '../Weights/best_tiny_400_16.trt'
	device = torch.device('cuda:0')
	Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
	logger = trt.Logger(trt.Logger.INFO)
	trt.init_libnvinfer_plugins(logger, namespace="")
	## Open Model
	with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
    	    model = runtime.deserialize_cuda_engine(f.read())
	bindings = OrderedDict()	
	for index in range(model.num_bindings):
	    name = model.get_binding_name(index)
	    dtype = trt.nptype(model.get_binding_dtype(index))
	    shape = tuple(model.get_binding_shape(index))
	    data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
	    bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
	binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
	context = model.create_execution_context()
	## names
	names = ['panseo']
	colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

	# Image Setting and save setting
	## Get directory of the handwritten image and original image
	ori_img_path = f'{self.img_path}/original'
	hw_img_path = f'{self.img_path}/handwritten'
	## Get List of the handwritten Image
	list_hand_img = os.listdir(hw_img_path)
	list_hand_img = natsort.natsorted(list_hand_img)
	print(f'List of images: {list_hand_img}')
	save_img_path = f'Result/{self.result_name}/YOLO' # Before, 'Result_Panseo'
	if not os.path.exists(save_img_path):
	    os.makedirs(save_img_path)
	    os.makedirs(f'{save_img_path}/original')
	    os.makedirs(f'{save_img_path}/handwritten')
	
	# Processing the image
	print("Processing the image")
	time.sleep(1)
	start_time = time.time()
	for img in list_hand_img:
	    img_name = img.split('.')[0]
	    original_img = cv2.imread(f'{ori_img_path}/{img}')
	    handwritten_img = cv2.imread(f'{hw_img_path}/{img}')
	    handwritten_img = cv2.cvtColor(handwritten_img, cv2.COLOR_BGR2RGB)
	    hw_image = handwritten_img.copy()
	    hw_image, ratio, dwdh = letterbox(image, auto=False)
	    hw_image = hw_image.transpose((2,0,1))
	    hw_image = np.expand_dims(hw_image, 0)
	    hw_image = np.ascontiguousarray(hw_image)
	    hw_im = image.astype(np.float32)
	    hw_im = torch.from_numpy(hw_im).to(device)
	    hw_im/=255
	
	    
	


















	
