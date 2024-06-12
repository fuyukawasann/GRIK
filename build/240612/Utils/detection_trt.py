########## Project Description ##########
# This is a python script for detection using TensorRT
# Required Library: cv2, torch, random, time, numpy, tensorrt, collections, os
# BUILD: Jun 10, 2024 (KST)
#########################################

########## CHANGE LOG ##########
# Revised the code for detection using TensorRT
# Jun 12, 2024 (KST)
###############################

## import the necessary library
import cv2
import os
import torch
import random
import time
import numpy as np
import tensorrt as trt
import natsort
from collections import OrderedDict, namedtuple

class detection_ps_trt:
    def __init__(self, img_path, result_name):
        self.img_path = img_path
        self.result_name = result_name
	
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
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

    def postprocess(self, boxes,r,dwdh):
        dwdh = torch.tensor(dwdh*2).to(boxes.device)
        boxes -= dwdh
        boxes /= r
        return boxes
    
    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im = torch.from_numpy(im).to(self.device)
        im/=255
        return im, ratio, dwdh

    def detection_panseo_trt(self):
        print("This is the detection_panseo_trt module")
        time.sleep(2)
        
        # LOAD ENGINE
        print("Load Engine...")
        time.sleep(1)
        #w = 'Weights/best_tiny_400_16.trt'
        w = 'Weights/best.trt'
        self.device = torch.device('cuda:0')
        print("Load Engine... SUCCESS")
        time.sleep(1)
        
        # Bindings
        print("Binding...")
        time.sleep(1)
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()
        print("Binding... SUCCESS")
        
        # name and color setting
        names = ['panseo']
        colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
        
        # Image Setting and save setting
        print("Image Setting and save setting...")
        ## Get directory of the handwritten image and original image
        ori_img_path = f'{self.img_path}/original'
        hw_img_path = f'{self.img_path}/handwritten'
        ## Get list of the handwritten Image (original image because file name is same)
        list_hand_img = os.listdir(hw_img_path)
        list_hand_img = natsort.natsorted(list_hand_img)
        print(f'List of images: {list_hand_img}')
        save_img_path = f'Result/{self.result_name}/YOLO'
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
            os.makedirs(f'{save_img_path}/original')
            os.makedirs(f'{save_img_path}/handwritten')
            os.makedirs(f'{save_img_path}/yolo')
        ## SUCCESS
        print("Image Setting and save setting... SUCCESS")
        time.sleep(1)
        
        # Warmup for 10 times
        print("Warmup for 10 times...")
        time.sleep(1)
        ## Warmup
        for _ in range(10):
            tmp = torch.randn(1,3,640,640).to(self.device)
            binding_addrs['images'] = int(tmp.data_ptr())
            context.execute_v2(list(binding_addrs.values()))
        ## SUCCESS
        print("Warmup for 10 times... SUCCESS")
        time.sleep(1)
        
        # Inference
        ## Set start time
        start_time = time.time()
        ## Make Iterator
        for img in list_hand_img:
            img_name = img.split('.')[0]
            original_img = cv2.imread(f'{ori_img_path}/{img}')
            handwritten_img = cv2.imread(f'{hw_img_path}/{img}')
            ### preprocess
            result_img, ratio, dwdh = self.preprocess(handwritten_img)
            ### Inference
            binding_addrs['images'] = int(result_img.data_ptr())
            context.execute_v2(list(binding_addrs.values()))
            ### results
            nums = bindings['num_dets'].data
            boxes = bindings['det_boxes'].data
            scores = bindings['det_scores'].data
            classes = bindings['det_classes'].data
            nums.shape,boxes.shape,scores.shape,classes.shape
            
            boxes = boxes[0,:nums[0][0]]
            scores = scores[0,:nums[0][0]]
            classes = classes[0,:nums[0][0]]
            
            iterate = 0
            ### postprocess
            for box,score,cl in zip(boxes,scores,classes):
                this_score = abs(round(float(score),2))
                if this_score < 0.3:
                    continue
                box = self.postprocess(box,ratio,dwdh).round().int()
                name = names[cl]
                color = colors[name]
                name += ' ' + str(round(float(score),3))
                #### Save part of the original image
                new_ori_img = original_img[box[1]:box[3], box[0]:box[2]]
                cv2.imwrite(f'{save_img_path}/original/{img_name}_detect_{iterate}.jpg', new_ori_img)
                #### Save part of the handwritten image
                new_hand_img = handwritten_img[box[1]:box[3], box[0]:box[2]]
                cv2.imwrite(f'{save_img_path}/handwritten/{img_name}_detect_{iterate}.jpg', new_hand_img)
                #### Increase Iterator
                iterate = iterate + 1
                #### Draw the rectangle and text
                cv2.rectangle(handwritten_img,box[:2].tolist(),box[2:].tolist(),color,2)
                cv2.putText(handwritten_img,name,(int(box[0]), int(box[1]) - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,thickness=2)
            ### Save the result image
            cv2.imwrite(f'{save_img_path}/yolo/{img_name}_detect.jpg', handwritten_img)
        
        ## Set end time
        end_time = time.time()
        eval_time = end_time - start_time
        print(f"Running Time: {eval_time} seconds")
        time.sleep(1)
        print("End of the detection_panseo_trt module")
        time.sleep(1)
        return save_img_path, eval_time
