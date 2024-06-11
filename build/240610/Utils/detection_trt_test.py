import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2



class detection_trt:
    def __init__(self, engine_path):
        self.engine_path = engine_path

    def load_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context
    
    def alloc_buf(self):
        self.inputs = []
        self.outputs = []
        self.allocations = []

        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            shape = self.context.get_binding_shape(i)

            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
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
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        
    def inference(self, input_image):
        #image = input_image.transpose(0, 3, 1, 2)
        #image = np.ascontiguousarray(image)
        cuda.memcpy_htod(self.inputs[0]["allocation"], image)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o]["host_allocation"], self.outputs[o]["allocation"])
        
        num_detections = self.outputs[0]['host_allocation']
        nmsed_boxes = self.outputs[1]['host_allocation']
        nmsed_scores = self.outputs[2]['host_allocation']
        nmsed_classes = self.outputs[3]['host_allocation']
        result = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]
        return result
        
    def det_trt(self):
        self.load_engine()
        self.alloc_buf()
        test_image = cv2.imread("../test05.jpg")
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        result = self.inference(test_image)
        print(result)