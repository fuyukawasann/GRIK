import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time




class detection_trt:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        
    def _SettingTRTEngine(self):
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.trt_logger)
        trt.init_libnvinfer_plugins(self.trt_logger, "")
        self.SettingTRTEngine_GetEngine(self.engine_path, runtime)
    
    def _SettingTRTEngine_GetEngine(self, engine_path, runtime, from_Tensor=True, to_Tensor=True):
        context = self.engine.create_execution_context()
    def det_trt(self):
        self._SettingTRTEngine()
