from detection_trt_test import detection_trt as dtt

path = "../Weights/best_tiny_400_16.trt"

dtt_obj = dtt(path)
dtt_obj.det_trt()
