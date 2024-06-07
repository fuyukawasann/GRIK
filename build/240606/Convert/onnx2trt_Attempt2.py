import onnx
import tensorrt as trt

# 1. onnx 모델 로드
onnx_model = onnx.load("../Weights/best_tiny_400_16.onnx")

# 2. TensorRT 엔진 생성
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
PLUGIN_REGISTRY = trt.get_plugin_registry()

batchedNMSPlugin = None
for creator in PLUGIN_CREATORS:
    if creator.name == "BatchedNMSDynamic_TRT":
        batchedNMSPlugin = creator
        break

if batchedNMSPlugin is None:
    raise RuntimeError("BatchedNMSDynamic_TRT plugin not found")

PLUGIN_REGISTRY.register_creator(batchedNMSPlugin, "BatchedNMSDynamic_TRT")

with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    
    # 4. onnx 모델 파싱
    parser.parse_from_file("../Weights/best_tiny_400_16.onnx")

    # 5. 엔진 빌드 설정
    builder.max_workspace_size = 1 << 30 # 1GB
    builder.max_batch_size = 1
    builder.fp16_mode = True # FP16 모드 활성화

    # 6. TensorRT 엔진 빌드 및 직렬화
    engine = builder.build_cuda_engine(network)
    buf = engine.serialize()
    with open("../Weights/yolov7.trt", "wb") as f:
        f.write(buf)
