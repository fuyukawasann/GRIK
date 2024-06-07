import tensorrt as trt

onnx_file = 'yolov7-tiny-custom.onnx'

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
success = parser.parse_from_file(onnx_file)

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30
serialized_engine = builder.build_serialized_network(network, config)

with open('yolov7-tiny-custom.trt', 'wb') as f:
    f.write(serialized_engine)
