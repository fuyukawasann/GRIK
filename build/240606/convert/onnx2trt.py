########## Project Description ##########
# This is a python script that converting onnx file to tensorrt
# BUILD: Jun 06, 2024 (KST)
#########################################

# Import necessary library
import tensorrt as trt

# Load onnx model
print("Load onnx model...")
onnx_model_path = "../Weights/best_tiny_400_16.onnx"
print("Load onnx model... Done!")

# Build TenserRT Engine
print("Build TRT Engine...")
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(TRT_LOGGER)
builder.max_batch_size = 1
## Create a TRT network object
print("Create TRT network object...")
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
print("Create TRT network object... Done!")
## Create a builder configuration object
print("Create Builder Configuration Object...")
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30 # 1GB
print("Create Builder Configuration Object... Done")

## Parse the contents of the ONNX file and adds the layers to the TRT network
parser = trt.OnnxParser(network, TRT_LOGGER)
with open(onnx_model_path, "rb") as f:
    print("Parsing ONNX file ...")
    parser.parse(f.read())
    print("Completed parsing ONNX file")
## Use FP16 mode if possible
if builder.platform_has_fast_fp16:
    print("FP16 Quantization is POSSIBLE")
    config.set_flag(trt.BuilderFlag.FP16)
else:
    print("FP16 Quantization is IMPOSSIBLE")

network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
engine = builder.build_engine(network, config)
print("Build TRT Engine... Done!")
print("SAVE Engine...")
ENGINE_PATH = "../Weights/best_tiny_400_16.trt"
with open(ENGINE_PATH, "wb") as f:
    f.write(engine.serialize())
print("SAVE Engine... Done!")
