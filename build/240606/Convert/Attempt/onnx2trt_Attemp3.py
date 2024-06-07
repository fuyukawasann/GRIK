import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    
    # Parsing ONNX model
    parser.parse_from_file("../Weights/best_tiny_400_16.onnx")

    # Consist BUILD ENGINE CONFIG
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    config.set_flag(trt.BuilderFlag.FP16)
    builder.max_batch_size = 1

    # Convert TRT and serialized
    engine = builder.build_engine(network, config)
    buf = engine.serialize()
    with open("../Weights/best_tiny_400_16.trt", "wb") as f:
        f.write(buf)
