#!/usr/bin/env python3
# Template auto-generated by polygraphy [v0.33.0] on 10/09/22 at 09:13:18
# Generation Command: /opt/conda/bin/polygraphy template trt-network model.onnx --output modifyNetwork.py
# Creates a TensorRT Network using the Network API.
import tensorrt as trt
from polygraphy import func
from polygraphy.backend.trt import NetworkFromOnnxPath

# Loaders
parse_network_from_onnx = NetworkFromOnnxPath('/work/gitlab/tensorrt-cookbook/08-Tool/Polygraphy/templateExample/model.onnx')

@func.extend(parse_network_from_onnx)
def load_network(builder, network, parser):
    pass # TODO: Set up the network here. This function should not return anything.
