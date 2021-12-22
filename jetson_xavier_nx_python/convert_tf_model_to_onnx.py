import os
import sys
import argparse

import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

import numpy as np
import tensorflow as tf
from tf2onnx import tfonnx, optimizer, tf_loader
from onnxmltools.utils.float16_converter import convert_float_to_float16

def main(saved_model_path=None, onnx_path=None,batch_size=1, input_size=0, experimental_float16_onnx=False):
    assert saved_model_path != None
    assert onnx_path != None

    graph_def, inputs, outputs, tensors_to_rename = tf_loader.from_saved_model(saved_model_path, None, None, "serve", ["serving_default"], return_tensors_to_rename=True)
    print("Tensors to rename: {}".format(tensors_to_rename))
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name="")
    with tf_loader.tf_session(graph=tf_graph):
        onnx_graph = tfonnx.process_tf_graph(tf_graph, input_names=inputs, output_names=outputs, opset=13)
    onnx_model = optimizer.optimize_graph(onnx_graph).make_model("Converted from {}".format(saved_model_path))
    if experimental_float16_onnx:
        onnx_model = convert_float_to_float16(onnx_model)
        print("Onnx model converted to float16")
    graph = gs.import_onnx(onnx_model)
    assert graph
    print("ONNX graph created successfully")
    
    # Set the I/O tensor shapes
    graph.inputs[0].shape[0] = batch_size
    graph.outputs[3].shape = [1, 100, 91]
    

    if input_size > 0:
        if graph.inputs[0].shape[3] == 3:
            # Format NHWC
            graph.inputs[0].shape[1] = input_size
            graph.inputs[0].shape[2] = input_size
        elif graph.inputs[0].shape[1] == 3:
            # Format NCHW
            graph.inputs[0].shape[2] = input_size
            graph.inputs[0].shape[3] = input_size
    
    print("ONNX input named '{}' with shape {}".format(graph.inputs[0].name, graph.inputs[0].shape))
    print("ONNX output named '{}' with shape {}".format(graph.outputs[0].name, graph.outputs[0].shape))

    for i in range(4):
        if type(graph.inputs[0].shape[i] != int or graph.inputs[0].shape[i] <= 0):
            if type(graph.inputs[0].shape[i]) != int or graph.inputs[0].shape[i] <= 0:
                print("The input shape of the graph is invalid, try overriding it by giving a fixed size with --input_size")
                sys.exit(1)
    
    #TEST FOR THIS PART ... FOR FLOAT32 and FLOAT16
    # Fix Clip Nodes (ReLU6)
    for node in [n for n in graph.nodes if n.op == "Clip"]:
        for input in node.inputs[1:]:
            # In TensorRT, the min/max inputs on a Clip op *must* have fp32 datatype
            input.values = np.float32(input.values)
    
    graph.cleanup().toposort()
    model = shape_inference.infer_shapes(gs.export_onnx(graph))
    graph = gs.import_onnx(model)

    #Save updated model
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    onnx.save(model, onnx_path)
    
    print("ONNX model saved to {}".format(onnx_path))

    


if __name__ == "__main__":
    saved_model_path = "/home/gorkem/Downloads/Compressed/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model"
    onnx_path = "/home/gorkem/Documents/acceleration/jetson_xavier_nx_python/models/efficientnet_d0_float16.onnx"
    main(saved_model_path, onnx_path=onnx_path, batch_size=1, input_size=224, experimental_float16_onnx=False)