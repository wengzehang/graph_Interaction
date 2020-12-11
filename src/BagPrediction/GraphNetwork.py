"""
Graph network module
"""

import tensorflow as tf
from tensorflow import keras
import sonnet as snt

import graph_nets as gn
from graph_nets import utils_tf

import SimulatedData
import GraphRepresentation


def make_mlp(layers):
    model = tf.keras.models.Sequential()
    for layer in layers:
        model.add(keras.layers.Dense(layer))
    return model


# TODO: First, we should create a simple graph network to test learning
# TODO: Then, we can try the Encode-Process-Decode architecture

# Create the graph network.
graph_net_module = gn.modules.GraphNetwork(
    edge_model_fn=lambda: snt.nets.MLP([32, 32]),
    node_model_fn=lambda: snt.nets.MLP([32, 32]),
    global_model_fn=lambda: snt.nets.MLP([32, 32]))

path_to_topodict = 'h5data/topo_train.pkl'
path_to_dataset = 'h5data/train_sphere_sphere_f_f_soft_out_scene1.h5'
data = SimulatedData.SimulatedData.load(path_to_topodict, path_to_dataset)

representation = GraphRepresentation.GraphRepresentation(SimulatedData.keypoint_indices, SimulatedData.keypoint_edges)

scenario = data.scenario(3)
frame = scenario.frame(0)

graph_dict = representation.to_graph_dict(frame)

# Test the conversion to tf.Tensor
input_graphs = utils_tf.data_dicts_to_graphs_tuple([graph_dict])

# Pass the input graphs to the graph network, and return the output graphs.
output_graphs = graph_net_module(input_graphs)
