"""
Graph network module
"""

import tensorflow as tf
from tensorflow import keras
import sonnet as snt

import graph_nets as gn
from graph_nets import utils_tf
from graph_nets.demos_tf2 import models

import SimulatedData
import GraphRepresentation
import GraphNetworkModules


def make_mlp(layers):
    model = tf.keras.models.Sequential()
    for layer in layers:
        model.add(keras.layers.Dense(layer))
    return model


def snt_mlp(layers):
    return lambda: make_mlp(layers)


def create_loss(target, outputs):
    losses = [
        tf.compat.v1.losses.mean_squared_error(target.nodes, output.nodes) +
        tf.compat.v1.losses.mean_squared_error(target.edges, output.edges)
        for output in outputs
    ]
    return tf.stack(losses)


def update_step(model: snt.Module, optimizer: snt.optimizers.Adam, inputs_tr, targets_tr):
    with tf.GradientTape() as tape:
        outputs_tr = model(inputs_tr)
        loss_tr = create_loss(targets_tr, outputs_tr)
        loss_tr = tf.math.reduce_sum(loss_tr)

    gradients = tape.gradient(loss_tr, model.trainable_variables)
    optimizer.apply(gradients, model.trainable_variables)

    return outputs_tr, loss_tr


# Optimizer.
learning_rate = 1e-3
optimizer = snt.optimizers.Adam(learning_rate)

# TODO: First, we should create a simple graph network to test learning
# TODO: Then, we can try the Encode-Process-Decode architecture

# Create the graph network.
module = GraphNetworkModules.EncodeProcessDecode(
    make_encoder_edge_model=snt_mlp([64, 64]),
    make_encoder_node_model=snt_mlp([64, 64]),
    make_encoder_global_model=snt_mlp([64]),
    make_core_edge_model=snt_mlp([64, 64]),
    make_core_node_model=snt_mlp([64, 64]),
    make_core_global_model=snt_mlp([64]),
    num_processing_steps=10,
    edge_output_size=3,
    node_output_size=3,
    global_output_size=1,
)

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
output_graphs = module(input_graphs)
