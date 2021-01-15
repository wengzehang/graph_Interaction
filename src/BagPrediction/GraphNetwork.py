"""
Graph network module
"""

import tensorflow as tf
import sonnet as snt
import time
import os

from graph_nets import utils_tf

import SimulatedData
import GraphRepresentation
import GraphNetworkModules
import DataGenerator


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
  except RuntimeError as e:
    print(e)


train_path_to_topodict = 'h5data/topo_train.pkl'
# train_path_to_dataset = 'h5data/train_sphere_sphere_f_f_soft_out_scene1.h5'
train_path_to_dataset = 'h5data/train_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

valid_path_to_topodict = 'h5data/topo_valid.pkl'
# valid_path_to_dataset = 'h5data/valid_sphere_sphere_f_f_soft_out_scene1.h5'
valid_path_to_dataset = 'h5data/valid_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

# xxx: None for normal fully graph; 1 for partially connected graph; 2 for fully connected graph but with copied edge attribute
#  Remember to modified the edge attr size for the network
graph_edgetype = 1

train_data = SimulatedData.SimulatedData.load(train_path_to_topodict, train_path_to_dataset)
train_generator = DataGenerator.DataGenerator(train_data, graph_edgetype)

valid_data = SimulatedData.SimulatedData.load(valid_path_to_topodict, valid_path_to_dataset)
valid_generator = DataGenerator.DataGenerator(valid_data, graph_edgetype)

representation = GraphRepresentation.GraphRepresentation(SimulatedData.keypoint_indices, SimulatedData.keypoint_edges)


def make_mlp(layers):
    return snt.Sequential([
        snt.nets.MLP(layers, activate_final=True),
        snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
    ])


def snt_mlp(layers):
    return lambda: make_mlp(layers)


def create_loss(target, outputs, edgetype=None):
    losses = None
    if edgetype == None:
        losses = [
            tf.compat.v1.losses.mean_squared_error(target.nodes[:, :3], output.nodes[:, :3]) +
            tf.compat.v1.losses.mean_squared_error(target.edges[:, :3], output.edges[:, :3])
            for output in outputs
        ]
    elif edgetype == 1:
        losses = [
            tf.compat.v1.losses.mean_squared_error(target.nodes[:, :3], output.nodes[:, :3]) +
            tf.compat.v1.losses.mean_squared_error(target.edges[:, :3], output.edges[:, :3])
            for output in outputs
        ]
    elif edgetype == 2:
        losses = [
            tf.compat.v1.losses.mean_squared_error(target.nodes[:,:3], output.nodes[:,:3]) +
            tf.compat.v1.losses.mean_squared_error(target.edges[:,:6], output.edges[:,:6])
            for output in outputs
        ]

    return tf.stack(losses)


# Optimizer.
learning_rate = 1e-3
optimizer = snt.optimizers.Adam(learning_rate)

# Create the graph network.
module = GraphNetworkModules.EncodeProcessDecode(
    make_encoder_edge_model=snt_mlp([64, 64]),
    make_encoder_node_model=snt_mlp([64, 64]),
    make_encoder_global_model=snt_mlp([64]),
    make_core_edge_model=snt_mlp([64, 64]),
    make_core_node_model=snt_mlp([64, 64]),
    make_core_global_model=snt_mlp([64]),
    num_processing_steps=5,
    edge_output_size= 4, # TODO: Modify the edge number for different edge types
    node_output_size= 5, # 5, # 5, 3
    global_output_size=4, # TODO: Modify the length of global feature vector
)


def compute_output_and_loss(inputs_tr, targets_tr):
    outputs_tr = module(inputs_tr)
    # loss_tr = create_loss(targets_tr, outputs_tr)
    loss_tr = create_loss(targets_tr, outputs_tr)
    loss_tr = tf.math.reduce_sum(loss_tr) / module.num_processing_steps
    return outputs_tr, loss_tr


def update_step(inputs_tr, targets_tr):
    with tf.GradientTape() as tape:
        outputs_tr, loss_tr = compute_output_and_loss(inputs_tr, targets_tr)

    gradients = tape.gradient(loss_tr, module.trainable_variables)
    optimizer.apply(gradients, module.trainable_variables)

    return outputs_tr, loss_tr


# Get some example data that resembles the tensors that will be fed
# into update_step():
example_input_data, example_target_data = train_generator.next_batch(32)

# Get the input signature for that function by obtaining the specs
input_signature = [
    utils_tf.specs_from_graphs_tuple(example_input_data),
    utils_tf.specs_from_graphs_tuple(example_target_data)
]

# Compile the update function using the input signature for speedy code.
compiled_update_step = tf.function(update_step, input_signature=input_signature)
compiled_compute_output_and_loss = tf.function(compute_output_and_loss, experimental_relax_shapes=True)

model_path = "./models/test-10"
checkpoint_root = model_path + "/checkpoints"
checkpoint_name = "checkpoint-1"
checkpoint_save_prefix = os.path.join(checkpoint_root, checkpoint_name)

# Make sure the model path exists
if not os.path.exists(model_path):
    os.makedirs(model_path)

checkpoint = tf.train.Checkpoint(module=module)
latest = tf.train.latest_checkpoint(checkpoint_root)
if latest is not None:
    print("Loading latest checkpoint: ", latest)
    checkpoint.restore(latest)
else:
    print("No checkpoint found. Beginning training from scratch.")

# How many training steps before we do a validation run (+ checkpoint)
train_steps_per_validation = 100
validation_batch_size = 512# valid_generator.num_samples

batch_size = 32
log_every_seconds = 1
min_loss = 100.0
for iteration in range(0, 2000):
    last_iteration = iteration

    for i in range(train_steps_per_validation):
        (inputs_tr, targets_tr) = train_generator.next_batch(batch_size)
        outputs_tr, loss_tr = compiled_update_step(inputs_tr, targets_tr)

    # Calculate validation loss
    (inputs_val, targets_val) = valid_generator.next_batch(validation_batch_size)
    outputs_val, loss_val = compiled_compute_output_and_loss(inputs_val, targets_val)
    loss_val_np = loss_val.numpy()
    if loss_val_np < min_loss:
        checkpoint.save(checkpoint_save_prefix)
        min_loss = loss_val_np

    print("# {:05d}, Loss Train {:.4f}, Valid {:.4f}, Min {:.4f}".format(iteration,
                                                                         loss_tr.numpy(),
                                                                         loss_val_np,
                                                                         min_loss))
