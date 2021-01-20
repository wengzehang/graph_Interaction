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
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
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
graph_edgetype = None # 2

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

def create_loss_mask(target, outputs, edgetype=None):
    losses = None
    if edgetype == None:
        losses = [
            tf.compat.v1.losses.mean_squared_error(target.nodes[:, :3], output[:, :3])
            for output in outputs
        ]
    elif edgetype == 1:
        losses = [
            tf.compat.v1.losses.mean_squared_error(target.nodes[:, :3], output[:, :3])
            for output in outputs
        ]
    elif edgetype == 2:
        losses = [
            tf.compat.v1.losses.mean_squared_error(target.nodes[:,:3], output[:,:3])
            for output in outputs
        ]

    return tf.stack(losses)

# Optimizer.
learning_rate = 1e-3
optimizer = snt.optimizers.Adam(learning_rate)


# Get some example data that resembles the tensors that will be fed
# into update_step():
example_input_data, example_target_data = train_generator.next_batch(32)

# Get the input signature for that function by obtaining the specs
input_signature = [
    utils_tf.specs_from_graphs_tuple(example_input_data),
    utils_tf.specs_from_graphs_tuple(example_target_data)
]


# Create the graph network.
module_dyn = GraphNetworkModules.EncodeProcessDecode(
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

def create_node_output_label():
    return snt.nets.MLP([2],
                        activation=tf.nn.softmax,
                        activate_final=True,
                        name="node_output")

# Create the graph network.
module_mask = GraphNetworkModules.EncodeProcessDecode(
    make_encoder_edge_model=snt_mlp([64, 64]),
    make_encoder_node_model=snt_mlp([64, 64]),
    make_encoder_global_model=snt_mlp([64]),
    make_core_edge_model=snt_mlp([64, 64]),
    make_core_node_model=snt_mlp([64, 64]),
    make_core_global_model=snt_mlp([64]),
    num_processing_steps=1,
    edge_output_size=4, #example_target_data.edges.shape[1],
    node_output_size=2, #example_target_data.nodes.shape[1],
    global_output_size=4, #example_target_data.globals.shape[1],
    node_output_fn=create_node_output_label
)


def compute_output_and_loss(inputs_tr, targets_tr, edgetype=None, softmask=True):
    outputs_tr = module_dyn(inputs_tr)
    nodenumber = tf.shape(inputs_tr.nodes)[0]
    probmask = compute_hasmovedmask(inputs_tr)
    # calculate the mask
    if softmask == False:
        movedmask = tf.argmax(probmask, axis=1)
        movedmask = tf.reshape(tf.cast(movedmask, tf.float32), [nodenumber,1])

        masked_output = [
            (1 - movedmask) * inputs_tr.nodes[:, :3] + \
            movedmask * output.nodes[:, :3]

            for output in outputs_tr
        ]

    elif softmask == True:
        movedmask = tf.cast(probmask, tf.float32)

        # TODO: changed to soft mask
        masked_output = [
            tf.reshape(movedmask[:,0],[nodenumber, 1]) * inputs_tr.nodes[:, :3] + \
            tf.reshape(movedmask[:,1], [nodenumber, 1]) * output.nodes[:, :3]

            for output in outputs_tr
        ]

    else:
        # give -1
        # non-masked output
        masked_output = [
            output.nodes[:, :3]

            for output in outputs_tr
        ]

    loss_tr = create_loss_mask(targets_tr, masked_output, edgetype=edgetype)
    loss_tr = tf.math.reduce_sum(loss_tr) / module_dyn.num_processing_steps
    return outputs_tr, loss_tr

def compute_hasmovedmask(inputs_tr):
    outputs_tr = module_mask(inputs_tr)[-1].nodes
    return outputs_tr

def update_step(inputs_tr, targets_tr):
    with tf.GradientTape() as tape:
        # outputs_tr, loss_tr = compute_output_and_loss(inputs_tr, targets_tr, movedmask=movedmask)
        outputs_tr, loss_tr = compute_output_and_loss(inputs_tr, targets_tr)

    gradients = tape.gradient(loss_tr, module_dyn.trainable_variables)
    optimizer.apply(gradients, module_dyn.trainable_variables)

    return outputs_tr, loss_tr



# Compile the update function using the input signature for speedy code.
compiled_update_step = tf.function(update_step, input_signature=input_signature)
compiled_compute_output_and_loss = tf.function(compute_output_and_loss, experimental_relax_shapes=True)

model_path_dyn = "./models/test-14" # root 14 for soft mask, root 13 for saving dynamics estimation module checkpoints
checkpoint_root_dyn = model_path_dyn + "/checkpoints"
checkpoint_name_dyn = "checkpoint-1"
checkpoint_save_prefix_dyn = os.path.join(checkpoint_root_dyn, checkpoint_name_dyn)

# Make sure the model path exists
if not os.path.exists(model_path_dyn):
    os.makedirs(model_path_dyn)

checkpoint_dyn = tf.train.Checkpoint(module=module_dyn)
latest_dyn = tf.train.latest_checkpoint(checkpoint_root_dyn)
if latest_dyn is not None:
    print("Loading latest checkpoint: ", latest_dyn)
    checkpoint_dyn.restore(latest_dyn)
else:
    print("No checkpoint found for dynamics estimation. Beginning training from scratch.")


# load the trained mask module
model_path_mask = "./models/test-11" # root for saving mask estimation module checkpoints
#model_path = "./models/has-moved-2"
checkpoint_root_mask = model_path_mask + "/checkpoints"
checkpoint_name_mask = "checkpoint-1"
checkpoint_save_prefix_mask = os.path.join(checkpoint_root_mask, checkpoint_name_mask)
checkpoint_mask = tf.train.Checkpoint(module=module_mask)
latest_mask = tf.train.latest_checkpoint(checkpoint_root_mask)
checkpoint_mask.restore(latest_mask)
print("Loading latest checkpoint for the mask module: ", latest_mask)


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
        checkpoint_dyn.save(checkpoint_save_prefix_dyn)
        min_loss = loss_val_np

    print("# {:05d}, Loss Train {:.4f}, Valid {:.4f}, Min {:.4f}".format(iteration,
                                                                         loss_tr.numpy(),
                                                                         loss_val_np,
                                                                         min_loss))
