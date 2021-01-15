"""
Graph network module
"""

import tensorflow as tf
import sonnet as snt
import os
import pickle

from graph_nets import utils_tf

import SimulatedData
import GraphNetworkModules
import DataGenerator

train_path_to_topodict = 'h5data/topo_train.pkl'
train_path_to_dataset = 'h5data/train_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

valid_path_to_topodict = 'h5data/topo_valid.pkl'
valid_path_to_dataset = 'h5data/valid_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

movement_threshold = 0.001

train_data = SimulatedData.SimulatedData.load(train_path_to_topodict, train_path_to_dataset)
train_generator = DataGenerator.DataGeneratorHasMoved(train_data, movement_threshold=movement_threshold)

valid_data = SimulatedData.SimulatedData.load(valid_path_to_topodict, valid_path_to_dataset)
valid_generator = DataGenerator.DataGeneratorHasMoved(valid_data, movement_threshold=movement_threshold)


def make_mlp(layers):
    return snt.Sequential([
        snt.nets.MLP(layers, activate_final=True),
        snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
    ])


def snt_mlp(layers):
    return lambda: make_mlp(layers)


def create_squared_error(target, output):
    return tf.reduce_sum((output - target) ** 2,
                         axis=-1)


def create_loss_op(target_op, output_op):
    # Average mean squared error over node (only a single attribte)
    #errors = create_squared_error(target_op.nodes, output_op.nodes)
    #loss = tf.reduce_mean(errors)
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(target_op.nodes, output_op.nodes)
    return loss


def create_accuracy_op(target_op, output_op):
    target_labels = target_op.nodes
    output_labels = tf.math.sign(output_op.nodes)

    equal_labels = tf.equal(target_labels, output_labels)

    count = tf.math.count_nonzero(equal_labels, dtype=tf.dtypes.float32)
    total = tf.shape(target_labels)[0]

    return tf.cast(count, tf.float32) / tf.cast(total, tf.float32)


def create_node_output_label():
    return snt.nets.MLP([2],
                        activation=tf.nn.softmax,
                        activate_final=True,
                        name="node_output")


# Optimizer.
learning_rate = 1e-4
optimizer = snt.optimizers.Adam(learning_rate)

# Get some example data that resembles the tensors that will be fed
# into update_step():
example_input_data, example_target_data = train_generator.next_batch(32)

# Create the graph network.
module = GraphNetworkModules.EncodeProcessDecode(
    make_encoder_edge_model=snt_mlp([64, 64]),
    make_encoder_node_model=snt_mlp([64, 64]),
    make_encoder_global_model=snt_mlp([64]),
    make_core_edge_model=snt_mlp([64, 64]),
    make_core_node_model=snt_mlp([64, 64]),
    make_core_global_model=snt_mlp([64]),
    num_processing_steps=2,
    edge_output_size=example_target_data.edges.shape[1],
    node_output_size=example_target_data.nodes.shape[1],
    global_output_size=example_target_data.globals.shape[1],
    node_output_fn=create_node_output_label
)


def predict(inputs_tr):
    return module(inputs_tr)


def compute_outputs(inputs_tr, targets_tr):
    outputs_tr = module(inputs_tr)
    # loss_tr = create_loss(targets_tr, outputs_tr)
    loss_tr = create_loss_op(targets_tr, outputs_tr[-1])
    loss_tr = tf.math.reduce_sum(loss_tr) / module.num_processing_steps

    return outputs_tr, loss_tr


def update_step(inputs_tr, targets_tr):
    with tf.GradientTape() as tape:
        outputs_tr, loss_tr, acc_tr = compute_outputs(inputs_tr, targets_tr)

    gradients = tape.gradient(loss_tr, module.trainable_variables)
    optimizer.apply(gradients, module.trainable_variables)

    return outputs_tr, loss_tr, acc_tr


# Get the input signature for that function by obtaining the specs
input_signature = [
    utils_tf.specs_from_graphs_tuple(example_input_data),
    utils_tf.specs_from_graphs_tuple(example_target_data)
]

input_signature_predict = [
    utils_tf.specs_from_graphs_tuple(example_input_data)
]

# Compile the update function using the input signature for speedy code.
compiled_update_step = tf.function(update_step, input_signature=input_signature)
compiled_compute_outputs = tf.function(compute_outputs, experimental_relax_shapes=True)
compiled_predict = tf.function(predict, experimental_relax_shapes=True)

# Checkpoint stuff
model_path = "./models/test-11"
#model_path = "./models/has-moved-2"
checkpoint_root = model_path + "/checkpoints"
checkpoint_name = "checkpoint-1"
checkpoint_save_prefix = os.path.join(checkpoint_root, checkpoint_name)

# Make sure the model path exists
if not os.path.exists(model_path):
    os.makedirs(model_path)

checkpoint = tf.train.Checkpoint(module=module)
latest = checkpoint_root + "/checkpoint-1-432"
#latest = tf.train.latest_checkpoint(checkpoint_root)
if latest is not None:
    print("Loading latest checkpoint: ", latest)
    checkpoint.restore(latest)
else:
    print("No checkpoint found. Beginning training from scratch.")

# How many training steps before we do a validation run (+ checkpoint)
train_steps_per_validation = 100
batch_size = 32
validation_batch_size = 512  # valid_generator.num_samples

# save_complete_model = True
# if save_complete_model:
#     print("Trying to save model, running one validation step first")
#     (inputs_tr, targets_tr) = train_generator.next_batch(batch_size)
#     outputs_tr = compiled_predict(inputs_tr)
#     (inputs_val, targets_val) = valid_generator.next_batch(validation_batch_size)
#     outputs_val = compiled_predict(inputs_val)
#
#     #to_save = snt.Module()
#     #to_save.predict = compiled_predict
#     #to_save.all_variables = list(module.variables)
#
#     print("Now, saving the model")
#     complete_model_path = model_path + "/pickle_model"
#     #tf.saved_model.save(to_save, complete_model_path)
#     with open(complete_model_path, 'wb') as file:
#         pickle.dump(module, file)
#     print("Saved model to", complete_model_path)
#     raise NotImplementedError()

train_accuracy = tf.keras.metrics.Accuracy()
valid_accuracy = tf.keras.metrics.Accuracy()

min_loss = 100.0
for iteration in range(0, 2000):
    last_iteration = iteration

    for i in range(train_steps_per_validation):
        (inputs_tr, targets_tr) = train_generator.next_batch(batch_size)
        outputs_tr, loss_tr = compiled_update_step(inputs_tr, targets_tr)

        train_accuracy.update_state(tf.argmax(targets_tr.nodes, axis=1),
                                    tf.argmax(outputs_tr[-1].nodes, axis=1))

    # Calculate validation loss
    (inputs_val, targets_val) = valid_generator.next_batch(validation_batch_size)
    outputs_val, loss_val = compiled_compute_outputs(inputs_val, targets_val)

    valid_accuracy.update_state(tf.argmax(targets_val.nodes, axis=1), tf.argmax(outputs_val[-1].nodes, axis=1))
    acc_tr = train_accuracy.result().numpy()
    acc_val = valid_accuracy.result().numpy()

    loss_val_np = loss_val.numpy()
    if loss_val_np < min_loss:
        checkpoint.save(checkpoint_save_prefix)
        min_loss = loss_val_np

    if train_generator.has_reshuffled:
        train_accuracy.reset_states()
    if valid_generator.has_reshuffled:
        valid_accuracy.reset_states()

    print("# {:05d}, Train Loss {:.4f}, Acc {:.4f}; Valid Loss {:.4f}, Min {:.4f}, Acc {:.4f}"
          .format(iteration,
                  loss_tr.numpy(),
                  acc_tr,
                  loss_val_np,
                  min_loss,
                  acc_val))
