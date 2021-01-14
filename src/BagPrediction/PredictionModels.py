"""
Implementations of the prediction interface
"""

import SimulatedData
import GraphRepresentation
from GraphNetworkModules import EncodeProcessDecode, snt_mlp
from PredictionInterface import PredictionInterface, PredictedFrame

from graph_nets import utils_tf
import tensorflow as tf
import numpy as np


class FullyConnectedPredictionModel(PredictionInterface):
    def __init__(self,
                 model_path="./models/test-11",
                 checkpoint_to_load="checkpoint-1-432"):

        self.model = self.create_model()

        checkpoint_root = model_path + "/checkpoints"
        if checkpoint_to_load is None:
            latest = tf.train.latest_checkpoint(checkpoint_root)
        else:
            latest = checkpoint_root + "/checkpoint-1-432"

        print("Loading checkpoint:", latest)
        checkpoint = tf.train.Checkpoint(module=self.model)
        checkpoint.restore(latest)

        self.representation = GraphRepresentation.GraphRepresentation_rigid_deformable(SimulatedData.keypoint_indices,
                                                                                       SimulatedData.keypoint_edges)

    def create_model(self):
        # This cannot change
        # If you train a new model with different architecture, then make a new subclass of PredictionModel
        return EncodeProcessDecode(
            make_encoder_edge_model=snt_mlp([64, 64]),
            make_encoder_node_model=snt_mlp([64, 64]),
            make_encoder_global_model=snt_mlp([64]),
            make_core_edge_model=snt_mlp([64, 64]),
            make_core_node_model=snt_mlp([64, 64]),
            make_core_global_model=snt_mlp([64]),
            num_processing_steps=5,
            edge_output_size=4,
            node_output_size=5,
            global_output_size=4,
        )

    def predict_frame(self, frame: SimulatedData.Frame, next_effector_position: np.array) -> PredictedFrame:
        # Prepare input graph tuples
        current_effector_position = frame.get_effector_pose()[0][:3]
        input_graph_dict = self.representation.to_graph_dict_global_4_align(frame, next_effector_position,
                                                                            current_effector_position, add_noise=False)
        input_graph_tuples = utils_tf.data_dicts_to_graphs_tuple([input_graph_dict])

        # Model prediction
        predicted_graph_tuples = self.model(input_graph_tuples)

        # Convert output graph tuple to PredictFrame
        predicted_nodes = predicted_graph_tuples[-1].nodes.numpy()
        # Add the previous effector position back to transform the center position to global coordinate
        predicted_nodes[:, :3] += current_effector_position

        # The first entries are cloth keypoints (followed by rigid body nodes)
        cloth_keypoint_positions = predicted_nodes[:len(self.representation.keypoint_indices), :3]

        # FIXME: Rigid body prediction not yet implemented
        rigid_body_positions = None

        return PredictedFrame(cloth_keypoint_positions, rigid_body_positions)


if __name__ == '__main__':
    train_path_to_topodict = 'h5data/topo_train.pkl'
    train_path_to_dataset = 'h5data/train_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

    valid_path_to_topodict = 'h5data/topo_valid.pkl'
    valid_path_to_dataset = 'h5data/valid_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

    movement_threshold = 0.001

    train_data = SimulatedData.SimulatedData.load(train_path_to_topodict, train_path_to_dataset)

    scenario = train_data.scenario(0)
    current_frame = scenario.frame(0)
    next_frame = scenario.frame(1)

    model = FullyConnectedPredictionModel()
    next_effector_position = next_frame.get_effector_pose()[0]
    predicted = model.predict_frame(current_frame, next_effector_position)
