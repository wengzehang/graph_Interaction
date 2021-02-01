"""
New prediction models based on model specification
"""
from abc import ABC

import SimulatedData
import ModelSpecification
from ModelTrainer import ModelLoader
from PredictionInterface import PredictionInterface, PredictedFrame

from graph_nets import utils_tf

import numpy as np
import itertools


def create_input_graph_dict(specification: ModelSpecification.ModelSpecification,
                            current_frame: SimulatedData.Frame,
                            effector_xyzr_next: np.array,
                            training: bool = False):
    keypoint_indices = specification.cloth_keypoints.indices
    fixed_keypoint_indices = specification.cloth_keypoints.fixed_indices

    # Cloth object node features
    # Format: Position (XYZ), Radius (R), InverseDense flag (0 if fixed, 1 if movable)
    cloth_data_current = np.float32(current_frame.get_cloth_keypoint_info(keypoint_indices, fixed_keypoint_indices))

    # Rigid object node features
    # Format: Position (XYZ), Radius (R), InverseDense flag (always 1 for movable)
    rigid_data_current = np.float32(current_frame.get_rigid_keypoint_info())

    # Data for all nodes is stacked (cloth first, rigid second)
    node_data_current = np.vstack((cloth_data_current, rigid_data_current))

    # TensorFlow expects float32 values, the dataset contains float64 values
    effector_xyzr_current = np.float32(current_frame.get_effector_pose()).reshape(4)

    input_global_format = specification.input_graph_format.global_format
    input_global_features = input_global_format.compute_features(effector_xyzr_current,
                                                                 effector_xyzr_next)

    # Move to ModelSpecification.py?
    position_frame = specification.position_frame
    if position_frame == ModelSpecification.PositionFrame.Global:
        # No transformation needed
        pass
    elif position_frame == ModelSpecification.PositionFrame.LocalToEndEffector:
        # Transform positions to local frame (current effector position)
        new_origin = effector_xyzr_current[:3]
        node_data_current[:, :3] -= new_origin
    else:
        raise NotImplementedError("Position frame not implemented")

    movement_threshold = specification.training_params.movement_threshold

    # Add input noise to the position data (only during training)
    positions_current = node_data_current[:, :3]
    noise_stddev = specification.training_params.input_noise_stddev
    if training and noise_stddev is not None:
        noise = np.random.normal([0.0, 0.0, 0.0], noise_stddev, positions_current.shape)
        positions_current += noise

    # Create input node features (after applying noise)
    input_node_format = specification.input_graph_format.node_format
    # Next node data is only required for HasMovedClasses node format
    node_data_next = None
    input_node_features = input_node_format.compute_features(node_data_current,
                                                             node_data_current, node_data_next,
                                                             movement_threshold)

    input_edge_format = specification.input_graph_format.edge_format
    input_edge_features = input_edge_format.compute_features(positions_current,
                                                             specification.cloth_keypoints.keypoint_edges_from,
                                                             specification.cloth_keypoints.keypoint_edges_to)

    num_nodes = node_data_current.shape[0]
    edge_index = [i for i in itertools.product(np.arange(num_nodes), repeat=2)]
    # all connected, bidirectional
    node_edges_to, node_edges_from = list(zip(*edge_index))

    input_graph_dict = {
        "globals": input_global_features,
        "nodes": input_node_features,
        "edges": input_edge_features,
        "senders": node_edges_from,
        "receivers": node_edges_to,
    }

    return input_graph_dict


class ModelFromSpecification(PredictionInterface, ABC):
    def __init__(self, model: ModelSpecification.ModelSpecification):
        self.model_loader = ModelLoader(model)
        # We do not need to give example input data if we do not train the network
        self.model_loader.initialize_graph_net(None, None)

        # Recompute cloth edges (if loaded data is out of date)
        ck = self.model_loader.model.cloth_keypoints
        self.model_loader.model.cloth_keypoints = ModelSpecification.ClothKeypoints(ck.indices, ck.edges,
                                                                                    ck.fixed_indices)

    def create_input_graph_tuples(self, frame: SimulatedData.Frame, next_effector_position: np.array):
        # Prepare input graph tuples
        input_graph_dict = create_input_graph_dict(self.model_loader.model,
                                                   frame, next_effector_position)
        input_graph_tuples = utils_tf.data_dicts_to_graphs_tuple([input_graph_dict])
        return input_graph_tuples

    def predict_output_graph_tuples(self, frame: SimulatedData.Frame, next_effector_position: np.array):
        input_graph_tuples = self.create_input_graph_tuples(frame, next_effector_position)

        # Model prediction
        predicted_graph_tuples = self.model_loader.compiled_predict(input_graph_tuples)
        return predicted_graph_tuples


class MotionModelFromSpecification(ModelFromSpecification):
    def __init__(self, model: ModelSpecification.ModelSpecification):
        super().__init__(model)

        assert self.model_loader.model.output_graph_format.node_format == ModelSpecification.NodeFormat.XYZ, \
            "Output node format must be XYZ for the motion model"

    def predict_frame(self, frame: SimulatedData.Frame, next_effector_position: np.array) -> PredictedFrame:

        predicted_graph_tuples = self.predict_output_graph_tuples(frame, next_effector_position)

        # Convert output graph tuple to PredictFrame
        predicted_nodes = predicted_graph_tuples[-1].nodes.numpy()

        if self.model_loader.model.position_frame == ModelSpecification.PositionFrame.LocalToEndEffector:
            # Add the current effector position back to transform the center position to global coordinate
            current_effector_position = frame.get_effector_pose()[0][:3]
            predicted_nodes[:, :3] += current_effector_position

        # The first entries are cloth keypoints (followed by rigid body nodes)
        num_keypoints = len(self.model_loader.model.cloth_keypoints.indices)
        cloth_keypoint_positions = predicted_nodes[:num_keypoints, :3]
        rigid_body_positions = predicted_nodes[num_keypoints:, :3]

        return PredictedFrame(cloth_keypoint_positions, rigid_body_positions)


def compute_moved_indices(predicted_nodes):
    has_moved_mask = np.argmax(predicted_nodes, axis=1)
    has_moved_indices = np.where(has_moved_mask > 0)
    return has_moved_indices


class HasMovedMaskModelFromSpecification(ModelFromSpecification):
    def __init__(self,
                 motion_model: PredictionInterface,
                 has_moved_spec: ModelSpecification.ModelSpecification):
        super().__init__(has_moved_spec)

        self.motion_model = motion_model

        output_node_format = self.model_loader.model.output_graph_format.node_format
        assert output_node_format == ModelSpecification.NodeFormat.HasMovedClasses, \
            "Output node format must be HasMovedClasses for the has_moved mask model"

    def predict_frame(self, frame: SimulatedData.Frame, next_effector_position: np.array) -> PredictedFrame:
        # Predict motion without mask
        motion_prediction = self.motion_model.predict_frame(frame, next_effector_position)
        unmasked_pos_cloth = motion_prediction.cloth_keypoint_positions
        unmasked_pos_rigid = motion_prediction.rigid_body_positions

        # Predict has_moved mask
        predicted_graph_tuples = self.predict_output_graph_tuples(frame, next_effector_position)
        predicted_nodes = predicted_graph_tuples[-1].nodes.numpy()

        keypoint_indices = self.model_loader.model.cloth_keypoints.indices
        num_keypoints = len(keypoint_indices)

        # The first indices are the cloth keypoints (the last indices are rigid bodies)
        has_moved_cloth = compute_moved_indices(predicted_nodes[:num_keypoints])
        has_moved_rigid = compute_moved_indices(predicted_nodes[num_keypoints:])

        # Only override positions that have been classified as moving
        original_pos_cloth = frame.get_cloth_keypoint_positions(keypoint_indices)
        predicted_pos_cloth = original_pos_cloth
        predicted_pos_cloth[has_moved_cloth] = unmasked_pos_cloth[has_moved_cloth]

        original_pos_rigid = frame.get_rigid_keypoint_info()[:, :3]
        predicted_pos_rigid = original_pos_rigid
        predicted_pos_rigid[has_moved_rigid] = unmasked_pos_rigid[has_moved_rigid]

        return PredictedFrame(predicted_pos_cloth, predicted_pos_rigid)
