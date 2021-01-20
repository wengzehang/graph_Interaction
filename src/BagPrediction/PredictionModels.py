"""
Implementations of the prediction interface
"""

import SimulatedData
import GraphRepresentation
from GraphNetworkModules import EncodeProcessDecode, snt_mlp, create_node_output_label
from PredictionInterface import PredictionInterface, PredictedFrame

from graph_nets import utils_tf
import tensorflow as tf
import numpy as np
import sonnet as snt


class FullyConnectedPredictionModel(PredictionInterface):
    def __init__(self,
                 model_path="./models/test-11",
                 checkpoint_to_load="checkpoint-1-432"):

        self.model = self.create_model()

        checkpoint_root = model_path + "/checkpoints"
        if checkpoint_to_load is None:
            latest = tf.train.latest_checkpoint(checkpoint_root)
        else:
            latest = checkpoint_root + "/" + checkpoint_to_load

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
                                                                            current_effector_position)
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


class FullyConnectedMaskedPredictionModel(PredictionInterface):
    def __init__(self,
                 model_dyn_path="./models/test-13",
                 model_mask_path="./models/test-11",
                 checkpoint_dyn_to_load="checkpoint-1-28",
                 checkpoint_mask_to_load="checkpoint-1-10", ):

        self.model_dyn, self.model_mask = self.create_model()

        checkpoint_root_dyn = model_dyn_path + "/checkpoints/"
        if checkpoint_dyn_to_load is None:
            latest_dyn = tf.train.latest_checkpoint(checkpoint_root_dyn)
        else:
            latest_dyn = checkpoint_root_dyn + checkpoint_dyn_to_load

        checkpoint_root_mask = model_mask_path + "/checkpoints/"
        if checkpoint_mask_to_load is None:
            latest_mask = tf.train.latest_checkpoint(checkpoint_root_mask)
        else:
            latest_mask = checkpoint_root_mask + checkpoint_mask_to_load

        print("Loading checkpoint for dyn:", latest_dyn)
        checkpoint_dyn = tf.train.Checkpoint(module=self.model_dyn)
        checkpoint_dyn.restore(latest_dyn)

        print("Loading checkpoint for mask:", latest_mask)
        checkpoint_mask = tf.train.Checkpoint(module=self.model_mask)
        checkpoint_mask.restore(latest_mask)

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
        ), EncodeProcessDecode(
            make_encoder_edge_model=snt_mlp([64, 64]),
            make_encoder_node_model=snt_mlp([64, 64]),
            make_encoder_global_model=snt_mlp([64]),
            make_core_edge_model=snt_mlp([64, 64]),
            make_core_node_model=snt_mlp([64, 64]),
            make_core_global_model=snt_mlp([64]),
            num_processing_steps=1,
            edge_output_size=4,
            node_output_size=2,
            global_output_size=4,
            node_output_fn=create_node_output_label
        )

    def predict_frame(self, frame: SimulatedData.Frame, next_effector_position: np.array) -> PredictedFrame:
        # Prepare input graph tuples
        current_effector_position = frame.get_effector_pose()[0][:3]
        input_graph_dict = self.representation.to_graph_dict_global_4_align(frame, next_effector_position,
                                                                            current_effector_position, add_noise=False)
        input_graph_tuples = utils_tf.data_dicts_to_graphs_tuple([input_graph_dict])

        # Mask prediction
        predicted_graph_mask_tuples = self.model_mask(input_graph_tuples)
        nonmovedIndices = predicted_graph_mask_tuples[0].nodes.numpy()

        # Model prediction
        predicted_graph_tuples = self.model_dyn(input_graph_tuples)

        # Convert output graph tuple to PredictFrame
        predicted_nodes = predicted_graph_tuples[-1].nodes.numpy()

        # renew the fixed/non-moved points
        predicted_nodes[SimulatedData.fix_keypoint_place, :3] = input_graph_dict['nodes'][
                                                                SimulatedData.fix_keypoint_place, :3]
        predicted_nodes[nonmovedIndices[:, 0] > 0.5, :3] = input_graph_dict['nodes'][nonmovedIndices[:, 0] > 0.5,
                                                           :3]

        # Add the previous effector position back to transform the center position to global coordinate
        predicted_nodes[:, :3] += current_effector_position

        # The first entries are cloth keypoints (followed by rigid body nodes)
        cloth_keypoint_positions = predicted_nodes[:len(self.representation.keypoint_indices), :3]

        # FIXME: Rigid body prediction not yet implemented
        rigid_body_positions = None

        return PredictedFrame(cloth_keypoint_positions, rigid_body_positions)


class HasMovedMaskPredictionModel(PredictionInterface):
    def __init__(self,
                 motion_model: PredictionInterface,
                 model_path="./models/has-moved-2",
                 checkpoint_to_load=None):
        self.motion_model = motion_model

        self.has_moved_model = self.create_has_moved_model()

        checkpoint_root = model_path + "/checkpoints"
        if checkpoint_to_load is None:
            latest = tf.train.latest_checkpoint(checkpoint_root)
        else:
            latest = checkpoint_root + "/" + checkpoint_to_load

        print("Loading checkpoint:", latest)
        checkpoint = tf.train.Checkpoint(module=self.has_moved_model)
        checkpoint.restore(latest)

        self.representation = GraphRepresentation.GraphRepresentation_rigid_deformable(SimulatedData.keypoint_indices,
                                                                                       SimulatedData.keypoint_edges)

    @staticmethod
    def create_node_output_label():
        return snt.nets.MLP([2],
                            activation=tf.nn.softmax,
                            activate_final=True,
                            name="node_output")

    def create_has_moved_model(self):
        return EncodeProcessDecode(
            make_encoder_edge_model=snt_mlp([64, 64]),
            make_encoder_node_model=snt_mlp([64, 64]),
            make_encoder_global_model=snt_mlp([64]),
            make_core_edge_model=snt_mlp([64, 64]),
            make_core_node_model=snt_mlp([64, 64]),
            make_core_global_model=snt_mlp([64]),
            num_processing_steps=2,
            edge_output_size=4,
            node_output_size=2,
            global_output_size=4,
            node_output_fn=HasMovedMaskPredictionModel.create_node_output_label
        )

    def predict_frame(self, frame: SimulatedData.Frame, next_effector_position: np.array) -> PredictedFrame:

        # Prepare input graph tuples
        current_effector_position = frame.get_effector_pose()[0][:3]
        input_graph_dict = self.representation.to_graph_dict_global_4_align(frame, next_effector_position,
                                                                            current_effector_position)
        input_graph_tuples = utils_tf.data_dicts_to_graphs_tuple([input_graph_dict])

        # Model prediction
        predicted_graph_tuples = self.has_moved_model(input_graph_tuples)

        # Convert output graph tuple to PredictFrame
        predicted_nodes = predicted_graph_tuples[-1].nodes.numpy()
        # The last indices are the cloth keypoints (the first indices are rigid bodies)
        has_moved_mask = np.argmax(predicted_nodes[:len(self.representation.keypoint_indices)], axis=1)
        has_moved_indices = np.where(has_moved_mask > 0)
        has_moved_indices_cloth = has_moved_indices

        # Predict motion without mask
        motion_prediction = self.motion_model.predict_frame(frame, next_effector_position)
        unmasked_pos = motion_prediction.cloth_keypoint_positions

        # Only override positions that have been classified as moving
        original_pos = frame.get_cloth_keypoint_positions(self.representation.keypoint_indices)
        predicted_pos = original_pos
        predicted_pos[has_moved_indices_cloth] = unmasked_pos[has_moved_indices_cloth]

        # FIXME: Rigid body prediction not yet implemented
        rigid_body_positions = None

        return PredictedFrame(predicted_pos, rigid_body_positions)


class FullyConnectedHorizonPredictionModel(PredictionInterface):
    def __init__(self,
                 single_prediction_model: PredictionInterface,
                 frame_step: int = 5,
                 motion_model_path="./models/masked-prediction-horizon-5",
                 motion_model_checkpoint=None,
                 has_moved_model_path="./models/has-moved-horizon-5",
                 has_moved_model_checkpoint=None):

        # We use a base prediction model for frame-wise prediction
        self.single_prediction_model = single_prediction_model
        self.frame_step = frame_step

        # These models were trained to predict 'frame_step' into the future
        # We use these predictions as anchor points for the frame-wise prediction
        self.horizon_motion_model, self.horizon_has_moved_model = self.create_models()

        motion_checkpoint_root = motion_model_path + "/checkpoints/"
        if motion_model_checkpoint is None:
            motion_latest = tf.train.latest_checkpoint(motion_checkpoint_root)
        else:
            motion_latest = motion_checkpoint_root + motion_model_checkpoint

        has_moved_checkpoint_root = has_moved_model_path + "/checkpoints/"
        if has_moved_model_checkpoint is None:
            has_moved_latest = tf.train.latest_checkpoint(has_moved_checkpoint_root)
        else:
            has_moved_latest = has_moved_checkpoint_root + has_moved_model_checkpoint

        print("Loading checkpoint for motion model:", motion_latest)
        motion_checkpoint = tf.train.Checkpoint(module=self.horizon_motion_model)
        motion_checkpoint.restore(motion_latest)

        print("Loading checkpoint for has_moved:", has_moved_latest)
        has_moved_checkpoint = tf.train.Checkpoint(module=self.horizon_has_moved_model)
        has_moved_checkpoint.restore(has_moved_latest)

        self.representation = GraphRepresentation.GraphRepresentation_rigid_deformable(
            SimulatedData.keypoint_indices,
            SimulatedData.keypoint_edges)

        self.anchor_frames = []

    def create_models(self):
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
        ), EncodeProcessDecode(
            make_encoder_edge_model=snt_mlp([64, 64]),
            make_encoder_node_model=snt_mlp([64, 64]),
            make_encoder_global_model=snt_mlp([64]),
            make_core_edge_model=snt_mlp([64, 64]),
            make_core_node_model=snt_mlp([64, 64]),
            make_core_global_model=snt_mlp([64]),
            num_processing_steps=1,
            edge_output_size=4,
            node_output_size=2,
            global_output_size=4,
            node_output_fn=create_node_output_label
        )

    def prepare_scenario(self, scenario: SimulatedData.Scenario):

        num_anchor_frames = scenario.num_frames() // self.frame_step
        anchor_frames = [None] * num_anchor_frames

        current_frame = scenario.frame(0)
        anchor_frames[0] = PredictedFrame(
            current_frame.get_cloth_keypoint_positions(self.representation.keypoint_indices),
            None)
        next_frame = scenario.frame(self.frame_step)

        next_effector_position = next_frame.get_effector_pose()[0]
        prev_predicted_frame = self.predict_horizon_frame(current_frame, next_effector_position)

        for anchor_index in range(1, num_anchor_frames):
            current_frame = next_frame
            current_frame.overwrite_keypoint_positions(prev_predicted_frame.cloth_keypoint_positions)


            frame_index = anchor_index * self.frame_step
            next_frame = scenario.frame(frame_index + self.frame_step)
            next_effector_position = next_frame.get_effector_pose()[0]

            # Evaluate single frame
            predicted_frame = self.predict_horizon_frame(current_frame, next_effector_position)
            anchor_frames[anchor_index] = predicted_frame

            prev_predicted_frame = predicted_frame

        self.anchor_frames = anchor_frames

    def predict_horizon_frame(self, frame: SimulatedData.Frame, next_effector_position: np.array) -> PredictedFrame:
        # Prepare input graph tuples
        current_effector_position = frame.get_effector_pose()[0][:3]
        input_graph_dict = self.representation.to_graph_dict_global_4_align(frame, next_effector_position,
                                                                            current_effector_position, add_noise=False)
        input_graph_tuples = utils_tf.data_dicts_to_graphs_tuple([input_graph_dict])

        # Mask prediction
        predicted_graph_mask_tuples = self.horizon_has_moved_model(input_graph_tuples)
        nonmovedIndices = predicted_graph_mask_tuples[0].nodes.numpy()

        # Model prediction
        predicted_graph_tuples = self.horizon_motion_model(input_graph_tuples)

        # Convert output graph tuple to PredictFrame
        predicted_nodes = predicted_graph_tuples[-1].nodes.numpy()

        # renew the fixed/non-moved points
        predicted_nodes[SimulatedData.fix_keypoint_place, :3] = input_graph_dict['nodes'][
                                                                SimulatedData.fix_keypoint_place, :3]
        predicted_nodes[nonmovedIndices[:, 0] > 0.5, :3] = input_graph_dict['nodes'][nonmovedIndices[:, 0] > 0.5,
                                                           :3]

        # Add the previous effector position back to transform the center position to global coordinate
        predicted_nodes[:, :3] += current_effector_position

        # The first entries are cloth keypoints (followed by rigid body nodes)
        cloth_keypoint_positions = predicted_nodes[:len(self.representation.keypoint_indices), :3]

        # FIXME: Rigid body prediction not yet implemented
        rigid_body_positions = None

        return PredictedFrame(cloth_keypoint_positions, rigid_body_positions)

    def predict_frame(self, frame: SimulatedData.Frame, next_effector_position: np.array) -> PredictedFrame:

        anchor_index = frame.frame_index // self.frame_step
        anchor_frame: PredictedFrame = self.anchor_frames[anchor_index]

        # If we hit an anchor frame directly, we just return it
        anchor_frame_index = anchor_index * self.frame_step
        if anchor_frame_index == frame.frame_index:
            return anchor_frame

        # Otherwise, we use the frame-wise prediction model
        return self.single_prediction_model.predict_frame(frame, next_effector_position)


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

    # model = FullyConnectedPredictionModel()
    # next_effector_position = next_frame.get_effector_pose()[0]
    # predicted = model.predict_frame(current_frame, next_effector_position)

    model_masked = FullyConnectedMaskedPredictionModel()
    next_effector_position = next_frame.get_effector_pose()[0]
    maskedpredicted = model_masked.predict_frame(current_frame, next_effector_position)
