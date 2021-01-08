"""
Generate batches of graph tuples from the simulated data
"""

import random
from typing import Tuple
from sklearn.utils import shuffle
from graph_nets import utils_tf
import numpy as np
import itertools

import SimulatedData
import GraphRepresentation


class DataGeneratorBase:
    def __init__(self, data: SimulatedData.SimulatedData):
        self.data = data

        # We can generate a sample from each adjacent frame pair
        self.num_samples = data.num_scenarios * (data.num_frames - 1)

        self.representation = GraphRepresentation.GraphRepresentation_rigid_deformable(SimulatedData.keypoint_indices,
                                                                                       SimulatedData.keypoint_edges)

        self.indices = [(scenario_index, frame_index)
                        for scenario_index in range(0, data.num_scenarios)
                        for frame_index in range(0, data.num_frames - 1)]

        self.generated_count = 0
        self.has_reshuffled = False

    def next_batch(self, batch_size: int) -> Tuple:
        dataset_size = self.num_samples
        start_index = random.randint(0, dataset_size - batch_size)
        end_index = start_index + batch_size

        self.generated_count = self.generated_count + batch_size
        if self.generated_count > 2 * dataset_size:
            print("Reshuffling")
            self.indices = shuffle(self.indices)
            self.generated_count = 0
            self.has_reshuffled = True
        else:
            self.has_reshuffled = False

        input_dicts = [None] * batch_size
        target_dicts = [None] * batch_size

        batch_indices = self.indices[start_index:end_index]
        for i, (scenario_index, frame_index) in enumerate(batch_indices):
            scenario = self.data.scenario(scenario_index)
            current_frame = scenario.frame(frame_index)
            next_frame = scenario.frame(frame_index + 1)

            input_dicts[i], target_dicts[i] = self.create_input_and_target_graph_dict(current_frame, next_frame)

        input_graph_tuples = utils_tf.data_dicts_to_graphs_tuple(input_dicts)
        target_graph_tuples = utils_tf.data_dicts_to_graphs_tuple(target_dicts)
        return input_graph_tuples, target_graph_tuples


class DataGenerator(DataGeneratorBase):
    def __init__(self, data: SimulatedData.SimulatedData):
        super().__init__(self, data)

    def create_input_and_target_graph_dict(self,
                                           current_frame: SimulatedData.Frame,
                                           next_frame: SimulatedData.Frame):
        # This method should be overridden by derived classes to define different attribute structures
        # TODO: Do we need access to the previous frame?
        # current_graph = self.representation.to_graph_dict(current_frame)
        # next_graph = self.representation.to_graph_dict(next_frame)

        # FIXME: Modify the node attributes, for the graph global features, we use the current and future effector POSE, with the radius
        # FIXME: For the target graph global features, we use the moving direction to calculate a pseudo effector position
        current_frame_effector = current_frame.get_effector_pose().reshape(4)
        next_frame_effector = next_frame.get_effector_pose().reshape(4)

        # try:
        potential_future_frame_effector = next_frame_effector * 2 - current_frame_effector  # be careful about the radius
        # except RuntimeWarning:
        #     import pdb;
        #     pdb.set_trace()

        potential_future_frame_effector[3] = next_frame_effector[3]
        # current_graph = self.representation.to_graph_dict_global_7(current_frame, next_frame_effector)
        # next_graph = self.representation.to_graph_dict_global_7(next_frame, potential_future_frame_effector)
        current_graph = self.representation.to_graph_dict_global_4_align(current_frame,
                                                                         next_frame_effector,
                                                                         current_frame_effector[:3],
                                                                         add_noise=True)
        next_graph = self.representation.to_graph_dict_global_4_align(next_frame, potential_future_frame_effector,
                                                                      current_frame_effector[:3])
        return current_graph, next_graph


class DataGeneratorHasMoved(DataGeneratorBase):
    def __init__(self, data: SimulatedData.SimulatedData,
                 movement_threshold=0.001,
                 noise_stddev=0.002):
        super().__init__(data)

        self.movement_threshold = movement_threshold
        self.noise_stddev = noise_stddev

    def create_input_and_target_graph_dict(self,
                                           current_frame: SimulatedData.Frame,
                                           next_frame: SimulatedData.Frame):
        # Non-zero global feature length: 7
        # We use the positions as nodes

        # Soft object node features
        info_soft = np.float32(current_frame.get_cloth_keypoint_info(self.representation.keypoint_indices,
                                                                     self.representation.fix_keypoint_indices))
        # Rigid object node features
        info_rigid = np.float32(current_frame.get_rigid_keypoint_info())
        info_all = np.vstack((info_soft, info_rigid))
        num_allpoints = info_all.shape[0]
        num_addedges = num_allpoints * num_allpoints

        # Soft object node features
        info_soft_next = np.float32(
            next_frame.get_cloth_keypoint_info(self.representation.keypoint_indices,
                                               self.representation.fix_keypoint_indices))
        # Rigid object node features
        info_rigid_next = np.float32(next_frame.get_rigid_keypoint_info())
        info_all_next = np.vstack((info_soft_next, info_rigid_next))

        # We use the current and future positions and radius of effector as global features
        effector_xyzr_current = current_frame.get_effector_pose()
        effector_xyzr_next = next_frame.get_effector_pose()
        # TensorFlow expects float32 values, the dataset contains float64 values
        effector_xyzr_current = np.float32(effector_xyzr_current).reshape(4)
        effector_xyzr_next = np.float32(effector_xyzr_next).reshape(4)
        effector_position_diff = effector_xyzr_next[:3] - effector_xyzr_current[:3]
        effector_radius = effector_xyzr_current[3]

        global_features = np.zeros(4, np.float32)
        global_features[:3] = effector_position_diff
        global_features[3] = effector_radius

        # See whether the nodes have moved or not
        positions_current = info_all[:, :3]
        positions_next = info_all_next[:, :3]
        positions_diff = np.linalg.norm(positions_next - positions_current, axis=-1).reshape(-1, 1)
        has_moved = (positions_diff > self.movement_threshold).astype(np.float32)
        # The has_moved label is -1.0 if the node did not move and 1.0 if it moved
        has_moved_label = 2.0 * has_moved - 1.0

        # Subtract the current effector position from all node positions
        info_all[:, :3] -= effector_xyzr_current[:3]

        positions = info_all[:, :3]
        if self.noise_stddev is not None:
            noise = np.random.normal([0.0, 0.0, 0.0], self.noise_stddev, positions.shape)
            positions = positions + noise

        edge_index = [i for i in itertools.product(np.arange(num_allpoints), repeat=2)]
        # all connected, bidirectional
        keypoint_edges_to_ALL, keypoint_edges_from_ALL = list(zip(*edge_index))

        # The distance between adjacent nodes are the edges
        keypoint_edges_from_ALL = list(keypoint_edges_from_ALL)
        keypoint_edges_to_ALL = list(keypoint_edges_to_ALL)

        distances = np.float32(np.zeros((len(keypoint_edges_from_ALL), 4)))  # DISTANCE 3D, CONNECTION TYPE 1.
        distances[:, :3] = positions[keypoint_edges_to_ALL] - positions[keypoint_edges_from_ALL]
        combineindices = self.representation.keypoint_edges_to * num_allpoints + self.representation.keypoint_edges_from
        distances[combineindices, 3] = 1  # denote the physical connection

        input_graph_dict = {
            "globals": global_features,  # TODO: Fill global field with action parameter
            "nodes": info_all,  # info_all,# info_all, # positions,
            "edges": distances,
            "senders": keypoint_edges_from_ALL,
            "receivers": keypoint_edges_to_ALL,
        }

        output_graph_dict = {
            "globals": global_features,  # TODO: Fill global field with action parameter
            "nodes": has_moved_label,
            "edges": distances,
            "senders": keypoint_edges_from_ALL,
            "receivers": keypoint_edges_to_ALL,
        }

        return input_graph_dict, output_graph_dict
