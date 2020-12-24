"""
Graph representation of simulated data
"""


from typing import List, Tuple, Dict
import numpy as np
from graph_nets import utils_tf

import SimulatedData


class GraphRepresentation:
    """
    Converts frames of simulated data into graph_dicts which are compatible with TensorFlow and Graph Nets
    """
    def __init__(self, keypoint_indices: List[int], keypoint_edges: List[Tuple[int, int]]):
        self.keypoint_indices = keypoint_indices
        self.keypoint_edges = keypoint_edges
        # Convert the edges to local indices in the range (0, len(keypoint_indices))
        self.keypoint_edges_from = np.array([keypoint_indices.index(f) for (f, _) in keypoint_edges])
        self.keypoint_edges_to = np.array([keypoint_indices.index(t) for (_, t) in keypoint_edges])
        self.keypoint_edges_indices = [(keypoint_indices.index(f), keypoint_indices.index(t))
                                       for (f, t) in keypoint_edges]

    def to_graph_dict(self, frame: SimulatedData.Frame) -> Dict[str, List]:
        # We use the positions as nodes
        positions = frame.get_cloth_keypoint_positions(self.keypoint_indices)
        # We use the positions and radius of effector as global features
        effector_positions = frame.get_effector_pose()
        # TensorFlow expects float32 values, the dataset contains float64 values
        positions = np.float32(positions)
        # The distance between adjacent nodes are the edges
        distances = positions[self.keypoint_edges_to] - positions[self.keypoint_edges_from]

        return {
            "globals": [0.0],  # TODO: Fill global field with action parameter
            "nodes": positions,
            "edges": distances,
            "senders": self.keypoint_edges_from,
            "receivers": self.keypoint_edges_to,
        }

    def to_graph_dict_global_4(self, frame: SimulatedData.Frame) -> Dict[str, List]:
        # Non-zero global feature length: 4
        # We use the positions as nodes
        positions = frame.get_cloth_keypoint_positions(self.keypoint_indices)
        # We use the positions and radius of effector as global features
        effector_positions = frame.get_effector_pose()
        # TensorFlow expects float32 values, the dataset contains float64 values
        positions = np.float32(positions)
        effector_positions = np.float32(effector_positions).reshape(4)

        # The distance between adjacent nodes are the edges
        distances = positions[self.keypoint_edges_to] - positions[self.keypoint_edges_from]

        return {
            "globals": effector_positions,  # TODO: Fill global field with action parameter
            "nodes": positions,
            "edges": distances,
            "senders": self.keypoint_edges_from,
            "receivers": self.keypoint_edges_to,
        }

    def to_graph_dict_global_7(self, frame_current: SimulatedData.Frame, effector_positions_future: np.float32([0.0, 0.0, 0.0])) -> Dict[str, List]:
        # Non-zero global feature length: 7
        # We use the positions as nodes
        positions = frame_current.get_cloth_keypoint_positions(self.keypoint_indices)
        # We use the current and future positions and radius of effector as global features
        effector_positions_current = frame_current.get_effector_pose()
        # TensorFlow expects float32 values, the dataset contains float64 values
        positions = np.float32(positions)
        effector_positions_current = np.float32(effector_positions_current).reshape(4)
        effector_positions_future = np.float32(effector_positions_future).reshape(4)

        global_feat = np.float32(np.zeros(7))

        global_feat[:3] = effector_positions_current[:3]
        global_feat[3:6] = effector_positions_future[:3]
        global_feat[6] = effector_positions_future[3] # radius

        # The distance between adjacent nodes are the edges
        distances = positions[self.keypoint_edges_to] - positions[self.keypoint_edges_from]

        return {
            "globals": global_feat,  # TODO: Fill global field with action parameter
            "nodes": positions,
            "edges": distances,
            "senders": self.keypoint_edges_from,
            "receivers": self.keypoint_edges_to,
        }

    def to_graph_dict_global_4_align(self, frame_current: SimulatedData.Frame, effector_positions_future: np.float32([0.0, 0.0, 0.0]), origin: np.float32([0.0, 0.0, 0.0])) -> Dict[str, List]:
        # Non-zero global feature length: 7
        # We use the positions as nodes
        positions = frame_current.get_cloth_keypoint_positions(self.keypoint_indices)
        # We use the current and future positions and radius of effector as global features
        effector_positions_current = frame_current.get_effector_pose()
        # TensorFlow expects float32 values, the dataset contains float64 values
        positions = np.float32(positions)
        effector_positions_current = np.float32(effector_positions_current).reshape(4)
        effector_positions_future = np.float32(effector_positions_future).reshape(4)

        global_feat = np.float32(effector_positions_future)

        # TODO: Align, move to origin
        if origin is None:
            # Move cloth mesh to origin
            positions -= effector_positions_current[:3]
            # Move global feat to origin, radius not affected
            global_feat[:3] -=  effector_positions_current[:3]
        else:
            # Let the target frame use the previous frame's origin
            # Move cloth mesh to origin
            positions -= origin
            # Move global feat to origin, radius not affected
            global_feat[:3] -=  origin

        # The distance between adjacent nodes are the edges
        distances = positions[self.keypoint_edges_to] - positions[self.keypoint_edges_from]

        return {
            "globals": global_feat,  # TODO: Fill global field with action parameter
            "nodes": positions,
            "edges": distances,
            "senders": self.keypoint_edges_from,
            "receivers": self.keypoint_edges_to,
        }


if __name__ == '__main__':
    path_to_topodict = 'h5data/topo_train.pkl'
    path_to_dataset = 'h5data/train_sphere_sphere_f_f_soft_out_scene1.h5'
    data = SimulatedData.SimulatedData.load(path_to_topodict, path_to_dataset)

    representation = GraphRepresentation(SimulatedData.keypoint_indices, SimulatedData.keypoint_edges)

    scenario = data.scenario(3)
    frame = scenario.frame(0)

    frame_target = scenario.frame(1)

    graph_dict = representation.to_graph_dict(frame)

    # Test the conversion to tf.Tensor
    input_graphs = utils_tf.data_dicts_to_graphs_tuple([graph_dict])
    print(input_graphs)
