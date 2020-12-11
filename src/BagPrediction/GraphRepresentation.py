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


if __name__ == '__main__':
    path_to_topodict = 'h5data/topo_train.pkl'
    path_to_dataset = 'h5data/train_sphere_sphere_f_f_soft_out_scene1.h5'
    data = SimulatedData.SimulatedData.load(path_to_topodict, path_to_dataset)

    representation = GraphRepresentation(SimulatedData.keypoint_indices, SimulatedData.keypoint_edges)

    scenario = data.scenario(3)
    frame = scenario.frame(0)

    graph_dict = representation.to_graph_dict(frame)

    # Test the conversion to tf.Tensor
    input_graphs = utils_tf.data_dicts_to_graphs_tuple([graph_dict])
    print(input_graphs)
