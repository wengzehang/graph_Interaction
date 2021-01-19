"""
Graph representation of simulated data
"""


from typing import List, Tuple, Dict
import numpy as np
from graph_nets import utils_tf

import SimulatedData
import itertools


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


class GraphRepresentation_rigid_deformable:
    """
    Converts frames of simulated data into graph_dicts which are compatible with TensorFlow and Graph Nets
    """
    def __init__(self, keypoint_indices: List[int], keypoint_edges: List[Tuple[int, int]]):
        self.keypoint_indices = keypoint_indices
        self.keypoint_node_indices = np.arange(len(self.keypoint_indices))
        self.fix_keypoint_indices = SimulatedData.fix_keypoint_indices
        self.keypoint_edges = keypoint_edges
        # Convert the edges to local indices in the range (0, len(keypoint_indices))
        self.keypoint_edges_from = np.array([keypoint_indices.index(f) for (f, _) in keypoint_edges])
        self.keypoint_edges_to = np.array([keypoint_indices.index(t) for (_, t) in keypoint_edges])
        self.keypoint_edges_indices = [(keypoint_indices.index(f), keypoint_indices.index(t))
                                       for (f, t) in keypoint_edges]
        self.num_allpoints = 0
        self.combineindices_1 = None
        self.combineindices_2 = None

        self.count_cloth_keypoint = len(self.keypoint_indices)

    def to_graph_dict(self, frame: SimulatedData.Frame) -> Dict[str, List]:
        # TODO: take all the rigid objects into account when constructs the input graph

        # We use the positions as nodes
        positions_soft = frame.get_cloth_keypoint_positions(self.keypoint_indices)
        info_soft = np.float32(frame.get_cloth_keypoint_info(self.keypoint_indices))
        info_rigid = np.float32(frame.get_rigid_keypoint_info())

        # We use the positions and radius of effector as global features
        effector_positions = frame.get_effector_pose()
        # TensorFlow expects float32 values, the dataset contains float64 values
        info_all = np.vstack((info_soft, info_rigid))
        num_allpoints = info_all.shape[0]
        positions = info_all[:,:3]

        # create new edge connection inside and between rigid and soft object particles
        # TODO: Create the new edge list, from - to, two ways:
        #  0. fully connected inside rigid graph, non-fully connected in soft graph, connect somewhere (tick
        #  0.1 fully connected inside rigid graph, non-fully connected in soft graph, all connected between two graph nodes (tick
        #  1. fully connected in each part, and connect two parts somewhere at a node
        #  2. all connected

        edge_index = [i for i in itertools.product(np.arange(num_allpoints), repeat=2)]
        # all connected, bidirectional
        self.keypoint_edges_from, self.keypoint_edges_to = list(zip(*edge_index))
        self.keypoint_edges_from = list(self.keypoint_edges_from)
        self.keypoint_edges_to = list(self.keypoint_edges_to)

        # The distance between adjacent nodes are the edges
        distances = positions[self.keypoint_edges_to] - positions[self.keypoint_edges_from]



        return {
            "globals": [0.0],  # TODO: Fill global field with action parameter
            "nodes": info_all, # positions
            "edges": distances,
            "senders": self.keypoint_edges_from,
            "receivers": self.keypoint_edges_to,
        }

    def to_graph_dict_global_4_align(self, frame_current: SimulatedData.Frame, effector_positions_future: np.float32([0.0, 0.0, 0.0]), origin: np.float32([0.0, 0.0, 0.0]), add_noise=False) -> Dict[str, List]:
        # Function: Convert the frames into fully connected graphs, which are aligned by the current effector position
        #  nodes (5): [x, y, z, radius, freeflag]
        #  edge (4): [dx, dy, dz, physicalflag=0/1]
        #  global (4): [gx, gy, gz, gradius]

        # xxx: Construct the nodes feature vectors
        # Get soft object nodes features
        info_soft = np.float32(frame_current.get_cloth_keypoint_info(self.keypoint_indices, self.fix_keypoint_indices))
        # Get rigid object nodes features
        info_rigid = np.float32(frame_current.get_rigid_keypoint_info())
        info_all = np.vstack((info_soft, info_rigid))
        num_allpoints = info_all.shape[0]

        # We use the current and future positions and radius of effector as global features
        effector_positions_current = frame_current.get_effector_pose()
        # TensorFlow expects float32 values, the dataset contains float64 values
        effector_positions_current = np.float32(effector_positions_current).reshape(4)
        effector_positions_future = np.float32(effector_positions_future).reshape(4)


        # xxx: Construct the global feature vector
        # xxx: Align/move the node and global features to origin
        global_feat = np.float32(effector_positions_future)
        if origin is None:
            # Move the scene to origin
            info_all[:,:3] -= effector_positions_current[:3]
            # Move global feat to origin, radius not affected
            global_feat[:3] -=  effector_positions_current[:3]
        else:
            # Let the target frame use the previous frame's origin
            # Move the scene to origin
            info_all[:, :3] -= origin
            # Move global feat to origin, radius not affected
            global_feat[:3] -=  origin

        # xxx: Construct the edge features
        positions = info_all[:, :3]
        edge_index = [i for i in itertools.product(np.arange(num_allpoints), repeat=2)]

        if add_noise:
            noise = np.random.normal([0.0,0.0,0.0],0.002, positions.shape)
            positions = positions + noise

        # all connected, bidirectional
        self.keypoint_edges_to_ALL, self.keypoint_edges_from_ALL = list(zip(*edge_index))

        # The distance between adjacent nodes are the edges
        self.keypoint_edges_from_ALL = list(self.keypoint_edges_from_ALL)
        self.keypoint_edges_to_ALL = list(self.keypoint_edges_to_ALL)

        distances = np.float32(np.zeros((len(self.keypoint_edges_from_ALL),4))) # DISTANCE 3D, CONNECTION TYPE 1.
        distances[:,:3] = positions[self.keypoint_edges_to_ALL] - positions[self.keypoint_edges_from_ALL]
        combineindices = self.keypoint_edges_to*num_allpoints+self.keypoint_edges_from
        distances[combineindices,3] = 1 # denote the physical connection
        # # bidirectional
        combineindices = self.keypoint_edges_from*num_allpoints+self.keypoint_edges_to
        # distances[combineindices,3] = 1 # denote the physical connection
        distances[combineindices,3] = 1 # denote the physical connection

        return {
            "globals": global_feat,  # TODO: Fill global field with action parameter
            "nodes": info_all, #info_all,# info_all, # positions,
            "edges": distances,
            "senders": self.keypoint_edges_from_ALL, #self.keypoint_edges_from,
            "receivers":self.keypoint_edges_to_ALL, # self.keypoint_edges_to,
        }

    def to_graph_dict_global_align_type1(self, frame_current: SimulatedData.Frame,
                                     effector_positions_future: np.float32([0.0, 0.0, 0.0]),
                                     origin: np.float32([0.0, 0.0, 0.0])) -> Dict[str, List]:
        # Function: Convert the frames into partially connected graphs, which are aligned by the current effector position
        #  nodes (5): [x, y, z, radius, freeflag]
        #  edge (4): [dx, dy, dz, physicalflag=-1/0/1]
        #  global (4): [gx, gy, gz, gradius]

        # xxx: Construct the nodes feature vectors
        # Get soft object nodes features
        info_soft = np.float32(
            frame_current.get_cloth_keypoint_info_partgraph(self.keypoint_indices, self.fix_keypoint_indices))
        # Get rigid object nodes features
        info_rigid = np.float32(frame_current.get_rigid_keypoint_info())
        info_all = np.vstack((info_soft, info_rigid))

        # We use the current and future positions and radius of effector as global features
        effector_positions_current = frame_current.get_effector_pose()
        # TensorFlow expects float32 values, the dataset contains float64 values
        effector_positions_current = np.float32(effector_positions_current).reshape(4)
        effector_positions_future = np.float32(effector_positions_future).reshape(4)

        # xxx: Construct the global feature vector
        global_feat = np.float32(effector_positions_future)

        # xxx: Align/move the node and global features to origin
        if origin is None:
            # Move the scene to origin
            info_all[:, :3] -= effector_positions_current[:3]
            # Move global feat to origin, radius not affected
            global_feat[:3] -= effector_positions_current[:3]
        else:
            # Let the target frame use the previous frame's origin
            # Move the scene to origin
            info_all[:, :3] -= origin
            # Move global feat to origin, radius not affected
            global_feat[:3] -= origin


        # xxx: Construct the edge features
        positions = info_all[:, :3]

        # cloth edges, edge 1
        distances_back = positions[self.keypoint_edges_to] - positions[self.keypoint_edges_from]
        distances_forth = positions[self.keypoint_edges_from] - positions[self.keypoint_edges_to]
        num_soft_edge = 2*distances_back.shape[0]
        # hidden edge, edge 2
        distance_hidden_back = positions[self.keypoint_node_indices] - positions[-1,:3]
        distance_hidden_forth = positions[-1,:3] - positions[self.keypoint_node_indices]
        num_hide_edge = 2*distance_hidden_back.shape[0]

        num_soft_node = info_soft.shape[0] - 1
        num_rigid_node = info_rigid.shape[0]
        num_rigid_h_node = num_rigid_node + 1
        # higher object edges, edge 3
        edge_index_rigid_h = [i for i in itertools.product(np.arange(num_soft_node,num_soft_node+num_rigid_h_node), repeat=2)]

        # all connected, bidirectional cloth edges
        self.keypoint_edges_to_ALL_rigid, self.keypoint_edges_from_ALL_rigid = list(zip(*edge_index_rigid_h))
        self.keypoint_edges_from_ALL_rigid = list(self.keypoint_edges_from_ALL_rigid)
        self.keypoint_edges_to_ALL_rigid = list(self.keypoint_edges_to_ALL_rigid)
        distance_rigid = positions[self.keypoint_edges_to_ALL_rigid] - positions[self.keypoint_edges_from_ALL_rigid]
        num_rigid_edge = len(self.keypoint_edges_from_ALL_rigid)

        # stack the distance matrices
        distances = np.float32(np.zeros((num_soft_edge+num_hide_edge+num_rigid_edge, 4)))  # DISTANCE 3D, CONNECTION TYPE 1.
        distances[:num_soft_edge, :3] = np.vstack((distances_back, distances_forth))
        distances[num_soft_edge:num_soft_edge+num_hide_edge, :3] = np.vstack((distance_hidden_back, distance_hidden_forth))
        distances[num_soft_edge+num_hide_edge:, :3] = distance_rigid

        # assign the edge flags
        distances[:num_soft_edge, 3] = -1  # denote the physical connection
        distances[num_soft_edge:num_soft_edge+num_hide_edge, 3] = 0  # denote the physical connection
        distances[num_soft_edge:num_soft_edge+num_hide_edge, 3] = 1  # denote the physical connection

        # xxx: receiver and sender indices
        receiver_indices = np.hstack((self.keypoint_edges_to, self.keypoint_edges_from, self.keypoint_node_indices,
                                    np.repeat(num_soft_node, num_hide_edge/2), self.keypoint_edges_to_ALL_rigid))
        sender_indices = np.hstack((self.keypoint_edges_from, self.keypoint_edges_to,
                                      np.repeat(num_soft_node, num_hide_edge/2), self.keypoint_node_indices,
                                    self.keypoint_edges_from_ALL_rigid))

        # a = positions[receiver_indices] - positions[sender_indices]

        return {
            "globals": global_feat,  # TODO: Fill global field with action parameter
            "nodes": info_all,  # info_all,# info_all, # positions,
            "edges": distances,
            "senders": sender_indices,  # self.keypoint_edges_from,
            "receivers": receiver_indices,  # self.keypoint_edges_to,
        }

    def to_graph_dict_global_align_type2(self,
                                         frame_current: SimulatedData.Frame,
                                         effector_positions_future: np.float32([0.0, 0.0, 0.0]),
                                         origin: np.float32([0.0, 0.0, 0.0]),
                                         add_noise=False) -> Dict[str, List]:
        # Function: Convert the frames into fully connected graphs, which are aligned by the current effector position
        #  nodes (5): [x, y, z, radius, freeflag]
        #  edge (7): [dx, dy, dz, dx', dy', dz', physicalflag=0/1]
        #  global (4): [gx, gy, gz, gradius]

        # xxx: Construct the nodes feature vectors
        # Get soft object nodes features
        info_soft = np.float32(frame_current.get_cloth_keypoint_info(self.keypoint_indices, self.fix_keypoint_indices))
        # Get rigid object nodes features
        info_rigid = np.float32(frame_current.get_rigid_keypoint_info())
        info_all = np.vstack((info_soft, info_rigid))
        self.num_allpoints = info_all.shape[0]

        # We use the current and future positions and radius of effector as global features
        effector_positions_current = frame_current.get_effector_pose()
        # TensorFlow expects float32 values, the dataset contains float64 values
        effector_positions_current = np.float32(effector_positions_current).reshape(4)
        effector_positions_future = np.float32(effector_positions_future).reshape(4)

        # xxx: Construct the global feature vector
        global_feat = np.float32(effector_positions_future)

        # xxx: Align/move the node and global features to origin
        if origin is None:
            # Move the scene to origin
            info_all[:,:3] -= effector_positions_current[:3]
            # Move global feat to origin, radius not affected
            global_feat[:3] -=  effector_positions_current[:3]
        else:
            # Let the target frame use the previous frame's origin
            # Move the scene to origin
            info_all[:, :3] -= origin
            # Move global feat to origin, radius not affected
            global_feat[:3] -=  origin

        # xxx: Construct the edge features
        #  assign physical connection with flag=1
        positions = info_all[:, :3]

        edge_index = [i for i in itertools.product(np.arange(self.num_allpoints), repeat=2)]
        if add_noise:
            noise = np.random.normal([0.0,0.0,0.0],0.002, positions.shape)
            positions = positions + noise

        # all connected, bidirectional
        self.keypoint_edges_to_ALL, self.keypoint_edges_from_ALL = list(zip(*edge_index))

        self.keypoint_edges_from_ALL = list(self.keypoint_edges_from_ALL)
        self.keypoint_edges_to_ALL = list(self.keypoint_edges_to_ALL)

        distances = np.float32(np.zeros((len(self.keypoint_edges_from_ALL),7))) # DISTANCE 3D, CONNECTION TYPE 1.
        distances[:,:3] = positions[self.keypoint_edges_to_ALL] - positions[self.keypoint_edges_from_ALL]
        self.combineindices_1 = self.keypoint_edges_to*self.num_allpoints+self.keypoint_edges_from
        distances[self.combineindices_1, 3:6] = distances[self.combineindices_1, :3]  # denote the physical connection
        distances[self.combineindices_1,6] = 1 # denote the physical connection
        # # bidirectional
        self.combineindices_2 = self.keypoint_edges_from*self.num_allpoints+self.keypoint_edges_to
        distances[self.combineindices_2, 3:6] = distances[self.combineindices_2, :3]  # denote the physical connection
        distances[self.combineindices_2,6] = 1 # denote the physical connection

        return {
            "globals": global_feat,  # TODO: Fill global field with action parameter
            "nodes": info_all, #info_all,# info_all, # positions,
            "edges": distances,
            "senders": self.keypoint_edges_from_ALL, #self.keypoint_edges_from,
            "receivers":self.keypoint_edges_to_ALL, # self.keypoint_edges_to,
        }

if __name__ == '__main__':
    path_to_topodict = 'h5data/topo_train.pkl'
    path_to_dataset = 'h5data/train_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'
    data = SimulatedData.SimulatedData.load(path_to_topodict, path_to_dataset)

    representation = GraphRepresentation_rigid_deformable(SimulatedData.keypoint_indices, SimulatedData.keypoint_edges)

    scenario = data.scenario(3)
    frame = scenario.frame(0)

    frame_target = scenario.frame(1)

    graph_dict = representation.to_graph_dict(frame)

    # Test the conversion to tf.Tensor
    input_graphs = utils_tf.data_dicts_to_graphs_tuple([graph_dict])
    print(input_graphs)
