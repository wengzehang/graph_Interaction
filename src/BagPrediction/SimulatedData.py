"""Loader for hdf5 data

This module contains function for loading the simulated data.
The simulated data contains ...

"""

import h5py
import pickle

from typing import List, Tuple


# Dataset keys:
MESH_KEY = 'posCloth'
RIGID_KEY = 'posRigid'
CLOTH_ID_KEY = 'clothid'
RIGID_NUM_KEY = 'numRigid'
EFFECTOR_KEY = 'posEffector'


class Frame:
    def __init__(self, data: 'SimulatedData', scenario_index: int, frame_index: int):
        assert scenario_index >= 0
        assert scenario_index < data.num_scenarios
        assert frame_index >= 0
        assert frame_index < data.num_frames

        self.data = data
        self.scenario_index = scenario_index
        self.frame_index = frame_index

    def get_cloth_keypoint_positions(self, indices):
        mesh_vertices = self.data.dataset[MESH_KEY][self.scenario_index][self.frame_index]
        return mesh_vertices[indices]


class Scenario:
    def __init__(self, data: 'SimulatedData', scenario_index: int):
        assert scenario_index >= 0
        assert scenario_index < data.num_scenarios

        self.data = data
        self.scenario_index = scenario_index

    def num_frames(self):
        return self.data.num_frames

    def frame(self, frame_index: int):
        return Frame(self.data, self.scenario_index, frame_index)


class SimulatedData:
    """This class contains the simulated data

    The topodict contains the topology description for each map.
    It is a dictionary mapping the cloth id to the faces of each cloth.
    The faces are represented as a 2D array with shape (#faces, 3) where
    each face is represented as 3 vertex indices.

    The dataset contains the simulation results:
    TODO: Explain relevant keys here
    """

    def __init__(self, dataset, topodict):
        self.dataset = dataset
        self.topodict = topodict

        # The 'posCloth' entry has shape ( #scenario_ids, #frames, #mesh_points, 4[xyz, r] )
        shape = self.dataset[MESH_KEY].shape
        self.num_scenarios = shape[0]
        self.num_frames = shape[1]
        self.num_mesh_points = shape[2]

    @staticmethod
    def load(path_to_topodict: str, path_to_dataset: str) -> 'SimulatedData':

        with open(path_to_topodict, 'rb') as pickle_file:
            topodict = pickle.load(pickle_file)

        dataset = h5py.File(path_to_dataset, 'r')
        return SimulatedData(dataset, topodict)

    def scenario(self, scenario_index: int) -> Scenario:
        return Scenario(self, scenario_index)


keypoint_indices = [
    # Front
    4, 127, 351, 380, 395, 557, 535, 550, 756, 783, 818, 1258,
    # Back
    150, 67, 420, 436, 920, 952, 1082, 1147, 1125, 1099, 929, 464,
    # Left
    142, 851, 1178,
    # Right
    49, 509, 1000,
    # Bottom
    641
]

keypoint_edges = [
    # Front edges
    (4, 351), (4, 1258),
    (351, 380), (351, 818),
    (380, 395), (380, 783),
    (395, 756),
    (127, 557), (127, 1258),
    (557, 818), (557, 535),
    (535, 783), (535, 550),
    (550, 756),
    (783, 818),
    (818, 1258),
    # Back edges
    (436, 1082), (436, 420),
    (1082, 952),
    (952, 920),
    (420, 1099), (420, 464),
    (1099, 920), (1099, 1125),
    (920, 929),
    (464, 1125), (464, 67),
    (1125, 929), (1125, 1147),
    (67, 1147),
    (150, 1147), (150, 929),
    # Left edges
    (920, 1178),
    (1178, 535), (1178, 851),
    (150, 142),
    (851, 557), (851, 142), (851, 929),
    (142, 127),
    # Right edges
    (509, 380), (509, 420), (509, 1000),
    (1000, 351), (1000, 464), (1000, 49),
    (49, 4), (49, 67),
    # Bottom edges
    (641, 127), (641, 4),
    (641, 67), (641, 150),
]


def validate_keypoint_graph(indices: List[int], edges: List[Tuple[int, int]]):
    for (e_from, e_to) in edges:
        if e_from not in indices:
            print("from:", e_from, "to:", e_to, " from is not in indices")
        if e_to not in indices:
            print("from:", e_from, "to:", e_to, " to is not in indices")
