"""
Generate batches of graph tuples from the simulated data
"""

import random
from typing import Tuple
from sklearn.utils import shuffle
from graph_nets import utils_tf

import SimulatedData
import GraphRepresentation


class DataGenerator:
    def __init__(self, data: SimulatedData.SimulatedData):
        self.data = data

        # We can generate a sample from each adjacent frame pair
        self.num_samples = data.num_scenarios * (data.num_frames - 1)

        self.representation = GraphRepresentation.GraphRepresentation(SimulatedData.keypoint_indices,
                                                                      SimulatedData.keypoint_edges)
        self.indices = [(scenario_index, frame_index)
                        for scenario_index in range(0, data.num_scenarios)
                        for frame_index in range(0, data.num_frames - 1)]
        self.indices = shuffle(range(self.num_samples))

        self.generated_count = 0

    def next_batch(self, batch_size: int) -> Tuple:
        dataset_size = self.num_samples
        start_index = random.randint(0, dataset_size - batch_size)
        end_index = start_index + batch_size

        self.generated_count = self.generated_count + batch_size
        if self.generated_count > 2 * dataset_size:
            print("Reshuffling")
            self.indices = shuffle(self.indices)
            self.generated_count = 0

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

    def create_input_and_target_graph_dict(self,
                                           current_frame: SimulatedData.Frame,
                                           next_frame: SimulatedData.Frame):

        # This method should be overridden by derived classes to define different attribute structures
        # TODO: Do we need access to the previous frame?
        current_graph = self.representation.to_graph_dict(current_frame)
        next_graph = self.representation.to_graph_dict(next_frame)
        return current_graph, next_graph

