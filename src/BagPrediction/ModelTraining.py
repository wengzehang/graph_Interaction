"""
Training a model given by a specification
"""

import random
from sklearn.utils import shuffle
from graph_nets import utils_tf

import SimulatedData
import Models
import ModelSpecification


class DataGenerator:
    def __init__(self,
                 data: SimulatedData.SimulatedData,
                 specification: ModelSpecification.ModelSpecification):
        self.data = data
        self.specification = specification

        # We can generate a sample from each adjacent frame pair
        frame_step = specification.training_params.frame_step
        self.num_samples = data.num_scenarios * (data.num_frames - frame_step)

        # We generate scenario and frame index pairs which represent the input data for training
        # Each epoch, we randomly shuffle these indices to generate a new training order
        self.indices = [(scenario_index, frame_index)
                        for scenario_index in range(0, data.num_scenarios)
                        for frame_index in range(0, data.num_frames - frame_step)]
        self.indices = shuffle(self.indices)

        # Number of generated epochs (increased when the complete dataset has been generated/returned)
        self.epoch_count = 0

        # How many samples have been generated since the last epoch reset?
        self.generated_count = 0

    def next_batch(self, training: bool = True,
                   batch_size: int = None):
        if batch_size is None:
            batch_size = self.specification.training_params.batch_size

        dataset_size = self.num_samples
        start_index = random.randint(0, dataset_size - batch_size)
        end_index = start_index + batch_size

        self.generated_count = self.generated_count + batch_size
        new_epoch = self.generated_count >= dataset_size
        if new_epoch:
            self.indices = shuffle(self.indices)
            self.epoch_count += 1

        input_dicts = [None] * batch_size
        target_dicts = [None] * batch_size

        frame_step = self.specification.training_params.frame_step

        batch_indices = self.indices[start_index:end_index]
        for i, (scenario_index, frame_index) in enumerate(batch_indices):
            scenario = self.data.scenario(scenario_index)
            current_frame = scenario.frame(frame_index)
            next_frame = scenario.frame(frame_index + frame_step)

            # input_dicts[i], target_dicts[i] = self.create_input_and_target_graph_dict(current_frame, next_frame)
            input_dicts[i], target_dicts[i] = self.create_input_and_target_graph_dict(current_frame, next_frame)

        input_graph_tuples = utils_tf.data_dicts_to_graphs_tuple(input_dicts)
        target_graph_tuples = utils_tf.data_dicts_to_graphs_tuple(target_dicts)
        return input_graph_tuples, target_graph_tuples

    def create_input_and_target_graph_dict(self,
                                           current_frame: SimulatedData.Frame,
                                           next_frame: SimulatedData.Frame):
        # TODO: Implement based on model specification
        pass


train_path_to_topodict = 'h5data/topo_train.pkl'
train_path_to_dataset = 'h5data/train_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

valid_path_to_topodict = 'h5data/topo_valid.pkl'
valid_path_to_dataset = 'h5data/valid_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

movement_threshold = 0.001
# For frame-wise prediction set frame_step to 1
# For long horizon prediction choose a value > 1
frame_step = 5

train_data = SimulatedData.SimulatedData.load(train_path_to_topodict, train_path_to_dataset)

valid_data = SimulatedData.SimulatedData.load(valid_path_to_topodict, valid_path_to_dataset)

motion_model = Models.specify_motion_model("MotionModel")
motion_net = motion_model.create_graph_net()
