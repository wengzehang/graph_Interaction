"""
Evaluation code for prediction models
"""

from PredictionInterface import PredictionInterface, PredictedFrame
from PredictionModels import FullyConnectedPredictionModel, FullyConnectedMaskedPredictionModel
from SimulatedData import SimulatedData, Scenario, Frame, keypoint_indices

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class EvaluationResult:
    def __init__(self,
                 keypoint_pos_error_mean: float,
                 keypoint_pos_error_stddev: float,
                 horizon_pos_error_mean: np.array):
        self.keypoint_pos_error_mean = keypoint_pos_error_mean
        self.keypoint_pos_error_stddev = keypoint_pos_error_stddev
        self.horizon_pos_error_mean = horizon_pos_error_mean


class Evaluation:
    def __init__(self, model: PredictionInterface,
                 max_scenario_index: int = 100):
        self.model = model
        self.max_scenario_index = max_scenario_index

    def evaluate_dataset(self, data: SimulatedData) -> EvaluationResult:

        keypoint_pos_error_mean, keypoint_pos_error_stddev = self.calculate_keypoint_pos_mean_error(data)
        horizon_pos_error_mean = self.calculate_horizon_pos_error(data)

        return EvaluationResult(keypoint_pos_error_mean, keypoint_pos_error_stddev, horizon_pos_error_mean)

    def calculate_keypoint_pos_mean_error(self, data: SimulatedData):
        num_scenarios = min(data.num_scenarios, self.max_scenario_index)
        errors = np.zeros(num_scenarios * (data.num_frames - 1))
        error_index = 0
        for scenario_index in range(num_scenarios):
            print("Scenario", scenario_index)
            scenario = data.scenario(scenario_index)
            next_frame = scenario.frame(0)
            for frame_index in range(data.num_frames - 1):
                current_frame = next_frame
                next_frame = scenario.frame(frame_index + 1)

                # Evaluate single frame
                next_effector_position = next_frame.get_effector_pose()[0]
                predicted_frame = self.model.predict_frame(current_frame, next_effector_position)
                errors[error_index] = self.calculate_keypoint_pos_error(predicted_frame, next_frame)

                error_index += 1

        return np.mean(errors), np.std(errors)

    def calculate_horizon_pos_error(self, data: SimulatedData):
        num_scenarios = min(data.num_scenarios, self.max_scenario_index)
        errors_per_step = np.zeros((data.num_frames - 1, num_scenarios))
        error_index = 0
        for scenario_index in range(num_scenarios):
            print("Scenario", scenario_index)
            scenario = data.scenario(scenario_index)

            current_frame = scenario.frame(0)
            next_frame = scenario.frame(1)

            next_effector_position = next_frame.get_effector_pose()[0]
            prev_predicted_frame = self.model.predict_frame(current_frame, next_effector_position)
            errors_per_step[0][scenario_index] = self.calculate_keypoint_pos_error(prev_predicted_frame, next_frame)

            for frame_index in range(1, data.num_frames - 1):
                # TODO: Calculate the current frame based on the prev_predicted frame
                current_frame = next_frame
                current_frame.overwrite_keypoint_positions(prev_predicted_frame.cloth_keypoint_positions)

                next_frame = scenario.frame(frame_index + 1)
                next_effector_position = next_frame.get_effector_pose()[0]

                # Evaluate single frame
                predicted_frame = self.model.predict_frame(current_frame, next_effector_position)
                errors_per_step[frame_index][scenario_index] = \
                    self.calculate_keypoint_pos_error(predicted_frame, next_frame)

        mean_error_per_step = np.mean(errors_per_step, axis=-1)
        return mean_error_per_step

    def calculate_keypoint_pos_error(self, predicted: PredictedFrame, ground_truth: Frame):
        pred_keypoint_pos = predicted.cloth_keypoint_positions
        gt_keypoint_pos = ground_truth.get_cloth_keypoint_positions(keypoint_indices)

        errors = np.linalg.norm(gt_keypoint_pos - pred_keypoint_pos, axis=-1)
        mean_error = np.mean(errors)
        return mean_error


if __name__ == '__main__':
    # motion_model = FullyConnectedPredictionModel()
    # model = HasMovedMaskPredictionModel(motion_model)
    model = FullyConnectedMaskedPredictionModel()

    eval = Evaluation(model)

    dataset_name = "train"
    if dataset_name == "train":
        train_path_to_topodict = 'h5data/topo_train.pkl'
        train_path_to_dataset = 'h5data/train_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'
        dataset = SimulatedData.load(train_path_to_topodict, train_path_to_dataset)
    else:
        valid_path_to_topodict = 'h5data/topo_valid.pkl'
        valid_path_to_dataset = 'h5data/valid_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'
        dataset = SimulatedData.load(valid_path_to_topodict, valid_path_to_dataset)

    result = eval.evaluate_dataset(dataset)
    print(result.keypoint_pos_error_mean, result.keypoint_pos_error_stddev)

    print("Frame-wise errors:")
    print(result.horizon_pos_error_mean)

    plt.bar(range(result.horizon_pos_error_mean.shape[0]), result.horizon_pos_error_mean)

    plt.show()
