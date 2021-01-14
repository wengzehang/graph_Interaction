"""
Evaluation code for prediction models
"""

from PredictionInterface import PredictionInterface, PredictedFrame
from PredictionModels import FullyConnectedPredictionModel
from SimulatedData import SimulatedData, Scenario, Frame, keypoint_indices

import numpy as np


class EvaluationResult:
    def __init__(self,
                 keypoint_pos_error_mean: float,
                 keypoint_pos_error_stddev: float):
        self.keypoint_pos_error_mean = keypoint_pos_error_mean
        self.keypoint_pos_error_stddev = keypoint_pos_error_stddev


class Evaluation:
    def __init__(self, model: PredictionInterface,
                 max_scenario_index: int = 10):
        self.model = model
        self.max_scenario_index = max_scenario_index

    def evaluate_dataset(self, data: SimulatedData) -> EvaluationResult:

        errors = np.zeros(data.num_scenarios * (data.num_frames - 1))
        error_index = 0
        for scenario_index in range(min(data.num_scenarios, self.max_scenario_index)):
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

        keypoint_pos_error_mean = np.mean(errors)
        keypoint_pos_error_stddev = np.std(errors)
        return EvaluationResult(keypoint_pos_error_mean, keypoint_pos_error_stddev)

    def calculate_keypoint_pos_error(self, predicted: PredictedFrame, ground_truth: Frame):
        pred_keypoint_pos = predicted.cloth_keypoint_positions
        gt_keypoint_pos = ground_truth.get_cloth_keypoint_positions(keypoint_indices)

        error = np.linalg.norm(gt_keypoint_pos - pred_keypoint_pos)
        return error


if __name__ == '__main__':
    model = FullyConnectedPredictionModel()
    eval = Evaluation(model)

    dataset_name = "valid"
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
