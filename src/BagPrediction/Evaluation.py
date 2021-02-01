"""
Evaluation code for prediction models
"""

from PredictionInterface import PredictionInterface, PredictedFrame
from PredictionModels import *
from NewPredictionModels import *
from SimulatedData import SimulatedData, Scenario, Frame, keypoint_indices

import matplotlib.pyplot as plt
import numpy as np
import argparse
import tqdm
import csv


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
        print("Evaluating statistics about position error")
        for scenario_index in tqdm.tqdm(range(num_scenarios)):
            scenario = data.scenario(scenario_index)
            self.model.prepare_scenario(scenario)

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

        print("Evaluating horizon position error")
        for scenario_index in tqdm.tqdm(range(num_scenarios)):
            scenario = data.scenario(scenario_index)
            self.model.prepare_scenario(scenario)

            current_frame = scenario.frame(0)
            next_frame = scenario.frame(1)

            next_effector_position = next_frame.get_effector_pose()[0]
            prev_predicted_frame = self.model.predict_frame(current_frame, next_effector_position)
            errors_per_step[0][scenario_index] = self.calculate_keypoint_pos_error(prev_predicted_frame, next_frame)

            for frame_index in range(1, data.num_frames - 1):
                current_frame = next_frame
                current_frame.overwrite_keypoint_positions(prev_predicted_frame.cloth_keypoint_positions)
                current_frame.overwrite_rigid_body_positions(prev_predicted_frame.rigid_body_positions)

                next_frame = scenario.frame(frame_index + 1)
                next_effector_position = next_frame.get_effector_pose()[0]

                # Evaluate single frame
                predicted_frame = self.model.predict_frame(current_frame, next_effector_position)
                errors_per_step[frame_index][scenario_index] = \
                    self.calculate_keypoint_pos_error(predicted_frame, next_frame)

                prev_predicted_frame = predicted_frame

        mean_error_per_step = np.mean(errors_per_step, axis=-1)
        return mean_error_per_step

    def calculate_keypoint_pos_error(self, predicted: PredictedFrame, ground_truth: Frame):
        pred_keypoint_pos = predicted.cloth_keypoint_positions
        gt_keypoint_pos = ground_truth.get_cloth_keypoint_positions(keypoint_indices)

        errors = np.linalg.norm(gt_keypoint_pos - pred_keypoint_pos, axis=-1)
        mean_error = np.mean(errors)
        return mean_error


def create_prediction_model(model_name: str):
    motion_model_1_spec = ModelSpecification.ModelSpecification(name="MotionModel_1")
    motion_model_1 = MotionModelFromSpecification(motion_model_1_spec)
    if model_name == "one-stage":
        return motion_model_1

    has_moved_model_1_spec = ModelSpecification.ModelSpecification(name="HasMovedModel_1")
    mask_model_1 = HasMovedMaskModelFromSpecification(motion_model_1,
                                                      has_moved_model_1_spec)
    if model_name == "two-stage":
        return mask_model_1

    motion_model_5_spec = ModelSpecification.ModelSpecification(name="MotionModel_5")
    motion_model_5 = MotionModelFromSpecification(motion_model_5_spec)
    has_moved_model_5_spec = ModelSpecification.ModelSpecification(name="HasMovedModel_5")
    mask_model_5 = HasMovedMaskModelFromSpecification(motion_model_5,
                                                      has_moved_model_5_spec)

    horizon_model = HorizonModel(mask_model_1, mask_model_5,
                                 start_horizon_frame=20)

    if model_name == "horizon":
        return horizon_model

    raise NotImplementedError("Model name was not handled in create_prediction_model()", model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a prediction model for deformable bag manipulation')
    parser.add_argument('--model', help='Specify the model name: one-stage, two-stage, horizon',
                        default='one-stage')
    parser.add_argument('--max_scenarios', type=int, default=None)
    parser.add_argument('--set_name', type=str, default="train")
    # TODO: Add parameters to choose a different dataset

    args, _ = parser.parse_known_args()

    set_name = args.set_name
    path_to_topodict = 'h5data/topo_%s.pkl' % set_name
    path_to_dataset = 'h5data/%s_sphere_sphere_f_f_soft_out_scene1_2TO5.h5' % set_name
    dataset = SimulatedData.load(path_to_topodict, path_to_dataset)

    max_scenarios = args.max_scenarios
    if max_scenarios is None:
        max_scenarios = dataset.num_scenarios

    model_name = args.model
    model = create_prediction_model(model_name)

    evaluation = Evaluation(model, max_scenario_index=max_scenarios)

    result = evaluation.evaluate_dataset(dataset)

    filename = f"eval_error_{set_name}_{model_name}.csv"
    with open(filename, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # TODO: Also output the rigid body error
        writer.writerow(["keypoint_pos_error_mean", "keypoint_pos_error_stddev"])
        writer.writerow([result.keypoint_pos_error_mean, result.keypoint_pos_error_stddev])

    filename = f"eval_horizon_{set_name}_{model_name}.csv"
    with open(filename, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["pos_error_mean"])
        row_count = result.horizon_pos_error_mean.shape[0]
        for i in range(row_count):
            writer.writerow([result.horizon_pos_error_mean[i]])
