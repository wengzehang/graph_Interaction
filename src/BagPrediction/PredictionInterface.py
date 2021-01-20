"""
Interface for prediction models.
"""

import SimulatedData

import numpy as np
from typing import List


class PredictedFrame:
    def __init__(self,
                 cloth_keypoint_positions: np.array,
                 rigid_body_positions: np.array):
        self.cloth_keypoint_positions = cloth_keypoint_positions
        self.rigid_body_positions = rigid_body_positions


class PredictionInterface:

    def predict_frame(self, frame: SimulatedData.Frame, next_effector_position: np.array) -> PredictedFrame:
        raise NotImplementedError()

    def predict_frame_list(self, frames: List[SimulatedData.Frame]):
        # Implement batch prediction in implementation if possible
        return [self.predict_frame(frame) for frame in frames]

    def prepare_scenario(self, scenario: SimulatedData.Scenario):
        # This method is used to predict anchor frames with a long horizon model
        # It is called before any predict_frame() call is made for that scenario
        pass
