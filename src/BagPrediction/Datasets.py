"""
Code for loading the different datasets.
Examples:
    s1_soft_ballin_f_f_ballmove_*
    s3_soft_ballin_open_f_ballnomove_noeffector_*
"""

import SimulatedData

from enum import Enum
import os


class BagContent(Enum):
    Empty = 0
    BallInside = 1

    def filename(self):
        switcher = {
            BagContent.Empty: "noballin",
            BagContent.BallInside: "ballin",
        }
        result = switcher.get(self, None)
        if result is None:
            raise ValueError("BagContent is not handled in filename() function:", self)
        else:
            return result


class HandMotion(Enum):
    Fixed = 0
    Released = 1
    Open = 2
    Circle = 3
    Lift = 4

    def filename(self):
        switcher = {
            HandMotion.Fixed: "f",
            HandMotion.Released: "r",
            HandMotion.Open: "open",
            HandMotion.Circle: "circle",
            HandMotion.Lift: "lift"
        }
        result = switcher.get(self, None)
        if result is None:
            raise ValueError("HandMotion is not handled in filename() function:", self)
        else:
            return result


class EffectorMotion(Enum):
    NoBall = 0
    Ball = 1

    def filename(self):
        switcher = {
            EffectorMotion.NoBall: "ballnomove_noeffector",
            EffectorMotion.Ball: "ballmove",
        }
        result = switcher.get(self, None)
        if result is None:
            raise ValueError("EffectorMotion is not handled in filename() function:", self)
        else:
            return result


class Subset(Enum):
    Train = 0
    Validation = 1
    Test = 2

    def filename(self):
        switcher = {
            Subset.Train: "train",
            Subset.Validation: "valid",
            Subset.Test: "test"
        }
        result = switcher.get(self, None)
        if result is None:
            raise ValueError("Subset is not handled in filename() function:", self)
        else:
            return result

    @staticmethod
    def from_name(set_name):
        for subset in Subset:
            if subset.filename() == set_name:
                return subset
        return None


class TaskDataset:
    def __init__(self,
                 index: int = 1,
                 bag_content: BagContent = BagContent.Empty,
                 left_hand_motion: HandMotion = HandMotion.Fixed,
                 right_hand_motion: HandMotion = HandMotion.Fixed,
                 effector_motion: EffectorMotion = EffectorMotion.NoBall
                 ):
        self.index = index
        self.bag_content = bag_content
        self.left_hand_motion = left_hand_motion
        self.right_hand_motion = right_hand_motion
        self.effector_motion = effector_motion

    def filename(self, subset: Subset) -> str:
        return f"s{self.index}_soft_{self.bag_content.filename()}_" + \
               f"{self.left_hand_motion.filename()}_{self.right_hand_motion.filename()}_" + \
               f"{self.effector_motion.filename()}_{subset.filename()}.h5"

    def path_to_dataset(self, root_path: str, subset: Subset) -> str:
        return os.path.join(root_path, self.filename(subset))

    def path_to_topodict(self, root_path: str, subset: Subset) -> str:
        topo_filename = f"topo_{subset.filename()}.pkl"
        return os.path.join(root_path, topo_filename)


s1 = TaskDataset(index=1,
                 bag_content=BagContent.BallInside,
                 left_hand_motion=HandMotion.Fixed,
                 right_hand_motion=HandMotion.Fixed,
                 effector_motion=EffectorMotion.Ball)

s3 = TaskDataset(index=3,
                 bag_content=BagContent.BallInside,
                 left_hand_motion=HandMotion.Open,
                 right_hand_motion=HandMotion.Fixed,
                 effector_motion=EffectorMotion.NoBall)

s5 = TaskDataset(index=5,
                 bag_content=BagContent.BallInside,
                 left_hand_motion=HandMotion.Circle,
                 right_hand_motion=HandMotion.Fixed,
                 effector_motion=EffectorMotion.NoBall)

s6 = TaskDataset(index=6,
                 bag_content=BagContent.BallInside,
                 left_hand_motion=HandMotion.Circle,
                 right_hand_motion=HandMotion.Released,
                 effector_motion=EffectorMotion.NoBall)

s7 = TaskDataset(index=7,
                 bag_content=BagContent.Empty,
                 left_hand_motion=HandMotion.Fixed,
                 right_hand_motion=HandMotion.Fixed,
                 effector_motion=EffectorMotion.Ball)

s9 = TaskDataset(index=9,
                 bag_content=BagContent.Empty,
                 left_hand_motion=HandMotion.Open,
                 right_hand_motion=HandMotion.Fixed,
                 effector_motion=EffectorMotion.NoBall)

s11 = TaskDataset(index=11,
                  bag_content=BagContent.Empty,
                  left_hand_motion=HandMotion.Circle,
                  right_hand_motion=HandMotion.Fixed,
                  effector_motion=EffectorMotion.NoBall)

s12 = TaskDataset(index=12,
                  bag_content=BagContent.Empty,
                  left_hand_motion=HandMotion.Circle,
                  right_hand_motion=HandMotion.Released,
                  effector_motion=EffectorMotion.NoBall)

s13 = TaskDataset(index=13,
                  bag_content=BagContent.BallInside,
                  left_hand_motion=HandMotion.Lift,
                  right_hand_motion=HandMotion.Released,
                  effector_motion=EffectorMotion.NoBall)

s14 = TaskDataset(index=14,
                  bag_content=BagContent.Empty,
                  left_hand_motion=HandMotion.Lift,
                  right_hand_motion=HandMotion.Released,
                  effector_motion=EffectorMotion.NoBall)

tasks = [
    s1, s3, s5, s6, s7, s9, s11, s12
]


def get_task_by_index(task_index: int):
    for task in tasks:
        if task.index == task_index:
            return task
    return None


if __name__ == '__main__':
    # Verify that the files for the task datasets exist
    tasks_path = "./h5data/tasks/"

    for s in Subset:
        # Check for topology files
        topo_path = os.path.join(tasks_path, f"topo_{s.filename()}.pkl")
        if not os.path.exists(topo_path):
            print("Could not find topo file:", topo_path)

        # Check for data files
        for task in tasks:
            filename = task.filename(s)
            full_path = os.path.join(tasks_path, filename)
            if not os.path.exists(full_path):
                print("Could not find data file:", full_path)
