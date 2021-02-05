"""
Training a model given by a specification
"""

import Models
from ModelTrainer import ModelTrainer
import Datasets

import argparse

parser = argparse.ArgumentParser(description='Train a prediction model for deformable bag manipulation')
parser.add_argument('--spec', help='Specify the model specification you want to train: motion, has_moved',
                    default='motion')
parser.add_argument('--frame_step', type=int, default=1)
parser.add_argument('--task_index', type=int, default=None)

args, _ = parser.parse_known_args()

# Specification of the model which we want to train
if args.spec == "motion":
    model = Models.specify_motion_model("MotionModel")
elif args.spec == "has_moved":
    model = Models.specify_has_moved_model("HasMovedModel")
else:
    raise NotImplementedError("Model specification is unknown", args.spec)

# Add a suffic for the frame_step to distinguish longer horizon prediction models
model.name = model.name + "_" + str(args.frame_step)

# For frame-wise prediction set frame_step to 1
# For long horizon prediction choose a value > 1
model.training_params.frame_step = args.frame_step

print("Training ", model.name, "with frame_step", model.training_params.frame_step)

if args.task_index is None:
    # Paths to training and validation datasets (+ topology of the deformable object)
    train_path_to_topodict = 'h5data/topo_train.pkl'
    train_path_to_dataset = 'h5data/train_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

    valid_path_to_topodict = 'h5data/topo_valid.pkl'
    valid_path_to_dataset = 'h5data/valid_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'
else:
    tasks_path = "./h5data/tasks"

    print("Chosen task:", args.task_index)
    task = Datasets.get_task_by_index(args.task_index)

    train_path_to_dataset = task.path_to_dataset(tasks_path, Datasets.Subset.Train)
    train_path_to_topodict = task.path_to_topodict(tasks_path, Datasets.Subset.Train)

    valid_path_to_dataset = task.path_to_dataset(tasks_path, Datasets.Subset.Validation)
    valid_path_to_topodict = task.path_to_topodict(tasks_path, Datasets.Subset.Validation)


# TODO: Define a specific output path for the model depending on task_index
trainer = ModelTrainer(model=model,
                       train_path_to_topodict='h5data/topo_train.pkl',
                       train_path_to_dataset='h5data/train_sphere_sphere_f_f_soft_out_scene1_2TO5.h5',
                       valid_path_to_topodict='h5data/topo_valid.pkl',
                       valid_path_to_dataset='h5data/valid_sphere_sphere_f_f_soft_out_scene1_2TO5.h5',
                       )

trainer.train()


