"""
Training a model given by a specification
"""

import Models
from ModelTrainer import ModelTrainer

import argparse

parser = argparse.ArgumentParser(description='Train a prediction model for deformable bag manipulation')
parser.add_argument('--spec', help='Specify the model specification you want to train: motion, has_moved',
                    default='motion')
parser.add_argument('--frame_step', type=int, default=1)

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


# Paths to training and validation datasets (+ topology of the deformable object)
train_path_to_topodict = 'h5data/topo_train.pkl'
train_path_to_dataset = 'h5data/train_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

valid_path_to_topodict = 'h5data/topo_valid.pkl'
valid_path_to_dataset = 'h5data/valid_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

trainer = ModelTrainer(model=model,
                       train_path_to_topodict='h5data/topo_train.pkl',
                       train_path_to_dataset='h5data/train_sphere_sphere_f_f_soft_out_scene1_2TO5.h5',
                       valid_path_to_topodict='h5data/topo_valid.pkl',
                       valid_path_to_dataset='h5data/valid_sphere_sphere_f_f_soft_out_scene1_2TO5.h5',
                       )

trainer.train()


