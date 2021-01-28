"""
Training a model given by a specification
"""

import Models
from ModelTrainer import ModelTrainer

# Specification of the model which we want to train
model = Models.specify_motion_model("MotionModel")

masked_model = Models.specify_has_moved_model("MaskModel")
# For frame-wise prediction set frame_step to 1
# For long horizon prediction choose a value > 1
model.training_params.frame_step = 1

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


