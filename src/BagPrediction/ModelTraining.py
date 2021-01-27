"""
Training a model given by a specification
"""

import SimulatedData
import Models
import ModelDataGenerator


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
