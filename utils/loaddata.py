from __future__ import print_function
import h5py
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from utils.util import R_2vect, Global2Local, Local2Global
import os
import itertools




# from torch_geometric.nn import MetaLayer

class BagDataset(Dataset):
    def __init__(self, h5_file=None, topo_file=None, kpset=None, mintimegap=1, float_type=np.float32, align_axis=np.array([1,0,0]), rand_seed=1234):

        # seed for data shuffling, deprecated
        np.random.seed(rand_seed)

        # read topo dict
        pkl_file = open(topo_file, 'rb')
        topodict = pickle.load(pkl_file)
        pkl_file.close()

        # read the h5 file
        self.full_data = h5py.File(h5_file, "r")
        self.mesh_key = "posCloth"
        self.meshvelo_key = "veloCloth"
        # self.rigid_key = "posRigid"
        self.clothid_key = "clothid"
        # self.rigidNum_key = "numRigid"
        self.effector_key = "posEffector"
        # self.effector_init_pos_key = "initPosEffector"
        self.effector_init_act_key = "initActEffector"
        # self.effector_init_speed_key = "initSpeedEffector"

        #
        self.mintimegap = mintimegap

        self.mesh_data = self.full_data[self.mesh_key]

        self.effector_dir = self.full_data[self.effector_init_act_key]
        self.effector_data = self.full_data[self.effector_key]

        # get the valid video index according to the preaction flag, defaultly read the whole video sets
        self.validVideoInd = np.arange(self.full_data[self.mesh_key].shape[0])

        # kpset for tracking
        if kpset is None:
            kpset = np.arange(self.full_data[self.mesh_key].shape[2]).sort()
            self.kpset = kpset
        else:
            kpset.sort()
            self.kpset = kpset

        self.numkp = len(kpset)
        self.numnode = self.numkp + 1 # no extra rigid object

        self.numVideo = len(self.validVideoInd)
        # number of frames per video
        self.numFrame = self.full_data[self.mesh_key].shape[1]
        # number of point per frame
        self.numPoint = self.full_data[self.mesh_key].shape[2]
        # number of total frames
        self.numTotalFrame = self.numVideo * self.numFrame
        self.float_type = float_type
        self.align_axis = align_axis

        # indices generation, use all frames for training, suppose we want to predict the keypoint after "mintimegap" steps
        self.indices = np.arange(self.numTotalFrame)
        self.pc_data_index = self.indices

        great_ind = []
        for i in range(len(self.pc_data_index)):
            # < wrong before? <= not !=
            if self.pc_data_index[i] % self.numFrame < (self.numFrame - self.mintimegap):
                great_ind.append(i)

        self.pc_data_index = self.pc_data_index[great_ind]

        return

    def __len__(self):
        # number of valid frame pairs for training
        return len(self.pc_data_index)

    def __getitem__(self, idx):
        edge_index = np.array([i for i in itertools.product(np.arange(self.numnode), repeat=2)]).transpose()
        # tensor_edge_index_conti = torch.tensor(edge_index).contiguous()
        # tensor_edge_index = torch.tensor(edge_index)

        # get the video id in the original h5 file
        sampleindex_seq = self.pc_data_index[idx] // self.numFrame
        sampleindex_seq = self.validVideoInd[sampleindex_seq]

        # get the frame id in the video
        sampleindex_frame = self.pc_data_index[idx] % self.numFrame

        # get cloth frame
        # current
        data_xt_current = self.mesh_data[sampleindex_seq,
                          sampleindex_frame, self.kpset, :].reshape(-1,3).astype(self.float_type)

        # future
        data_xt_future = self.mesh_data[sampleindex_seq,
                          sampleindex_frame+self.mintimegap, self.kpset, :].reshape(-1,3).astype(self.float_type)
        # past
        # if sampleindex_frame == 0:
        if sampleindex_frame < self.mintimegap:
            data_xt_past = data_xt_current
        else:
            data_xt_past = self.mesh_data[sampleindex_seq,
                           sampleindex_frame - self.mintimegap, self.kpset, :].reshape(-1, 3).astype(self.float_type)

        ##############################3
        # effector
        # current
        move_direction = self.effector_dir[sampleindex_seq][0]
        R = R_2vect(self.align_axis, move_direction)
        effector_info_current = self.effector_data[sampleindex_seq, sampleindex_frame][0]

        pos_effector_current = effector_info_current[:3].reshape(1,3)
        radius_effector_current = effector_info_current[3]

        # future
        effector_info_future = self.effector_data[sampleindex_seq, sampleindex_frame+self.mintimegap][0]
        pos_effector_future = effector_info_future[:3].reshape(1,3)
        # past
        # if sampleindex_frame == 0:
        if sampleindex_frame < self.mintimegap:
            effector_info_past = effector_info_current

            # pos_effector_speed = np.zeros_like(pos_effector)
        else:
            effector_info_past = self.effector_data[sampleindex_seq, sampleindex_frame-self.mintimegap][0]
        pos_effector_past = effector_info_current[:3]
            # pos_effector_speed = pos_effector - pos_effector_past

        pos_effector_past = effector_info_past[:3].reshape(1,3)


        # convert global info to local info

        data_xt_past = Global2Local(pos_effector_current, R, data_xt_past)
        data_xt_current = Global2Local(pos_effector_current, R, data_xt_current)
        data_xt_future = Global2Local(pos_effector_current, R, data_xt_future)

        pos_effector_current = Global2Local(pos_effector_current, R, pos_effector_current) # 0,0,0
        pos_effector_future = Global2Local(pos_effector_current, R, pos_effector_future)
        pos_effector_past = Global2Local(pos_effector_current, R, pos_effector_past)


        data_xt_current_speed = data_xt_current - data_xt_past
        data_xt_future_speed = data_xt_future - data_xt_current
        pos_effector_current_speed = pos_effector_current - pos_effector_past
        pos_effector_future_speed = pos_effector_future - pos_effector_current


        # construct the cloth particle graph node with attributes
        radiusarray = 0.001*np.ones(self.numkp).reshape(-1,1)
        undercontrol = np.zeros(self.numkp).reshape(-1,1)
        # future pos, assume it is unknown
        data_xt_movetowards = np.zeros((self.numkp, 3))

        clothnode = np.hstack((data_xt_current,data_xt_current_speed, radiusarray, undercontrol, data_xt_movetowards))

        # effector
        effector_node_array = np.zeros((1,11))
        effector_node_array[0, 0:3] = pos_effector_current
        effector_node_array[0, 3:6] = pos_effector_current_speed
        effector_node_array[0, 6] = radius_effector_current
        effector_node_array[0, 7] = 1.0
        effector_node_array[0, 8:11] = pos_effector_future_speed
        effectornode = effector_node_array

        # source input graph
        sourcegraph = np.vstack((clothnode, effectornode))

        # target subgraph, predicting ground truth
        # targetgraph = data_xt_future
        targetgraph = np.vstack((data_xt_future, pos_effector_future))
        # targetgraph_sub = np.hstack((data_xt_future,data_xt_future_speed, radiusarray, undercontrol, data_xt_movetowards))
        # targetgraph = np.hstack((data_xt_future,data_xt_future_speed, radiusarray, undercontrol, data_xt_movetowards))


        nodepos_source = sourcegraph[:,:3]
        nodepos_target = targetgraph[:,:3]
        source_edge_attr = nodepos_source[edge_index[0]] - nodepos_source[edge_index[1]]
        target_edge_attr = nodepos_target[edge_index[0]] - nodepos_target[edge_index[1]]


        return sourcegraph, targetgraph, edge_index, source_edge_attr, target_edge_attr
        # return data_xt, data_xt_future, data_xt_speed, pos_effector_future, radius_effector, move_direction, R, sourcegraph, targetgraph_sub, targetgraph

if __name__ == '__main__':
    trainmode = "test"
    objtype = "sphere"
    effectortype = "sphere"

    ltype = "f"
    rtype = "f"
    materialtype = "soft"
    ballin = "out"
    folder_dir = "/media/zehang/New Volume/code/H5Bag_G/data/"
    dataset_filename = os.path.join(folder_dir, "{}_{}_{}_{}_{}_{}_{}_scene1.h5".format(trainmode,objtype, effectortype,ltype,rtype,materialtype,ballin))
    pklname = 'topo_{}.pkl'.format(trainmode)
    topopath = os.path.join(folder_dir, pklname)
    # keypoint index set
    kpset = [ 759, 545, 386, 1071, 429, 943,  820,1013, 1124,1212, 1269, 674, 685, 1236]#,       632, 203, 250, 699  ]

    mintimegap = 1

    bagSet = BagDataset(dataset_filename,  topopath, kpset=kpset, mintimegap=mintimegap, float_type=np.float32, rand_seed=1234)

    train_loader = torch.utils.data.DataLoader(
        dataset=bagSet,
        batch_size=5,
        shuffle=True  # False
    )

    for i in range(100):
        print(i)
        data = next(iter(train_loader))
        print(data[0].shape)
        print(data[1].shape)