from __future__ import print_function
import h5py
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from utils.util import R_2vect, Global2Local, Local2Global
import os
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
        # get the video id in the original h5 file
        sampleindex_seq = self.pc_data_index[idx] // self.numFrame
        sampleindex_seq = self.validVideoInd[sampleindex_seq]

        # get the frame id in the video
        sampleindex_frame = self.pc_data_index[idx] % self.numFrame

        # pushing direction
        move_direction = self.effector_dir[sampleindex_seq][0]
        R = R_2vect(self.align_axis, move_direction)
        effector_info = self.effector_data[sampleindex_seq, sampleindex_frame][0]
        pos_effector = effector_info[:3]
        radius_effector = effector_info[3]

        effector_info_future = self.effector_data[sampleindex_seq, sampleindex_frame+self.mintimegap][0]
        pos_effector_future = effector_info_future[:3]


        # return a single frame
        # data_xt = self.full_data[self.mesh_key][sampleindex_seq,
        #                   sampleindex_frame, :].reshape(self.numPoint,3).T.astype(self.float_type)

        data_xt = self.mesh_data[sampleindex_seq,
                          sampleindex_frame, self.kpset, :].reshape(-1,3).astype(self.float_type)

        data_xt_future = self.mesh_data[sampleindex_seq,
                          sampleindex_frame+self.mintimegap, self.kpset, :].reshape(-1,3).astype(self.float_type)

        data_xt = Global2Local(pos_effector, R, data_xt)
        data_xt_future = Global2Local(pos_effector, R, data_xt_future)

        if sampleindex_frame == 0:
            data_xt_speed = np.zeros_like(data_xt)
        else:
            data_xt_history = self.mesh_data[sampleindex_seq,
                          sampleindex_frame-1, self.kpset, :].reshape(-1,3).astype(self.float_type)
            data_xt_history = Global2Local(pos_effector, R, data_xt_history)
            data_xt_speed = data_xt - data_xt_history


        return data_xt, data_xt_future, data_xt_speed, pos_effector, pos_effector_future, radius_effector, move_direction, R

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