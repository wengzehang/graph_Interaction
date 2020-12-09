'''
tools for reading data
'''

from __future__ import print_function
import h5py
import numpy as np
import os
import pickle
import torch
from utils.loaddata import BagDataset
from global_local import R_2vect, Global2Local, Local2Global

if __name__ == '__main__':
    trainmode = "test"
    objtype = "sphere"
    effectortype = "sphere"

    ltype = "f"
    rtype = "f"
    materialtype = "soft"
    ballin = "out"

    # Change to correct $PATH$
    folder = "/Users/cat/Downloads/dataset/bag_scene1/"
    h5name = "{}_{}_{}_{}_{}_{}_{}_scene1.h5".format(trainmode,objtype, effectortype,ltype,rtype,materialtype,ballin)
    pklname = 'topo_{}.pkl'.format(trainmode)

    dataset_filename = os.path.join(folder, h5name)
    topopath = os.path.join(folder, pklname)

    # keypoint index set
    kpset = [ 759, 545, 386, 1071, 429, 943,  820,1013, 1124,1212, 1269, 674, 685, 1236]#,       632, 203, 250, 699  ]
    mintimegap = 1

    bagSet = BagDataset(dataset_filename,  topopath, kpset=kpset, mintimegap=mintimegap, float_type=np.float32, rand_seed=1234)

    train_loader = torch.utils.data.DataLoader(
        dataset=bagSet,
        batch_size=5,
        shuffle=True  # False
    )