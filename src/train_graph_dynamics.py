'''
tools for reading data
'''

from __future__ import print_function
import numpy as np
import os
import torch
from utils.loaddata import BagDataset
from torch_geometric.data import Data, DataLoader
from models.GraphModel import EncoderCoreDecoder
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LayerNorm
import itertools

if __name__ == '__main__':
    trainmode = "test"
    objtype = "sphere"
    effectortype = "sphere"

    ltype = "f"
    rtype = "f"
    materialtype = "soft"
    ballin = "out"

    # Change to correct $PATH$
    # folder = "/Users/cat/Downloads/dataset/bag_scene1/"
    folder = "/media/zehang/New Volume/code/H5Bag_G/data/"


    # Network hyper-parameters
    # # encoder
    # encoder_mlp_in_dim = 15
    # encoder_mlp_nlayer = 2
    # encoder_mlp_nnode = 64

    # encoder decoder input output dim for v and e
    # encoder_in_dims = (15, 225, None)
    # encoder_in_dims = (15, 225, 1)
    # encoder_in_dims = (15, 3, 64)
    encoder_in_dims = (11, 3, 32)
    encoder_out_dims = (64, 64, 64)
    # dec_out_dims = (25, 225, None)
    # dec_out_dims = (25, 225, 1)
    dec_out_dims = (25, 225, 64)
    # core
    core_out_dims = (96,96,64)
    core_step = 10
    # decoder
    decoder_mlp_nlayer = 2
    decoder_mlp_nnode = 64

    # network output dim
    out_dims = (14, 225, None)
    out_dims = (14, 225, None) # wrong
    out_dims = (14, 225, None)

    independent_block_layers = 1
    nhidden = 64

    core_block_layers = 2

    h5name = "{}_{}_{}_{}_{}_{}_{}_scene1.h5".format(trainmode,objtype, effectortype,ltype,rtype,materialtype,ballin)
    pklname = 'topo_{}.pkl'.format(trainmode)

    dataset_filename = os.path.join(folder, h5name)
    topopath = os.path.join(folder, pklname)

    # keypoint index set
    kpset = [ 759, 545, 386, 1071, 429, 943,  820,1013, 1124,1212, 1269, 674, 685, 1236]#,       632, 203, 250, 699  ]
    mintimegap = 1

    bagSet = BagDataset(dataset_filename, topopath, kpset=kpset, mintimegap=mintimegap, float_type=np.float32,
                        rand_seed=1234)
    # edge_index = torch.arange(14)
    # edge_index = torch.combinations(edge_index, with_replacement=False)


    edge_index = torch.tensor([i for i in itertools.product(np.arange(5), repeat=2)]).t().contiguous()

    x = torch.randn(5, 8)

    data = Data(x=x, edge_index=edge_index)

    print(data)



    train_loader = torch.utils.data.DataLoader(
        dataset=bagSet,
        batch_size=5,
        shuffle=True  # False
    )

    # define the network, learn from scratch
    net = EncoderCoreDecoder(
        in_dims = encoder_in_dims, # dim for encoder input
        core_out_dims=core_out_dims, # dim for core output
        out_dims=(14, 225, None), # dim for output graph
        core_steps=core_step,
        dec_out_dims=dec_out_dims,
        encoder_out_dims=encoder_out_dims,
        save_name=None,
        e2v_agg="sum",
        n_hidden=core_block_layers,
        hidden_size=nhidden, # 64
        activation=ReLU,
        independent_block_layers=independent_block_layers,
    ).cuda()

    print(net)


    data = next(iter(train_loader))
    data[0][0].shape
    data[2][0].shape

    net.forward(data[0][0].float().cuda(), data[2][0].cuda(), data[3][0].float().cuda(), u=torch.randn(32).float().cuda())

    # for batchid, data in enumerate(train_loader):
    #
    #     inputGraph_batch = [Data(x=data[0][i], edge_index=data[2][i], edge_attr=data[3][i]) for i in range(data[0].shape[0])]# Data(x=data[0], edge_index=edge_index)
    #     targetGraph_batch = [Data(x=data[1][i], edge_index=data[2][i], edge_attr=data[4][i]) for i in range(data[1].shape[0])]
    #     print(inputGraph_batch)
    #     print(targetGraph_batch)
    #     # construct the input graph, 14 nodes.
    #     print(data[0].shape)
    #     print(data[1].shape)
    #     # edge_index = torch.tensor([])
    #     loader = DataLoader(inputGraph_batch, batch_size=2)
    #     a = next(iter(loader))
    #     print(a)

