'''
    python tools/ethucy/test_model.py --gpu 0 --dataset DAIR --model Social_SGNet_CVAE --checkpoint /data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/best_model/xx.pt
    改文件名在lib/utils/dair_train_utils_cvae.py中
'''
import sys
import os
# 添加 lib 模块的父目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import os.path as osp
import shutil
import numpy as np
import time
import random
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data

import lib.utils as utl
from configs.dair import dair_parse_sgnet_args as parse_args
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.dair_train_utils_cvae import train, val, test_save

def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', args.dataset, model_name, str(args.dropout), str(args.seed), 'visual')
    print("save_dir: ", save_dir)
    # Check if the directory exists; if it does, delete it
    if osp.isdir(save_dir):
        shutil.rmtree(save_dir)
    
    # Create a new directory
    os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))
    model = build_model(args)

    model = nn.DataParallel(model)
    model = model.to(device)
    
    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        del checkpoint

    criterion = rmse_loss().to(device)

    # Build test data loader
    test_gen = utl.build_data_loader(args, 'test', batch_size=1)
    print("Number of test samples:", test_gen.__len__())

    first_batch = next(iter(test_gen))
    print("First batch keys:", first_batch.keys())
    
    # Run test and save predictions and ground truths within the test function
    test_loss, ADE_08, FDE_08, ADE_12, FDE_12 = test_save(model, test_gen, criterion, device, save_dir)
    print("Test Loss: {:.4f}".format(test_loss))
    print("ADE_08: {:.4f}; FDE_08: {:.4f}; ADE_12: {:.4f}; FDE_12: {:.4f}\n".format(ADE_08, FDE_08, ADE_12, FDE_12))

if __name__ == '__main__':
    main(parse_args())
