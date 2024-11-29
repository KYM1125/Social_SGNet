import sys
import os
# 添加 lib 模块的父目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
import lib.utils as utl
from configs.dair import dair_parse_sgnet_args as parse_args
from lib.utils.dair_train_utils_cvae import visualize_trajectories  # 导入可视化函数

def main(args):
    this_dir = osp.dirname(__file__)
    save_dir = osp.join(this_dir, 'checkpoints', args.dataset, args.model, str(args.dropout), str(args.seed), 'visualisation_file','plot')
    # print("save_dir: ", save_dir)
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))

    # Build test data loader
    test_gen = utl.build_data_loader(args, 'test', batch_size=1)
    # print("Number of test samples:", test_gen.__len__())

    # 可视化输入和目标轨迹
    visualize_trajectories(test_gen, device, save_dir)

if __name__ == '__main__':
    main(parse_args())
