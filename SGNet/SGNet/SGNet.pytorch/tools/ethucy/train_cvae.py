import sys
import os
import os.path as osp
import numpy as np
import time
import random
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
sys.path.append('.')

import lib.utils as utl
from configs.ethucy import parse_sgnet_args as parse_args
from configs.dair import dair_parse_sgnet_args as dair_parse_args
from lib.models import build_model
from lib.losses import rmse_loss
# from lib.utils.ethucy_train_utils_cvae import train, val, test
from lib.utils.dair_train_utils_cvae import train, val, test

def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', args.dataset,model_name,str(args.dropout), str(args.seed))
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    utl.set_seed(int(args.seed))
    model = build_model(args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
                                                           min_lr=1e-10, verbose=1)
    model = nn.DataParallel(model)
    model = model.to(device)
    if osp.isfile(args.checkpoint):

        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch += checkpoint['epoch']
        del checkpoint

    criterion = rmse_loss().to(device)

    train_gen = utl.build_data_loader(args, 'train', batch_size = 1)
    # print(len(train_gen))
    val_gen = utl.build_data_loader(args, 'val', batch_size = 1)
    test_gen = utl.build_data_loader(args, 'test', batch_size = 1)
    # Print batch_size and num_workers
    print(f"Train DataLoader - Batch Size: {train_gen.batch_size}, Num Workers: {train_gen.num_workers}")
    print(f"Val DataLoader - Batch Size: {val_gen.batch_size}, Num Workers: {val_gen.num_workers}")
    print(f"Test DataLoader - Batch Size: {test_gen.batch_size}, Num Workers: {test_gen.num_workers}")
    
    print("Number of validation samples:", val_gen.__len__())
    print("Number of test samples:", test_gen.__len__())



    # train
    min_loss = 1e6
    min_ADE_08 = 10e5
    min_FDE_08 = 10e5
    min_ADE_12 = 10e5
    min_FDE_12 = 10e5
    best_model = None
    best_model_metric = None
    best_ADE_08 = float('inf')  # 初始化为正无穷大，表示一开始没有最优ADE_08
    best_FDE_12 = float('inf')

    for epoch in range(args.start_epoch, args.epochs+args.start_epoch):
        print("Number of training samples:", len(train_gen))

        # train
        train_goal_loss, train_cvae_loss, train_KLD_loss = train(model, train_gen, criterion, optimizer, device)
        # print('Train Epoch: ', epoch, 'Goal loss: ', train_goal_loss, 'Decoder loss: ', train_dec_loss, 'CVAE loss: ', train_cvae_loss, \
        #     'KLD loss: ', train_KLD_loss, 'Total: ', total_train_loss) 
        print('Train Epoch: {} \t Goal loss: {:.4f}\t  CVAE loss: {:.4f}\t KLD loss: {:.10f}\t Total: {:.4f}'.format(
                epoch,train_goal_loss, train_cvae_loss, train_KLD_loss, train_goal_loss + train_cvae_loss + train_KLD_loss ))


        # val
        val_loss = val(model, val_gen, criterion, device)
        lr_scheduler.step(val_loss)
        print("Test Loss: {:.4f}".format(val_loss))

        # test
        test_loss, ADE_08, FDE_08, ADE_12, FDE_12 = test(model, test_gen, criterion, device)
        print("Test Loss: {:.4f}".format(test_loss))
        print("ADE_08: %4f;  FDE_08: %4f;  ADE_12: %4f;   FDE_12: %4f\n" % (ADE_08, FDE_08, ADE_12, FDE_12))

        # 检查当前的 ADE_08 是否是最小值
        if FDE_12 < best_FDE_12:
            best_FDE_12 = FDE_12  # 更新最优ADE_08
            best_model_path = osp.join(save_dir, 'best_model_FDE_12.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'FDE_12': FDE_12,
            }, best_model_path)
            print(f"Best model with FDE_12: {FDE_12:.4f} saved to {best_model_path}")

        # Save the model
        save_path = osp.join(save_dir, f'model_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, save_path)
        print(f"Model saved to {save_path}")

if __name__ == '__main__':
    # main(parse_args())#训练ETH时打开注释
    main(dair_parse_args())#训练DAIR时打开注释
