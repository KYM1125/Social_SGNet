'''
Defined classes:
    class BiTraPNP()
Some utilities are cited from Trajectron++
'''
import sys
import numpy as np
import copy
from collections import defaultdict
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn
from torch.distributions import Normal

def reconstructed_probability(x):
    # 创建一个标准正态分布
    recon_dist = Normal(0, 1)
    # log_prob(x)计算x的对数概率密度值
    # exp()计算e的x次方，将其还原成概率密度值
    # mean(dim=-1)计算每个样本的平均概率
    p = recon_dist.log_prob(x).exp().mean(dim=-1)  # [batch_size, K]
    return p

class BiTraPNP(nn.Module):
    def __init__(self, args):
        super(BiTraPNP, self).__init__()
        self.args = copy.deepcopy(args)
        self.param_scheduler = None
        self.input_dim = self.args.input_dim
        self.pred_dim = self.args.pred_dim
        self.hidden_size = self.args.hidden_size
        self.nu = args.nu
        self.sigma = args.sigma
        # node_future_encoder_h用于将输入的节点特征编码到一个隐藏表示
        self.node_future_encoder_h = nn.Sequential(nn.Linear(self.input_dim, self.hidden_size//2),nn.ReLU())
        # 对目标轨迹进行编码，它将未来轨迹经过一个GRU编码成一个隐藏表示
        self.gt_goal_encoder = nn.GRU(input_size=self.pred_dim,
                                        hidden_size=self.hidden_size//2,
                                        bidirectional=True,
                                        batch_first=True)
        # 用于生成潜在变量的先验均值和对数方差
        self.p_z_x = nn.Sequential(nn.Linear(self.hidden_size,  
                                            128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.args.LATENT_DIM*2))
        # posterior
        # 用于生成潜在变量的后验均值和对数方差
        self.q_z_xy = nn.Sequential(nn.Linear(self.hidden_size + self.hidden_size,
                                            128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.args.LATENT_DIM*2))
        
        

    def gaussian_latent_net(self, enc_h, cur_state, K,  target=None, z_mode=None):
        # enc_h是当前时间步的隐藏状态，形状为 (batch_size, hidden_size)
        # cur_state是raw_inputs中的当前时间步的数据，形状为 (batch_size, input_dim)
        # K是cvae中的采样次数，代表多模态
        # target是raw_targets中的未来目标轨迹，形状为 (batch_size, pred_len, pred_dim)
        # z_mode是一个布尔值，如果为True，则将先验均值作为采样结果的第一个模态

        # 从先验分布中抽样潜在变量z
        z_mu_logvar_p = self.p_z_x(enc_h)# 使用神经网络得到一个包含均值和对数方差的张量
        z_mu_p = z_mu_logvar_p[:, :self.args.LATENT_DIM]# 张量的前一部分是均值 μ
        z_logvar_p = z_mu_logvar_p[:, self.args.LATENT_DIM:]# 张量的后一部分是对数方差 log(σ^2)
        # 只是约定了这个张量的前半部分是均值，后半部分是对数方差，具体实际是靠神经网络p_z_x学习出来的,这就是VAE算法学习的部分

        if target is not None:

            # 从后验分布中抽样潜在变量z，仅用于训练
            initial_h = self.node_future_encoder_h(cur_state)
            # 创建双向GRU的初始隐藏状态
            # 输入中的initial_h是正向GRU的初始隐藏状态
            torch.zeros_like(initial_h, device=initial_h.device)#是创建一个与initial_h相同形状的全零张量，用于初始化反向GRU的初始隐藏状态
            # 堆叠后的张量形状为 (2, batch_size, hidden_size//2)
            initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=initial_h.device)], dim=0)
            # flatten_parameters() 通过重新排列内存，使这些参数变得连续，从而加快模型的前向传播速度
            self.gt_goal_encoder.flatten_parameters()
            # 调用双向GRU模块，将目标轨迹编码成隐藏表示
            # _是GRU所有时间步的输出序列和最后的隐藏状态，所以用_表示忽略
            # target_h是GRU处理完整个序列后得到的最终隐藏状态，形状为(2, batch_size, hidden_size//2)
            _, target_h = self.gt_goal_encoder(target, initial_h)
            # 将target_h的形状从(2, batch_size, hidden_size//2)变为(batch_size, 2, hidden_size//2)
            target_h = target_h.permute(1,0,2)
            # 将target_h的形状从(batch_size, 2, hidden_size//2)变为(batch_size, hidden_size)
            target_h = target_h.reshape(-1, target_h.shape[1] * target_h.shape[2])
            # 将当前时间步的隐藏状态enc_h和未来目标轨迹的隐藏表示target_h拼接起来
            z_mu_logvar_q = self.q_z_xy(torch.cat([enc_h, target_h], dim=-1))
            z_mu_q = z_mu_logvar_q[:, :self.args.LATENT_DIM]
            z_logvar_q = z_mu_logvar_q[:, self.args.LATENT_DIM:]
            Z_mu = z_mu_q
            Z_logvar = z_logvar_q

            # 量化后验分布与先验分布之间的差异。它在变分自编码器的训练过程中用作正则化项，鼓励模型的后验分布接近先验分布。
            # 这样可以确保生成的样本具有良好的分布性质，并且避免过拟合。
            KLD = 0.5 * ((z_logvar_q.exp()/z_logvar_p.exp()) + \
                        (z_mu_p - z_mu_q).pow(2)/z_logvar_p.exp() - \
                        1 + \
                        (z_logvar_p - z_logvar_q))
            KLD = KLD.sum(dim=-1).mean()
            # 限制KLD的最小值为0.001
            KLD = torch.clamp(KLD, min=0.001)
            
        else:
            Z_mu = z_mu_p
            Z_logvar = z_logvar_p
            KLD = torch.as_tensor(0.0, device=Z_logvar.device)
        
        # 采样次数K决定了模型可以生成多少种不同的潜在表示Z
        # 这些不同的潜在表示经过解码器的处理后，表现为模型的多模态输出
        # 采样次数是技术层面上的操作，用于从潜在空间中抽取不同的潜在变量；
        # 而多模态是这些采样操作在最终输出上的表现，即生成多个不同的、合理的结果。
        with torch.set_grad_enabled(False):# 避免在采样时计算不必要的梯度
            # 从高斯分布中抽样K个样本，将用于构造潜在变量 Z，以帮助CVAE模型生成多模态预测。
            # 注：高斯分布就是不标准的正态分布，均值不为0，方差不为1
            K_samples = torch.normal(self.nu, self.sigma, size = (enc_h.shape[0], K, self.args.LATENT_DIM)).cuda()#gpu训练时打开
            # K_samples = torch.normal(self.nu, self.sigma, size = (enc_h.shape[0], K, self.args.LATENT_DIM))#cpu训练时打开
        # 计算所有采样样本在标准正态分布的平均概率
        probability = reconstructed_probability(K_samples)
        # 计算标准差
        Z_std = torch.exp(0.5 * Z_logvar)
        # 从高斯分布中采样Z=μ+σ⋅ϵ，其中μ是均值，σ是标准差，ϵ是从高斯分布中的采样噪声
        # 高斯噪声可以被视为对潜在空间中不确定性的建模，并不是指有害或不需要的东西
        Z = Z_mu.unsqueeze(1).repeat(1, K, 1) + K_samples * Z_std.unsqueeze(1).repeat(1, K, 1)
        if z_mode:
            Z = torch.cat((Z_mu.unsqueeze(1), Z), dim=1)
        return Z, KLD, probability


    def forward(self, h_x, last_input, K, target_y=None):
        '''
        Params:
            h_x: hidden state of the current time step
            last_input: raw_inputs of the current time step
            K: number of samples
            target_y: raw_targets of the current time step
        '''
        Z, KLD, probability = self.gaussian_latent_net(h_x, last_input, K, target_y, z_mode=False)
        # 将当前时间步的隐藏状态h_x和潜在变量Z拼接起来，作为解码器的输入
        enc_h_and_z = torch.cat([h_x.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1)
        # 如果解码器不使用潜在变量Z，则将当前时间步的隐藏状态h_x作为解码器的输入，否则将enc_h_and_z作为解码器的输入
        dec_h = enc_h_and_z if self.args.DEC_WITH_Z else h_x
        return dec_h, KLD, probability
