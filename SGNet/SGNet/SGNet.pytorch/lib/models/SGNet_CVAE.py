import torch
import torch.nn as nn
from .feature_extractor import build_feature_extractor
from .bitrap_np import BiTraPNP
import torch.nn.functional as F

class SGNet_CVAE(nn.Module):
    def __init__(self, args):
        super(SGNet_CVAE, self).__init__()
        self.cvae = BiTraPNP(args)
        self.hidden_size = args.hidden_size # GRU hidden size
        self.enc_steps = args.enc_steps # observation step
        self.dec_steps = args.dec_steps # prediction step
        self.dataset = args.dataset
        self.dropout = args.dropout
        self.feature_extractor = build_feature_extractor(args)
        self.pred_dim = args.pred_dim
        self.K = args.K
        self.map = False
        if self.dataset in ['JAAD','PIE']:
            # the predict shift is in pixel
            self.pred_dim = 4
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                     self.pred_dim),
                                                     nn.Tanh())
            self.flow_enc_cell = nn.GRUCell(self.hidden_size*2, self.hidden_size)
        elif self.dataset in ['DAIR','ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
            self.pred_dim = 2
            # the predict shift is in meter
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                        self.pred_dim))   
        self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))
        self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))

        self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size//4),
                                                nn.ReLU(inplace=True))
        self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
        self.cvae_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size + args.LATENT_DIM,
                                                self.hidden_size),
                                                nn.ReLU(inplace=True))
        self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size),
                                                nn.ReLU(inplace=True))

        self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.dec_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
        self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.enc_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)
        self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size//4, self.hidden_size//4)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
        print("SGNet_CVAE model is initialized")

    def SGE(self, goal_hidden):
        # initial goal input with zero
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size//4))
        # initial trajectory tensor
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_input), goal_hidden)
            # next step input is generate by hidden
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)
            # regress goal traj for loss
            # goal_traj_hidden形状为(batch_size, hidden_size//4)
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            # goal_traj形状为(batch_size, dec_steps, pred_dim)
            goal_traj[:,dec_step,:] = self.regressor(goal_traj_hidden)
        # get goal for decoder and encoder
        # goal_for_dec形状为(batch_size, dec_steps, hidden_size//4)
        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        # goal_for_enc 是由多个时间步的 goal_hidden 通过 goal_to_enc 映射而来，并堆叠成一个张量
        # 其维度为 (batch_size, dec_steps, hidden_size//4)
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list],dim = 1)
        # tanh(goal_for_enc)每个时间步的特征通过 tanh 函数进行非线性变换，(batch_size, dec_steps, hidden_size//4)
        # enc_goal_attn 通过全连接层得到，(batch_size, dec_steps, 1)
        # squeeze(-1)去掉最后一个维度，(batch_size, dec_steps)
        enc_attn= self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        # enc_attn 通过 softmax 函数进行归一化，(batch_size, dec_steps)
        # unsqueeze(1)在第二个维度增加一个维度，(batch_size, 1, dec_steps)
        enc_attn = F.softmax(enc_attn, dim =1).unsqueeze(1)
        # (batch_size, 1, dec_steps) * (batch_size, dec_steps, hidden_size//4) -> (batch_size, 1, hidden_size//4)
        # squeeze(1)去掉第二个维度，(batch_size, hidden_size//4)
        goal_for_enc  = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
        # goal_traj 用于计算损失函数，goal_for_enc 用于传递给下一个时间步，goal_for_dec 用于传递给 decoder
        return goal_for_dec, goal_for_enc, goal_traj

    def cvae_decoder(self, dec_hidden, goal_for_dec):
        batch_size = dec_hidden.size(0)# batch_size
       
        K = dec_hidden.shape[1]# K是随机性，多模态，代表了cvae的多样性
        # 将 dec_hidden 重新形状化为 (batch_size * K, hidden_size)
        dec_hidden = dec_hidden.view(-1, dec_hidden.shape[-1])
        # 初始化一个形状为 (batch_size, dec_steps, K, pred_dim) 的张量，用于存储解码器预测的轨迹
        dec_traj = dec_hidden.new_zeros(batch_size, self.dec_steps, K, self.pred_dim)
        for dec_step in range(self.dec_steps):
            # incremental goal for each time step
            # 为每个时间步创建一个形状为 (batch_size, dec_steps, hidden_size//4) 的全零张量
            goal_dec_input = dec_hidden.new_zeros(batch_size, self.dec_steps, self.hidden_size//4)
            # 从 goal_for_dec 中选择从当前时间步到最后一个时间步的目标隐藏状态，形状为 (batch_size, dec_steps-dec_step, hidden_size//4)
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:],dim=1)
            # 将这些目标隐藏状态放入 goal_dec_input 中，从当前时间步到最后一个时间步的部分，形状为 (batch_size, dec_steps-dec_step, hidden_size//4)
            goal_dec_input[:,dec_step:,:] = goal_dec_input_temp
            # dec_goal_attn全连接后(batch_size, dec_steps-dec_step, hidden_size//4)
            # squeeze(-1)去掉最后一个维度，(batch_size, dec_steps-dec_step)
            dec_attn= self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            # unsqueeze(1)在第二个维度增加一个维度，(batch_size, 1, dec_steps-dec_step)
            dec_attn = F.softmax(dec_attn, dim =1).unsqueeze(1)
            # (batch_size, 1, dec_steps-dec_step) * (batch_size, dec_steps-dec_step, hidden_size//4)
            # -> (batch_size, 1, hidden_size//4)
            # squeeze(1)去掉第二个维度，(batch_size, hidden_size//4)
            goal_dec_input  = torch.bmm(dec_attn,goal_dec_input).squeeze(1)
            # 将 goal_dec_input 扩展为 (batch_size, K, hidden_size//4) 并展平为 (batch_size * K, hidden_size//4)
            goal_dec_input = goal_dec_input.unsqueeze(1).repeat(1, K, 1).view(-1, goal_dec_input.shape[-1])
            # dec_dec_input经过全连接层后，形状仍为 (batch_size * K, hidden_size)
            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            # 将 goal_dec_input 和 dec_dec_input 拼接在一起，形状为 (batch_size * K, hidden_size + hidden_size//4)
            dec_input = self.dec_drop(torch.cat((goal_dec_input,dec_dec_input),dim = -1))
            # dec_hidden经过GRUCell后，形状为 (batch_size * K, hidden_size)
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            # regress dec traj for loss
            # dec_hidden经过regressor后，形状为 (batch_size * K, pred_dim)
            batch_traj = self.regressor(dec_hidden)
            # batch_traj经过展开后，形状为 (batch_size, K, pred_dim)
            batch_traj = batch_traj.view(-1, K, batch_traj.shape[-1])
            dec_traj[:,dec_step,:,:] = batch_traj
        return dec_traj


    def encoder(self, raw_inputs, raw_targets, traj_input, flow_input=None, start_index = 0):
        # initial output tensor
        # all_goal_traj形状为(batch_size, enc_steps, dec_steps, pred_dim)
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        # all_cvae_dec_traj形状为(batch_size, enc_steps, dec_steps, K, pred_dim)
        all_cvae_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.K, self.pred_dim)
        # initial encoder goal with zeros
        # goal_for_enc形状为(batch_size, hidden_size//4)
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size//4))
        # initial encoder hidden with zeros
        # traj_enc_hidden形状为(batch_size, hidden_size)
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        # total_probabilities形状为(batch_size, enc_steps, K)
        total_probabilities = traj_input.new_zeros((traj_input.size(0), self.enc_steps, self.K))
        total_KLD = 0
        for enc_step in range(start_index, self.enc_steps):
            # traj_input[:,enc_step,:]是当前时间步的特征，goal_for_enc是上一个时间步的目标特征
            # traj_enc_hidden经过GRUCell后，形状为 (batch_size, hidden_size)
            # traj_enc_hidden是当前时间步的隐藏状态
            traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:,enc_step,:], goal_for_enc), 1)), traj_enc_hidden)
            enc_hidden = traj_enc_hidden
            # enc_hidden经过全连接层后，形状为 (batch_size, hidden_size//4)
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            goal_for_dec, goal_for_enc, goal_traj = self.SGE(goal_hidden)

            all_goal_traj[:,enc_step,:,:] = goal_traj
            # dec_hidden是当前时间步的隐藏状态，形状为 (batch_size, hidden_size)
            dec_hidden = self.enc_to_dec_hidden(enc_hidden) 
 
            # cvae就是BiTraPNP，一个多模态的预测模型

            if self.training:
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, raw_inputs[:,enc_step,:], self.K, raw_targets[:,enc_step,:,:])
            else:
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, raw_inputs[:,enc_step,:], self.K)
            total_probabilities[:,enc_step,:] = probability
            total_KLD += KLD
            # cvae_hidden经过全连接层后，形状为 (batch_size, hidden_size)
            cvae_dec_hidden= self.cvae_to_dec_hidden(cvae_hidden)
            if self.map:
                map_input = flow_input
                cvae_dec_hidden = (cvae_dec_hidden + map_input.unsqueeze(1))/2
            # 记录当前时间步的解码器输出
            # all_cvae_dec_traj的最终形状为 (batch_size, enc_steps, dec_steps, K, pred_dim)
            all_cvae_dec_traj[:,enc_step,:,:,:] = self.cvae_decoder(cvae_dec_hidden, goal_for_dec)
            '''
                all_goal_traj是通过 SGE 生成的目标轨迹，形状为 (batch_size, enc_steps, dec_steps, pred_dim)，用于计算损失值
                all_cvae_dec_traj是通过 cvae_decoder 生成的多模态轨迹，形状为 (batch_size, enc_steps, dec_steps, K, pred_dim)
                total_KLD是所有时间步的 KLD 损失之和
                total_probabilities是所有时间步的概率分布
            '''

        return all_goal_traj, all_cvae_dec_traj, total_KLD, total_probabilities
            
    def forward(self, inputs, map_mask=None, targets = None, start_index = 0, training=True):
        self.training = training
        if torch.is_tensor(start_index):
            start_index = start_index[0].item()
        if self.dataset in ['JAAD','PIE']:
            traj_input = self.feature_extractor(inputs)
            all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, targets, traj_input)
            return all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities
        elif self.dataset in ['DAIR','ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
            # inputs的形状为 (batch_size, obs_steps, input_dim)
            # feature_extractor里就是一个线性层加个ReLU激活函数

            traj_input_temp = self.feature_extractor(inputs[:,start_index:,:])
            # traj_input初始化为全零张量，形状为 (batch_size, enc_steps, hidden_size)
            traj_input = traj_input_temp.new_zeros((inputs.size(0), inputs.size(1), traj_input_temp.size(-1)))

            traj_input[:,start_index:,:] = traj_input_temp
            all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, targets, traj_input, None, start_index)
            '''
                all_goal_traj用于计算损失值
                all_cvae_dec_traj是预测的多模态轨迹，形状为 (batch_size, enc_steps, dec_steps, K, pred_dim)
                total_KLD是所有时间步的 KLD 损失之和，用于loss
                total_probabilities是所有时间步的概率分布
            '''

            # enc_steps是观测步数，dec_steps是预测步数
            return all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities