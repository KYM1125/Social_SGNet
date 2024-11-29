import torch
import torch.nn as nn
from .feature_extractor import build_feature_extractor
from .bitrap_np import BiTraPNP
import torch.nn.functional as F

class Social_SGNet_CVAE(nn.Module):
    def __init__(self, args):
        super(Social_SGNet_CVAE, self).__init__()
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
        self.ob_radius = 5
        self.hidden_dim_by = 256
        self.feature_dim = 256
        self.hidden_dim_fx = 256
        self.self_embed_dim = 256 #128
        self.neighbor_embed_dim = 128 #128
        self.social_weight = nn.Parameter(torch.tensor(1.0))

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
        self.embed_k = nn.Sequential(
            nn.Linear(1, self.feature_dim),    # mpd
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True)#
        )
        self.embed_s = nn.Sequential(
            nn.Linear(4, 64),             # v, a
            nn.ReLU(inplace=True),
            nn.Linear(64, self.self_embed_dim),
            nn.ReLU(inplace=True)#
        )
        self.embed_n = nn.Sequential(
            nn.Linear(4, self.neighbor_embed_dim),
            nn.ReLU(inplace=True)#
        )
        self.embed_q = nn.Sequential(
            nn.Linear(self.hidden_size, self.feature_dim),
            nn.ReLU(inplace=True)#
        )
        self.attention_nonlinearity = nn.LeakyReLU(0.2)

        self.rnn_fx = nn.GRU(self.self_embed_dim+self.neighbor_embed_dim, self.hidden_dim_fx)
        self.rnn_fx_init = nn.Sequential(
            nn.Linear(2, self.hidden_dim_fx), # dp
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim_fx, self.hidden_dim_fx*self.rnn_fx.num_layers),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim_fx*self.rnn_fx.num_layers, self.hidden_dim_fx*self.rnn_fx.num_layers),
        )
        self.rnn_by = nn.GRU(self.self_embed_dim+self.neighbor_embed_dim, self.hidden_dim_by)

        self.embed = nn.Sequential(nn.Linear(6, self.hidden_size), 
                                        nn.ReLU()) 

        self.enc_drop = nn.Dropout(self.dropout)
        self.social_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)
        self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4 + self.neighbor_embed_dim, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size//4, self.hidden_size//4)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
        print("Social_SGNet_CVAE model is initialized")

    def attention(self, q, k):
        # q: N x d
        # k: N x N x d
        # mask: N x Nn
        e = (k @ q.unsqueeze(-1)).squeeze(-1)           # N x Nn
        e = self.attention_nonlinearity(e)              # N x Nn
        # e[~mask] = -float("inf")
        att = nn.functional.softmax(e, dim=-1)    # N x Nn
        att = torch.where(torch.isnan(att), torch.tensor(0.0, device=att.device), att)
        return att

    def SGE(self, goal_hidden):
        # initial goal input with zero
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size//4))
        # initial trajectory tensor
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_hidden), goal_hidden)
            # next step input is generate by hidden
            goal_input = self.goal_hidden_to_input(goal_hidden)
            
            goal_list.append(goal_hidden)
            # regress goal traj for loss
            # goal_traj_hidden形状为(batch_size, hidden_size)
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            # print("goal_traj_hidden.shape:",goal_traj_hidden.shape)
            # goal_traj形状为(batch_size, dec_steps, pred_dim)
            goal_traj[:,dec_step,:] = self.regressor(goal_traj_hidden)
            # print("goal_traj.shape:",goal_traj.shape)
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

    def radial_basis_function(self, dist, sigma):
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))

    def SocialEncoder(self, raw_inputs, traj_enc_hidden):
        # 提取位置、速度和加速度信息
        x = raw_inputs[:, :2]  # [num_pedestrian, 2], 提取x, y坐标
        v = raw_inputs[:, 2:4] # [num_pedestrian, 2], 提取速度 (v_x, v_y)
        
        num_pedestrians = raw_inputs.size(0)  # 行人数量
        dp = x.unsqueeze(1) - x.unsqueeze(0)  # [num_pedestrian, num_pedestrian, 2], 行人之间的坐标差
        dv = v.unsqueeze(1) - v.unsqueeze(0)  # [num_pedestrian, num_pedestrian, 2], 行人之间的速度差

        # 初始化能量图
        em_ped_rep = torch.zeros((num_pedestrians, num_pedestrians), device=raw_inputs.device)  # 能量图
        
        # 位置差与速度差的点积
        dot_dp_dv = (dp.unsqueeze(-2) @ dv.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # [num_pedestrian, num_pedestrian], 位置差和速度差的点积
        ttc = -dot_dp_dv / (dv.norm(dim=-1))**2  # [num_pedestrian, num_pedestrian], 加权时间 (tau)

        # 如果 ttc 为负数，表示两者不会相撞，则将 ttc 设置为 delta_t
        delta_t = 12.0  # 自定义的最大时间阈值
        ttc = torch.where(ttc < 0, torch.tensor(delta_t, device=ttc.device), ttc)

        # 如果 ttc 大于 delta_t，则限制 ttc 最大值为 delta_t
        ttc = torch.clamp(ttc, max=delta_t)  # 限制 ttc 的上限为 delta_t
        ttc = torch.where(torch.isnan(ttc), torch.tensor(0.0, device=ttc.device), ttc)

        # 调整后的位置和速度差的模
        # collision_distance = (dp + ttc.unsqueeze(-1) * dv).norm(dim=-1)  # [num_pedestrian, num_pedestrian], 调制后的位置和速度差的和的模

        # 构建collision_distance势场
        # 计算 collision_distance 的标准差作为 sigma
        sigma = ttc.std()  # 使用碰撞距离的标准差

        rbf_weighted_md = self.radial_basis_function(ttc, sigma)
        rbf_weighted_md.fill_diagonal_(0)


        # 在能量图中累加能量值
        for i in range(num_pedestrians):
            for j in range(num_pedestrians):
                em_ped_rep[i, j] += rbf_weighted_md[i, j]

        # 特征嵌入
        k = self.embed_k(em_ped_rep.unsqueeze(-1))                          # [num_pedestrian, num_pedestrian, feature_dim]

        n = self.embed_n(torch.cat((dp, dv), -1))           # [num_pedestrian, num_pedestrian, neighbor_embed_dim]

        # Query矩阵是dp的嵌入，Key矩阵是特征嵌入，Value矩阵是邻居的嵌入
        q = self.embed_q(traj_enc_hidden)                             # [num_pedestrian, feature_dim]，使用上一个时间步的隐藏状态
        att = self.attention(q, k)                     # [num_pedestrian, num_pedestrian]
        social_feature = att.unsqueeze(-2) @ n                          # [num_pedestrian, 1, neighbor_embed_dim]
        social_feature = social_feature.squeeze(-2)                                # [num_pedestrian, neighbor_embed_dim]

        return social_feature

    
    def encoder(self, raw_inputs, traj_input, raw_targets, flow_input=None, start_index = 0):

        # all_goal_traj形状为(batch_size, enc_steps, dec_steps, pred_dim)
        all_goal_traj = raw_inputs.new_zeros(raw_inputs.size(0), self.enc_steps, self.dec_steps, self.pred_dim)

        # all_cvae_dec_traj形状为(batch_size, enc_steps, dec_steps, K, pred_dim)
        all_cvae_dec_traj = raw_inputs.new_zeros(raw_inputs.size(0), self.enc_steps, self.dec_steps, self.K, self.pred_dim)

        # goal_for_enc形状为(batch_size, hidden_size//4)
        goal_for_enc = raw_inputs.new_zeros((raw_inputs.size(0), self.hidden_size//4))

        # traj_enc_hidden形状为(batch_size, hidden_size)
        traj_enc_hidden = raw_inputs.new_zeros((raw_inputs.size(0), self.hidden_size))

        # total_probabilities形状为(batch_size, enc_steps, K)
        total_probabilities = raw_inputs.new_zeros((raw_inputs.size(0), self.enc_steps, self.K))

        total_KLD = 0

        for enc_step in range(start_index, self.enc_steps):
            # 计算全局交互信息
            social_feature = self.SocialEncoder(raw_inputs[:, enc_step, :],traj_enc_hidden)  # 使用所有行人的信息
             
            # 将当前时间步的输入与全局交互信息拼接
            combined_input = torch.cat((traj_input[:, enc_step, :], self.social_weight * social_feature, goal_for_enc), 1)

            # 将拼接后的输入送入 GRUCell
            traj_enc_hidden = self.traj_enc_cell(self.enc_drop(combined_input), traj_enc_hidden)

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

            all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, traj_input, targets, None, start_index)
            '''
                all_goal_traj用于计算损失值
                all_cvae_dec_traj是预测的多模态轨迹，形状为 (batch_size, enc_steps, dec_steps, K, pred_dim)
                total_KLD是所有时间步的 KLD 损失之和，用于loss
                total_probabilities是所有时间步的概率分布
            '''
            # enc_steps是观测步数，dec_steps是预测步数
            return all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities