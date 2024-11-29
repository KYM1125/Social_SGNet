import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F


class JAADFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(JAADFeatureExtractor, self).__init__()
        self.embbed_size = args.hidden_size
        self.box_embed = nn.Sequential(nn.Linear(4, self.embbed_size), 
                                        nn.ReLU()) 
    def forward(self, inputs):
        box_input = inputs
        embedded_box_input= self.box_embed(box_input)

        return embedded_box_input

class SocialFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(SocialFeatureExtractor, self).__init__()
        
    def forward(self, inputs):
        box_input = inputs
        return box_input

class ETHUCYFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(ETHUCYFeatureExtractor, self).__init__()
        self.embbed_size = args.hidden_size
        self.embed = nn.Sequential(nn.Linear(6, self.embbed_size), 
                                        nn.ReLU()) 


    def forward(self, inputs):
        box_input = inputs

        embedded_box_input= self.embed(box_input)

        return embedded_box_input

class DAIRFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(DAIRFeatureExtractor, self).__init__()
        self.embbed_size = args.hidden_size
        
        # 分别为位置、速度、加速度创建线性编码层
        self.position_embed = nn.Sequential(
            nn.Linear(2, self.embbed_size//2),  
            nn.ReLU()
        )
        
        self.velocity_embed = nn.Sequential(
            nn.Linear(2, self.embbed_size//2),  
            nn.ReLU()
        )
        

    def forward(self, inputs):
        # 输入形状假设为 (batch_size, 6)，分别为 [x, y, v_x, v_y, a_x, a_y]
        position_input = inputs[:, :, :2]      # 位置部分 (x, y)
        velocity_input = inputs[:, :, 2:4]     # 速度部分 (v_x, v_y)

        # 分别对位置、速度、加速度进行编码
        embedded_position = self.position_embed(position_input)
        embedded_velocity = self.velocity_embed(velocity_input)

        # 拼接编码后的特征
        embedded_box_input = torch.cat((embedded_position, embedded_velocity), dim=2)

        return embedded_box_input


class PIEFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(PIEFeatureExtractor, self).__init__()

        self.embbed_size = args.hidden_size
        self.box_embed = nn.Sequential(nn.Linear(4, self.embbed_size), 
                                        nn.ReLU()) 
    def forward(self, inputs):
        box_input = inputs
        embedded_box_input= self.box_embed(box_input)
        return embedded_box_input

_FEATURE_EXTRACTORS = {
    'PIE': PIEFeatureExtractor,
    'JAAD': JAADFeatureExtractor,
    'ETH': ETHUCYFeatureExtractor,
    'HOTEL': ETHUCYFeatureExtractor,
    'UNIV': ETHUCYFeatureExtractor,
    'ZARA1': ETHUCYFeatureExtractor,
    'ZARA2': ETHUCYFeatureExtractor,
    'DAIR': ETHUCYFeatureExtractor,
    'Social': SocialFeatureExtractor,
}

def build_feature_extractor(args):
    func = _FEATURE_EXTRACTORS[args.dataset]
    return func(args)
