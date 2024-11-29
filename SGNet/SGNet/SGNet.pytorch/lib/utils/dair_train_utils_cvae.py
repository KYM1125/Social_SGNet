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
import matplotlib.pyplot as plt
import time

from lib.utils.eval_utils import eval_ethucy, eval_ethucy_cvae
from lib.losses import cvae, cvae_multi

def check_none(**kwargs):
    for var_name, var_value in kwargs.items():
        if var_value is None:
            print(f"Warning: {var_name} is None")

# def stack_trajectories_by_scene(data, key, device):
#     traj_dict = {}
#     # print("scene_name: ", data['scene_name'] )
#     # print("len(data['scene_name'])", len(data['scene_name']))
#     for idx in range(len(data['scene_name'])):
#         sn = data['scene_name'][idx]
#         if sn not in traj_dict:
#             traj_dict[sn] = []
#         traj_dict[sn].append(data[key][idx])

#     # Stack trajectories for each scene_name
#     traj_dict_stacked = {sn: torch.stack(trajs).to(device) for sn, trajs in traj_dict.items()}
#     return traj_dict_stacked

def stack_trajectories_by_scene(data, key, traj_dict, device):
    # 遍历每个 sample 并根据 scene_name 追加轨迹
    for idx in range(len(data['scene_name'])):
        sn = data['scene_name'][idx]
        if sn not in traj_dict:
            traj_dict[sn] = []  # 初始化为列表
        traj_dict[sn].append(data[key][idx])  # 追加轨迹

    # 累加后，返回当前的轨迹字典，暂时不堆叠，直到所有批次处理完
    return traj_dict

# 在所有 batch 处理完后再堆叠
def finalize_trajectory_stacking(traj_dict, device):
    traj_dict_stacked = {}
    for sn, trajs in traj_dict.items():
        # print(f"Final Scene: {sn}, Total trajectories: {len(trajs)}")
        # for traj in trajs:
            # print(f"Trajectory type: {type(traj)}, shape: {traj.shape}")

        if len(trajs) > 1:
            traj_dict_stacked[sn] = torch.stack(trajs).to(device)
        else:
            traj_dict_stacked[sn] = trajs[0].unsqueeze(0).to(device)

    return traj_dict_stacked


def train(model, train_gen, criterion, optimizer, device):
    model.train()  # Sets the module in training mode.
    count = 0
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    # total_Social_loss = 0

    valid_scene_count = 0  # 用来统计轨迹数大于 1 的场景数量

    input_traj_dict = {}
    target_traj_dict = {}

    # 遍历所有 batch
    for batch_idx, data in enumerate(tqdm(train_gen, total=len(train_gen), desc="Loading Data")):
        first_history_index = data['first_history_index']
        assert torch.unique(first_history_index).shape[0] == 1

        # 堆叠每个场景的输入和目标轨迹
        input_traj_dict = stack_trajectories_by_scene(data, 'input_x', input_traj_dict, device)
        target_traj_dict = stack_trajectories_by_scene(data, 'target_y', target_traj_dict, device)

    # 批次处理完成后，进行堆叠
    input_traj_dict_stacked = finalize_trajectory_stacking(input_traj_dict, device)
    target_traj_dict_stacked = finalize_trajectory_stacking(target_traj_dict, device)

    scene_count = 0  # 用于跟踪处理的场景数量

    with torch.set_grad_enabled(True):
        # 创建针对场景处理的进度条
        scene_loader = tqdm(input_traj_dict_stacked.keys(), desc="Processing Scenes")
        
        # 对每个场景进行处理
        for scene_name in scene_loader:

            input_traj = input_traj_dict_stacked[scene_name]
            target_traj = target_traj_dict_stacked[scene_name]

            # 如果场景中只有一条轨迹，则跳过该场景
            if input_traj.shape[0] <= 1:
                # print(f"Skipping scene {scene_name} as it has only one trajectory.")
                continue

            scene_count += 1  # 计数当前处理的场景
            
            # 每10个场景中取一个进行训练
            if scene_count % 10 != 0:
                continue

            valid_scene_count += 1

            # 获取当前场景的 batch_size
            current_batch_size = input_traj.shape[0]
            count += current_batch_size  # 这里根据实际的场景轨迹数量累加 count
            
            # 调用模型进行预测
            all_goal_traj, cvae_dec_traj, KLD_loss, _  = model(input_traj, map_mask=None, targets=target_traj, start_index=first_history_index, training=True)
            
            # 计算损失
            cvae_loss = cvae_multi(cvae_dec_traj, target_traj, first_history_index[0])
            goal_loss = criterion(all_goal_traj[:, first_history_index[0]:, :, :], target_traj[:, first_history_index[0]:, :, :])
            train_loss = goal_loss + cvae_loss + KLD_loss.mean() 
            
            # 累积损失
            total_goal_loss += goal_loss.item() * current_batch_size
            total_cvae_loss += cvae_loss.item() * current_batch_size
            total_KLD_loss += KLD_loss.mean().item() * current_batch_size
            # total_Social_loss += Social_loss.mean().item() * current_batch_size
            
            # 优化步骤
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # 在所有 batch 处理完后，计算平均损失
        total_goal_loss /= count
        total_cvae_loss /= count
        total_KLD_loss /= count
        # total_Social_loss /= count

    print(f"Total valid scenes: {valid_scene_count}")
    return total_goal_loss, total_cvae_loss, total_KLD_loss


def val(model, val_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    # total_Social_loss = 0
    count = 0
    model.eval()
    loader = tqdm(val_gen, total=len(val_gen))

    valid_scene_count = 0

    input_traj_dict = {}
    target_traj_dict = {}

    # 遍历所有 batch
    for batch_idx, data in enumerate(tqdm(val_gen, total=len(val_gen), desc="Loading Data")):
        first_history_index = data['first_history_index']
        assert torch.unique(first_history_index).shape[0] == 1

        # 堆叠每个场景的输入和目标轨迹
        input_traj_dict = stack_trajectories_by_scene(data, 'input_x', input_traj_dict, device)
        target_traj_dict = stack_trajectories_by_scene(data, 'target_y', target_traj_dict, device)

    # 批次处理完成后，进行堆叠
    input_traj_dict_stacked = finalize_trajectory_stacking(input_traj_dict, device)
    target_traj_dict_stacked = finalize_trajectory_stacking(target_traj_dict, device)

    scene_count = 0  # 用于跟踪处理的场景数量

    with torch.set_grad_enabled(False):
        # 创建针对场景处理的进度条
        scene_loader = tqdm(input_traj_dict_stacked.keys(), desc="Processing Scenes")
        
        # 对每个场景进行处理
        for scene_name in scene_loader:
            

            input_traj = input_traj_dict_stacked[scene_name]
            target_traj = target_traj_dict_stacked[scene_name]
            

            # 如果场景中只有一条轨迹，则跳过该场景
            if input_traj.shape[0] <= 1:
                # print(f"Skipping scene {scene_name} as it has only one trajectory.")
                continue
            
            scene_count += 1  # 计数当前处理的场景
            
            # 每10个场景中取一个进行训练
            if scene_count % 10 != 0:
                continue

            valid_scene_count += 1

            # 获取当前场景的 batch_size
            current_batch_size = input_traj.shape[0]
            count += current_batch_size  # 这里根据实际的场景轨迹数量累加 count
            
            # 调用模型进行预测
            all_goal_traj, cvae_dec_traj, KLD_loss, _= model(input_traj, map_mask=None, targets=None, start_index=first_history_index, training=False)
            
            # 计算损失
            cvae_loss = cvae_multi(cvae_dec_traj, target_traj, first_history_index[0])
            goal_loss = criterion(all_goal_traj[:, first_history_index[0]:, :, :], target_traj[:, first_history_index[0]:, :, :])

            # 累积损失
            total_goal_loss += goal_loss.item() * current_batch_size
            total_cvae_loss += cvae_loss.item() * current_batch_size
            total_KLD_loss += KLD_loss.mean().item() * current_batch_size
            # total_Social_loss += Social_loss.mean().item() * current_batch_size

        # 在所有 batch 处理完后，计算平均损失
        total_goal_loss /= count
        total_cvae_loss /= count
        total_KLD_loss /= count
        # total_Social_loss /= count

        val_loss = total_goal_loss + total_cvae_loss + total_KLD_loss

    print(f"Total valid scenes: {valid_scene_count}")
    
    return val_loss

def test(model, test_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    total_Social_loss = 0
    ADE_08 = 0
    ADE_12 = 0 
    FDE_08 = 0 
    FDE_12 = 0 
    count = 0
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))

    valid_scene_count = 0    

    input_traj_dict = {}
    target_traj_dict = {}

    # 遍历所有 batch
    for batch_idx, data in enumerate(tqdm(test_gen, total=len(test_gen), desc="Loading Data")):
        first_history_index = data['first_history_index']
        assert torch.unique(first_history_index).shape[0] == 1

        # 堆叠每个场景的输入和目标轨迹
        input_traj_dict = stack_trajectories_by_scene(data, 'input_x', input_traj_dict, device)
        target_traj_dict = stack_trajectories_by_scene(data, 'target_y', target_traj_dict, device)

    # 批次处理完成后，进行堆叠
    input_traj_dict_stacked = finalize_trajectory_stacking(input_traj_dict, device)
    target_traj_dict_stacked = finalize_trajectory_stacking(target_traj_dict, device)

    scene_count = 0

    with torch.set_grad_enabled(False):
        # 创建针对场景处理的进度条
        scene_loader = tqdm(input_traj_dict_stacked.keys(), desc="Processing Scenes")
        
        # 对每个场景进行处理
        for scene_name in scene_loader:
            

            input_traj = input_traj_dict_stacked[scene_name]
            target_traj = target_traj_dict_stacked[scene_name]

            # 如果场景中只有一条轨迹，则跳过该场景
            if input_traj.shape[0] <= 1:
                # print(f"Skipping scene {scene_name} as it has only one trajectory.")
                continue
            
            scene_count += 1  # 计数当前处理的场景
            
            # 每10个场景中取一个进行训练
            if scene_count % 10 != 0:
                continue
            
            valid_scene_count += 1

            # 获取当前场景的 batch_size
            current_batch_size = input_traj.shape[0]
            count += current_batch_size  # 累加 count
            
            # 调用模型进行预测
            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(input_traj, map_mask=None, targets=None, start_index=first_history_index, training=False)
            
            # 计算损失
            cvae_loss = cvae_multi(cvae_dec_traj, target_traj, first_history_index[0])
            goal_loss = criterion(all_goal_traj[:, first_history_index[0]:, :, :], target_traj[:, first_history_index[0]:, :, :])

            # 累积损失
            total_goal_loss += goal_loss.item() * current_batch_size
            total_cvae_loss += cvae_loss.item() * current_batch_size
            total_KLD_loss += KLD_loss.mean().item() * current_batch_size
            # total_Social_loss += Social_loss.mean().item() * current_batch_size

            # 将结果转为 numpy 进行评估
            cvae_dec_traj_np = cvae_dec_traj.to('cpu').numpy()
            all_goal_traj_np = all_goal_traj.to('cpu').numpy()
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()

            # 调用评估函数计算 ADE 和 FDE
            batch_results = eval_ethucy_cvae(input_traj_np, target_traj_np[:, -1, :, :], cvae_dec_traj_np[:, -1, :, :, :])
            ADE_08 += batch_results['ADE_08']
            ADE_12 += batch_results['ADE_12']
            FDE_08 += batch_results['FDE_08']
            FDE_12 += batch_results['FDE_12']

        # 计算平均损失和评估指标
        ADE_08 /= count
        ADE_12 /= count
        FDE_08 /= count
        FDE_12 /= count

        test_loss = total_goal_loss / count + total_cvae_loss / count + total_KLD_loss / count 

    # print("Test Loss %4f\n" % (test_loss))
    # print("ADE_08: %4f;  FDE_08: %4f;  ADE_12: %4f;   FDE_12: %4f\n" % (ADE_08, FDE_08, ADE_12, FDE_12))
    print(f"Total valid scenes: {valid_scene_count}")
    return test_loss, ADE_08, FDE_08, ADE_12, FDE_12



def test_save(model, test_gen, criterion, device, save_dir):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    # total_Social_loss = 0
    ADE_08 = 0
    ADE_12 = 0 
    FDE_08 = 0 
    FDE_12 = 0 
    count = 0

    scene_predictions = {}
    scene_ground_truths = {}
    scene_past_trajs = {}

    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))

    valid_scene_count = 0

    input_traj_dict = {}
    target_traj_dict = {}
    past_traj_dict = {}

    # Add variables to record timing
    total_inference_time = 0
    total_frames = 0

    # 遍历所有 batch
    for batch_idx, data in enumerate(loader):
        first_history_index = data['first_history_index']
        assert torch.unique(first_history_index).shape[0] == 1

        # 堆叠每个场景的输入和目标轨迹
        input_traj_dict = stack_trajectories_by_scene(data, 'input_x', input_traj_dict, device)
        target_traj_dict = stack_trajectories_by_scene(data, 'target_y', target_traj_dict, device)

    # 批次处理完成后，进行堆叠
    input_traj_dict_stacked = finalize_trajectory_stacking(input_traj_dict, device)
    target_traj_dict_stacked = finalize_trajectory_stacking(target_traj_dict, device)

    with torch.set_grad_enabled(False):
        # 对每个场景进行处理，添加进度条
        for scene_name in tqdm(input_traj_dict_stacked.keys(), desc="Processing scenes"):
            input_traj = input_traj_dict_stacked[scene_name]
            target_traj = target_traj_dict_stacked[scene_name]
            past_traj = input_traj_dict_stacked[scene_name]  # 获取当前场景的过去轨迹

            # # 如果场景中只有一条轨迹，则跳过该场景
            # if input_traj.shape[0] == 1:
            #     # print(f"Skipping scene {scene_name} as it has only one trajectory.")
            #     continue

            valid_scene_count += 1

            # 获取当前场景的 batch_size
            current_batch_size = input_traj.shape[0]
            count += current_batch_size  # 累加 count

            # Measure model inference time
            start_time = time.time()
            
            # 调用模型进行预测
            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(input_traj, map_mask=None, targets=None, start_index=first_history_index, training=False)
            
            end_time = time.time()
            
            # Calculate the inference time for this batch and accumulate it
            inference_time = end_time - start_time
            total_inference_time += inference_time
            total_frames += current_batch_size * input_traj.shape[1]  # Add the number of frames processed

            # 计算损失
            cvae_loss = cvae_multi(cvae_dec_traj, target_traj, first_history_index[0])
            goal_loss = criterion(all_goal_traj[:, first_history_index[0]:, :, :], target_traj[:, first_history_index[0]:, :, :])

            # 累积损失
            total_goal_loss += goal_loss.item() * current_batch_size
            total_cvae_loss += cvae_loss.item() * current_batch_size
            total_KLD_loss += KLD_loss.mean().item() * current_batch_size
            # total_Social_loss += Social_loss.mean().item() * current_batch_size

            # 将结果转为 numpy 进行评估
            cvae_dec_traj_np = cvae_dec_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()

            # 组织场景预测和真实值
            if scene_name not in scene_predictions:
                scene_predictions[scene_name] = []
                scene_ground_truths[scene_name] = []
                scene_past_trajs[scene_name] = []

            scene_predictions[scene_name].append(cvae_dec_traj_np)
            scene_ground_truths[scene_name].append(target_traj_np)
            scene_past_trajs[scene_name].append(past_traj.to('cpu').numpy())
            # print("scene_name", scene_name)
            # print("cvae_dec_traj_np.shape", cvae_dec_traj_np.shape)

            # 调用评估函数计算 ADE 和 FDE
            batch_results = eval_ethucy_cvae(input_traj.to('cpu').numpy(), target_traj_np[:, -1, :, :], cvae_dec_traj_np[:, -1, :, :])
            ADE_08 += batch_results['ADE_08']
            ADE_12 += batch_results['ADE_12']
            FDE_08 += batch_results['FDE_08']
            FDE_12 += batch_results['FDE_12']

    # 计算平均损失和评估指标
    ADE_08 /= count
    ADE_12 /= count
    FDE_08 /= count
    FDE_12 /= count

    test_loss = total_goal_loss / count + total_cvae_loss / count + total_KLD_loss / count 

    # Calculate the average inference time per frame
    avg_inference_time_per_frame = total_inference_time / total_frames
    print(f"Average inference time per frame: {avg_inference_time_per_frame:.6f} seconds")

    # 保存场景预测、真实值和过去轨迹
    for scene_name in scene_predictions:
        np.save(osp.join(save_dir, f'{scene_name}_predictions.npy'), np.array(scene_predictions[scene_name], dtype=object))
        np.save(osp.join(save_dir, f'{scene_name}_ground_truths.npy'), np.array(scene_ground_truths[scene_name], dtype=object))
        np.save(osp.join(save_dir, f'{scene_name}_past_trajectories.npy'), np.array(scene_past_trajs[scene_name], dtype=object))

    print(f"Predictions and ground truths saved to {save_dir} by scene")
    print(f"Total valid scenes: {valid_scene_count}")
    return test_loss, ADE_08, FDE_08, ADE_12, FDE_12

def visualize_trajectories(test_gen, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录

    scene_trajectories = {}

    with torch.set_grad_enabled(False):
        loader = tqdm(test_gen, total=len(test_gen))
        for batch_idx, data in enumerate(loader):
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1

            # 堆叠每个场景的输入和目标轨迹
            input_traj_dict_stacked = stack_trajectories_by_scene(data, 'input_x', device)
            target_traj_dict_stacked = stack_trajectories_by_scene(data, 'target_y', device)

            # 对每个场景进行处理
            for scene_name in input_traj_dict_stacked.keys():
                input_traj = input_traj_dict_stacked[scene_name]
                target_traj = target_traj_dict_stacked[scene_name]

                if scene_name not in scene_trajectories:
                    scene_trajectories[scene_name] = []

                # 收集轨迹数据
                scene_trajectories[scene_name].append((input_traj, target_traj))

    # 可视化每个场景的轨迹并保存图像
    for scene_name, trajectories in scene_trajectories.items():
        plt.figure(figsize=(10, 8))
        for _, (input_traj, target_traj) in enumerate(trajectories):
            # print("input_traj.shape = ",input_traj.shape)
            print("target_traj.shape = ",target_traj.shape)
            # Convert tensors to numpy arrays
            input_traj_np = input_traj.cpu().numpy()
            target_traj_np = target_traj.cpu().numpy()
            num_pedestrian = len(input_traj_np)
            # print("num_pedestrian: ", num_pedestrian)
            for p_idx in range(num_pedestrian):
                # print("p_idx : ", p_idx)
                # 绘制输入轨迹
                plt.plot(input_traj_np[p_idx, :, 0], input_traj_np[p_idx, :, 1], marker='o', color='blue', alpha=0.5, label='Input Trajectories' if scene_name == list(scene_trajectories.keys())[0] else "")
                # 绘制目标轨迹
                # target_traj_np[p_idx, -1, 0,0] = target_traj_np[p_idx, -1, 0,0] + input_traj_np[p_idx, -1, 0]
                # target_traj_np[p_idx, -1, 0,1] = target_traj_np[p_idx, -1, 0,1] + input_traj_np[p_idx, -1, 1]
                length_of_dec = target_traj.size(2)
                for dec_step in range(0,length_of_dec):
                    # print("dec_step",dec_step)
                    target_traj_np[p_idx, -1, dec_step,0] = input_traj_np[p_idx, -1, 0] + target_traj_np[p_idx, -1, dec_step,0]
                    target_traj_np[p_idx, -1, dec_step,1] =  input_traj_np[p_idx, -1, 1] + target_traj_np[p_idx, -1, dec_step,1]
                plt.plot(target_traj_np[p_idx, -1, :,0], target_traj_np[p_idx, -1, :, 1], marker='o', color='green', linestyle='-', label='Target Trajectories' if scene_name == list(scene_trajectories.keys())[0] else "")

        plt.title(f'Trajectories in Scene: {scene_name}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid()
        plt.axis('equal')

        # 保存图像
        save_path = os.path.join(save_dir, f'{scene_name}_trajectories.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f'Saved trajectory plot to {save_path}')

def evaluate(model, test_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    ADE_08 = 0
    ADE_12 = 0 
    FDE_08 = 0 
    FDE_12 = 0 
    count = 0
    all_file_name = []
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):            
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_traj_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)
            scene_name = data['scene_name'] 
            timestep = data['timestep']
            current_img = timestep
            #import pdb; pdb.set_trace()
            # filename = datapath + '/test/biwi_eth.txt'
            # data = pd.read_csv(filename, sep='\t', index_col=False, header=None)
            # data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
            # frame_id_min = data['frame_id'].min()
            # filename path = os.path.join(datapath, dataset ,str((current_img[1][0]+int(frame_id_min)//10)*10).zfill(5) + '.png')

            all_goal_traj, cvae_dec_traj, KLD_loss = model(input_traj, target_traj, first_history_index, False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)
            goal_loss = criterion(all_goal_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])
            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

            cvae_dec_traj_np = cvae_dec_traj.to('cpu').numpy()
            cvae_dec_traj = cvae_dec_traj.to('cpu').numpy()

            all_goal_traj_np = all_goal_traj.to('cpu').numpy()
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()
            #import pdb;pdb.set_trace()
            # Decoder
            # batch_MSE_15, batch_MSE_05, batch_MSE_10, batch_FMSE, batch_CMSE, batch_CFMSE, batch_FIOU =\
            #     eval_jaad_pie(input_traj_np, target_traj_np, all_dec_traj_np)
            
            batch_results =\
                eval_ethucy_cvae(input_traj_np, target_traj_np[:,-1,:,:], cvae_dec_traj[:,-1,:,:,:])
            ADE_08 += batch_results['ADE_08']
            ADE_12 += batch_results['ADE_12']
            FDE_08 += batch_results['FDE_08']
            FDE_12 += batch_results['FDE_12']

            if batch_idx == 0:
                all_input = input_traj_np
                all_target = target_traj_np
                all_prediction = cvae_dec_traj_np
            else:
                all_input = np.vstack((all_input,input_traj_np))
                all_target = np.vstack((all_target,target_traj_np))
                all_prediction = np.vstack((all_prediction,cvae_dec_traj_np))
            all_file_name.extend(current_img)

    ADE_08 /= count
    ADE_12 /= count
    FDE_08 /= count
    FDE_12 /= count
    
    print("ADE_08: %4f;  FDE_08: %4f;  ADE_12: %4f;   FDE_12: %4f\n" % (ADE_08, FDE_08, ADE_12, FDE_12))

    return all_input,all_target,all_prediction,all_file_name

def weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.001)
    elif isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
