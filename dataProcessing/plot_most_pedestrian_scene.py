'''
    此程序绘制包含最多行人的场景的静态图
'''
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 提取文件名中的中心点坐标
def extract_center_point_from_filename(filename):
    match = re.search(r'(\d+\.\d+)_(\d+\.\d+)_predictions.npy', filename)
    if match:
        center_x, center_y = float(match.group(1)), float(match.group(2))
        return center_x, center_y
    return None

# 统计每个场景的行人数量
def count_pedestrians_in_scenes(base_dir):
    scene_counts = defaultdict(int)
    scene_files = defaultdict(list)

    for file_name in os.listdir(base_dir):
        match = re.match(r'dair_infra_interpolated_train_(\d+)_(\d+)_', file_name)
        if match and '_predictions.npy' in file_name:
            scene_id = match.group(1)
            scene_files[scene_id].append(file_name)
            
            # 加载对应的 past 轨迹文件来统计行人数量
            past_trajectories_file = os.path.join(base_dir, file_name.replace('_predictions.npy', '_past_trajectories.npy'))
            if os.path.exists(past_trajectories_file):
                past_trajectories = np.load(past_trajectories_file, allow_pickle=True)
                pedestrian_count = sum(len(batch) for batch in past_trajectories)
                scene_counts[scene_id] += pedestrian_count

    # 获取行人数量最多的场景ID
    max_scene_id = max(scene_counts, key=scene_counts.get)
    print(f"Scene with the most pedestrians: {max_scene_id} ({scene_counts[max_scene_id]} pedestrians)")
    return max_scene_id, scene_files[max_scene_id]

# 绘制并还原最多行人的场景
def plot_trajectory_for_all_subscenes(scene_files, base_dir, save_dir, max_scene_id):
    plt.figure(figsize=(10, 6))
    os.makedirs(save_dir, exist_ok=True)

    for file_name in scene_files:
        # 提取中心点坐标
        center_point = extract_center_point_from_filename(file_name)
        if center_point is None:
            continue
        
        center_x, center_y = center_point
        print(f"Processing file: {file_name}, center_point: {center_point}")

        # 加载该子场景的预测、ground truth 和 past 轨迹
        predictions_file = os.path.join(base_dir, file_name)
        ground_truths_file = predictions_file.replace('_predictions.npy', '_ground_truths.npy')
        past_trajectories_file = predictions_file.replace('_predictions.npy', '_past_trajectories.npy')

        predictions = np.load(predictions_file, allow_pickle=True)
        ground_truths = np.load(ground_truths_file, allow_pickle=True)
        past_trajectories = np.load(past_trajectories_file, allow_pickle=True)

        # 遍历所有 batch 的数据
        batch_size = ground_truths.shape[0]
        for batch_idx in range(batch_size):
            pedestrian_num = len(past_trajectories[batch_idx])
            for p_idx in range(pedestrian_num):
                # 提取并还原过去的轨迹
                past_dec = past_trajectories[batch_idx, p_idx][:, :2] + [center_x, center_y]
                plt.plot(past_dec[:, 0], past_dec[:, 1], marker='o', linestyle='-', color='b', alpha=0.5, label='Observation Steps' if batch_idx == 0 and p_idx == 0 else "")

                # 解码轨迹并还原
                traj_dec = ground_truths[batch_idx, p_idx, -1, :, :]
                target_traj_state = np.zeros((12, 2))
                start_position = past_dec[-1]  # 获取过去轨迹的最后位置
                for dec_step in range(len(traj_dec)):
                    target_traj_state[dec_step, 0] = start_position[0] + traj_dec[dec_step, 0]
                    target_traj_state[dec_step, 1] = start_position[1] + traj_dec[dec_step, 1]
                
                plt.plot(target_traj_state[:, 0], target_traj_state[:, 1], marker='o', linestyle='-', color='r', alpha=0.5, label='Prediction Steps' if batch_idx == 0 and p_idx == 0 else "")

                # 还原并绘制预测轨迹
                traj_dec_pred = predictions[batch_idx, p_idx, -1, :, :, :]
                traj_dec_pred_state = np.zeros((12, 20, 2))
                for k in range(traj_dec_pred.shape[1]):  # 遍历每条预测轨迹
                    for pred_step in range(traj_dec_pred.shape[0]):
                        traj_dec_pred_state[pred_step, k, 0] = start_position[0] + traj_dec_pred[pred_step, k, 0]
                        traj_dec_pred_state[pred_step, k, 1] = start_position[1] + traj_dec_pred[pred_step, k, 1]

                for i in range(traj_dec_pred_state.shape[1]):  # 遍历每条预测轨迹
                    plt.plot(traj_dec_pred_state[:, i, 0], traj_dec_pred_state[:, i, 1], linestyle='--', color='green', alpha=0.3)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Predicted Trajectories for All Pedestrians in Scene {max_scene_id}')
    plt.legend()

    # 保存图像
    save_path = os.path.join(save_dir, f"all_subscenes_{max_scene_id}_no_social.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot for all subscenes to {save_path}")

# 基础路径和保存路径
base_dir = '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/tools/ethucy/checkpoints/DAIR/SGNet_CVAE/0.5/1/visual/'
save_dir = 'each_traj/'

# 获取行人最多的场景ID和对应的文件列表
max_scene_id, scene_files = count_pedestrians_in_scenes(base_dir)

# 绘制该场景的所有子场景
plot_trajectory_for_all_subscenes(scene_files, base_dir, save_dir, max_scene_id)
