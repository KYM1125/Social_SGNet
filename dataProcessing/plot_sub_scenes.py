'''
    此程序绘制出每个子场景的静态图
'''
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# 提取文件名中的中心点坐标
def extract_center_point_from_filename(filename):
    match = re.search(r'(\d+\.\d+)_(\d+\.\d+)_predictions.npy', filename)
    if match:
        center_x, center_y = float(match.group(1)), float(match.group(2))
        return center_x, center_y
    return None

# 获取所有符合场景ID的文件
def get_scene_files(base_dir, target_scene_id):
    scene_files = []
    for file_name in os.listdir(base_dir):
        # 使用正则表达式匹配以 test_<sceneID>_<subsceneID> 格式开头的文件名
        match = re.match(rf'dair_infra_interpolated_train_{target_scene_id}_(\d+)_', file_name)
        if match and '_predictions.npy' in file_name:
            scene_files.append(file_name)
    return scene_files

# 绘制每个子场景的轨迹
def plot_trajectory_for_each_subscene(scene_files, base_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for file_name in scene_files:
        # 提取子场景ID和中心点坐标
        subscene_id = file_name.split('_')[5]
        center_point = extract_center_point_from_filename(file_name)
        if center_point is None:
            continue
        
        center_x, center_y = center_point
        print(f"Processing file: {file_name}, subscene_id: {subscene_id}, center_point: {center_point}")

        # 创建新的图像
        plt.figure(figsize=(10, 6))
        
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
                plt.plot(past_dec[:, 0], past_dec[:, 1], marker='o', linestyle='-', color='b', alpha=1, label='Observation Steps' if batch_idx == 0 and p_idx == 0 else "")
                # 获取过去轨迹的最后一个点
                last_past_point = past_dec[-1]

                # 解码轨迹并还原
                traj_dec = ground_truths[batch_idx, p_idx, -1, :, :]
                target_traj_state = np.zeros((12, 2))
                start_position = past_dec[-1]  # 获取过去轨迹的最后位置
                for dec_step in range(len(traj_dec)):
                    target_traj_state[dec_step, 0] = start_position[0] + traj_dec[dec_step, 0]
                    target_traj_state[dec_step, 1] = start_position[1] + traj_dec[dec_step, 1]

                # 连接过去轨迹的最后一个点到解码轨迹的第一个点
                plt.plot([last_past_point[0], target_traj_state[0, 0]], [last_past_point[1], target_traj_state[0, 1]], color='r', linestyle='--', alpha=1)
                plt.plot(target_traj_state[:, 0], target_traj_state[:, 1], marker='o', linestyle='-', color='r', alpha=1, label='Prediction Steps' if batch_idx == 0 and p_idx == 0 else "")

                # plt.plot(target_traj_state[:, 0], target_traj_state[:, 1], marker='o', linestyle='-', color='r', alpha=0.5, label='Prediction Steps' if batch_idx == 0 and p_idx == 0 else "")

                # 还原并绘制预测轨迹
                traj_dec_pred = predictions[batch_idx, p_idx, -1, :, :, :]
                traj_dec_pred_state = np.zeros((12, 20, 2))
                for k in range(traj_dec_pred.shape[1]):  # 遍历每条预测轨迹
                    for pred_step in range(traj_dec_pred.shape[0]):
                        traj_dec_pred_state[pred_step, k, 0] = start_position[0] + traj_dec_pred[pred_step, k, 0]
                        traj_dec_pred_state[pred_step, k, 1] = start_position[1] + traj_dec_pred[pred_step, k, 1]

                # 连接过去轨迹的最后一个点到预测轨迹的第一个点
                for i in range(traj_dec_pred_state.shape[1]):  # 遍历每条预测轨迹
                    plt.plot([last_past_point[0], traj_dec_pred_state[0, i, 0]], [last_past_point[1], traj_dec_pred_state[0, i, 1]], marker='o',linestyle='--', color='green', alpha=0.1)
                    plt.plot(traj_dec_pred_state[:, i, 0], traj_dec_pred_state[:, i, 1], marker='o',linestyle='--', color='green', alpha=0.1)
                
                # for i in range(traj_dec_pred_state.shape[1]):  # 遍历每条预测轨迹
                #     plt.plot(traj_dec_pred_state[:, i, 0], traj_dec_pred_state[:, i, 1], linestyle='--', color='green', alpha=0.3)

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Subscene {subscene_id}')
        plt.legend()

        # 保存当前子场景的图像
        save_path = os.path.join(save_dir, f"social_{file_name}_trajectory_plot.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved trajectory plot for subscene {file_name} to {save_path}")

# 基础路径和保存路径
base_dir = '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/tools/ethucy/checkpoints/DAIR/Social_SGNet_CVAE/0.5/1/visual/'
save_dir = 'each_traj/'

# 获取所有包含场景 42440 的文件
scene_files = get_scene_files(base_dir, "42440")

# 绘制每个子场景
plot_trajectory_for_each_subscene(scene_files, base_dir, save_dir)
