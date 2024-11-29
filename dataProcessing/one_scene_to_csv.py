'''
    将每个场景转换成csv文件
'''
import os
import re
import numpy as np
import csv

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
        if re.match(rf'dair_infra_interpolated_train_{target_scene_id}_(\d+)_', file_name) and '_predictions.npy' in file_name:
            scene_files.append(file_name)
    return scene_files

# 将所有子场景的轨迹数据保存到CSV文件中
def save_trajectory_data_to_csv(scene_files, base_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "trajectory_data_14841_prediction_k_.csv")
    
    pedestrian_counter = 0  # 初始化行人计数器

    # 打开CSV文件写入数据
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # 写入表头
        headers = ["city", "timestamp", "id", "type", "sub_type", "tag", 
                   "x", "y", "z", "length", "width", "height", 
                   "theta", "v_x", "v_y", "intersect_id"]
        
        # 添加多模态预测轨迹列
        num_modes = 20  # 假设多模态预测有20个模式
        for k in range(num_modes):
            headers.extend([f"x{k+1}", f"y{k+1}"])

        writer.writerow(headers)
        
        for file_idx, file_name in enumerate(scene_files):
            center_point = extract_center_point_from_filename(file_name)
            if center_point is None:
                continue
            center_x, center_y = center_point

            # 加载预测、ground truth和过去的轨迹数据
            predictions_file = os.path.join(base_dir, file_name)
            ground_truths_file = predictions_file.replace('_predictions.npy', '_ground_truths.npy')
            past_trajectories_file = predictions_file.replace('_predictions.npy', '_past_trajectories.npy')

            predictions = np.load(predictions_file, allow_pickle=True)
            ground_truths = np.load(ground_truths_file, allow_pickle=True)
            past_trajectories = np.load(past_trajectories_file, allow_pickle=True)
            
            batch_size = ground_truths.shape[0]
            for batch_idx in range(batch_size):
                pedestrian_num = len(past_trajectories[batch_idx])
                for p_idx in range(pedestrian_num):
                    pedestrian_id = pedestrian_counter  # 使用计数器的值作为行人ID
                    pedestrian_counter += 1  # 每次递增计数器

                    # 处理过去的轨迹（观察步骤）
                    past_dec = past_trajectories[batch_idx, p_idx][:, :2] + [center_x, center_y]
                    for idx, (x, y) in enumerate(past_dec):
                        writer.writerow(["PEK", idx, pedestrian_id, None, "PEDESTRIAN", None, 
                                         x, y, 0, None, None, None, None, None, None, None] + [None] * (num_modes * 2))
                    
                    # 处理ground truth的未来轨迹（预测目标）
                    traj_dec = ground_truths[batch_idx, p_idx, -1, :, :]
                    traj_dec_pred = predictions[batch_idx, p_idx, -1, :, :, :]
                    start_position = past_dec[-1]  # 以过去轨迹的最后一个位置作为未来预测的起始点
                    for dec_step in range(traj_dec.shape[0]):
                        multi_modal_coords = []
                        for k in range(traj_dec_pred.shape[1]):  # 遍历每条多模预测轨迹
                            x_pred = start_position[0] + traj_dec_pred[dec_step, k, 0]
                            y_pred = start_position[1] + traj_dec_pred[dec_step, k, 1]
                            multi_modal_coords.extend([x_pred, y_pred])  # 将每个模式的x和y坐标加入列表
                        x = start_position[0] + traj_dec[dec_step, 0]
                        y = start_position[1] + traj_dec[dec_step, 1]
                        # writer.writerow(["PEK", len(past_dec) + dec_step, pedestrian_id, None, "PEDESTRIAN", None, 
                        #                  x, y, 0, None, None, None, None, None, None, None] + [None] * (num_modes * 2))
                        writer.writerow(["PEK", len(past_dec) + dec_step, pedestrian_id, None, "PEDESTRIAN", None, 
                                         x, y, 0, None, None, None, None, None, None, None] + multi_modal_coords)


    print(f"轨迹数据已保存到 {csv_path}")

# 基础路径和保存路径
base_dir = '../../SGNet/SGNet/SGNet/SGNet.pytorch/tools/ethucy/checkpoints/DAIR/Social_SGNet_CVAE/0.5/1/visual/'
save_dir = 'csv/'

# 获取所有包含场景 14841 的文件
scene_files = get_scene_files(base_dir, "14841")

# 保存轨迹数据到CSV
save_trajectory_data_to_csv(scene_files, base_dir, save_dir)
