import os
import numpy as np
import pandas as pd
import dill
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import sys
from sklearn.cluster import KMeans

sys.path.append("../../trajectron")
from environment import Environment, MyScene, Node
from utils import maybe_makedirs
from environment import derivative_of

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
dt = 0.1

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = MyScene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        data_dict = {('position', 'x'): x,
                     ('position', 'y'): y,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay}

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug

def process_file(subdir, file, data_class, env):
    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
    input_data_dict = dict()
    full_data_path = os.path.join(subdir, file)

    data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None, dtype={2: np.float64, 3: np.float64})
    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')
    data.sort_values('frame_id', inplace=True)

    max_timesteps = data['frame_id'].max()
    unique_track_ids = data['track_id'].unique()
    individual_trajectories = []
    
    # 筛选轨迹(每帧为有效帧)
    # valid_track_ids = []
    # for track_id in unique_track_ids:
    #     traj = data[data['track_id'] == track_id].values
    #     if 50 + 12 in traj[:, 0]:
    #         frame_indices = np.where(traj[:, 0] == 50)[0]
    #         if frame_indices.size > 0:  
    #             current_index = frame_indices[0]
    #         if current_index >= 7 and current_index + 12 < len(traj):
    #             individual_trajectories.append(traj[current_index - 7:current_index + 13, 2:4])
    #             valid_track_ids.append(track_id)

    # 筛选轨迹(每4帧为一帧有效帧)
    valid_track_ids = []
    for track_id in unique_track_ids:
        traj = data[data['track_id'] == track_id].values
        if 50 + 12 * 4 in traj[:, 0]:  # 保证有足够的未来帧
            frame_indices = np.where(traj[:, 0] == 50)[0]
            if frame_indices.size > 0:  
                current_index = frame_indices[0]
                if current_index >= 7 * 4 and current_index + 12 * 4 < len(traj):
                    # 每4帧取一帧
                    downsampled_traj = traj[current_index - 7 * 4:current_index + 13 * 4:4, 2:4]
                    individual_trajectories.append(downsampled_traj)
                    valid_track_ids.append(track_id)
    print("len(valid_track_ids) = ",len(valid_track_ids))


    # 计算质心
    centroids = [np.mean(traj, axis=0) for traj in individual_trajectories]
    sse = []
    for k in range(2, 10 + 1):
        kmeans = KMeans(n_clusters=k)
        # 检查质心数量是否足够用于当前的 k 值聚类
        if len(centroids) < k:
            continue
        kmeans.fit(centroids)
        sse.append(kmeans.inertia_)

    # 检查是否成功计算 SSE
    if len(sse) >= 2:
        # 计算 SSE 的斜率变化
        slopes = np.diff(sse)
        slope_changes = np.diff(slopes)
        
        # 确保 slope_changes 存在
        if len(slope_changes) > 0:
            optimal_k = np.argmax(np.abs(slope_changes)) + 2 + 1  # 根据变化找到最佳的 k 值

            while optimal_k >= 2:
                if len(centroids) >= optimal_k:
                    # 进行当前 optimal_k 的聚类
                    kmeans = KMeans(n_clusters=optimal_k)
                    labels = kmeans.fit_predict(centroids)
                    break  # 找到合适的聚类数后退出循环
                else:
                    optimal_k -= 1  # 如果质心不足，尝试减少聚类数

            # 如果 optimal_k 仍然小于 2，放弃分类
            if optimal_k < 2:
                labels = np.zeros(len(centroids), dtype=int)  # 所有质心都分配到 0 类
        else:      
            distances = [
                np.linalg.norm(centroids[0] - centroids[1]),
                np.linalg.norm(centroids[0] - centroids[2]),
                np.linalg.norm(centroids[1] - centroids[2])
            ]

            if all(dist < 10 for dist in distances):
                labels = np.zeros(len(centroids), dtype=int)  # 所有质心分配到同一类
            elif any(dist < 10 for dist in distances) and not all(dist < 10 for dist in distances):
                kmeans = KMeans(n_clusters=2)
                labels = kmeans.fit_predict(centroids)  # 直接使用 KMeans 将三个轨迹分成两类
            else:
                kmeans = KMeans(n_clusters=3)
                labels = kmeans.fit_predict(centroids)  # 直接使用 KMeans 将三个轨迹分成三类
    else:
        if len(sse) == 0:
            labels = np.zeros(len(centroids), dtype=int)  # 所有质心都分配到 0 类
        elif len(sse) == 1:
            # 计算质心之间的距离
            distance = np.linalg.norm(centroids[0] - centroids[1])
            if distance < 10:
                labels = np.zeros(len(centroids), dtype=int)  # 所有质心分配到同一类
            else:
                # 进行当前 optimal_k 的聚类
                kmeans = KMeans(n_clusters=2)
                labels = kmeans.fit_predict(centroids)

    scenes = []
    # 初始化计数器
    skipped_count = 0
    valid_count = 0
    for label in np.unique(labels):
        scene_name = f"{desired_source}_{data_class}_{file.split('.')[0]}_{label}"
        cluster_indices = np.where(labels == label)[0]
        cluster_mean = np.mean(np.array([centroids[i] for i in cluster_indices]), axis=0)
        scene_name += f"_{cluster_mean[0]:.9f}_{cluster_mean[1]:.9f}"  # 将聚类中心添加到场景名称

        scene = MyScene(timesteps=max_timesteps + 1, dt=dt*4, name=scene_name) # 每4帧为一帧有效帧（如果需要每帧都为有效帧则去掉*4）

        for track_id in valid_track_ids:
            traj_label = labels[valid_track_ids.index(track_id)]
            if traj_label == label:
                node_df = data[data['track_id'] == track_id]
                # frame_diff = np.diff(node_df['frame_id'])
                # if np.any(frame_diff != 4):  # Check if frame difference is not 4
                #     skipped_count += 1
                #     print(f"Non-continuous frames detected for track_id {track_id}, skipping this trajectory.")
                #     continue  # Skip this trajectory


                # # assert np.all(np.diff(node_df['frame_id']) == 1)
                # print(f"Non-continuous frames detected for track_id {track_id}, skipping this trajectory.")
                # continue  # Skip this iteration if frames are non-continuous
                node_values = node_df[['pos_x', 'pos_y']].values

                new_first_idx = node_df['frame_id'].iloc[0]
                x = node_values[:, 0] - cluster_mean[0]  # 以聚类中心进行标准化
                y = node_values[:, 1] - cluster_mean[1]

                vx = derivative_of(x, scene.dt)
                vy = derivative_of(y, scene.dt)
                ax = derivative_of(vx, scene.dt)
                ay = derivative_of(vy, scene.dt)

                data_dict = {('position', 'x'): x,
                             ('position', 'y'): y,
                             ('velocity', 'x'): vx,
                             ('velocity', 'y'): vy,
                             ('acceleration', 'x'): ax,
                             ('acceleration', 'y'): ay}

                node_data = pd.DataFrame(data_dict, columns=data_columns)
                node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=track_id, data=node_data)
                node.first_timestep = new_first_idx

                scene.nodes.append(node)
                valid_count += 1  # 增加有效的轨迹数

        scenes.append(scene)


    # 最后打印或返回跳过和有效的轨迹数量
    print(f"Total skipped trajectories: {skipped_count}")
    print(f"Total valid trajectories: {valid_count}")
    return scenes


def process_data_class(desired_source, data_class):
    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    env.attention_radius = attention_radius

    scenes = []
    data_dict_path = os.path.join('../processed', '_'.join([desired_source, data_class]) + '.pkl')


    # 自动查找该文件夹下所有以 .txt 结尾的文件
    specific_dir = os.path.join('raw', desired_source, data_class)
    specific_scene_files = [f for f in os.listdir(specific_dir) if f.endswith('.txt')]
    file_paths = [(specific_dir, scene) for scene in specific_scene_files]

    # 使用多进程处理文件，并显示进度条
    with mp.Pool(mp.cpu_count()) as pool:
        
        process_func = partial(process_file, data_class=data_class, env=env)

        for scenes_list in tqdm(pool.starmap(process_func, file_paths), total=len(file_paths), desc=f"Processing {data_class}"):

            if len(scenes_list) > 0:
                scenes.extend(scenes_list)  # 合并当前文件的所有场景

    env.scenes = scenes


    if len(scenes) > 0:
        with open(data_dict_path, 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    maybe_makedirs('../processed')
    for desired_source in ['dair_infra']:
        for data_class in ['interpolated_train', 'interpolated_val', 'interpolated_test']:
            process_data_class(desired_source, data_class)
