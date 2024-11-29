import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import umap

def calculate_curvature(x, y):
    """计算曲线的曲率"""
    n = len(x)
    if n < 3:
        return np.array([np.nan])  # 返回一个包含NaN的数组

    # 计算曲率的近似值
    curvature = np.zeros(n)
    for i in range(1, n-1):
        x1, x2, x3 = x[i-1], x[i], x[i+1]
        y1, y2, y3 = y[i-1], y[i], y[i+1]

        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x3 - x2, y3 - y2
        d2x = dx2 - dx1
        d2y = dy2 - dy1
        denominator = (dx1**2 + dy1**2) * (dx2**2 + dy2**2)
        if denominator == 0:
            curvature[i] = 0
        else:
            curvature[i] = abs(d2x * dy1 - d2y * dx1) / denominator**0.5
    return curvature[1:-1] if n > 2 else np.array([np.nan])  # 返回中间部分的曲率或NaN


def calculate_angle(x1, y1, x2, y2):
    """计算从点(x1, y1)到点(x2, y2)的朝向角度（以度为单位）"""
    return np.arctan2(y2 - y1, x2 - x1) * (180 / np.pi)

def calculate_angle_change(start_angle, end_angle):
    """计算角度变化，考虑正负角度"""
    angle_change = end_angle - start_angle
    if angle_change > 180:
        angle_change -= 360
    elif angle_change < -180:
        angle_change += 360
    return angle_change

def calculate_average_acceleration(x, y, time):
    """计算轨迹的平均加速度"""
    velocities = []
    accelerations = []

    # 计算速度
    for i in range(1, len(time)):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        dt = (time[i] - time[i-1]) / 10.0
        if dt > 0:
            velocities.append(np.sqrt(dx**2 + dy**2) / dt)
    
    # 计算加速度
    for i in range(1, len(velocities)):
        dv = velocities[i] - velocities[i-1]
        dt = (time[i] - time[i-1]) / 10.0
        if dt > 0:
            accelerations.append(dv / dt)

    avg_acceleration = np.mean(accelerations) if accelerations else np.nan
    return avg_acceleration

# def extract_features(group):
#     start_x, start_y = group.iloc[0][['x', 'y']]
#     end_x, end_y = group.iloc[-1][['x', 'y']]
#     total_displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)  # 特征1：总位移
#     total_time = group['time'].max() - group['time'].min()
#     # avg_speed = total_displacement / total_time if total_time > 0 else np.nan  # 特征2：平均速度
#     avg_acceleration = calculate_average_acceleration(group['x'].values, group['y'].values, group['time'].values)  # 特征3：平均加速度
    
#     # 计算曲率
#     curvature = calculate_curvature(group['x'].values, group['y'].values)
#     avg_curvature = np.nan if np.all(np.isnan(curvature)) else np.mean(curvature)  # 特征4：平均曲率
    
#     # 计算角度变化
#     if len(group) >= 2:
#         start_x1, start_y1 = group.iloc[0][['x', 'y']]
#         start_x2, start_y2 = group.iloc[1][['x', 'y']]
#         start_angle = calculate_angle(start_x1, start_y1, start_x2, start_y2)
        
#         end_x1, end_y1 = group.iloc[-2][['x', 'y']]
#         end_x2, end_y2 = group.iloc[-1][['x', 'y']]
#         end_angle = calculate_angle(end_x1, end_y1, end_x2, end_y2)
        
#         angle_change = calculate_angle_change(start_angle, end_angle)
#     else:
#         angle_change = np.nan  # 特征5：角度变化
    
#     return [avg_curvature,angle_change,total_displacement,total_time,avg_acceleration]
def extract_features(group):
    start_x, start_y = group.iloc[0][['x', 'y']]
    end_x, end_y = group.iloc[-1][['x', 'y']]
    total_displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    total_time = group['time'].max() - group['time'].min()
    avg_acceleration = calculate_average_acceleration(group['x'].values, group['y'].values, group['time'].values)
    
    curvature = calculate_curvature(group['x'].values, group['y'].values)
    avg_curvature = np.nan if np.all(np.isnan(curvature)) else np.mean(curvature)
    
    if len(group) >= 2:
        start_x1, start_y1 = group.iloc[0][['x', 'y']]
        start_x2, start_y2 = group.iloc[1][['x', 'y']]
        start_angle = calculate_angle(start_x1, start_y1, start_x2, start_y2)
        
        end_x1, end_y1 = group.iloc[-2][['x', 'y']]
        end_x2, end_y2 = group.iloc[-1][['x', 'y']]
        end_angle = calculate_angle(end_x1, end_y1, end_x2, end_y2)
        
        angle_change = calculate_angle_change(start_angle, end_angle)
    else:
        angle_change = np.nan

    # 计算速度的变化率
    velocities = [np.sqrt((group['x'].iloc[i] - group['x'].iloc[i-1])**2 + (group['y'].iloc[i] - group['y'].iloc[i-1])**2) / (group['time'].iloc[i] - group['time'].iloc[i-1]) for i in range(1, len(group))]
    velocity_change_rate = np.mean(np.diff(velocities)) if len(velocities) > 1 else np.nan
    
    return [angle_change, total_time, avg_acceleration]
    # return [avg_curvature, angle_change, total_displacement, total_time, avg_acceleration, velocity_change_rate]


def plot_trajectories_by_cluster(folder_path, clustered_data, sample_size=50):
    all_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path, delimiter='\t', header=None, names=['time', 'person_id', 'x', 'y'])
            data['person_id'] = data['person_id'].apply(lambda x: f"{file_name}_{int(x)}")  # 确保person_id唯一
            all_data.append(data)
    
    combined_data = pd.concat(all_data, ignore_index=True)  # 合并所有数据，忽略原始索引
    
    # 将聚类结果映射到原始数据
    combined_data['Cluster'] = combined_data['person_id'].map(clustered_data.set_index('person_id')['Cluster'])

    plt.figure(figsize=(12, 10))

    unique_clusters = clustered_data['Cluster'].unique()
    colors = plt.cm.get_cmap('viridis', len(unique_clusters))  # 获取与聚类数目相同的颜色

    for cluster_id in unique_clusters:
        cluster_data = combined_data[combined_data['Cluster'] == cluster_id]
        
        # 获取聚类内的颜色
        cluster_color = colors(cluster_id)
        
        # 对每个聚类的数据进行采样
        sampled_data = cluster_data.groupby('person_id').apply(lambda x: x.sample(min(sample_size, len(x))))
        sampled_data = sampled_data.reset_index(drop=True)  # 重置索引以避免歧义

        for person_id, person_data in sampled_data.groupby('person_id'):
            # 绘制每个人的轨迹，用线连接
            plt.plot(person_data['x'], person_data['y'], marker='o', linestyle='-', color=cluster_color, alpha=0.7, label=f'Person {person_id}' if person_id in sampled_data['person_id'].unique() else "")

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectories Colored by Cluster')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()

# def analyze_trajectory_data(folder_path, n_clusters=3):
#     all_data = []

#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.txt'):
#             file_path = os.path.join(folder_path, file_name)
#             data = pd.read_csv(file_path, delimiter='\t', header=None, names=['time', 'person_id', 'x', 'y'])
#             data['person_id'] = data['person_id'].apply(lambda x: f"{file_name}_{int(x)}")
#             all_data.append(data)
    
#     combined_data = pd.concat(all_data, ignore_index=True)
    
#     grouped = combined_data.groupby('person_id')
#     features = [extract_features(group) for person_id, group in grouped]
    
#     # features_df = pd.DataFrame(features, columns=['Total Displacement', 'Average Acceleration', 'Average Curvature', 'Angle Change'])
#     features_df = pd.DataFrame(features, columns=['Average Curvature','Angle Change','Total Displacement','Total Time','Average Acceleration','Velocity Change Rate'])
#     features_df['person_id'] = [person_id for person_id, group in grouped]
    
#     features_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # 替换正负无穷
#     features_df.dropna(inplace=True)  # 删除缺失值

#     scaler = StandardScaler()  # 标准化
#     features_scaled = scaler.fit_transform(features_df.drop('person_id', axis=1))  # 删除person_id列
    
#     kmeans = KMeans(n_clusters=n_clusters, random_state=2024)  # K-means聚类，随机种子为2024
#     labels = kmeans.fit_predict(features_scaled)
    
#     features_df['Cluster'] = labels
    
#     pca = PCA(n_components=2)  # 主成分分析
#     pca_result = pca.fit_transform(features_scaled)
    
#     plt.figure(figsize=(10, 6))
#     plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis')
#     plt.colorbar(label='Cluster')
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.title('K-means Clustering of Pedestrian Trajectories')
#     plt.grid(True)
#     plt.show()
    
#     for cluster_id in range(n_clusters):
#         print(f"Cluster {cluster_id}: {sum(labels == cluster_id)} samples")
    
#     return features_df
def analyze_trajectory_data(folder_path, n_clusters=3):
    all_data = []

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            # 读取数据文件，分隔符为制表符
            data = pd.read_csv(file_path, delimiter='\t', header=None, names=['time', 'person_id', 'x', 'y'])
            # 修改 person_id 以避免 ID 冲突
            data['person_id'] = data['person_id'].apply(lambda x: f"{file_name}_{int(x)}")
            all_data.append(data)
    
    # 合并所有数据
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 按 person_id 分组，并提取特征
    grouped = combined_data.groupby('person_id')
    features = [extract_features(group) for person_id, group in grouped]
    
    # 创建特征 DataFrame
    features_df = pd.DataFrame(features, columns=['Angle Change', 'Total Time', 'Average Acceleration'])
    features_df['person_id'] = [person_id for person_id, group in grouped]
    
    # 处理无穷大和缺失值
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.dropna(inplace=True)

    # 数据标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.drop('person_id', axis=1))
    
    # K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=2024)
    labels = kmeans.fit_predict(features_scaled)
    
    
    
    # 将聚类标签添加到特征 DataFrame
    features_df['Cluster'] = labels
    
    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=2024)
    tsne_result = tsne.fit_transform(features_scaled)
    
    # 绘制 t-SNE 结果图
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
    plt.colorbar(label='Cluster')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of K-means Clustering of Pedestrian Trajectories')
    plt.grid(True)
    plt.show()

    # 计算 Silhouette Score
    silhouette_avg = silhouette_score(features_scaled, labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")

    # 打印每个聚类中的样本数量
    for cluster_id in range(n_clusters):
        print(f"Cluster {cluster_id}: {sum(labels == cluster_id)} samples")
    
    return features_df

def analyze_trajectory_data_with_umap(folder_path, n_clusters=3):
    all_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path, delimiter='\t', header=None, names=['time', 'person_id', 'x', 'y'])
            data['person_id'] = data['person_id'].apply(lambda x: f"{file_name}_{int(x)}")
            all_data.append(data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    grouped = combined_data.groupby('person_id')
    features = [extract_features(group) for person_id, group in grouped]
    
    features_df = pd.DataFrame(features, columns=['Average Curvature', 'Angle Change', 'Total Displacement', 'Total Time', 'Average Acceleration', 'Velocity Change Rate'])
    features_df['person_id'] = [person_id for person_id, group in grouped]
    
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.dropna(inplace=True)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.drop('person_id', axis=1))
    
    # 尝试UMAP降维
    umap_model = umap.UMAP(n_components=2, random_state=2024)
    umap_result = umap_model.fit_transform(features_scaled)
    
    # 使用GMM进行聚类
    gmm = GaussianMixture(n_components=n_clusters, random_state=2024)
    labels = gmm.fit_predict(features_scaled)
    
    features_df['Cluster'] = labels
    
    plt.figure(figsize=(10, 6))
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='viridis')
    plt.colorbar(label='Cluster')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP Visualization of GMM Clustering of Pedestrian Trajectories')
    plt.grid(True)
    plt.show()
    
    silhouette_avg = silhouette_score(features_scaled, labels)
    print(f"Silhouette Score: {silhouette_avg}")

    for cluster_id in range(n_clusters):
        print(f"Cluster {cluster_id}: {sum(labels == cluster_id)} samples")
    
    return features_df

if __name__ == "__main__":
    folder_path = '/home/ubuntu220403/data/DAIR_example/converted_data'
    clustered_data = analyze_trajectory_data(folder_path, n_clusters=4)
    # clustered_data = analyze_trajectory_data_with_umap(folder_path, n_clusters=4)
    print(clustered_data.head())
    
    plot_trajectories_by_cluster(folder_path, clustered_data, sample_size=20)
