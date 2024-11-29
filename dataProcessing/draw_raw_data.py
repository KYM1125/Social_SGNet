'''
    绘制原始数据中的行人轨迹
'''
import pandas as pd
import matplotlib.pyplot as plt

def plot_pedestrian_trajectory(filepath):
    # 读取CSV文件
    df = pd.read_csv(filepath)
    
    # 筛选出PEDESTRIAN的数据
    pedestrian_data = df[df['sub_type'] == 'PEDESTRIAN']

    # 按 ID 分组并绘制轨迹
    grouped = pedestrian_data.groupby('id')
    plt.figure(figsize=(10, 8))
    
    for pedestrian_id, group in grouped:
        plt.plot(group['x'], group['y'], marker='o', label=f'Pedestrian {pedestrian_id}', linestyle='-', alpha=0.7)
    
    plt.title('Pedestrian Trajectories')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize='small', markerscale=0.5)
    plt.grid(True)
    plt.show()

# 示例：绘制CSV文件中的行人轨迹
csv_filepath = '10095.csv'
plot_pedestrian_trajectory(csv_filepath)
