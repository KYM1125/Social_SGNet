'''
    将DAIR格式数据转换成ETH格式数据，用于尝试
'''
import pandas as pd
import matplotlib.pyplot as plt

def process_csv_file(filepath):
    # 读取 CSV 文件
    df = pd.read_csv(filepath, delimiter=',')
    
    # 筛选出行人数据
    pedestrian_data = df[df['sub_type'] == 'PEDESTRIAN']
    
    # 按 id 排序并重新分配新的 ID
    sorted_ids = sorted(pedestrian_data['id'].unique())
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_ids, start=1)}
    
    # 替换原有的行人 ID 为新的 ID
    pedestrian_data['new_id'] = pedestrian_data['id'].map(id_mapping)
    
    # 按新 ID 进行分组
    grouped = pedestrian_data.groupby('new_id')
    
    # 存储转换后的数据
    eth_format_data = []
    for new_id, group in grouped:
        # 按 timestamp 排序
        sorted_group = group.sort_values(by='timestamp')
        # 选择该行人轨迹中的第一个时间戳作为基准时间
        base_time = sorted_group['timestamp'].iloc[0]
        # 转换为 ETH 格式
        for idx, row in sorted_group.iterrows():
            eth_time = convert_to_eth_time(row['timestamp'], base_time)
            eth_format_data.append(f"{eth_time:.1f}\t{new_id:.1f}\t{row['x']:.2f}\t{row['y']:.2f}")
    
    return eth_format_data, pedestrian_data

def convert_to_eth_time(timestamp, base_time, interval=0.1):
    return (timestamp - base_time) / interval

def save_to_txt(eth_data, output_filepath):
    with open(output_filepath, 'w') as file:
        for line in eth_data:
            file.write(line + '\n')

def plot_all_trajectories(df):
    """可视化所有行人的轨迹"""
    unique_ids = df['new_id'].unique()
    
    plt.figure(figsize=(12, 8))
    
    for pedestrian_id in unique_ids:
        group = df[df['new_id'] == pedestrian_id]
        
        # 绘制轨迹
        plt.plot(group['x'], group['y'], label=f'行人 {pedestrian_id}')
        # 标记所有点
        plt.scatter(group['x'], group['y'], s=30, alpha=0.5)

    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('所有行人的轨迹')
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例：处理并保存一个 CSV 文件
csv_filepath = "10098.csv"
output_filepath = "eth_data_10098.txt"

eth_data, cleaned_data = process_csv_file(csv_filepath)
save_to_txt(eth_data, output_filepath)

# 可视化所有行人的轨迹
if not cleaned_data.empty:
    plot_all_trajectories(cleaned_data)
