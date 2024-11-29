import os
import shutil
import math
import random
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def get_unique_pedestrian_count(file_path):
    unique_pedestrians = set()
    file_prefix = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, 'r') as file:
        for line in file:
            _, person_id, _, _ = line.strip().split()
            unique_person_id = f"{file_prefix}_{person_id}"
            unique_pedestrians.add(unique_person_id)
    return len(unique_pedestrians)

def calculate_pedestrian_counts(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.txt')]
    file_counts = []
    for file_name in files:
        file_path = os.path.join(folder, file_name)
        pedestrian_count = get_unique_pedestrian_count(file_path)
        file_counts.append((file_path, pedestrian_count))
    return file_counts

def distribute_files_by_pedestrians(source_folder, train_folder, val_folder, test_folder):
    # 如果目标文件夹已经存在，则删除
    for folder in [train_folder, val_folder, test_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    # 获取所有源文件信息
    all_files = calculate_pedestrian_counts(source_folder)
    random.shuffle(all_files)

    # 计算目标分配人数
    total_pedestrians = sum(count for _, count in all_files)
    target_train = math.floor(total_pedestrians * 0.7)
    target_test = math.floor(total_pedestrians * 0.15)
    target_val = total_pedestrians - target_train - target_test

    # 分配文件
    assigned_train, assigned_test, assigned_val = 0, 0, 0
    file_distribution = []

    for file_path, pedestrian_count in all_files:
        # 按照剩余需求分配文件
        if assigned_train + pedestrian_count <= target_train:
            dst_folder = train_folder
            assigned_train += pedestrian_count
        elif assigned_test + pedestrian_count <= target_test:
            dst_folder = test_folder
            assigned_test += pedestrian_count
        else:
            dst_folder = val_folder
            assigned_val += pedestrian_count

        dst_path = os.path.join(dst_folder, os.path.basename(file_path))
        
        # 检查文件是否已经存在，避免重复
        if os.path.exists(dst_path):
            print(f"File {dst_path} already exists. Skipping move.")
            continue
        
        file_distribution.append((file_path, dst_path))

    # 使用多进程移动文件
    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(move_file, file_distribution), total=len(file_distribution), desc="Moving files"):
            pass

    # 打印每个文件夹中的总行人数量
    print(f"Train pedestrians: {assigned_train}")
    print(f"Val pedestrians: {assigned_val}")
    print(f"Test pedestrians: {assigned_test}")

def move_file(file_paths):
    """将文件从源路径移动到目标路径"""
    src_path, dst_path = file_paths
    if os.path.exists(src_path):  # 检查源文件是否存在
        shutil.move(src_path, dst_path)
    else:
        print(f"Source file {src_path} does not exist.")

if __name__ == "__main__":
    # 数据源文件夹，包含所有文件的原始文件夹路径
    source_folder = '../dair_infra/interpolated_all'
    # 目标文件夹路径
    train_folder = '../dair_infra/interpolated_train'
    val_folder = '../dair_infra/interpolated_val'
    test_folder = '../dair_infra/interpolated_test'
    
    distribute_files_by_pedestrians(source_folder, train_folder, val_folder, test_folder)
