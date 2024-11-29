import os
import shutil
import math
import random
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def distribute_files(train_folder, val_folder, test_folder):
    # 获取文件夹中所有 txt 文件
    train_files = [f for f in os.listdir(train_folder) if f.endswith('.txt')]
    val_files = [f for f in os.listdir(val_folder) if f.endswith('.txt')]
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.txt')]

    # 合并文件列表
    all_files = train_files + val_files + test_files
    random.shuffle(all_files)  # 打乱文件顺序以进行随机分配

    # 计算分配的数量
    total_files = len(all_files)
    num_train = math.floor(total_files * 0.7)
    num_test = math.floor(total_files * 0.15)
    num_val = total_files - num_train - num_test

    # 创建目标文件夹，如果不存在的话
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 准备文件分配数据
    file_distribution = []
    for i, file_name in enumerate(all_files):
        src_path = os.path.join(train_folder if file_name in train_files else (val_folder if file_name in val_files else test_folder), file_name)
        if i < num_train:
            dst_path = os.path.join(train_folder, file_name)
        elif i < num_train + num_test:
            dst_path = os.path.join(test_folder, file_name)
        else:
            dst_path = os.path.join(val_folder, file_name)
        file_distribution.append((src_path, dst_path))

    # 使用多进程移动文件，并显示进度条
    with Pool(cpu_count()) as pool:
        # tqdm 进度条
        for _ in tqdm(pool.imap_unordered(move_file, file_distribution), total=len(file_distribution), desc="Moving files"):
            pass

    # 打印每个文件夹里的文件数
    print(f"Train folder: {len(os.listdir(train_folder))} files")
    print(f"Val folder: {len(os.listdir(val_folder))} files")
    print(f"Test folder: {len(os.listdir(test_folder))} files")

def move_file(file_paths):
    """将文件从源路径移动到目标路径"""
    src_path, dst_path = file_paths
    shutil.move(src_path, dst_path)

if __name__ == "__main__":
    # 文件夹路径
    train_folder = 'raw/dair_infra/interpolated_train'
    val_folder = 'raw/dair_infra/interpolated_val'
    test_folder = 'raw/dair_infra/interpolated_test'
    
    distribute_files(train_folder, val_folder, test_folder)
