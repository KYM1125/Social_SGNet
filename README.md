# 环境搭建

模型所需要的环境在 `../SGNet/SGNet/SGNet/SGNet.pytorch/SGNet_env.yml` 中。

# 构建数据集

目前打包好的 `train`、`val`、`test` 数据集分别位于以下文件夹中：

- `../SGNet/SGNet/SGNet/SGNet.pytorch/data/DAIR/dair/train`
- `../SGNet/SGNet/SGNet/SGNet.pytorch/data/DAIR/dair/val`
- `../SGNet/SGNet/SGNet/SGNet.pytorch/data/DAIR/dair/test`

如果需要重新制作数据集 `pkl` 文件，请遵循以下步骤：

1. 首先 `cd` 到 `../SGNet/dataprocessing` 目录。
2. 运行 `1_convert_all_data.py` 将 DAIR 格式数据集转换成 ETH/UCY 格式。
3. 运行 `2_interpolation_for_all_missing_data.py` 补齐缺失帧（可选，因为插值功能已经集成到 `1_convert_all_data.py` 中）。
4. 运行 `3_redistribution.py` 将数据按照行人总量分配成 `train`、`val`、`test`。
5. 最后，`cd` 进入 `../SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/pedestrians` 目录。
6. 运行其中的 `process_data.py` 即可生成处理后的 `pkl` 文件。

处理后的 `pkl` 文件会存储在 `../SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/processed` 文件夹中。

# 训练模型

1. 进入 `SGNet.Pytorch` 目录。
2. 运行以下命令开始训练不包含社交模块的模型：
    ```bash
    python tools/ethucy/train_cvae.py --gpu 0 --dataset DAIR --model SGNet_CVAE
    ```
3. 如果训练带社交模块的模型，运行：
    ```bash
    python tools/ethucy/train_cvae.py --gpu 0 --dataset DAIR --model Social_SGNet_CVAE
    ```

训练好的权重参数会保存在以下文件夹中：

- 不包含社交模块的模型：`../SGNet/SGNet/SGNet/SGNet.pytorch/tools/ethucy/checkpoints/DAIR/SGNet_CVAE/0.5/1`
- 带社交模块的模型：`../SGNet/SGNet/SGNet/SGNet.pytorch/tools/ethucy/checkpoints/DAIR/Social_SGNet_CVAE/0.5/1`

训练好的模型会保存在 `../SGNet/SGNet/SGNet/SGNet.pytorch/best_model` 文件夹中。

# 测试与可视化

训练完成后，如果需要进行测试和后续可视化，可以运行以下命令：

1. 测试不包含社交模块的模型：
    ```bash
    python tools/ethucy/test_model.py --gpu 0 --dataset DAIR --model SGNet_CVAE --checkpoint (pt文件所在地址)
    ```
2. 测试带社交模块的模型：
    ```bash
    python tools/ethucy/test_model.py --gpu 0 --dataset DAIR --model Social_SGNet_CVAE --checkpoint (pt文件所在地址)
    ```

运行完毕后，终端会输出测试参数 `ADE12` 和 `FDE12`，并在以下目录中生成用于可视化的中间文件：

- 不包含社交模块的模型：`kym/SGNet/SGNet/SGNet/SGNet.pytorch/tools/ethucy/checkpoints/DAIR/SGNet_CVAE/0.5/1/visual`
- 带社交模块的模型：`kym/SGNet/SGNet/SGNet/SGNet.pytorch/tools/ethucy/checkpoints/DAIR/Social_SGNet_CVAE/0.5/1/visual`

通过运行以下脚本，可以获取整个场景行人轨迹的真值和预测值的可视化结果：

1. 整个场景的可视化：
    ```bash
    ../SGNet/dataProcessing/restore_to_one_secen.py
    ```
2. 指定场景下每个子场景的可视化：
    ```bash
    ../SGNet/dataProcessing/plot_sub_scenes.py
    ```

可视化结果会保存在 `../SGNet/dataProcessing/each_traj` 文件夹中。

如果需要将可视化中间文件转换回 DAIR 格式的 CSV 文件，可以运行以下命令：

- 整个场景转换为 CSV：
    ```bash
    ../SGNet/dataProcessing/one_scene_to_csv.py
    ```
- 子场景转换为 CSV：
    ```bash
    ../SGNet/dataProcessing/sub_scenes_to_csv.py
    ```

CSV 文件将存储在 `kym/SGNet/dataProcessing/csv` 文件夹中，并可用于后续的 DAIR-V2X-Seq 官方提供的可视化程序，带有地图信息的可视化。
