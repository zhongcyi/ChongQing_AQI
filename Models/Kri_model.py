import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging

def kriging_with_error(group):
    """
    对每个分组进行克里金建模，选择给定 gid 列表中的点作为训练集和测试集
    """
    # 根据给定的 gid 列表选择训练集和测试集
    train_data = group[group['gid'].isin(train_gids)]  # 选择训练集中的点
    test_data = group[group['gid'].isin(test_gids)]    # 选择测试集中的点

    # 提取训练集的坐标和观测值
    train_coords = train_data[['longitude', 'latitude']].values
    train_values = train_data['iPM25IAQI'].values
    'spherical'
    'linear'
    'gaussian'
    'power'
    'nugget'
    'exponential'
    # 创建克里金插值模型
    OK = OrdinaryKriging(
        train_coords[:, 0],  # 经度
        train_coords[:, 1],  # 纬度
        train_values,         # 目标值（例如 'iPM25IAQI'）
        variogram_model='gaussian',  # 选择变异函数类型
        verbose=False,
        enable_plotting=False
    )

    # 提取测试集的坐标
    test_coords = test_data[['longitude', 'latitude']].values

    # 对测试集进行插值
    predicted_values, ss = OK.execute('points', test_coords[:, 0], test_coords[:, 1])

    # 计算插值误差（预测值与实际值的差异）
    errors = np.abs(predicted_values - test_data['iPM25IAQI'].values ) # 预测误差
    return errors  # 返回误差
df = pd.read_csv(r'../data_process/Data/Kri_and_Knn_data.csv')
gids = [[1081231, 1082762, 1072047, 1074594], [1065920, 1067459, 1068967, 1070999],
        [1071033, 1072037, 1073059, 1073070], [1073572, 1075090, 1075100, 1075581],
        [1077136, 1077156, 1077665, 1076099], [1078676, 1086819, 1081725, 1080204, ],
        [1074053, 1082252, 1065915, 1070507], [1087837, 1087842, 1074583, 1066944, ]]

for num in range(1, 9):
    print(f"\nFold {num} ...")

    test_gids = gids[num - 1]
    train_gids = np.concatenate(gids[:num - 1] + gids[num:])

    errors_list = (
        df.groupby('time_encoded')
        .apply(kriging_with_error)
        .reset_index(drop=True)
    )

    valid_errors = [e for e in errors_list.values if not np.all(np.array(e) == 0)]
    if len(valid_errors) == 0:
        print(f"Fold {num} 无有效误差，跳过")
        continue

    all_errors = np.concatenate(valid_errors)

    mae = np.mean(np.abs(all_errors))
    mse = np.sqrt(np.mean(all_errors ** 2))

    print("平均绝对误差 (MAE):", mae)
    print("均方根误差 (RMSE):", mse)

