import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


def knn_interpolation_error(group):
    train_data = group[group['gid'].isin(train_gids)].dropna()
    test_data = group[group['gid'].isin(test_gids)].dropna()

    if train_data.empty or test_data.empty:
        return [0]

    train_coords = train_data.iloc[:, 3:-5].values
    train_values = train_data['iPM25IAQI'].values

    test_coords = test_data.iloc[:, 3:-5].values
    test_values = test_data['iPM25IAQI'].values

    knn = KNeighborsRegressor(n_neighbors=3, weights="distance")
    knn.fit(train_coords, train_values)

    predicted_values = knn.predict(test_coords)
    errors = predicted_values - test_values
    return errors


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
        .apply(knn_interpolation_error)
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



