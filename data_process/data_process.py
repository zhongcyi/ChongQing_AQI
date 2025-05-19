import os
import pandas as pd
import tensorflow as tf
import numpy as np

def fill_nan_with_last_step(data, start_idx=25, end_idx=31):
    """
    用每个样本最后一个时间步的值填充指定通道范围内的 NaN。
    参数：
    - data: 4维数组，形状为 (样本数, 空间维, 时间步, 特征维)
    - start_idx: 特征起始索引（含）
    - end_idx: 特征结束索引（不含）
    """
    for i in range(data.shape[0]):
        mask_nan = np.isnan(data[i, :, :, start_idx:end_idx])
        if np.any(mask_nan):
            last_step_values = np.expand_dims(data[i, :, -1, start_idx:end_idx], axis=1)  # (空间维, 1, 特征维)
            fill_values = np.tile(last_step_values, (1, data.shape[2], 1))  # (空间维, 时间步, 特征维)
            data[i, :, :, start_idx:end_idx] = np.where(mask_nan, fill_values, data[i, :, :, start_idx:end_idx])

    return data  # 可选返回，视是否想链式调用
if __name__ == '__main__':
    df=pd.read_csv('../../data_process/merged_data.csv')
    columns=df.columns
    nei_gird_size = 3 #网格领域大小
    array = df.to_numpy()#将df转化为numpy
    num_features = array.shape[1]#特征数
    sample_len = array.shape[0]//9#样本数量
    # #将数据重塑（3*3网格大小）
    array_reshaped=tf.reshape(array,(-1,nei_gird_size,nei_gird_size,num_features))
    # # # 展平每个样本的数据为一个一维向量
    array_flattened = array_reshaped[:,1,1,:]
    # # 创建一个 DataFrame
    array_flattened = pd.DataFrame(array_flattened.numpy())
    # # 为了给每个列命名，可以根据需要设置列名
    array_flattened.columns = columns
    # # 对 DataFrame 按gid和第dt进行排序
    # # 删除 gid 为1086834.0 的行，因为这个站点只有一个数据
    df_flattened = array_flattened[array_flattened['gid'] != 1086833.0]

    gids =[[1081231, 1082762,1072047, 1074594],[1065920,  1067459, 1068967,  1070999],[1071033, 1072037,1073059, 1073070],[1073572, 1075090, 1075100, 1075581],
           [ 1077136,1077156, 1077665,  1076099],[1078676,1086819,1081725,1080204,],[1074053, 1082252,1065915,1070507],[1087837, 1087842,1074583,1066944,]]
    for num in range(1,9):
        # 时间长度为8760，这里的滑动窗口大小为8
        Train=[]
        Test=[]
        seq_len=8
        test_gids = gids[num - 1]
        train_gids = gids[:num - 1] + gids[num:]
        train_gids = np.concatenate(train_gids)
        #按gid进行分组
        array_grouped = array_flattened.groupby('gid', group_keys=True)
        # # 修改索引的名称
        #
        train_station = []  # 存储训练站点的 DataFrame
        test_station = []   # 存储测试站点的 DataFrame

        for name, group in array_grouped:
            # print(f"Group: {name}")
            group_sorted = group.sort_values(by='time_encoded')
            if name in test_gids:
                test_station.append(group_sorted)  # 存入测试站点列表
            if name in train_gids:
                train_station.append(group_sorted)  # 存入训练站点列表
        # # 合并 DataFrame
        train_df = pd.concat(train_station, ignore_index=True)
        test_df = pd.concat(test_station, ignore_index=True)

        #按时间进行分组
        test_df_grouped = test_df.groupby('time_encoded', group_keys=True)
        train_df_grouped = train_df.groupby('time_encoded', group_keys=True)
        for name, group in train_df_grouped:
            # print(f"Group: {name}") # [B, S, T, F]
            group_sorted=group.sort_values(by='gid')
            Train.append(group_sorted)
        for name, group in test_df_grouped:
            # print(f"Group: {name}")
            group_sorted=group.sort_values(by='gid')
            Test.append(group_sorted)

        # # 转换为 NumPy 数组
        Train = np.array(Train)
        Test = np.array(Test)

        """构造时序数据"""
        num_samples = 8760 - seq_len  # 计算样本数量，即 8752
        Train = np.array([
            Train[i:i + seq_len] for i in range(num_samples)
        ])  # 形状 (8752, , 16, 36)
        # 调整维度顺序 (8752, 16, 8, 36)->(num_samples,train_station_num,seq_leb,num_fetures)
        Train = Train.transpose(0, 2, 1, 3)

        Test = np.array([
            Test[i:i + seq_len] for i in range(num_samples)
        ])  # 形状 (8752, 8, 16, 36)
        # 调整维度顺序 (8752, 16, 8, 36)->(num_samples,train_station_num,seq_leb,num_fetures)
        Test = Test.transpose(0, 2, 1, 3)
        '''过滤aqi为0的样本'''
        Train_data = [] #用于训练的数据
        for i in range(len(Train)):
            Train_data.append(Train[i,:,:,:])
            for j in range(Train.shape[1]):
                if Train[i,j,-1,-1]==0:
                    Train_data.pop()
                    break
        Train_data=np.array(Train_data)
        "用于测试的数据"
        Test_station = []
        Test_local = []
        for i in range(len(Test)):
            Test_local.append(Test[i,:,:,:])
            Test_station.append(Train[i,:,:,:])
            flag = 0
            for j in range(Test.shape[1]):
                if Test[i,j,-1,-1]==0:
                    flag=1
                    break
            for j in range(Train.shape[1]):
                if Train[i,j,-1,-1]==0:
                    flag=1
                    break
            if flag:
                Test_local.pop()
                Test_station.pop()
                flag=0
        Test_local, Test_station=np.array(Test_local), np.array(Test_station)
        print('========================')
        print("Train_data:",Train_data.shape)
        print("Test_local",Test_local.shape)
        print(",Test_station:",Test_station.shape)
        """对监测站的存在的气象数据进行缺失值处理，如果监测站存在缺失值，就删除掉这个样本"""
        #检查每个样本的第 7 时间步（索引为 7）和位置 (1, 1) 的第 25-31 个特征（索引为 24 到 30）是否为 NaN
        # 选择出 [7, 1, 1, 24:30] 这个范围
        Train_data_filter_meteorology =[]
        Test_station_filter_meteorology = []
        Test_local_filter_meteorology = []
        for i in range(Train_data.shape[0]):
            if np.any(np.isnan(Train_data[i, :, -1, 25:31])):  # 检查是否有NaN
                continue  # 如果有NaN，跳过该样本
            Train_data_filter_meteorology.append(Train_data[i])  # 否则保留该样本
        for i in range(Test_station.shape[0]):
            if np.any(np.isnan(Test_station[i,:,-1,25:31])) and np.any(np.isnan(Test_local[i, :, -1, 25:31])):
                continue
            Test_station_filter_meteorology.append(Test_station[i])
            Test_local_filter_meteorology.append(Test_local[i])
        #转化为numpy类型
        Train_data_filter_meteorology = np.array(Train_data_filter_meteorology)
        Test_local_filter_meteorology = np.array(Test_local_filter_meteorology)
        print('===================================')
        Test_station_filter_meteorology = np.array(Test_station_filter_meteorology)
        print("Train_data_filter_meteorology:",Train_data_filter_meteorology.shape)
        print("Test_local_filter_meteorology:",Test_local_filter_meteorology.shape)
        print("Test_station_filter_meteorology:",Test_station_filter_meteorology.shape)
        # print(Train_filtered_nan_data.shape,Test_filtered_nan_data.shape)
        """若非监测站的气象特征数据进存在缺失，用监测站的数据进行填充"""
        Train_data_filled=fill_nan_with_last_step(Train_data_filter_meteorology)
        Test_local_filled=fill_nan_with_last_step(Test_local_filter_meteorology)
        Test_station_filled=fill_nan_with_last_step(Test_station_filter_meteorology)
        print('=========================')
        print('Train_data_filled:',Train_data_filled.shape)
        print('Test_local_filled:',Test_local_filled.shape)
        print('Test_station_filled:',Test_station_filled.shape)
        for name, data in zip(['Train', 'Test_local', 'Test_station'],
                              [Train_data_filled, Test_local_filled, Test_station_filled]):
            print(f"{name} missing count:", np.isnan(data).sum())

        os.makedirs('../../data_process/Data/', exist_ok=True)
        # 保存为 .npy 文件
        np.save(f'./Data/Train_data_{num}.npy', Train_data_filled)
        np.save(f'./Data/Test_local_{num}.npy',Test_local_filled)
        np.save(f'./Data/Test_station_{num}.npy', Test_station_filled)
        print(f"第{num}个样本保存成功")
