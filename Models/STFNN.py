import tensorflow as tf
from keras.layers import Softmax
from tensorflow.keras.layers import Layer, Dense,TimeDistributed
import numpy as np
import Transformer
import copy
class SpatioTemporalEncoder1(Layer):
    def __init__(self, hidden_dim=64, num_meteo_features=5):
        super().__init__()
        self.num_meteo_features = num_meteo_features  # 气象特征数（例如温度、湿度、风速）
        # 编码气象特征的MLP
        self.meteo_encoder = TimeDistributed(Transformer.EncoderLayer(d_model=num_meteo_features,num_heads=1,hidden_units=hidden_dim))
        # 原有时空编码MLP
        self.spatiotemporal_mlp = Dense(hidden_dim, activation="relu")

    def call(self, coord,meteo):
        # 输入形状: [..., 3 + M]（坐标 + 气象特征）
        coordinates = coord # 经度、纬度、时间
        meteo_features = meteo # 气象特征
        # 原有时空编码（公式5）
        t,lng, lat= tf.split(coordinates, [1, 1, 1], axis=-1)
        p_S = tf.concat([lng, lat], axis=-1)
        p_T = []
        for period in [1.0, 7.0, 30.5, 365.0]:
            sin_term = tf.sin(2 * np.pi * t / (period*24))
            cos_term = tf.cos(2 * np.pi * t / (period*24))
            p_T.extend([sin_term, cos_term])
        p_T = tf.concat(p_T, axis=-1)
        spatiotemporal_code = tf.concat([p_S, p_T], axis=-1)
        # 气象特征编码
        meteo_code = self.meteo_encoder(meteo_features)[:,:,-1,:]  # [..., hidden_dim]

        # 拼接时空编码与气象编码
        combined_code = tf.concat([spatiotemporal_code, meteo_code], axis=-1)
        final_code = self.spatiotemporal_mlp(combined_code)  # [..., hidden_dim]
        return final_code

class SpatioTemporalEncoder2(Layer):
    def __init__(self, hidden_dim=64, num_meteo_features=5):
        super().__init__()
        self.num_meteo_features = num_meteo_features  # 气象特征数（例如温度、湿度、风速）
        # 编码气象特征的MLP
        self.meteo_encoder = Transformer.EncoderLayer(d_model=num_meteo_features,num_heads=1,hidden_units=hidden_dim)
        # 原有时空编码MLP
        self.spatiotemporal_mlp = Dense(hidden_dim, activation="relu")

    def call(self, coord,meteo):
        # 输入形状: [..., 3 + M]（坐标 + 气象特征）
        coordinates = coord # 经度、纬度、时间
        meteo_features = meteo # 气象特征

        # 原有时空编码（公式5）
        t,lng, lat= tf.split(coordinates, [1, 1, 1], axis=-1)
        p_S = tf.concat([lng, lat], axis=-1)
        p_T = []
        for period in [1.0, 7.0, 30.5, 365.0]:
            sin_term = tf.sin(2 * np.pi * t / (period*24))
            cos_term = tf.cos(2 * np.pi * t / (period*24))
            p_T.extend([sin_term, cos_term])
        p_T = tf.concat(p_T, axis=-1)
        spatiotemporal_code = tf.concat([p_S, p_T], axis=-1)
        # 气象特征编码
        meteo_code = self.meteo_encoder(meteo_features)[:,-1,:]  # [..., hidden_dim]

        # 拼接时空编码与气象编码
        combined_code = tf.concat([spatiotemporal_code, meteo_code], axis=-1)
        final_code = self.spatiotemporal_mlp(combined_code)  # [..., hidden_dim]
        return final_code

class RingEstimation(Layer):
    def __init__(self, hidden_dim=64, m=16):
        super().__init__()
        self.mlp = tf.keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(hidden_dim, activation='relu'),
            Dense(3)  # 输出梯度向量
        ])
        self.m = m

    def call(self, src_coords, tar_coords, src_meteo):
        # 计算单位方向向量
        delta_raw = tar_coords[:, tf.newaxis, :] - src_coords  # [B, K, 3]
        direction = delta_raw / (tf.norm(delta_raw, axis=-1, keepdims=True) + 1e-8)
        step_size = delta_raw / self.m  # 总增量分成m步

        integral = 0.0
        for j in range(self.m):
            current_coords = src_coords + step_size * j  # [B, K, 3]
            # 拼接当前坐标与源气象特征（假设气象在路径上不变）
            inputs = tf.concat([current_coords, src_meteo[:,:,-1,:]], axis=-1)
            D_ij = self.mlp(inputs)  # [B, K, 3]
            # 沿单位方向投影
            integral += tf.reduce_sum(D_ij * direction, axis=-1)  # [B, K]
        return integral  # [B, K]
class NeighborAggregation(Layer):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = tf.keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(1)
        ])
        self.softmax = Softmax(axis=1)

    def call(self, src_enc, tar_enc):
        tar_enc = tf.expand_dims(tar_enc, 1)  # [B, 1, H]
        combined = tf.concat([src_enc, tf.tile(tar_enc, [1, tf.shape(src_enc)[1], 1])], axis=-1)
        logits = self.mlp(combined)  # [B, K, 1]
        return self.softmax(logits)  # [B, K, 1]
class STFNNWithMeteo(tf.keras.Model):
    def __init__(self, K=6, m=16, hidden_dim=64, num_meteo_features=6):
        super().__init__()
        self.encoder1 = SpatioTemporalEncoder1(hidden_dim, num_meteo_features)
        self.encoder2 = SpatioTemporalEncoder2(hidden_dim, num_meteo_features)
        self.ring_estimator = RingEstimation(hidden_dim, m)
        self.aggregator = NeighborAggregation(K)
        self.K = K

    def call(self, inputs):
        # 输入调整为包含气象特征
        c_scr_coord,c_scr_meo,c_tar_coord,c_tar_meo,c_scr_y = inputs

        # 编码（输入已包含气象特征）
        p_src = self.encoder1(c_scr_coord,c_scr_meo)  # [B, K, hidden_dim]
        p_tar = self.encoder2(c_tar_coord,c_tar_meo)  # [B, hidden_dim]

        # 环估计（输入包含气象特征）
        integral = self.ring_estimator(c_scr_coord,c_tar_coord,c_scr_meo)  # [B, K]
        y_estimates = c_scr_y + integral # [B, K, 1]
        # 邻居聚合
        weights = self.aggregator(p_src, p_tar)  # [B, K, 1]

        y_estimates = tf.expand_dims(y_estimates, axis=-1)

        y_tar = tf.reduce_sum(weights * y_estimates, axis=1)  # [B, 1]
        return y_tar
def build_dataset(data_local, data_station, station_indices, is_train=True):
    """
    构建数据集

    参数：
        data_local: 本地站点数据，形状为 [样本数, 站点数, 时间步数, 特征数]
        data_station: 监测站数据（仅测试时使用），同样形状
        station_indices: 要处理的站点索引列表
        is_train: 是否为训练数据集（决定 station_mask 处理方式）

    返回：
        X_dict: 包含各类特征的字典
        y_array: 标签数组
    """
    X_list, y_list = [], []

    for i in station_indices:
        local_static = data_local[:, i, -1, 3:25]  # 本地静态特征
        local_seq = data_local[:, i, :, 25:35]  # 本地时序特征
        T = data_local[:, i, -1, 0]  # 时间特征
        y = data_local[:, i, -1, -1]  # 标签

        if is_train:
            station_mask = np.setdiff1d(station_indices, i)
            stations_static = data_local[:, station_mask, -1, 3:25]
            stations_seq = data_local[:, station_mask, :, 25:]
        else:
            stations_static = data_station[:, :, -1, 3:25]
            stations_seq = data_station[:, :, :, 25:]

        X_list.append({
            'local_static': local_static,
            'local_seq': local_seq,
            'stations_static': stations_static,
            'stations_seq': stations_seq,
            'T': T
        })
        y_list.append(y)

    # 合并为 numpy 数组
    X_dict = {key: np.concatenate([x[key] for x in X_list], axis=0) for key in X_list[0]}
    y_array = np.concatenate(y_list, axis=0)

    return X_dict, y_array

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
def evaluate_all(model, X_test, y_test, num_train_stations):
    best_result = float('inf')
    best_index = -1
    all_results = []

    for i in range(num_train_stations):
        x_test = copy.deepcopy(X_test)
        n = i
        m = n + 1

        # 移除当前站点
        x_test['c_scr_test'] = tf.concat([X_test['c_scr_test'][:, :n, :], X_test['c_scr_test'][:, m:, :]], axis=1)

        # 构造输入
        c_scr_coord = x_test['c_scr_test'][:, :, -1, :3]
        c_scr_meo = x_test['c_scr_test'][:, :, :, 3:-1]
        c_tar_coord = x_test['c_tar_test'][:, -1, :3]
        c_tar_meo = x_test['c_tar_test'][:, :, 3:-1]
        c_scr_y = x_test['c_scr_test'][:, :, -1, -1]

        inputs = [c_scr_coord, c_scr_meo, c_tar_coord, c_tar_meo, c_scr_y]
        result = model.evaluate(inputs, y_test, verbose=0)  # 设置 verbose=0 防止输出过多

        all_results.append(result)

        if result[0] < best_result:  # 以第一个指标（如 MSE）为标准
            best_result = result[0]
            best_index = i

        print(f'======= Station {i} Evaluation =======')
        print(f'Results: {result}')

    print('\n======== Best Result ========')
    print(f'Best Station Index: {best_index}')
    print(f'Best Evaluation Result: {all_results[best_index]}')

    return best_index, all_results[best_index], all_results
if __name__ == '__main__':
    x=1
    Train = np.load(rf"..\data_process\Data\Train_data_{x}.npy")
    Test_local = np.load(rf'..\data_process\Data\Test_local_{x}.npy')
    Test_station = np.load(rf'..\data_process\Data\Test_station_{x}.npy')
    num_total_stations = 32
    num_train_stations = 28
    num_test_stations = 4
    train_stations = np.arange(num_train_stations)
    test_stations = np.arange(num_test_stations)


    def build_samples(data, stations, is_train=True):
        X_list, y_list = [], []
        for i in stations:
            c_tar = tf.concat([
                tf.expand_dims(data[:, i, :, 0], axis=-1),
                data[:, i, :, 3:5],
                data[:, i, :, 25:31],
                tf.expand_dims(data[:, i, :, -1], axis=-1)
            ], axis=-1)

            if is_train:
                station_mask = np.setdiff1d(stations, i)
                c_scr = tf.concat([
                    tf.expand_dims(data[:, station_mask, :, 0], axis=-1),
                    tf.transpose(data[:, station_mask, :, 3:5], perm=[1, 0, 2, 3]),
                    tf.transpose(data[:, station_mask, :, 25:31], perm=[1, 0, 2, 3]),
                    tf.expand_dims(data[:, station_mask, :, -1], axis=-1)
                ], axis=-1)
                c_scr = tf.transpose(c_scr, perm=[1, 0, 2, 3])
            else:
                c_scr = tf.concat([
                    tf.expand_dims(Test_station[:, :, :, 0], axis=-1),
                    Test_station[:, :, :, 3:5],
                    Test_station[:, :, :, 25:31],
                    tf.expand_dims(Test_station[:, :, :, -1], axis=-1)
                ], axis=-1)

            X_list.append({
                'c_tar_train' if is_train else 'c_tar_test': c_tar,
                'c_scr_train' if is_train else 'c_scr_test': c_scr
            })
            y = data[:, i, -1, -1]
            y_list.append(y)

        X = {key: np.array([d[key] for d in X_list]) for key in X_list[0]}
        y = np.array(y_list)
        for key in X:
            X[key] = np.concatenate(X[key], axis=0)
        y = np.concatenate(y, axis=0)
        return X, y


    X_train, y_train = build_samples(Train, train_stations, is_train=True)
    X_test, y_test = build_samples(Test_local, test_stations, is_train=False)


    # 加载模型
    model = STFNNWithMeteo(K=27, m=16, hidden_dim=64,num_meteo_features=6)

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss=rmse,
        metrics=['mae']
    )
    import os
    # 构建模型
    checkpoint_save_path = f"./checkpoint/STFNN_{x}.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print("==========load the model==========")
        model.load_weights(checkpoint_save_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_save_path,
        # monitor="val_accuracy",#分类
        monitor='val_loss',  # 回归
        save_best_only=True,
        save_weights_only=True,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # 监视验证集上的损失
        patience=10,  # 如果连续5个epoch验证集上的损失没有改善，则停止训练
        restore_best_weights=True  # 训练停止时，恢复最优权重
    )
    c_scr_coord = X_train['c_scr_train'][:,:,-1,:3]
    c_scr_meo = X_train['c_scr_train'][:,:,:,3:-1]
    c_tar_coord = X_train['c_tar_train'][:,-1,:3]
    c_tar_meo = X_train['c_tar_train'][:,:,3:-1]
    c_scr_y = X_train['c_scr_train'][:,:,-1,-1]

    # 训练模型
    # history = model.fit([c_scr_coord,c_scr_meo,c_tar_coord,c_tar_meo,c_scr_y],y_train,validation_split=0.14 , epochs=128,batch_size=128,
    #                     callbacks=[checkpoint_callback,early_stopping])
    best_index, best_result, all_results = evaluate_all(model, X_test, y_test, num_train_stations)



