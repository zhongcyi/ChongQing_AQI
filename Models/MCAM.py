import copy
import os

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class BilateralAttention(tf.keras.layers.Layer):
    """静态图通道的双边滤波注意力"""
    def __init__(self, sigma_d=1.0, sigma_r=1.0):
        super(BilateralAttention, self).__init__()
        self.sigma_d = tf.Variable(initial_value=sigma_d, trainable=True, dtype=tf.float32)
        self.sigma_r = tf.Variable(initial_value=sigma_r, trainable=True, dtype=tf.float32)
    def call(self, target_static, stations_static):
        """
        target_static: [B, 22]
        stations_static: [B, 23, 22]
        """
        # 提取经纬度
        target_coord = target_static[:, :2]  # [B, 2]
        stations_coord = stations_static[:, :, :2]  # [B, 23, 2]
        # 计算欧氏距离项
        dist = tf.norm(target_coord[:, None] - stations_coord, axis=-1)  # [B, 23]
        c = tf.exp(-0.5 * (dist / self.sigma_d) ** 2)
        # 计算地理特征相似性（Pearson相关系数）
        target_feat = target_static[:, 2:]  # [B, 20]
        stations_feat = stations_static[:, :, 2:]  # [B, 23, 20]
        rho = tf.keras.losses.cosine_similarity(target_feat[:, None], stations_feat, axis=-1)  # 近似Pearson
        s = tf.exp(-0.5 * (rho / self.sigma_r) ** 2)

        # 计算注意力权重
        e = c * s  # [B, 23]
        weights = e / tf.reduce_sum(e, axis=1, keepdims=True)
        return weights  # [B, 23]


class DynamicAttention(tf.keras.layers.Layer):
    """动态图通道的注意力机制（含滞后项和大气项）"""

    def __init__(self, hidden_dim):
        super(DynamicAttention, self).__init__()
        self.fc_hysteresis = layers.Dense(hidden_dim, activation='relu')
        self.fc_atmospheric = layers.Dense(hidden_dim, activation='relu')
        self.w_d = layers.Dense(1, use_bias=False)

    def call(self, h_target, h_station_cur, h_station_prev, weather_features):
        """
        h_target: [B, hidden]
        h_station_cur: [B, 23, hidden]
        h_station_prev: [B, 23, hidden]
        weather_features: [B, 23, 3] (u, v, angle)
        """
        # 滞后项
        h_hysteresis = self.fc_hysteresis(tf.concat([h_station_cur, h_station_prev], axis=-1))  # [B,23,hidden]

        # 大气项
        h_atmo = self.fc_atmospheric(weather_features)  # [B,23,hidden]

        # 注意力得分
        energy = tf.concat([h_target[:, None, :] + h_hysteresis, h_atmo], axis=-1)  # [B,23,2*hidden]
        e = tf.squeeze(self.w_d(energy), axis=-1)  # [B,23]
        weights = tf.nn.softmax(e, axis=1)
        return weights


class MCAM(Model):
    def __init__(self, static_feat_dim=22, dynamic_feat_dim=10,
                 lstm_hidden=300, fusion_dim=200):
        super(MCAM, self).__init__()

        # 静态通道
        self.static_attn = BilateralAttention()
        self.static_fc = tf.keras.Sequential([
            layers.Dense(fusion_dim, activation='relu')
        ])

        # 动态通道
        self.lstm_station = layers.LSTM(lstm_hidden, return_sequences=False)
        self.lstm_target = layers.LSTM(lstm_hidden, return_sequences=False)
        self.dynamic_attn = DynamicAttention(lstm_hidden)
        self.dynamic_fc = tf.keras.Sequential([
            layers.Dense(fusion_dim, activation='relu')
        ])

        # 融合网络
        self.fusion = layers.Dense(1)  # 预测单一污染物

    def call(self,x):
        """
        target_static: [B, 22]
        target_dynamic: [B, T, 10]
        stations_static: [B, 23, 22]
        stations_dynamic: [B, 23, T, 11]
        """
        target_static=x['target_static']
        target_dynamic=x['target_dynamic']
        stations_static=x['stations_static']
        stations_dynamic=x['stations_dynamic']
        B = target_static.shape[0]

        # --- 静态通道 ---
        static_weights = self.static_attn(target_static, stations_static)  # [B,23]
        static_feat = self.static_fc(stations_static[:, :, 2:])  # [B,23,fusion]
        static_out = tf.reduce_sum(static_weights[:, :, None] * static_feat, axis=1)  # [B,fusion]

        # --- 动态通道 ---
        # LSTM处理
        h_target = self.lstm_target(target_dynamic)  # [B,hidden]

        h_stations = []
        for i in range(27):
            x = tf.concat([stations_dynamic[:, i, :,:4], tf.expand_dims(stations_dynamic[:, i, :,-1],axis=-1)],axis=-1)
            h = self.lstm_station(x)  # [B,hidden]
            h_stations.append(h)
        h_stations = tf.stack(h_stations, axis=1)  # [B,23,hidden]

        # 注意力计算（需提取天气特征）
        weather_features = stations_dynamic[:, :, -1, 4:6]  # 假设最后三个是u, v, angle [B,23,3]
        dynamic_weights = self.dynamic_attn(h_target, h_stations, h_stations, weather_features)

        # 动态特征融合
        dynamic_feat = self.dynamic_fc(tf.concat([h_stations, h_stations], axis=-1))  # [B,23,fusion]
        dynamic_out = tf.reduce_sum(dynamic_weights[:, :, None] * dynamic_feat, axis=1)  # [B,fusion]

        # --- 融合 ---
        combined = tf.concat([static_out, dynamic_out], axis=1)
        pred = self.fusion(combined)  # [B,1]
        return tf.squeeze(pred, axis=-1)
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
def prepare_train_test_data(Train, Test_local, Test_station, num_train_stations=28, num_test_stations=4):
    """
    构造训练和测试数据（local_static、local_seq、stations_static、stations_seq 和 y）

    参数:
        Train: 训练数据，形状为 [样本数, 站点数, 时间步, 特征数]
        Test_local: 测试点数据，形状为 [样本数, 站点数, 时间步, 特征数]
        Test_station: 训练站点在测试集中的数据，形状为 [样本数, 训练站点数, 时间步, 特征数]
        num_train_stations: 训练站点数量
        num_test_stations: 测试站点数量

    返回:
        X_train, y_train, X_test, y_test: 处理好的训练与测试集数据
    """
    train_stations = np.arange(0, num_train_stations)
    test_stations = np.arange(0, num_test_stations)

    X_train_list, y_train_list = [], []
    for i in train_stations:
        local_static = Train[:, i, -1, 3:25]           # [样本数, 静态维度]
        local_seq = Train[:, i, :, 25:35]              # [样本数, 时间步, 动态维度]
        station_mask = np.setdiff1d(train_stations, i)
        stations_static = Train[:, station_mask, -1, 3:25]  # [样本数, 其他站点数, 静态维度]
        stations_seq = Train[:, station_mask, :, 25:]       # [样本数, 其他站点数, 时间步, 动态维度]
        y = Train[:, i, -1, -1]                         # [样本数]

        X_train_list.append({
            'local_static': local_static,
            'local_seq': local_seq,
            'stations_static': stations_static,
            'stations_seq': stations_seq
        })
        y_train_list.append(y)

    X_train = {key: np.array([x[key] for x in X_train_list]) for key in X_train_list[0]}
    y_train = np.array(y_train_list)

    for key in X_train:
        X_train[key] = np.concatenate(X_train[key], axis=0)
    y_train = np.concatenate(y_train, axis=0)

    X_test_list, y_test_list = [], []
    for i in test_stations:
        local_static = Test_local[:, i, -1, 3:25]
        local_seq = Test_local[:, i, :, 25:35]
        stations_static = Test_station[:, :, -1, 3:25]
        stations_seq = Test_station[:, :, :, 25:]
        y = Test_local[:, i, -1, -1]

        X_test_list.append({
            'local_static': local_static,
            'local_seq': local_seq,
            'stations_static': stations_static,
            'stations_seq': stations_seq
        })
        y_test_list.append(y)

    X_test = {key: np.array([x[key] for x in X_test_list]) for key in X_test_list[0]}
    y_test = np.array(y_test_list)

    for key in X_test:
        X_test[key] = np.concatenate(X_test[key], axis=0)
    y_test = np.concatenate(y_test, axis=0)

    return X_train, y_train, X_test, y_test
def evaluate_leave_one_station_out(X_test, y_test, model):
    """
    对每个站点进行 leave-one-out 测试，评估模型性能，输出最优评估结果。

    参数:
        X_test: dict，包含 local_static、local_seq、stations_static、stations_seq
        y_test: 测试集真实值
        model: 已训练好的模型，需支持 evaluate(x, y) 方法

    返回:
        best_station_idx: 最优的剔除站点编号
        best_score: 最优评估得分
        all_scores: 所有站点剔除后的评估分数列表
    """
    all_scores = []
    best_score = float('inf')
    best_station_idx = -1

    for i in range(X_test['stations_static'].shape[1]):  # stations 有 28 个，测试点有 4 个（可扩展）
        x_test = copy.deepcopy(X_test)
        if i < X_test['stations_static'].shape[1]:
            n = i
            m = n + 1
            x_test['stations_static'] = tf.concat(
                [X_test['stations_static'][:, :n, :], X_test['stations_static'][:, m:, :]], axis=1
            )
            x_test['stations_seq'] = tf.concat(
                [X_test['stations_seq'][:, :n, :], X_test['stations_seq'][:, m:, :]], axis=1
            )

        print(f"======= Leave out station {i} =======")

        target_static = x_test['local_static']
        target_dynamic = x_test['local_seq']
        stations_static = x_test['stations_static']
        stations_dynamic = x_test['stations_seq']

        eval_input = {
            'target_static': target_static,
            'target_dynamic': target_dynamic,
            'stations_static': stations_static,
            'stations_dynamic': stations_dynamic
        }

        score = model.evaluate(eval_input, y_test, verbose=0)  # 设置 verbose=0 避免重复输出
        print("rmse:",score[0],'mae:',score[1])
        all_scores.append(score)

        if isinstance(score, list):
            primary_score = score[0]
        else:
            primary_score = score

        if primary_score < best_score:
            best_score = primary_score
            best_station_idx = i

    print("\n======= 最优结果 =======")
    print(f"最优站点编号: {best_station_idx}")
    print(f"最优评估得分: {best_score}")

    return best_station_idx, best_score, all_scores
# 数据导入
if __name__ == '__main__':
    for x in range(1, 9):
        Train = np.load(rf"../data_process/Data/Train_data_{x}.npy")
        Test_local = np.load(rf'../data_process/Data/Test_local_{x}.npy')
        Test_station = np.load(rf'../data_process/Data/Test_station_{x}.npy')
        # 站点划分（假设前 24 个站点用于训练，后 8 个用于测试）

        X_train, y_train, X_test, y_test = prepare_train_test_data(
            Train=Train,
            Test_local=Test_local,
            Test_station=Test_station,
            num_train_stations=28,
            num_test_stations=4
        )

        target_static = X_train['local_static']  # 目标点静态特征
        target_dynamic = X_train['local_seq']  # 目标点动态特征
        stations_static = X_train['stations_static']  # 站点静态特征
        stations_dynamic = X_train['stations_seq'] # 站点动态特征
        #建立模型
        model = MCAM()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss=rmse,
            metrics=['mae']
        )
        checkpoint_save_path = f"./checkpoint/MACM_{x}.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print("==========load the model==========")
            model.load_weights(checkpoint_save_path)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_save_path,
                    # monitor="val_accuracy",#分类
                    monitor='val_loss',#回归
                    save_best_only=True,
                    save_weights_only=True,
                )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',   # 监视验证集上的损失
            patience=5,           # 如果连续5个epoch验证集上的损失没有改善，则停止训练
            restore_best_weights=True # 训练停止时，恢复最优权重
        )
        # 使用model.fit()进行训练
        history = model.fit(
            x={
                'target_static': target_static,
                'target_dynamic': target_dynamic,
                'stations_static': stations_static,
                'stations_dynamic': stations_dynamic
            },
            y=y_train,  # 标签
            batch_size=128,  # 批次大小
            epochs=128,  # 训练轮数
            validation_split=0.14,  # 使用20%的数据作为验证集
            verbose=1,  # 打印训练过程
            callbacks=[checkpoint_callback,early_stopping],

        )

        best_station_idx, best_score, all_scores = evaluate_leave_one_station_out(X_test, y_test, model)



