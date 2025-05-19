import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, concatenate
from tensorflow.keras.regularizers import l2
import numpy as np

# Attention 层
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, l2_reg=1e-4, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attn_weights',
            shape=(input_shape[-1], self.hidden_dim),
            initializer='glorot_uniform',
            regularizer=l2(self.l2_reg)
        )
        self.b = self.add_weight(
            name='attn_bias',
            shape=(self.hidden_dim,),
            initializer='zeros'
        )
        self.v = self.add_weight(
            name='attn_v',
            shape=(self.hidden_dim, 1),
            initializer='glorot_uniform'
        )
        super().build(input_shape)

    def call(self, inputs):
        scores = tf.tanh(tf.matmul(inputs, self.W) + self.b)  # [B, S, H]
        scores = tf.matmul(scores, self.v)  # [B, S, 1]
        weights = tf.nn.softmax(scores, axis=1)  # [B, S, 1]
        weighted = tf.reduce_sum(inputs * weights, axis=1)  # [B, F]
        return weighted


# ADAIN模型 class实现
class ADAINModel(tf.keras.Model):
    def __init__(self, dropout_rate=0.3, l2_reg=1e-4, **kwargs):
        super(ADAINModel, self).__init__(**kwargs)

        self.dense_1 = Dense(200, activation='relu', kernel_regularizer=l2(l2_reg))
        self.dropout_1 = Dropout(dropout_rate)
        self.lstm_local = LSTM(300, return_sequences=False, kernel_regularizer=l2(l2_reg))
        self.time_distributed_lstm = tf.keras.layers.TimeDistributed(
            LSTM(300, return_sequences=False, kernel_regularizer=l2(l2_reg))
        )

        self.dense_2 = Dense(200, activation='relu', kernel_regularizer=l2(l2_reg))
        self.dropout_2 = Dropout(dropout_rate)

        self.dense_3 = Dense(200, activation='relu', kernel_regularizer=l2(l2_reg))
        self.dropout_3 = Dropout(dropout_rate)

        # Attention layer
        self.attention_layer = AttentionLayer(hidden_dim=64, l2_reg=l2_reg)

        # Output layer
        self.output_layer = Dense(1, activation='linear')

    def call(self, inputs):
        local_static, local_seq, stations_static, stations_seq = inputs['local_static'], inputs['local_seq'], inputs['stations_static'], inputs['stations_seq']
        # 本地特征处理B,200
        local_fnn = self.dense_1(local_static)
        'B,300'
        local_fnn = self.dropout_1(local_fnn)
        'B,300'
        local_lstm = self.lstm_local(local_seq)
        # 监测站时序特征 LSTM
        'B,23,300'
        stations_lstm = self.time_distributed_lstm(stations_seq)
        # 静态特征处理
        'B,23,200'
        stations_fnn = self.dense_2(stations_static)
        stations_fnn = self.dropout_2(stations_fnn)
        'B,23,200'
        # 合并静态和时序特征
        'B,23,500'
        station_features = concatenate([stations_fnn, stations_lstm], axis=-1)
        # 注意力机制
        'B,500'
        weighted_stations = self.attention_layer(station_features)
        # 特征融合
        merged = concatenate([local_fnn, local_lstm, weighted_stations],axis=-1)
        merged = self.dense_3(merged)
        merged = self.dropout_3(merged)
        # 输出层
        output = self.output_layer(merged)
        return output
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
# 使用方法
if __name__ == '__main__':
    # 载入数据
    for x in range(1,29):
        Train = np.load(rf"..\data_process\Data\Train_data_{x}.npy")
        Test_local = np.load(rf'..\data_process\Data\Test_local_{x}.npy')
        Test_station = np.load(rf'..\data_process\Data\Test_station_{x}.npy')
        num_total_stations = 32
        num_train_stations = 28
        num_test_stations = 4
        train_stations = np.arange(num_train_stations)
        test_stations = np.arange(num_test_stations)
        # 构造训练集
        X_train, y_train = build_dataset(Train, None, train_stations, is_train=True)
        # 构造测试集
        X_test, y_test = build_dataset(Test_local, Test_station, test_stations, is_train=False)
        # 创建模型实例
        model = ADAINModel()
        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=rmse,
            metrics=['mae']
        )
        # 构建模型
        checkpoint_save_path = f"./checkpoint/ADAIN_{x}.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print("==========load the model==========")
            model.load_weights(checkpoint_save_path)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_save_path,
            monitor='val_loss',  # 回归
            save_best_only=True,
            save_weights_only=True,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # 监视验证集上的损失
            patience=5,  # 如果连续5个epoch验证集上的损失没有改善，则停止训练
            restore_best_weights=True  # 训练停止时，恢复最优权重
        )
        # 训练模型
        model.fit(
            x=X_train,
            y=y_train,
            epochs=140,
            batch_size=128,
            validation_split=0.14,
            callbacks=[checkpoint_callback, early_stopping],
        )
        # 评估模型
        print(f'第{x}折模型的评估：')
        model.evaluate(X_test, y_test)
