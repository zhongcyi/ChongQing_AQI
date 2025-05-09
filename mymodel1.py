import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import MultiHeadAttention, Dense, Add, Multiply, TimeDistributed, Dropout, \
    LayerNormalization, Flatten, GlobalAveragePooling1D
import Transformer
import numpy as np
class Expert(keras.Model):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = layers.Dense(hidden_dim, activation='gelu')
        self.fc2 = layers.Dense(hidden_dim, activation='gelu')

    def call(self, x):
        x = self.fc1(x)
        return self.fc2(x)


class AttentionGate(keras.Model):
    def __init__(self, num_experts):
        super().__init__()
        self.attention = layers.Attention()
        self.fc = layers.Dense(num_experts, activation='softmax')

    def call(self, infer_point, stations):
        combined = tf.concat([tf.expand_dims(infer_point, axis=1), stations],
                             axis=1)  # (batch, num_stations+1, input_dim)
        attn_output = self.attention([combined, combined])
        logits = self.fc(attn_output)  # (batch, num_stations+1, num_experts)
        return logits


class MoE(keras.layers.Layer):
    def __init__(self, hidden_dim, num_stations):
        super().__init__()
        self.infer_expert = Expert(hidden_dim)  # 推断点专家
        self.station_experts = [Expert(hidden_dim) for _ in range(num_stations)]
        self.gate = AttentionGate(num_stations + 1)  # 站点数 + 推断点

    def call(self, infer_point, stations):
        station_outputs = tf.stack([expert(stations[:, i, :]) for i, expert in enumerate(self.station_experts)], axis=1)
        infer_output = tf.expand_dims(self.infer_expert(infer_point), axis=1)  # (batch, 1, hidden_dim)
        all_outputs = tf.concat([infer_output, station_outputs], axis=1)  # (batch, num_stations+1, hidden_dim)

        gates = self.gate(infer_point, stations)  # (batch, num_stations+1, num_stations+1)
        gated_output = tf.einsum('bij,bik->bk', gates, all_outputs)  # (batch, hidden_dim)
        return gated_output


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, reduction=16):
        super(ChannelAttention, self).__init__()
        self.reduction = reduction  # 降维比例

    def build(self, input_shape):
        channels = input_shape[-1]  # 获取通道数 C
        self.dense1 = Dense(channels // self.reduction, activation='gelu', use_bias=False)
        self.dense2 = Dense(channels, activation='sigmoid', use_bias=False)

    def call(self, inputs):
        # 平均池化 & 最大池化
        avg_pool = tf.reduce_mean(inputs, axis=1, keepdims=True)  # (B, 1, C)
        max_pool = tf.reduce_max(inputs, axis=1, keepdims=True)  # (B, 1, C)

        # MLP 计算注意力分数
        avg_out = self.dense2(self.dense1(avg_pool))  # (B, 1, C)
        max_out = self.dense2(self.dense1(max_pool))  # (B, 1, C)

        # 加和融合注意力权重
        attention = Add()([avg_out, max_out])  # (B, 1, C)

        # 加权输入特征
        return Multiply()([inputs, attention])  # (B, C)

    def compute_output_shape(self, input_shape):
        return input_shape


def positional_encoding(sequence_length, d_model):
    """
    Generate positional encoding using sinusoidal functions.

    Args:
        sequence_length: Length of the sequence.
        d_model: Dimensionality of the encoding.

    Returns:
        A tensor of shape (sequence_length, d_model) with positional encodings.
    """
    position = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]  # Shape: (seq_len, 1)
    div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(np.log(10000.0) / d_model))
    # Compute sine and cosine values
    sin_terms = tf.sin(position * div_term)  # Shape: (seq_len, d_model/2)
    cos_terms = tf.cos(position * div_term)  # Shape: (seq_len, d_model/2)
    # Concatenate sine and cosine terms along the last dimension
    pos_encoding = tf.concat([sin_terms, cos_terms], axis=-1)  # Shape: (seq_len, d_model)
    return pos_encoding


class LearnableFeatureLayer(tf.keras.layers.Layer):
    def __init__(self):
        """
        自定义层：为输入数据增加一个可学习的特征，并最终用于预测输出。
        """
        super(LearnableFeatureLayer, self).__init__()

        # 创建一个 learnable feature 参数
        self.learnable_feature = self.add_weight(
            shape=(1, 1),  # 只有一个 learnable feature
            initializer="random_normal",
            trainable=True,  # 让它可以被训练优化
            name="learnable_feature"
        )

    def call(self, inputs):
        """
        在输入数据的基础上拼接 learnable_feature
        """
        batch_size = tf.shape(inputs)[0]  # 获取 batch size
        # 复制 batch_size 和 seq_len 维度，使其变成 (batch_size, seq_len, 1)
        expanded_feature = tf.tile(self.learnable_feature, [batch_size, 1])
        # 将 learnable_feature 拼接到输入特征
        outputs = tf.concat([inputs, expanded_feature], axis=-1)  # (batch_size, seq_len, 21)

        return outputs, expanded_feature  # 额外返回 learnable_feature 作为最终的预测值


class Model(tf.keras.models.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.learnable_feature = LearnableFeatureLayer()
        self.se1 = ChannelAttention()
        self.se2 = TimeDistributed(ChannelAttention())
        self.se = TimeDistributed(ChannelAttention())
        self.output_dense = Dense(1, activation='linear')
        self.output_dense2 = Dense(1, activation='linear')
        self.local_encoder = Transformer.EncoderLayer(d_model=14, num_heads=2, hidden_units=32)
        self.stations_encoder = TimeDistributed(Transformer.EncoderLayer(d_model=14, num_heads=2, hidden_units=32))

        # 添加BatchNormalization
        self.moe = MoE(64, 27)
        self.flatten1 = Flatten()
        self.flatten2 = TimeDistributed(Flatten())
        self.out = Dense(32, activation='gelu')

        self.encoder = Transformer.EncoderLayer(d_model=37, num_heads=1, hidden_units=128)
        self.dropout = Dropout(0.1)

    def call(self, inputs):
        local_static, local_seq, stations_static, stations_seq, T = inputs['local_static'], inputs['local_seq'][...,
                                                                                            :6], \
            inputs['stations_static'], inputs['stations_seq'], inputs['T']
        # local_static = local_static[:,1,1,:]
        # local_seq = local_seq[:,:,1,1,:]
        # stations_static = stations_static[:,:,1,1,:]
        # stations_seq = stations_seq[:,:,:,1,1,:]
        t = tf.expand_dims(T, axis=-1)
        p_T = []
        for period in [1.0, 7.0, 30.5, 365.0]:
            sin_term = tf.sin(2 * np.pi * t / (period * 24))
            cos_term = tf.cos(2 * np.pi * t / (period * 24))
            p_T.extend([sin_term, cos_term])
        p_T = tf.concat(p_T, axis=-1)
        print('=======', p_T.shape)
        aqi = tf.expand_dims(stations_seq[:, :, -1, -1], axis=-1)
        stations_seq = stations_seq[..., :6]
        print(local_seq.shape)
        # 确保输入序列的维度是 (batch_size, time_steps, lstm_units)
        # local_position = positional_encoding(8, 6)
        local_seq = tf.concat([local_seq, tf.tile(tf.expand_dims(p_T, axis=1), [1, 8, 1])], axis=-1)
        local_encoder = self.local_encoder(local_seq)[:, -1, :]
        # local_encoder = self.flatten1(local_encoder)
        # stations_position = positional_encoding(8, 6)
        # stations_position = tf.expand_dims(stations_position, axis=0)
        station_p_T = tf.expand_dims(tf.expand_dims(p_T, axis=1), axis=1)
        station_p_T = tf.tile(station_p_T, [1, 27, 8, 1])  # 复制到形状 [?,27,8]
        stations_seq = tf.concat([stations_seq, station_p_T], axis=-1)
        station_encoder = self.stations_encoder(stations_seq)[:, :, -1, :]
        # station_encoder = self.flatten2(station_encoder)

        local_features = tf.concat([local_encoder, local_static], axis=-1)
        local_features, pred = self.learnable_feature(local_features)

        stations_feature = tf.concat([station_encoder, stations_static, aqi], axis=-1)


        station_se = self.se2(stations_feature)
        # station_se = self.dense_se2(station_se)
        local_se = self.se1(local_features)

        local_se = tf.expand_dims(local_se,axis=1)
        concat_se = tf.concat([station_se,local_se],axis=1)
        concat_se = self.encoder(concat_se)
        station_se = concat_se[:, :-1, :]
        local_se = concat_se[:, -1, :]
        # station_se = self.encoder(station_se)
        final_out = self.moe(local_se, station_se)
        final_out = self.out(final_out)
        final_out = tf.concat([final_out, pred], axis=-1)

        output = self.output_dense(final_out)

        return output


if __name__ == '__main__':
        x=7
        Train = np.load(
            rf'train_processed_data_4_{x}.npy')
        Test = np.load(
            rf'test_processed_data_4_{x}.npy')
        station1 = np.load(
            rf'train_processed_data_4_{x}_S1.npy')
        station2 = np.load(
            rf'test_processed_data_4_{x}_S2.npy')
        # 站点划分（假设前 24 个站点用于训练，后 8 个用于测试）
        num_total_stations = 32
        num_train_stations = 28
        num_test_stations = 4

        train_stations = np.arange(0, num_train_stations)  # 训练站点索引
        test_stations = np.arange(0, num_test_stations)  # 测试站点索引

        # 训练数据构造
        X_train_list = []
        y_train_list = []
        for i in train_stations:
            local_static = Train[:, i, -1, 3:25]  # 本地静态特征
            local_seq = Train[:, i, :, 25:35]  # 本地时序特征
            station_mask = np.setdiff1d(train_stations, i)  # 其余 23 个站点作为 stations
            stations_static = Train[:, station_mask, -1, 3:25]  # 监测站静态特征
            stations_seq = Train[:, station_mask, :, 25:]  # 监测站时序特征
            y = Train[:, i, -1, -1]  # 目标值
            T = Train[:, i, -1, 0]
            X_train_list.append({'local_static': local_static,
                                 'local_seq': local_seq,
                                 'stations_static': stations_static,
                                 'stations_seq': stations_seq,
                                 'T': T})

            y_train_list.append(y)

        X_train = {key: np.array([x[key] for x in X_train_list]) for key in X_train_list[0]}
        y_train = np.array(y_train_list)
        # 合并所有样本并调整维度
        for key in X_train:
            X_train[key] = np.concatenate(X_train[key], axis=0)  # [样本数*24, ...]
            print(X_train[key].shape)
        print(X_train['T'])
        y_train = np.concatenate(y_train, axis=0)  # [样本数*24]

        X_test_list = []
        y_test_list = []
        for i in test_stations:
            local_static = station1[:, i, -1, 3:25]  # 本地静态特征
            local_seq = station1[:, i, :, 25:35]  # 本地时序特征
            stations_static = station2[:, :, -1, 3:25]  # 24 个训练站点作为 stations
            stations_seq = station2[:, :, :, 25:]  # 训练站点的时序特征
            y = station1[:, i, -1, -1]  # 目标值
            T = station1[:, i, -1, 0]
            X_test_list.append({'local_static': local_static,
                                'local_seq': local_seq,
                                'stations_static': stations_static,
                                'stations_seq': stations_seq,
                                'T': T}
                               )

            y_test_list.append(y)

        X_test = {key: np.array([x[key] for x in X_test_list]) for key in X_test_list[0]}
        y_test = np.array(y_test_list)

        # 合并所有测试样本并调整维度
        for key in X_test:
            X_test[key] = np.concatenate(X_test[key], axis=0)  # [样本数*8, ...]

        y_test = np.concatenate(y_test, axis=0)  # [样本数*8]
        # 创建模型实例
        model = Model()


        def rmse(y_true, y_pred):
            return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss=rmse,
            metrics=['mae']
        )

        # 构建模型
        checkpoint_save_path = f"./checkpoint2/my_model_{x}.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print("==========load the model==========")
            model.load_weights(checkpoint_save_path)
        #
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

        # 训练模型
        model.fit(
            x=X_train,
            y=y_train,
            epochs=128,
            batch_size=128,
            validation_split=0.14,
            callbacks=[checkpoint_callback, early_stopping],
        )
        import copy

        x_test = copy.deepcopy(X_test)
        for i in range(28):
            n = i
            m = n + 1
            x_test['stations_static'] = tf.concat(
                [X_test['stations_static'][:, :n, :], X_test['stations_static'][:, m:, :]],
                axis=1)
            x_test['stations_seq'] = tf.concat([X_test['stations_seq'][:, :n, :], X_test['stations_seq'][:, m:, :]], axis=1)

            # 评估模型
            print('=======', i, '=======')
            model.evaluate(x_test, y_test)