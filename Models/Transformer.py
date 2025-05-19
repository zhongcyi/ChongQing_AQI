import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers



"""多头注意力机制"""
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self,d_model,num_heads):#d_model 模型的特征维度
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads ==0 #保证d_model维度能被num_head整除

        self.depth=self.d_model // self.num_heads #每个注意力头的维度
        #batch_size:批次大小 seq_q_len:序列长度，输入的单词数，d_model:模型特征维度
        self.wq=keras.layers.Dense(self.d_model) #线性层，用于生成Q （batch_size,seq_q_len,d_model）
        self.wk=keras.layers.Dense(self.d_model) #线性层，用于生成K  (batch_size,seq_k_len,d_model)
        self.wv=keras.layers.Dense(self.d_model) #线性层，用于生成V  (batch_size,seq_v_len,d_model)

        self.dense=keras.layers.Dense(self.d_model)#线性输出层

    def split_heads(self,x,batch_size):#将输入的x划为num_heads个头
        """拆分头部，将d_model分成（num_heads,depth）"""
        x=tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
        return tf.transpose(x,perm=[0,2,1,3]) # 重新排列张量的顺序，（batch_size,num_heads,seq_len,self.depth）
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """缩放的点积方法"""
        #q:(batch_size,seq_q_len,depth)
        matmul_qk=tf.matmul(q,k,transpose_b=True) #transpose_b=True 的作用是 在进行矩阵乘法时对矩阵 k（Key）进行转置
        dk=tf.cast(tf.shape(k)[-1],tf.float32)#转化为float类型
        scaled_attention_logit=matmul_qk/tf.math.sqrt(dk)
        #加入mask操作
        if mask is not None:#mask是一个0，1矩阵，1表示掩盖，0表示不掩盖
            scaled_attention_logit+=(mask * -1e9)

        attention_weights=tf.nn.softmax(scaled_attention_logit,axis=-1)#对每个 Query 和所有 Key 的相似度得分进行概率分布
        output=tf.matmul(attention_weights,v)
        return output,attention_weights
    def call(self,q,k,v,mask):
        batch_size=tf.shape(q)[0]

        q=self.wq(q)
        k=self.wk(k)
        v=self.wv(v)

        q=self.split_heads(q,batch_size)
        k=self.split_heads(k,batch_size)
        v=self.split_heads(v,batch_size)
        #scaled_attention:(batch_size,num_heads,seq_q_len,depth)
        #scaled_weight:(batch_size,num_heads,seq_q_len,seq_k_len)
        scaled_attention,attention_weights=self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention=tf.transpose(scaled_attention,perm=[0,2,1,3])#(batch_size,seq_q_len,num_heads,depth)
        #将多头的输出连接起来
        concat_attention=tf.reshape(scaled_attention,(batch_size,-1,self.d_model))
        #concat_attention:(batch_size,seq_len,d_model) 使其输出和输入一样
        output=self.dense(concat_attention)
        #映射为指定维度d_model output:(batch_size,seq_len,d_model)
        return output,attention_weights

# def feed_forward_network(d_model,hidden_units):
#     return keras.Sequential([
#                 dense1=keras.layers.Dense(hidden_units,activation='relu')
                # dense2=keras.layers.Dense(d_model)])

class feed_forward_network(keras.layers.Layer):
    """
    :param d_model: 输入模型维度，即特征维度
    :param hidden_units: 隐藏层神经元个数
    """
    def __init__(self,d_model,hidden_units):
        super(feed_forward_network,self).__init__()
        self.d_model=d_model
        self.hidden_units=hidden_units
        #升维
        self.dense1=keras.layers.Dense(hidden_units,activation='relu')
        #还原为输入维度
        self.dense2=keras.layers.Dense(d_model)
    def call(self,x):
        x=self.dense1(x)
        x=self.dense2(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    """
       :param d_model:模型维度，即输入特征维度
       :param num_heads: 多头注意力机制中的头数
       :param hidden_units: 前馈神经网络的隐藏层
       :param dropout_rate: 丢弃率
       input_size:(batch_size,seq_len,d_model)
       output_size:(batch_size,seq_len,d_model)
    """
    def __init__(self, d_model,num_heads,hidden_units,num_expert=6,dropout_rate=0.2):
        super(EncoderLayer, self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.hidden_units=hidden_units
        self.dropout_rate=dropout_rate
        self.num_expert=num_expert
        #多头自注意力层
        self.multiheadattention = MultiHeadAttention(self.d_model,self.num_heads)
        #归一化层，即layernormalization
        self.layernormalization=keras.layers.LayerNormalization(epsilon=1e-6)
        #因为数据为气象数据，特征长度一致，所以这里利用batchnormalization,而不是layernormlization
        # self.batchnormalization= keras.layers.BatchNormalization(epsilon=1e-6)
        self.dropout=keras.layers.Dropout(self.dropout_rate)
        self.feed_forward=feed_forward_network(self.d_model,self.hidden_units)

    def call(self,x,mask=None,training=False):#traning为bool值
        #x:(batch_size,seq_len,d_model)
        x1,_=self.multiheadattention(x,x,x,mask)
        #x1:(batch_size,seq_len,d_model)
        x1=self.dropout(x1,training=training)#只在训练时丢弃
        x_norm=self.layernormalization(x1+x)#残差连接
        x_feed_out=self.feed_forward(x_norm)
        x_feed_out=self.dropout(x_feed_out,training=training)
        x_out=self.layernormalization(x_norm+x_feed_out)
        return x_out

    def compute_output_shape(self, input_shape):
        # 计算输出形状，假设输出形状是 (batch_size, seq_len, feature_dim)
        batch_size, seq_len, feature_dim = input_shape
        return (batch_size, seq_len, feature_dim)  # 根据具体情况调整输出形状

class DecoderLayer(keras.layers.Layer):
    """
    :param d_model:模型维度，即输入特征维度
    :param num_heads: 多头注意力机制中的头数
    :param hidden_units: 前馈神经网络的隐藏层
    :param dropout_rate: 丢弃率
    """
    def __init__(self,d_model,num_heads,hidden_units,dropout_rate=0.2):
        super(DecoderLayer, self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.dropout_rate=dropout_rate
        self.hidden_units=hidden_units

        self.masked_multi_head_attention1=MultiHeadAttention(self.d_model,self.num_heads)
        self.dropout1=keras.layers.Dropout(self.dropout_rate)
        self.layernormlization1=keras.layers.LayerNormalization(epsilon=1e-6)

        self.multi_head_attention=MultiHeadAttention(self.d_model,self.num_heads)
        self.dropout2=keras.layers.Dropout(self.dropout_rate)
        self.layernormlization2=keras.layers.LayerNormalization(epsilon=1e-6)

        self.feed_forward=feed_forward_network(self.d_model,self.hidden_units)
        self.dropout3=keras.layers.Dropout(self.dropout_rate)
        self.layernormlization3=keras.layers.LayerNormalization(epsilon=1e-6)
    def call(self,x,encoder_out,mask=None,training=False):

        x1,_=self.masked_multi_head_attention1(x,x,x,mask)
        x1=self.dropout1(x1,training=training)
        x_norm1=self.layernormlization1(x1+x)
        #交互层，k,v来自encoder,q来自decoder
        x2,_=self.multi_head_attention(x_norm1,encoder_out,encoder_out,mask)
        x2=self.dropout2(x2,training=training)
        x_norm2=self.layernormlization2(x2+x_norm1)

        x3=self.feed_forward(x_norm2)
        x3=self.dropout3(x3,training=training)
        x_norm3=self.layernormlization3(x3+x_norm2)

        return x_norm3
class Outputlayer(keras.layers.Layer):
    def __init__(self,hidden_units,out_labels,dropout_rate=0.2):
        super(Outputlayer,self).__init__()
        self.hidden_units=hidden_units
        self.out_labels=out_labels
        self.dropout_rate=dropout_rate

        self.flatten=keras.layers.Flatten()
        self.dense1=keras.layers.Dense(self.hidden_units,activation='relu')
        self.dropout1=keras.layers.Dropout(self.dropout_rate)
        self.dense2=keras.layers.Dense(self.hidden_units//2,activation='relu')
        self.dropout2=keras.layers.Dropout(self.dropout_rate)
        self.dense3=keras.layers.Dense(self.out_labels,activation='softmax')

    def call(self,x):
        x=self.flatten(x)
        x=self.dense1(x)
        x=self.dropout1(x)
        x=self.dense2(x)
        x=self.dropout2(x)
        x=self.dense3(x)
        return x

class Transformer(keras.Model):
    def __init__(self,d_model,num_heads,hidden_units,out_labels,droupout_rate=0.2):
        super(Transformer,self).__init__()
        self.d_model=d_model
        self.num_heads = num_heads
        self.hidden_units=hidden_units
        self.droupout_rate=droupout_rate
        self.out_labels=out_labels
        #可以每个encoder的参数不一样 shape都是（batch_size,seq_len,d_model）
        self.encoder1=EncoderLayer(self.d_model,self.num_heads,self.hidden_units,self.droupout_rate)
        self.dcoder1=DecoderLayer(self.d_model,self.num_heads,self.hidden_units,self.droupout_rate)
        self.outputlayer=Outputlayer(self.hidden_units,self.out_labels,self.droupout_rate)
    def call(self,x,mask=None):
        enc=self.encoder1(x,mask)
        dnc=self.dcoder1(enc,enc,mask)
        out=self.outputlayer(dnc)
        return out

