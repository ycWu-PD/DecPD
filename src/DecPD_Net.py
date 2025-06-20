import torch
import numpy as np
import torch.nn as nn
from torch.nn import init


class MultiHeadAttention(nn.Module):  # out [64, 326, 256]  out [64, 326, 256]  out [64, 326, 256]
    '''
    MultiHeadAttention 多头注意力机制
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=0.2):
        '''
        :param d_model: Output dimensionality of the model 输出模型的维度
        :param d_k: Dimensionality of queries and keys 查询和键的维度
        :param d_v: Dimensionality of values 值的维度
        :param h: Number of heads  头的数量
        '''
        super(MultiHeadAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model) 查询向量 b_s = batch_size nq：查询数量
        :param keys: Keys (b_s, nk, d_model)   nk：键的数量
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking. 注意力值的掩码
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).  权重
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        # q = self.fc_q(queries).view(-1, 128, 8, 128).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        # k = self.fc_k(keys).view(-1, 128, 8, 128).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        # v = self.fc_v(values).view(-1, 128, 8, 128).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


def conv_block(in_channel, out_channel):  # 一个卷积块
    layer = nn.Sequential(
        nn.BatchNorm1d(in_channel),
        nn.ReLU(),
        nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer


class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super().__init__()  # growth_rate => k => out_channel
        block = []
        channel = in_channel  # channel => in_channel
        for i in range(num_layers):
            # block.append(conv_block(growth_rate * i + channel, growth_rate))  # 修改 卷积层通道数即学习率*i
            block.append(conv_block(channel, growth_rate))  # 这样的操作减少了参数量，但增加连接特征，提高模型性能
            channel += growth_rate  # 连接每层的特征
        self.net = nn.Sequential(*block)  # 实现简单的顺序连接模型
        # 必须确保前一个模块的输出大小和下一个模块的输入大小是一致的

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)  # contact同维度拼接特征，stack(是把list扩维连接
            # torch.cat()是为了把多个tensor进行拼接，在给定维度上对输入的张量序列seq 进行连接操作
            # inputs : 待连接的张量序列，可以是任意相同Tensor类型的python 序列
            # dim : 选择的扩维, 必须在0到len(inputs[0])之间，沿着此维连接张量序列
        return x


def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm1d(in_channel),
        nn.ReLU(),
        nn.Conv1d(in_channel, out_channel, 1),  # kernel_size = 1 1x1 conv
        nn.AvgPool1d(2, 2)  # 2x2 pool
    )
    return trans_layer


class DenseNet(nn.Module):  # 修改num逻辑迭代 降低稠密块的卷积层数
    # block_layers 每个稠密块的卷积层数 [6, 12, 24, 16] # in_channel = hide_size1 =128
    def __init__(self, in_channel, num_classes, num_channels=64, growth_rate=32, block_layers=[4, 4, 4, 4]):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channel, 64, 7, 2, 3),  # padding=3 参数要熟悉
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(3, 2, padding=1)
        )
        blks = []
        for i, num in enumerate(block_layers):
            blks.append(dense_block(num_channels, growth_rate, num))
            num_channels += num * growth_rate  # 64+4*32=192  :64 192 \96 224\ 112 240 \120 248
            if i != len(block_layers) - 1:
                blks.append(transition(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        self.blks = nn.Sequential(*blks)
        # self.DB1 = self._make_dense_block(64, growth_rate, num=block_layers[0])  # 64
        # self.TL1 = self._make_transition_layer(256)
        # self.DB2 = self._make_dense_block(128, growth_rate, num=block_layers[1])  # 128
        # self.TL2 = self._make_transition_layer(512)
        # self.DB3 = self._make_dense_block(256, growth_rate, num=block_layers[2])  # 256
        # self.TL3 = self._make_transition_layer(1024)
        # self.DB4 = self._make_dense_block(512, growth_rate, num=block_layers[3])   # 512
        self.global_avgpool = nn.Sequential(  # 全局平均池化
            nn.BatchNorm1d(num_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(num_channels, num_classes)  # fc层

    def forward(self, x):
        x = self.block1(x)  # [64,128,326] ->[64,64,82] 128->64：稠密块  326->82：池化
        x = self.blks(x)  # [64,64,82] ->[64,248,10]
        # x = self.DB1(x)
        # x = self.TL1(x)
        # x = self.DB2(x)
        # x = self.TL2(x)
        # x = self.DB3(x)
        # x = self.TL3(x)
        # x = self.DB4(x)
        x = self.global_avgpool(x)  # [64,248,10] -> [64,248,1]
        x = self.classifier(x.squeeze(2))  # [64,248,1] ->[64,248]->[64,num_classes:6]
        return x  # [32,6]
    # def _make_dense_block(self, channels, growth_rate, num):  # num是块的个数
    #     block = []
    #     block.append(dense_block(channels, growth_rate, num))
    #     channels += num * growth_rate  # 特征变化 # 这里记录下即可，生成时dense_block()中也做了变化
    #     return nn.Sequential(*block)
    #
    # def _make_transition_layer(self, channels):
    #     block = []
    #     block.append(transition(channels, channels // 2))  # channels // 2就是为了降低复杂度 θ = 0.5 （减半高度和宽度，减半通道数）
    #     return nn.Sequential(*block)


class SoftmaxClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 第一层全连接
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(128, num_classes)  # 输出层

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return nn.Softmax(dim=1)(x)  # 使用 Softmax 进行分类


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.4)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Bidirectional LSTM, so *2
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)  # Softmax along the sequence dimension
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)

        # Attention mechanism
        attention_weights = self.attention(out)
        attended_out = torch.sum(attention_weights * out, dim=1)

        # Fully connected layer
        output = self.fc(attended_out)

        return output


class DecPD(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(DecPD, self).__init__()
        self.device = device
        self.hidden_size_1 = hidden_size
        self.hidden_size_2 = hidden_size * 2
        self.hidden_size_3 = hidden_size * 4
        self.num_layers = num_layers
        self.dense1 = DenseNet(self.hidden_size_1, num_classes).to(device)
        self.CNN = nn.Sequential(
            nn.Conv1d(input_size, self.hidden_size_1, 3, padding=1),
            nn.BatchNorm1d(self.hidden_size_1, momentum=0.1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv1d(self.hidden_size_1, self.hidden_size_2, 3, padding=1),
            nn.BatchNorm1d(self.hidden_size_2, momentum=0.1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv1d(self.hidden_size_2, self.hidden_size_3, 3, padding=1),
            nn.BatchNorm1d(self.hidden_size_3, momentum=0.1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv1d(self.hidden_size_3, self.hidden_size_2, 3, padding=1),
            nn.BatchNorm1d(self.hidden_size_2, momentum=0.1),
            nn.ReLU(True),
            nn.Dropout(0.2)  # 原来0.2

            # # 使用 He 初始化
            # nn.init.kaiming_uniform_(self.fc.weight)
            # nn.init.zeros_(self.fc.bias)
        )
        self.lstm = nn.LSTM(input_size, self.hidden_size_2, num_layers=num_layers, batch_first=True, dropout=0.2).to(device)  # bidirectional BiLSTM bidirectional=True

        self.gru_layer = nn.GRU(self.hidden_size_2, self.hidden_size_1).to(device)

        self.selfattention = MultiHeadAttention(d_model=self.hidden_size_1, d_k=self.hidden_size_1,
                                                d_v=self.hidden_size_1, h=8).to(device)

        self.fc = nn.Linear(hidden_size, num_classes).to(device)

    def forward(self, x):
        input_X = x.transpose(1, 2)  # x[64,326,36] [batch_size, seq_length, input_size(feature)]  input[64,36,326]
        out_CNN = self.CNN(input_X)  # out[64,256,326]
        out_CNN = out_CNN.transpose(1, 2)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size_2).to(self.device)  # h0[10,64,128]
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size_2).to(self.device)  # c0[10,64,128]
        # Forward propagate LSTM
        out_BiLSTM, _ = self.lstm(x, (h0, c0))  # out_BiLSTM[64,326,256]  (batch_size, seq_length, hidden_size)

        out2 = out_BiLSTM + out_CNN  # out2[64,326,256]
        out3 = nn.ReLU(True)(out2)  # out3[64,326,256]
        # gru
        out4, _ = self.gru_layer(out3)  # out4[64,326,128]
        # ATTENTION
        out5 = self.selfattention(out4, out4, out4)  # out5[64,326,128]
        # dense
        finally_out = self.dense1(out5.transpose(1, 2))  # finally_out[64,7]

        return finally_out

