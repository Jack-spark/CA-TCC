'''
Author: Jack-spark 1411035134@qq.com
Date: 2025-03-18 13:15:11
LastEditors: Jack-spark 1411035134@qq.com
LastEditTime: 2025-03-30 21:41:25
FilePath: \CA-TCC\models\model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torch import nn
import torch.nn.functional as F
import torch

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.max1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # 第三个卷积块
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
        )
        self.reduce_dim = nn.Conv1d(
            in_channels=configs.final_out_channels,
            out_channels=16,
            kernel_size=1
        )

        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.bilstm = nn.LSTM(
            input_size=configs.final_out_channels,  # 输入特征维度
            hidden_size=configs.lstm_hidden,       # 隐层单元数
            num_layers=1,        # LSTM层数
            bidirectional=True,                    # 启用双向
            batch_first=True                       # 输入格式为(batch, seq, feature)
        )
        
        self.logits = nn.Linear(
            128, 
            configs.num_classes
        )

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        x_in = self.reduce_dim(x)
        x_in = self.pool(x_in)
        # 调整维度输入 BiLSTM
        x_in = x.permute(0, 2, 1)  # (batch, seq_len, features)
        output, _ = self.bilstm(x_in)

        lstm_out = output[:, -1, :]  # 取最后一个时间步的输出
        logits = self.logits(lstm_out)
        
        return logits, x


