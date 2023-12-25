import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary


from .attention import Seq_Transformer


class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device

        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4,
                                               heads=4, mlp_dim=64).to(device)

    def forward(self, z_aug1, z_aug2):
        seq_len = z_aug1.shape[2]# seq_len = 127

        z_aug1 = z_aug1.transpose(1, 2)
        z_aug2 = z_aug2.transpose(1, 2)# z_aug2.shape = [128, 127, 128]

        batch = z_aug1.shape[0]# batch = 128
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(
            self.device)  # randomly pick time stamps, t_samples = 24

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)# encode_samples.shape = [50, 128, 128]

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)# encode_samples的第一维都是z_aug2的特征的每个维度，127个从24->24+50,这个相当于Z_t+k
        forward_seq = z_aug1[:, :t_samples + 1, :]# [128, 25, 128], 这里的第二维每个都是特征的0->24

        c_t = self.seq_transformer(forward_seq)# [128, 64]

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)# c_t经过线性层，pred.shape = [50, 128, 128]
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))# 每一个c_t的经过线性层的第一个都和t之后第一个进行乘法，总共进行五十次，到t+k
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)

if __name__ == "__main__":
    configs = sleepEDF_Configs.Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temporal_contr_model = TC(configs, device).to(device)
    summary(temporal_contr_model, [(128,127), (128,127)])
    #summary(temporal_contr_model, input_size=features.size())