from torch import nn
from torchsummary import summary

import torch

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x

if __name__ == "__main__":
    class Config(object):
        def __init__(self):
            # model configs
            self.input_channels = 1
            self.final_out_channels = 128
            self.num_classes = 5
            self.dropout = 0.35

            self.kernel_size = 25
            self.stride = 3
            self.features_len = 127

            # training configs
            self.num_epoch = 40

            # optimizer parameters
            self.optimizer = 'adam'
            self.beta1 = 0.9
            self.beta2 = 0.99
            self.lr = 3e-4

            # data parameters
            self.drop_last = True
            self.batch_size = 128

            self.Context_Cont = Context_Cont_configs()
            self.TC = TC()
            self.augmentation = augmentations()
    class augmentations(object):
        def __init__(self):
            self.jitter_scale_ratio = 1.5
            self.jitter_ratio = 2
            self.max_seg = 12


    class Context_Cont_configs(object):
        def __init__(self):
            self.temperature = 0.2
            self.use_cosine_similarity = True


    class TC(object):
        def __init__(self):
            self.hidden_dim = 64
            self.timesteps = 50

    configs = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = base_Model(configs).to(device)
    predictions, features = model(torch.randn(1, 1, 3000).to(device))
    print(features.size())#torch.Size([1，128，127】）
    #temporal_contr_model = TC(configs, device).to(device)
    summary(model, input_size=(1, 3000))
    #summary(temporal_contr_model, input_size=features.size())