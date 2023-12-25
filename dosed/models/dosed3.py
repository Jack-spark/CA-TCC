import warnings
from collections import OrderedDict
import sys
sys.path.append("..")
import torch.nn as nn

from ..functions import Detection
from .base import BaseNet, get_overlerapping_default_events


class DOSED3(BaseNet):

    def __init__(self,
                 input_shape,
                 number_of_classes,
                 detection_parameters,
                 default_event_sizes,
                 k_max=6,
                 kernel_size=5,
                 pdrop=0.1,
                 fs=256):

        super(DOSED3, self).__init__()
        self.number_of_channels, self.window_size = input_shape
        self.number_of_classes = number_of_classes + 1  # eventless, real events

        detection_parameters["number_of_classes"] = self.number_of_classes
        self.detector = Detection(**detection_parameters)

        self.k_max = k_max
        self.kernel_size = kernel_size
        self.pdrop = pdrop

        if max(default_event_sizes) > self.window_size:
            warnings.warn("Detected default_event_sizes larger than"
                          " input_shape! Consider reducing them")

        # Localizations to default tensor
        self.localizations_default = get_overlerapping_default_events(
            window_size=self.window_size,
            default_event_sizes=default_event_sizes#窗口尺寸和默认事件的尺寸
        )#

        # model
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict([
                        ("conv_{}".format(k - 1), nn.Conv1d(
                            in_channels=4 * (2 ** (k - 1)) if k > 1 else self.number_of_channels,
                            out_channels=4 * (2 ** k),
                            kernel_size=self.kernel_size,
                            padding=2
                        )),
                        ("batchnorm_{}".format(k - 1), nn.BatchNorm1d(4 * (2 ** k))),
                        ("relu_{}".format(k), nn.ReLU()),
                        ("dropput_{}".format(k), nn.Dropout(self.pdrop)),
                        ("max_pooling_{}".format(k), nn.MaxPool1d(kernel_size=2)),
                    ])
                ) for k in range(1, self.k_max + 1)
            ]
        )
        self.localizations = nn.Conv1d(
            in_channels=4 * (2 ** (self.k_max)),
            out_channels=2 * len(self.localizations_default),
            kernel_size=int(self.window_size / (2 ** (self.k_max))),
            padding=0,
        )

        self.classifications = nn.Conv1d(
            in_channels=4 * (2 ** (self.k_max)),
            out_channels=self.number_of_classes * len(self.localizations_default),
            kernel_size=int(self.window_size / (2 ** (self.k_max))),
            padding=0,
        )

        #self.print_info_architecture(fs)

    def forward(self, x):#x.size=8,2,320
        batch = x.size(0)#batch=8
        for block in self.blocks:
            x = block(x)
        #经过blocks后x.size=8，128，10
        #self.localizations=8,126,1
        #self.localizations(x).squeeze().size=8,126
        feature = x
        localizations = self.localizations(x).squeeze().view(batch, -1, 2)#localizations.size=8,3,2
        classifications = self.classifications(x).squeeze().view(batch, -1, self.number_of_classes)#classifications.size=8,3,2
        
        return localizations, classifications, self.localizations_default, feature

    def print_info_architecture(self, fs):

        size = self.window_size
        receptive_field = 0
        print("\nInput feature map size: {}".format(size))#输入有320个点
        print("Input receptive field: {}".format(receptive_field))
        print("Input size in seconds: {} s".format(size / fs))
        print("Input receptive field in seconds: {} s \n".format(receptive_field / fs))

        kernal_size = self.kernel_size

        size //= 2
        receptive_field = kernal_size + 1
        print("After layer 1:")
        print("\tFeature map size: {}".format(size))
        print("\tReceptive field: {}".format(receptive_field))
        print("\tReceptive field in seconds: {} s".format(receptive_field / fs))

        for layer in range(2, self.k_max + 1):
            size //= 2
            receptive_field += (kernal_size // 2) * 2 * 2 ** (layer - 1)  # filter
            receptive_field += 2 ** (layer - 1)  # max_pool
            print("After layer {}:".format(layer))
            print("\tFeature map size: {}".format(size))
            print("\tReceptive field: {}".format(receptive_field))
            print("\tReceptive field in seconds: {} s".format(
                receptive_field / fs))
        print("\n")

# if __name__ == "__main__":
#     default_event_sizes = [0.7, 1, 1.3]#如何选取默认事件的大小，要检测的事件是脑电活动（spindle），大小约为1s
#     k_max = 5
#     kernel_size = 5
#     probability_dropout = 0.1
#     sampling_frequency = 32
#     net_parameters = {
#     "detection_parameters": {
#         "overlap_non_maximum_suppression": 0.5,
#         "classification_threshold": 0.7
#     },
#     "default_event_sizes": [
#         default_event_size * sampling_frequency
#         for default_event_size in default_event_sizes
#     ],
#     "k_max": k_max,
#     "kernel_size": kernel_size,
#     "pdrop": probability_dropout,
#     "fs": sampling_frequency,   # just used to print architecture info with right time
#     "input_shape": (2, 320),# 2，10s，采样率32hz
#     "number_of_classes": 1,# 1

# }