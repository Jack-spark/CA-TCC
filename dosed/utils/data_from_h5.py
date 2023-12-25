"""Transform a folder with h5 files into a dataset for dosed"""

import numpy as np
import sys
sys.path.append("..")
import h5py
from dosed.preprocessing import normalizers
from scipy.interpolate import interp1d


def get_h5_data(filename, signals, fs):
    with h5py.File(filename, "r") as h5:

        signal_size = int(fs * min(
            set([h5[signal["h5_path"]].size / signal['fs'] for signal in signals])
        ))# 在这里完成里降采样操作，将signal除去原始采样率，在乘上降采样率，得到降采样后的数据尺寸

        t_target = np.cumsum([1 / fs] * signal_size)#创建一个时间目标数组t_target，它是一个等差数列，差值为1/fs，长度为signal_size
        data = np.zeros((len(signals), signal_size))
        for i, signal in enumerate(signals):
            t_source = np.cumsum([1 / signal["fs"]] *
                                 h5[signal["h5_path"]].size)#创建一个时间源数组t_source，它是一个等差数列，差值为1/信号的采样频率，长度为信号的大小
            normalizer = normalizers[signal['processing']["type"]](**signal['processing']['args'])
            data[i, :] = interp1d(t_source, normalizer(h5[signal["h5_path"]][:]),
                                  fill_value="extrapolate")(t_target)# 使用插值函数interp1d，将源数据（时间源数组和正规化后的信号数据）插值到目标数据（时间目标数组）上，并将结果存储在data数组中
    return data# 完成了降采样操作，返回data数组


def get_h5_events(filename, event, fs):
    with h5py.File(filename, "r") as h5:
        starts = h5[event["h5_path"]]["start"][:]
        durations = h5[event["h5_path"]]["duration"][:]
        assert len(starts) == len(durations), "Inconsistents event durations and starts"

        data = np.zeros((2, len(starts)))
        data[0, :] = starts * fs
        data[1, :] = durations * fs
    return data#将开始时间和持续时间乘以采样率 fs，然后存储到 data 中。开始时间存储在 data 的第一行，持续时间存储在 data 的第二行。



if __name__ == '__main__':
    filename = 'D:/Desktop/research/github repo/dosed/data/h5' + '/a6624e57-c003-4c32-8ac5-03fc5770ccf8.h5'
    signals = [
    {
        'h5_path': '/eeg_0',
        'fs': 64,
        'processing': {
            "type": "clip_and_normalize",
            "args": {
                    "min_value": -150,
                "max_value": 150,
            }
        }
    },
    {
        'h5_path': '/eeg_1',
        'fs': 64,
        'processing': {
            "type": "clip_and_normalize",
            "args": {
                    "min_value": -150,
                "max_value": 150,
            }
        }
    }
    ]
    events = [
        {
            "name": "spindle",
            "h5_path": "spindle",
        },
    ]
    fs = 32
    get_h5_data(filename=filename, signals=signals, fs=fs)
    get_h5_events(filename=filename, event=events, fs=fs)