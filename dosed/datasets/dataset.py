"""Dataset Class for DOSED training"""

import os
import h5py
import numpy as np
from numpy.lib.stride_tricks import as_strided
from matplotlib import gridspec
from joblib import Memory, Parallel, delayed

import torch
from torch.utils.data import Dataset

from ..utils import get_h5_data, get_h5_events

from ..functions.augmentations import DataTransform 

class EventDataset(Dataset):

    """Extract data and events from h5 files and provide efficient way to retrieve windows with
    their corresponding events.

    args
    ====

    h5_directory:
        Location of the generic h5 files.
    signals:
        The signals from the h5 we want to include together with their normalization
    events:
        The events from the h5 we want to train on
    window:
        Window size in seconds
    downsampling_rate:
        Downsampling rate to apply to signals
    records:
        Use to select subset of records from h5_directory, default is None and uses all available recordings
    n_jobs:
        Number of process used to extract and normalize signals from h5 files.
    cache_data:
        Cache results of extraction and normalization of signals from h5_file in h5_directory + "/.cache"
        We strongly recommend to keep the default value True to avoid memory overhead.
    minimum_overlap:
        For an event on the edge to be considered included in a window
    ratio_positive:
        Sample within a training batch will have a probability of "ratio_positive" to contain at least one spindle

    """

    def __init__(self,
                 h5_directory,
                 signals,
                 window,
                 fs,
                 events=None,
                 records=None,
                 n_jobs=1,
                 cache_data=True,
                 minimum_overlap=0.25,
                 transformations=None
                 ):

        if events:
            self.number_of_classes = len(events)
        self.transformations = transformations

        # window parameters
        self.window = window#10

        # records (all of them by default)
        if records is not None:
            for record in records:
                assert record in os.listdir(h5_directory)
            self.records = records
        else:
            self.records = [x for x in os.listdir(h5_directory) if x != ".cache"]

        ###########################
        # Checks on H5
        self.fs = fs#32,下采样率

        # check event names
        if events:
            assert len(set([event["name"] for event in events])) == len(events)

        # ### joblib cache
        get_data = get_h5_data
        get_events = get_h5_events
        if cache_data:
            memory = Memory(h5_directory + "/.cache/", mmap_mode="r", verbose=0)
            get_data = memory.cache(get_h5_data)
            get_events = memory.cache(get_h5_events)

        self.window_size = int(self.window * self.fs)# 10s * 32 = 320，下采样窗口
        self.number_of_channels = len(signals)#2，信号通道
        # used in network architecture
        self.input_shape = (self.number_of_channels, self.window_size)#2,320，这是我应该一个batch的输入大小
        self.minimum_overlap = minimum_overlap  # for events on the edge of window_size,0.5

        # Open signals and events,初始化四个属性
        self.signals = {}
        self.events = {}
        self.index_to_record = []# 信号的索引，这里应该是时间，指一块一块的索引
        self.index_to_record_event = []  # link index to record，事件的索引

        # Preprocess signals from records,对信号进行预处理
#         这段代码使用了 joblib 库的 Parallel 和 delayed 函数来并行读取数据。

# Parallel 函数用于创建一个并行计算的环境，n_jobs 参数指定了并行任务的数量，prefer="threads" 表示优先使用线程而不是进程进行并行计算。

# delayed 函数用于包装需要并行执行的函数，使其可以在 Parallel 环境中延迟执行。

# 在这段代码中，get_data 函数被 delayed 包装，然后在 Parallel 环境中并行执行。get_data 函数的参数是一个文件名（由 h5_directory 和 record 组成），一个信号列表 signals，和一个采样率 fs。

# for record in self.records 是一个生成器表达式，它为 self.records 中的每个记录生成一个 delayed(get_data) 任务。这些任务被并行执行，结果被收集到一个列表 data 中。

# 注意，这段代码假设 get_data 函数已经定义，且可以接受 filename、signals 和 fs 三个参数。此外，这段代码也假设 h5_directory、signals、fs 和 self.records 已经定义。
        data = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(get_data)(#并行读取数据signals,5个样本，每个样本2*691200
            filename="{}/{}".format(h5_directory, record),
            signals=signals,
            fs=fs
        ) for record in self.records)
    # get_data接受filename、signals 和 fs 三个参数，返回的是降采样之后的数组
        for record, data in zip(self.records, data):
            signal_size = data.shape[-1]#691200
            number_of_windows = signal_size // self.window_size#691200//320=2160,有这么多个窗口

            self.signals[record] = {#键值对，每个record对应一个字典,这里突然降维了
                "data": data,
                "size": signal_size,
            }

            self.index_to_record.extend([#一个record，一个index，一个窗口，index是窗口索引，比如说0，320，640之类的，一个record好多个索引，这里就是切割的时候
                {
                    "record": record,
                    "index": x * self.window_size
                } for x in range(number_of_windows)
            ])

            if events:
                self.events[record] = {}
                number_of_events = 0
                events_indexes = set()
                max_index = signal_size - self.window_size

                for label, event in enumerate(events):
                    data = get_events(# 得到事件的数据，每个样本的事件数据是2*事件个数，2，130
                        filename="{}/{}".format(h5_directory, record),
                        event=event,
                        fs=self.fs,
                    )

                    number_of_events += data.shape[-1]#事件个数
                    self.events[record][event["name"]] = {
                        "data": data,#events也有一个record，record里面的data是开始时间和持续时间，已经乘以采样率了，还有一个标签
                        "label": label,
                    }

                    for start, duration in zip(*data):#同时便利start和duration两个列表，计算每个事件的开始和结束索引，然后更新事件索引集合。
                        #然后，它计算没有事件的索引，并将每个事件的记录，最大索引，事件索引和没有事件的索引
                        stop = start + duration#停止时间
                        duration_overlap = duration * self.minimum_overlap#持续事件的overlap，至少要包括事件一半，才算有标签
                        # 这两个索引可以用于从一个大的数据数组中提取出一个窗口，这个窗口包含了一个事件，并且事件的开始和结束位置都在窗口的范围内
                        start_valid_index = int(round(
                            max(0, start + duration_overlap - self.window_size + 1)))#这个索引也是乘以采样率的了，可以直接当作时间点
                        end_valid_index = int(round(
                            min(max_index + 1, stop - duration_overlap)))

                        indexes = list(range(start_valid_index, end_valid_index))
                        events_indexes.update(indexes)

                no_events_indexes = set(range(max_index + 1))
                no_events_indexes = list(no_events_indexes.difference(events_indexes))#没有事件的索引就是所有索引减去事件索引
                events_indexes = list(events_indexes)

                self.index_to_record_event.extend([
                    {
                        "record": record,#样本名字
                        "max_index": max_index,#最大索引signal_size - self.window_size
                        "events_indexes": events_indexes,
                        "no_events_indexes": no_events_indexes,
                    } for _ in range(number_of_events)
                ])

    def __len__(self):
        return len(self.index_to_record)

    def __getitem__(self, idx):
        signal, events = self.get_sample(
            record=self.index_to_record[idx]["record"],
            index=self.index_to_record[idx]["index"])

        if self.transformations is not None:
            signal = self.transformations(signal)
        return signal, events

    def get_valid_events_index(self, index, starts, durations):
        """Return the events' indexes that have enough overlap with the given time index
           ex: index = 155
               starts =   [10 140 150 165 2000]
               duration = [4  20  10  10   40]
               minimum_overlap = 0.5
               window_size = 15
           return: [2 3]
        """
        # Relative start stop

        starts_relative = (starts - index) / self.window_size
        durations_relative = durations / self.window_size
        stops_relative = starts_relative + durations_relative

        # Find valid start or stop
        valid_starts_index = np.where((starts_relative > 0) *
                                      (starts_relative < 1))[0]
        valid_stops_index = np.where((stops_relative > 0) *
                                     (stops_relative < 1))[0]

        valid_inside_index = np.where((starts_relative <= 0) *
                                      (stops_relative >= 1))[0]

        # merge them
        valid_indexes = set(list(valid_starts_index) +
                            list(valid_stops_index) +
                            list(valid_inside_index))

        # Annotations contains valid index with minimum overlap requirement
        events_indexes = []
        for valid_index in valid_indexes:
            if (valid_index in valid_starts_index) \
                    and (valid_index in valid_stops_index):
                events_indexes.append(valid_index)
            elif valid_index in valid_starts_index:
                if ((1 - starts_relative[valid_index]) /
                        durations_relative[valid_index]) > self.minimum_overlap:
                    events_indexes.append(valid_index)
            elif valid_index in valid_stops_index:
                if ((stops_relative[valid_index]) / durations_relative[valid_index]) \
                        > self.minimum_overlap:
                    events_indexes.append(valid_index)
            elif valid_index in valid_inside_index:
                if self.window_size / durations[valid_index] > self.minimum_overlap:
                    events_indexes.append(valid_index)
        return events_indexes

    def get_record_events(self, record):# 返回一个record对应的事件，以列表形式，包括开始时间和终止时间

        events = [[] for _ in range(self.number_of_classes)]

        for event_data in self.events[record].values():
            events[event_data["label"]].extend([
                [start, start + duration]
                for start, duration in event_data["data"].transpose().tolist()
            ])

        return events

    def get_record_batch(self, record, batch_size, stride=None):# 这段代码定义了一个名为 get_record_batch 的方法，它从特定的记录中获取一批连续的窗口作为信号数据。
        """Return signal data from a specific record as a batch of continuous
           windows. Overlap in seconds allows overlapping among windows in the
           batch. The last data points will be ignored if their length is
           inferior to window_size.
        """
        # 窗口之间可以有重叠，重叠大小以秒为单位
        # stride = overlap_size
        # batch_size = batch

        stride = int((stride if stride is not None else self.window) * self.fs)#步长
        batch_overlap_size = stride * batch_size  # stride at a batch level，批次级别重叠大小
        read_size = (batch_size - 1) * stride + self.window_size# 每个批次的读取大小
        signal_size = self.signals[record]["size"]#一个record的信号大小
        t = np.arange(signal_size)
        number_of_batches_in_record = (signal_size - read_size) // batch_overlap_size + 1#计算记录中的批次数量（number_of_batches_in_record）

        for batch in range(number_of_batches_in_record):#对于每个批次，计算开始和结束的位置，然后从信号数据中提取对应的部分
            start = batch_overlap_size * batch#一个批次的开始位置
            stop = batch_overlap_size * batch + read_size#一个批次的结束位置
            signal = self.signals[record]["data"][:, start:stop]#一个批次的信号数据

            signal_strided = torch.FloatTensor(#使用 as_strided 函数创建一个新的 numpy 数组。这个数组是 signal 的一个视图，
                #它有 batch_size 个窗口，每个窗口的大小是 self.window_size。窗口之间的步长是 stride
                #这里是一个batch里面的所有窗口，每个窗口是一个信号，其中包含了重叠步长
                np.copy(
                    as_strided(
                    x=signal,
                    shape=(batch_size, signal.shape[0], self.window_size),
                    strides=(signal.strides[1] * stride, signal.strides[0],
                             signal.strides[1]),
                    )
                )  
            )
            time = t[start:stop]
            t_strided = as_strided(
                x=time,
                shape=(batch_size, self.window_size),
                strides=(time.strides[0] * stride, time.strides[0]),
            )

            yield signal_strided, t_strided

        batch_end = (
            signal_size - number_of_batches_in_record * batch_overlap_size - self.window_size
        ) // stride + 1
        if batch_end > 0:

            read_size_end = (batch_end - 1) * stride + self.window_size
            start = batch_overlap_size * number_of_batches_in_record
            end = batch_overlap_size * number_of_batches_in_record + read_size_end
            signal = self.signals[record]["data"][:, start:end]

            signal_strided = torch.FloatTensor(
                np.copy(
                    as_strided(
                    x=signal,
                    shape=(batch_end, signal.shape[0], self.window_size),
                    strides=(signal.strides[1] * stride, signal.strides[0],
                             signal.strides[1]),
                    )
                )
                
            )
            time = t[start:end]
            t_strided = as_strided(
                x=time,
                shape=(batch_end, self.window_size),
                strides=(time.strides[0] * stride, time.strides[0]),
            )

            yield signal_strided, t_strided

    def plot(self, idx, channels):
        """Plot events and data from channels for record and index found at
           idx"""

        import matplotlib.pyplot as plt
        signal, events = self.extract_balanced_data(
            record=self.index_to_record_event[idx]["record"],
            max_index=self.index_to_record_event[idx]["max_index"])

        non_valid_indexes = np.where(np.array(channels) is None)[0]
        signal = np.delete(signal, non_valid_indexes, axis=0)
        channels = [channel for channel in channels if channel is not None][::-1]

        num_signals = len(channels)
        signal_size = len(signal[0])
        events_numpy = events.numpy()
        plt.figure(figsize=(10 * 4, 2 * num_signals))
        gs = gridspec.GridSpec(num_signals, 1)
        gs.update(wspace=0., hspace=0.)
        for channel_num, channel in enumerate(channels):
            assert signal_size == len(signal[channel_num])
            signal_mean = signal.numpy()[channel_num].mean()
            ax = plt.subplot(gs[channel_num, 0])
            ax.set_ylim(-0.55, 0.55)
            ax.plot(signal.numpy()[channel_num], alpha=0.3)
            for event in events_numpy:
                ax.fill([event[0] * signal_size, event[1] * signal_size],
                        [signal_mean, signal_mean],
                        alpha=0.5,
                        linewidth=30,
                        color="C{}".format(int(event[-1])))
            if channel_num == 0:
                # print(EVENT_DICT[event[2]])
                offset = (1. / num_signals) * 1.1
                step = (1. / num_signals) * 0.78
            plt.gcf().text(0.915, offset + channel_num * step,
                           channel, fontsize=14)
        plt.show()
        plt.close()

    def get_sample(self, record, index):
        """Return a sample [sata, events] from a record at a particularindex"""

        signal_data = self.signals[record]["data"][:, index: index + self.window_size]#这个索引是事件的索引，从抽的可能是事件索引开始，10s
        events_data = []

        for event_name, event in self.events[record].items():
            starts, durations = event["data"][0, :], event["data"][1, :]#事件的开始时间和持续时间

            # Relative start stop
            starts_relative = (starts - index) / self.window_size
            durations_relative = durations / self.window_size
            stops_relative = starts_relative + durations_relative

            for valid_index in self.get_valid_events_index(index, starts, durations):
                events_data.append((max(0, float(starts_relative[valid_index])),
                                    min(1, float(stops_relative[valid_index])),
                                    event["label"]))# 调用 self.get_valid_events_index 方法，获取在当前窗口内的事件的索引。

        return torch.FloatTensor(signal_data), torch.FloatTensor(events_data)#相对开始时间、相对结束时间和标签添加到 events_data 中


class BalancedEventDataset(EventDataset):
    """
    Same as EventDataset but with the possibility to choose the probability to get at least
    one event when retrieving a window.

    """

    def __init__(self,
                 h5_directory,
                 signals,
                 window,
                 fs,
                 events=None,
                 records=None,
                 minimum_overlap=0.5,
                 transformations=None,
                 ratio_positive=0.5,
                 n_jobs=1,
                 cache_data=True,
                 training_mode=None,#之后用来安排是否存在自监督的参数
                 ):
        super(BalancedEventDataset, self).__init__(
            h5_directory=h5_directory,
            signals=signals,
            events=events,
            window=window,
            fs=fs,
            records=records,
            minimum_overlap=minimum_overlap,
            transformations=transformations,
            n_jobs=n_jobs,
            cache_data=cache_data,
        )
        self.ratio_positive = ratio_positive
    

    def __len__(self):
        return len(self.index_to_record_event)

    def __getitem__(self, idx):

        signal, events = self.extract_balanced_data(
            record=self.index_to_record_event[idx]["record"],
            max_index=self.index_to_record_event[idx]["max_index"],
            events_indexes=self.index_to_record_event[idx]["events_indexes"],
            no_events_indexes=self.index_to_record_event[idx]["no_events_indexes"]
        )

        if self.transformations is not None:
            signal = self.transformations(signal)

        return signal, events

    def extract_balanced_data(self, record, max_index, events_indexes, no_events_indexes):
        """Extracts an index at random"""

        choice = np.random.choice([0, 1], p=[1 - self.ratio_positive, self.ratio_positive])# 随机抽样本，可能抽没事件的也可能是有事件的

        if choice == 0:
            index = no_events_indexes[np.random.randint(len(no_events_indexes))]
        else:
            index = events_indexes[np.random.randint(len(events_indexes))]

        signal_data, events_data = self.get_sample(record, index)

        return signal_data, events_data
