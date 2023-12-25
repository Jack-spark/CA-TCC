""" Trainer class basic with SGD optimizer """

import copy
import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ..datasets import collate
from ..functions import (loss_functions, available_score_functions, compute_metrics_dataset)
from ..utils import (match_events_localization_to_default_localizations, Logger)

from ..functions import augmentations
from ..functions.loss import NTXentLoss, SupConLoss


class TrainerBase:
    """Trainer class basic """

    def __init__(
            self,
            net,
            temporal_contr_model=None,
            optimizer_parameters={
                "lr": 0.001,
                "weight_decay": 1e-8,
            },
            temp_cont_optimizer=None,
            loss_specs={
                "type": "focal",
                "parameters": {
                    "number_of_classes": 1,
                    "alpha": 0.25,
                    "gamma": 2,
                    "device": torch.device("cuda"),
                }
            },
            metrics=["precision", "recall", "f1"],
            epochs=100,
            metric_to_maximize="f1",
            patience=None,
            save_folder=None,
            logger_parameters={
                "num_events": 1,
                "output_dir": 'None',
                "output_fname": 'train_history.json',
                "metrics": ["precision", "recall", "f1"],
                "name_events": ["event_type_1"],
            },
            threshold_space={
                "upper_bound": 0.85,
                "lower_bound": 0.55,
                "num_samples": 5,
                "zoom_in": False,
            },
            matching_overlap=0.5,
    ):
        self.net = net
        self.temporal_contr_model = temporal_contr_model#TC模块
        print("Device: ", net.device)
        self.loss_function = loss_functions[loss_specs["type"]](
            **loss_specs["parameters"])
        self.optimizer = optim.SGD(net.parameters(), **optimizer_parameters)
        self.temp_cont_optimizer = temp_cont_optimizer# TC的优化器
        self.metrics = {
            score: score_function for score, score_function in
            available_score_functions.items()
            if score in metrics + [metric_to_maximize]
        }
        self.epochs = epochs
        self.threshold_space = threshold_space
        self.metric_to_maximize = metric_to_maximize
        self.patience = patience if patience else epochs
        self.save_folder = save_folder
        self.matching_overlap = matching_overlap
        self.matching = match_events_localization_to_default_localizations
        if logger_parameters is not None:
            self.train_logger = Logger(**logger_parameters)

    def on_batch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def validate(self, validation_dataset, threshold_space):
        """
        Compute metrics on validation_dataset net for test_dataset and
        select best classification threshold
        """

        best_thresh = -1
        best_metrics_epoch = {
            metric: -1
            for metric in self.metrics.keys()
        }

        # Compute predicted_events
        thresholds = np.sort(
            np.random.uniform(threshold_space["upper_bound"],
                              threshold_space["lower_bound"],
                              threshold_space["num_samples"]))

        for threshold in thresholds:
            metrics_thresh = compute_metrics_dataset(
                self.net,
                validation_dataset,
                threshold,
            )

            # If 0 events predicted, all superiors thresh's will also predict 0
            if metrics_thresh == -1:
                if best_thresh in (self.threshold_space["upper_bound"],
                                   self.threshold_space["lower_bound"]):
                    print(
                        "Best classification threshold is " +
                        "in the boundary ({})! ".format(best_thresh) +
                        "Consider extending threshold range")
                return best_metrics_epoch, best_thresh

            # Add to logger
            if "train_logger" in vars(self):
                self.train_logger.add_new_metrics((metrics_thresh, threshold))

            # Compute mean metric to maximize across events
            mean_metric_to_maximize = np.nanmean(
                [m[self.metric_to_maximize] for m in metrics_thresh])

            if mean_metric_to_maximize >= best_metrics_epoch[
                    self.metric_to_maximize]:
                best_metrics_epoch = {
                    metric: np.nanmean(
                        [m[metric] for m in metrics_thresh])
                    for metric in self.metrics.keys()
                }

                best_thresh = threshold

        if best_thresh in (threshold_space["upper_bound"],
                           threshold_space["lower_bound"]):
            print("Best classification threshold is " +
                  "in the boundary ({})! ".format(best_thresh) +
                  "Consider extending threshold range")

        return best_metrics_epoch, best_thresh

    def get_batch_loss(self, data, config=None, training_mode=None):
        """ Single forward and backward pass """

        # Get signals and labels
        signals, events = data#signal=128，2，320，events=128，每一个是1，3，这个data是dataloader里的,events有起始时间，持续时间和标签
        x = signals.to(self.net.device)
        # Forward
        localizations, classifications, localizations_default, feature = self.net.forward(x)
        if training_mode == "self_supervised":
            aug1, aug2 = augmentations.DataTransform(x, config) 
            aug1, aug2 = aug1.float(), aug2.float()
            aug1, aug2 = aug1.to(self.net.device), aug2.to(self.net.device)
            feature_aug1, feature_aug2 = self.net.forward(aug1)[-1], self.net.forward(aug2)[-1]
            feature_aug1 = F.normalize(feature_aug1, dim=1)
            feature_aug2 = F.normalize(feature_aug2, dim=1)
            temp_cont_loss1, temp_cont_feat1 = self.temporal_contr_model(feature_aug1, feature_aug2)
            temp_cont_loss2, temp_cont_feat2 = self.temporal_contr_model(feature_aug2, feature_aug1)
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(self.net.device, 64, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                   nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2
            return loss
        else:
            #classification=128，63，2
            # Matching
            localizations_target, classifications_target = self.matching(
                localizations_default=localizations_default,
                events=events,
                threshold_overlap=self.matching_overlap)
            localizations_target = localizations_target.to(self.net.device)
            classifications_target = classifications_target.to(self.net.device)

            # Loss，真实和预测之间的比较
            (loss_classification_positive,
            loss_classification_negative,
            loss_localization) = (
                self.loss_function(localizations,
                                    classifications,
                                    localizations_target,
                                    classifications_target))
            return loss_classification_positive, \
                loss_classification_negative, \
                loss_localization

    def train(self, train_dataset, validation_dataset, batch_size=64, training_mode=None, 
              config=None, drop_last=False):
        """ Metwork training with backprop """
    # 这里是训练的函数，包含着输入输出
        dataloader_parameters = {
            "num_workers": 6,
            "shuffle": True,
            "collate_fn": collate,
            "pin_memory": True,
            "batch_size": batch_size,
        }
        dataloader_train = DataLoader(train_dataset, **dataloader_parameters, drop_last=drop_last)
        dataloader_val = DataLoader(validation_dataset, **dataloader_parameters, drop_last=drop_last)

        metrics_final = {
            metric: 0
            for metric in self.metrics.keys()
        }
        # metrics_final = {0}，包括召回率，精确率，f1

        best_value = -np.inf
        best_threshold = None
        best_net = None
        counter_patience = 0
        last_update = None
        t = tqdm.tqdm(range(self.epochs,))#tqdm是一个进度条库，range是一个迭代器
        for epoch, _ in enumerate(t):
            if epoch != 0:
                t.set_postfix(
                    best_metric_score=best_value,
                    threshold=best_threshold,
                    last_update=last_update,
                )

            epoch_loss_classification_positive_train = 0.0
            epoch_loss_classification_negative_train = 0.0
            epoch_loss_localization_train = 0.0

            epoch_loss_classification_positive_val = 0.0
            epoch_loss_classification_negative_val = 0.0
            epoch_loss_localization_val = 0.0
            # 模型开始训练放在
            self.net.train()
            if (self.temporal_contr_model is not None):
                self.temporal_contr_model.train()
            for i, data in enumerate(dataloader_train, 0):
                # On batch start
                self.on_batch_start()
                self.optimizer.zero_grad()
                if (self.temp_cont_optimizer is not None):
                    self.temp_cont_optimizer.zero_grad()# TC的优化器
                # Set network to train mode
                if training_mode == "self_supervised":
                    loss = self.get_batch_loss(data, config, training_mode)
                else:    
                    (loss_classification_positive,
                    loss_classification_negative,
                    loss_localization) = self.get_batch_loss(data)

                    epoch_loss_classification_positive_train += \
                        loss_classification_positive
                    epoch_loss_classification_negative_train += \
                        loss_classification_negative
                    epoch_loss_localization_train += loss_localization

                    loss = loss_classification_positive \
                        + loss_classification_negative \
                        + loss_localization
                loss.backward()
                # gradient descent
                self.optimizer.step()
                if (self.temp_cont_optimizer is not None):
                    self.temp_cont_optimizer.step()
            epoch_loss_classification_positive_train /= (i + 1)
            epoch_loss_classification_negative_train /= (i + 1)
            epoch_loss_localization_train /= (i + 1)

            
            if training_mode == None:
                for i, data in enumerate(dataloader_val, 0):

                    (loss_classification_positive,
                    loss_classification_negative,
                    loss_localization) = self.get_batch_loss(data)

                    epoch_loss_classification_positive_val += \
                        loss_classification_positive
                    epoch_loss_classification_negative_val += \
                        loss_classification_negative
                    epoch_loss_localization_val += loss_localization
                # 计算每个epoch的平均损失
                epoch_loss_classification_positive_val /= (i + 1)
                epoch_loss_classification_negative_val /= (i + 1)
                epoch_loss_localization_val /= (i + 1)
                # self.validate方法对当前模型进行验证，获取当前的评估指标和阈值
                metrics_epoch, threshold = self.validate(
                    validation_dataset=validation_dataset,
                    threshold_space=self.threshold_space,
                )

                if self.threshold_space["zoom_in"] and threshold != -1:
                    threshold_space_size = self.threshold_space["upper_bound"] - \
                        self.threshold_space["lower_bound"]
                    zoom_metrics_epoch, zoom_threshold = self.validate(
                        validation_dataset=validation_dataset,
                        threshold_space={
                            "upper_bound": threshold + 0.1 * threshold_space_size,
                            "lower_bound": threshold - 0.1 * threshold_space_size,
                            "num_samples": self.threshold_space["num_samples"],
                        })
                    # 更新最佳评估指标、最佳阈值和最佳模型，这是这个epoch的评估指标
                    if zoom_metrics_epoch[self.metric_to_maximize] > metrics_epoch[
                            self.metric_to_maximize]:
                        metrics_epoch = zoom_metrics_epoch
                        threshold = zoom_threshold
                net_parameters = self.net.state_dict()
                if self.save_folder:
                    self.net.save(self.save_folder + str(epoch) + ".pth", net_parameters)
                # 当前epoch没有优于最佳评估指标，counter_patience+1
                if metrics_epoch[self.metric_to_maximize] > best_value:
                    best_value = metrics_epoch[self.metric_to_maximize]
                    best_threshold = threshold
                    last_update = epoch
                    best_net = copy.deepcopy(self.net)
                    metrics_final = {
                        metric: metrics_epoch[metric]
                        for metric in self.metrics.keys()
                    }
                    counter_patience = 0
                else:
                    counter_patience += 1
                # 多个epoch都没有提高指标的话，就停止训练，self.patience是最大的epoch数
                if counter_patience > self.patience:
                    break
                # 停止训练的处理    
                self.on_epoch_end()

                if "train_logger" in vars(self):
                    self.train_logger.add_new_loss(
                        epoch_loss_localization_train.item(),
                        epoch_loss_classification_positive_train.item(),
                        epoch_loss_classification_negative_train.item(),
                        mode="train"
                    )
                    self.train_logger.add_new_loss(
                        epoch_loss_localization_val.item(),
                        epoch_loss_classification_positive_val.item(),
                        epoch_loss_classification_negative_val.item(),
                        mode="validation"
                    )
                    self.train_logger.add_current_metrics_to_history()
                    self.train_logger.dump_train_history()
            else:
                best_net = copy.deepcopy(self.net)
                metrics_final = 0
                best_threshold = 0
        return best_net, metrics_final, best_threshold
        
