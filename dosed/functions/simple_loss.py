import torch.nn as nn
import torch.nn.functional as F


class DOSEDSimpleLoss(nn.Module):
    """Loss function inspired from https://github.com/amdegroot/ssd.pytorch"""

    def __init__(self,
                 number_of_classes,
                 device,
                 ):
        super(DOSEDSimpleLoss, self).__init__()
        self.device = device
        self.number_of_classes = number_of_classes + 1  # eventlessness

    def localization_loss(self, positive, localizations, localizations_target):
        # Localization Loss (Smooth L1)
        positive_expanded = positive.unsqueeze(positive.dim()).expand_as(
            localizations)
        loss_localization = F.smooth_l1_loss(
            localizations[positive_expanded].view(-1, 2),
            localizations_target[positive_expanded].view(-1, 2),
            reduction="sum")
        return loss_localization

    def get_negative_index(self, positive, classifications,
                           classifications_target):
        negative = (classifications_target == 0)
        return negative

    def get_classification_loss(self, index, classifications,
                                classifications_target):
        index_expanded = index.unsqueeze(2).expand_as(classifications)

        loss_classification = F.cross_entropy(
            classifications[index_expanded.gt(0)
                            ].view(-1, self.number_of_classes),
            classifications_target[index.gt(0)],
            reduction="sum",
        )
        return loss_classification

    def forward(self, localizations, classifications, localizations_target,
                classifications_target):
        #一个batch的损失
        positive = classifications_target > 0#1024，63，spindle分为positive，等于1被标为true
        negative = self.get_negative_index(positive, classifications,#获取负样本的索引。这个方法可能会根据positive（正样本的索引）、classifications
        #（模型的分类预测结果）和classifications_target（真实的类别标签）来确定哪些样本是负样本。
                                           classifications_target)
        #等于0被标为true
        number_of_positive_all = positive.long().sum().float()#2247个正样本，真实的
        number_of_negative_all = negative.long().sum().float()#62265个负样本
        #1064*63=62265+2247=64512
        # loc loss
        loss_localization = self.localization_loss(positive, localizations,
                                                   localizations_target)

        # + Classification loss
        loss_classification_positive = 0
        if number_of_positive_all > 0:
            loss_classification_positive = self.get_classification_loss(#计算分类损失
                positive, classifications, classifications_target)#正样本，预测分类，真实分类

        # - Classification loss
        loss_classification_negative = 0
        if number_of_negative_all > 0:#负样本的损失
            loss_classification_negative = self.get_classification_loss(
                negative, classifications, classifications_target)

        # Loss: sum
        loss_classification_positive_normalized = (
            loss_classification_positive /
            number_of_positive_all)
        loss_classification_negative_normalized = (
            loss_classification_negative /
            number_of_negative_all)
        loss_localization_normalized = loss_localization / number_of_positive_all
        #这里应该是可以加一个分类准确率的东西的
        
        return (loss_classification_positive_normalized,
                loss_classification_negative_normalized,
                loss_localization_normalized)
