"""inspired from https://github.com/amdegroot/ssd.pytorch"""
import torch.nn as nn

from ..utils import non_maximum_suppression, decode


class Detection(nn.Module):
    """"""

    def __init__(self,
                 number_of_classes,
                 overlap_non_maximum_suppression,
                 classification_threshold,
                 ):
        super(Detection, self).__init__()
        self.number_of_classes = number_of_classes
        self.overlap_non_maximum_suppression = overlap_non_maximum_suppression
        self.classification_threshold = classification_threshold

    def forward(self, localizations, classifications, localizations_default):# 这里是模型返回结果，然后弄输出的地方的
        batch = localizations.size(0)
        scores = nn.Softmax(dim=2)(classifications)#这里是预测每一类的概率值,128,63,2
        results = []
        for i in range(batch):#每一个batch
            result = []
            localization_decoded = decode(localizations[i], localizations_default)#这里可能是那个偏离值
            for class_index in range(1, self.number_of_classes):  # we remove class 0
                scores_batch_class = scores[i, :, class_index]#选择第二个标签，也就是spindle类的概率
                scores_batch_class_selected = scores_batch_class[#selected是23，batch_class是63
                    scores_batch_class > self.classification_threshold]#选择概率大于lassification_threshold的
                if len(scores_batch_class_selected) == 0:
                    continue
                localizations_decoded_selected = localization_decoded[
                    (scores_batch_class > self.classification_threshold)
                    .unsqueeze(1).expand_as(localization_decoded)].view(-1, 2)

                events = non_maximum_suppression(
                    localizations_decoded_selected,
                    scores_batch_class_selected,
                    overlap=self.overlap_non_maximum_suppression,
                )
                result.extend([(event[0].item(), event[1].item(), class_index - 1)
                               for event in events])
            result = [event for event in result if event[0] > -10 and event[1] < 10]
            results.append(result)
        return results
