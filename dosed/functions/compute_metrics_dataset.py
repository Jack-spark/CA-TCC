import numpy as np

from .metrics import precision_function, recall_function, f1_function

available_score_functions = {
    "precision": precision_function(),
    "recall": recall_function(),
    "f1": f1_function(),
}


def compute_metrics_dataset(
        network,
        test_dataset,
        threshold,
        test_metrics=["precision", "recall", "f1"],
):
    """
    Computes metrics on current net for test_dataset, using threshold
    as classification threshold
    """

    metrics = {
        score: score_function for score, score_function in
        available_score_functions.items() if score in test_metrics
    }# 三个度量指标和其对应的度量函数

    metrics_test = [{
        metric: []
        for metric in metrics.keys()
    } for _ in range(network.number_of_classes - 1)]

    all_predicted_events = network.predict_dataset(
        test_dataset,
        threshold,
        batch_size=64)

    found_some_events = False

    for event_num in range(network.number_of_classes - 1):

        for record in test_dataset.records:

            # Select current event predictions，预测事件居然有4319个
            predicted_events = all_predicted_events[record][event_num]
            # 一个record预测的一个events，因为只有一个事件，event_num只有一个
            # If no predictions skip record, else some_events = 1
            if len(predicted_events) == 0:
                continue

            found_some_events = True#找到了事件

            # Select current true events，选择真实事件存在，88个真实事件
            events = test_dataset.get_record_events(record)[event_num]

            # Compute_metrics(events, predicted_events, threshold)
            # 这里进行预测事件和真实事件的计算
            for metric in metrics.keys():
                metrics_test[event_num][metric].append(metrics[metric](
                    predicted_events,
                    events))

    # If for any event and record the network predicted events, return -1
    if found_some_events is False:
        return -1

    for event_num in range(network.number_of_classes - 1):
        for metric in metrics.keys():
            metrics_test[event_num][metric] = np.nanmean(np.array(metrics_test[event_num][metric]))

    return metrics_test
