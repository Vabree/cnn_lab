import os
import sys 
import argparse
sys.path.insert(0, os.path.join(os.getcwd()))

import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, MultilabelAccuracy

def get_metric(task:str, metric_name: str, class_number: int):
    metricCollection = []
    if task == "image_classification":
        if metric_name['binary_accuracy']:
            metricCollection.append(BinaryAccuracy())
        if metric_name['multiclass_accuracy']:
            metricCollection.append(MulticlassAccuracy(class_number))
        if metric_name['multilabel_accuracy']:
            metricCollection.append(MultilabelAccuracy(class_number))
    elif task == "semseg":
        if metric_name['multiclass_accuracy']:
            metricCollection.append(MulticlassAccuracy(num_classes = class_number, ignore_index = 255))
    else:
        raise NameError("Task doesn't exist")
    
    return MulticlassAccuracy(num_classes=class_number) #MetricCollection(metricCollection)


def get_metric_test():
    task = "image_classification"
    metric_name = "accuracy"
    get_metric(task, metric_name)
if __name__ == "__main__":
    get_metric_test()