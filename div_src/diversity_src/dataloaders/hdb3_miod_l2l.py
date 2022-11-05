"""


note: there is a learn2learn_TO_torchmeta data loader, so you can implement everything in l2l.

delaunay: https://paperswithcode.com/dataset/delaunay
Delaunay authors say:  We believe
the unique properties of this dataset make it useful for both
machine learning as well as psychophsyics research, for example to investigate the hypothesis that
**sample efficiency scales inversely with the statistical similarity of samples to natural images for humans but not for DNNs.**
"""
import os
import random
from typing import Callable

import learn2learn as l2l
import numpy as np
import torch
from learn2learn.data import MetaDataset, FilteredMetaDataset
from learn2learn.data.transforms import TaskTransform
from learn2learn.vision.benchmarks import BenchmarkTasksets
from torch import nn
from torchvision.transforms import Compose, Normalize, ToPILImage, RandomCrop, ColorJitter, RandomHorizontalFlip, \
    ToTensor

from diversity_src.dataloaders.common import IndexableDataSet, ToRGB, DifferentTaskTransformIndexableForEachDataset

from torchvision import transforms
from PIL.Image import LANCZOS


def hdb3_mi_omniglot_delauny_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,
        root='~/data/l2l_data/',
        device=None,
        **kwargs,
) -> BenchmarkTasksets:
    pass


# -- Run experiment

def loop_through_hdb3_miod_delaunay():
    # - for determinism
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # - options for number of tasks/meta-batch size
    batch_size = 2

    # - get benchmark
    benchmark: BenchmarkTasksets = hdb3_mi_omniglot_delauny_tasksets()
    splits = ['train', 'validation', 'test']
    tasksets = [getattr(benchmark, split) for split in splits]

    # - loop through tasks
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    # from models import get_model
    # model = get_model('resnet18', pretrained=False, num_classes=5).to(device)
    # model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V2")
    # model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
    model, _ = get_default_learner_and_hps_dict()  # 5cnn
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    for i, taskset in enumerate(tasksets):
        print(f'-- {splits[i]=}')
        for task_num in range(batch_size):
            print(f'{task_num=}')

            X, y = taskset.sample()
            print(f'{X.size()=}')
            print(f'{y.size()=}')
            print(f'{y=}')

            y_pred = model(X)
            loss = criterion(y_pred, y)
            print(f'{loss=}')
            print()


if __name__ == "__main__":
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    loop_through_hdb3_miod_delaunay()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
