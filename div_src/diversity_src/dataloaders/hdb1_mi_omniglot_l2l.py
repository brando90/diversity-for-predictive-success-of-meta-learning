"""
HDB1 := MI + Omni

Note:
    - train_samples != k_shots and test_samples != k_eval. Instead train_samples = test_samples = k_shots + k_eval.
ref:
    - https://wandb.ai/brando/entire-diversity-spectrum/reports/Copy-of-brando-s-Div-ala-task2vec-HDB1-mi-omniglot-HDB2-cifarfs-omniglot---VmlldzoyMDc3OTAz
    - discussion: https://github.com/learnables/learn2learn/issues/333

Diversity: (div, ci)=(0.2161356031894684, 0.038439472241579925)

Issues with lambda pickling data transforms:
ref:
    - my answer: https://stackoverflow.com/a/74282085/1601580
"""
from pathlib import Path

import os
import random
from typing import Callable, Union

import learn2learn as l2l
import numpy as np
import torch
from learn2learn.data import MetaDataset, FilteredMetaDataset
from learn2learn.data.transforms import TaskTransform
from learn2learn.vision.benchmarks import BenchmarkTasksets
from torch import nn
from torchvision.transforms import Compose, Normalize, ToPILImage, RandomCrop, ColorJitter, RandomHorizontalFlip, \
    ToTensor, RandomResizedCrop, Resize

from diversity_src.dataloaders.common import IndexableDataSet, ToRGB, DifferentTaskTransformIndexableForEachDataset

from torchvision import transforms
from PIL.Image import LANCZOS

import os
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.omniglot import Omniglot

from uutils import expanduser, report_times, download_and_extract
from uutils.torch_uu.dataloaders.meta_learning.mini_imagenet_mi_l2l import Task_transform_mi, get_mi_datasets
from uutils.torch_uu.dataloaders.meta_learning.omniglot_l2l import get_omniglot_datasets, Task_transform_omniglot


def get_mi_and_omniglot_list_data_set_splits(
        root: str = '~/data/l2l_data/',
        data_augmentation: str = 'hdb1',
        device=None,
):
    dataset_list_train: list[MetaDataset, MetaDataset, MetaDataset] = []
    dataset_list_validation: list[MetaDataset, MetaDataset, MetaDataset] = []
    dataset_list_test: list[MetaDataset, MetaDataset, MetaDataset] = []

    #
    train_dataset, validation_dataset, test_dataset = get_mi_datasets(root, data_augmentation, device)
    assert isinstance(train_dataset, Dataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    train_dataset, validation_dataset, test_dataset = get_omniglot_datasets(root, data_augmentation, device)
    assert isinstance(train_dataset, Dataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    # - print the number of classes for all splits (but only need train for usl model final layer)
    print('-- Printing num classes')
    from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_test, verbose=True)
    return dataset_list_train, dataset_list_validation, dataset_list_test


def get_indexable_list_of_datasets_mi_and_omniglot(
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb1',
        device=None,
        **kwargs,
) -> tuple[IndexableDataSet]:
    """
    note:
        - omniglot has no img data (i.e. x) data augmentation but mi does (random crop, color jitter, h flip).
        Omniglot's img (i.e. x) transform is only to match size of mi images.
    """
    dataset_list_train, dataset_list_validation, dataset_list_test = get_mi_and_omniglot_list_data_set_splits(root,
                                                                                                              data_augmentation,
                                                                                                              device)

    train_dataset = IndexableDataSet(dataset_list_train)
    validation_dataset = IndexableDataSet(dataset_list_validation)
    test_dataset = IndexableDataSet(dataset_list_test)
    _datasets = (train_dataset, validation_dataset, test_dataset)
    return _datasets


def hdb1_mi_omniglot_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,  # let it be -1 for continual tasks https://github.com/learnables/learn2learn/issues/315
        root='~/data/l2l_data/',
        data_augmentation='hdb1',
        # device=None,
        **kwargs,
) -> BenchmarkTasksets:
    root = os.path.expanduser(root)

    #
    _datasets: tuple[IndexableDataSet] = get_indexable_list_of_datasets_mi_and_omniglot(root, data_augmentation)
    train_dataset, validation_dataset, test_dataset = _datasets
    # assert isinstance(train_dataset[0].dataset, l2l.vision.datasets.mini_imagenet.MiniImagenet)
    # assert isinstance(train_dataset[1].dataset.dataset, l2l.vision.datasets.full_omniglot.FullOmniglot)
    # assert isinstance(validation_dataset[0].dataset, l2l.vision.datasets.mini_imagenet.MiniImagenet)
    # assert isinstance(validation_dataset[1].dataset.dataset, l2l.vision.datasets.full_omniglot.FullOmniglot)
    # assert isinstance(test_dataset[0].dataset, l2l.vision.datasets.mini_imagenet.MiniImagenet)
    # assert isinstance(test_dataset[1].dataset.dataset, l2l.vision.datasets.full_omniglot.FullOmniglot)
    # TODO addition assert to check its the right split by checking the name

    # - get task transforms
    train_task_transform_mi: Callable = Task_transform_mi(train_ways, train_samples)
    test_task_transform_mi: Callable = Task_transform_mi(test_ways, test_samples)

    train_task_transform_omni: Callable = Task_transform_omniglot(train_ways, train_samples)
    test_task_transform_omni: Callable = Task_transform_omniglot(test_ways, test_samples)

    dict_cons_remaining_task_transforms: dict = {
        train_dataset[0].name: train_task_transform_mi,
        train_dataset[1].name: train_task_transform_omni,
        validation_dataset[0].name: test_task_transform_mi,
        validation_dataset[1].name: test_task_transform_omni,
        test_dataset[0].name: test_task_transform_mi,
        test_dataset[1].name: test_task_transform_omni
    }

    train_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(train_dataset,
                                                                                    dict_cons_remaining_task_transforms)
    validation_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(validation_dataset,
                                                                                         dict_cons_remaining_task_transforms)
    test_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(test_dataset,
                                                                                   dict_cons_remaining_task_transforms)

    # Instantiate the tasksets
    train_tasks = l2l.data.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )
    validation_tasks = l2l.data.TaskDataset(
        dataset=validation_dataset,
        task_transforms=validation_transforms,
        num_tasks=num_tasks,
    )
    test_tasks = l2l.data.TaskDataset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=num_tasks,
    )
    return BenchmarkTasksets(train_tasks, validation_tasks, test_tasks)


# - test

def loop_through_l2l_indexable_benchmark_with_model_test():
    # - for determinism
    # random.seed(0)
    # torch.manual_seed(0)
    # np.random.seed(0)

    # - options for number of tasks/meta-batch size
    batch_size = 5

    # - get benchmark
    benchmark: BenchmarkTasksets = hdb1_mi_omniglot_tasksets()
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

    print('-- end of test --')


def check_if_omniglots_labels_are_consistent():
    _datasets = get_omniglot_datasets()
    train, val, test = _datasets
    print(train.labels)
    print(val.labels)
    print(test.labels)
    _datasets = get_omniglot_datasets()
    train, val, test = _datasets
    print(train.labels)
    print(val.labels)
    print(test.labels)


def next_omniglot_and_mi_normal_dataloader():
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import get_mi_and_omniglot_list_data_set_splits
    from torch.utils.data import DataLoader
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders

    # - params
    root: str = '~/data/l2l_data/'
    data_augmentation: str = 'hdb1'

    # - test if data sets can be created into pytorch dataloader
    _, _, dataset_list = get_mi_and_omniglot_list_data_set_splits(root, data_augmentation)
    mi, omni = dataset_list

    loader = DataLoader(omni, num_workers=1)
    next(iter(loader))
    print()

    loader = DataLoader(mi, num_workers=1)
    next(iter(loader))
    print()


# -- Run experiment

if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    # next_omniglot_and_mi_normal_dataloader()
    loop_through_l2l_indexable_benchmark_with_model_test()
    # check_if_omniglots_labels_are_consistent()
    # download_mini_imagenet_fix()
    # download_mini_imagenet_fix_use_gdrive()
    # loop_through_mi_local()
    # download_mini_imagenet_brandos_download_from_zenodo()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
