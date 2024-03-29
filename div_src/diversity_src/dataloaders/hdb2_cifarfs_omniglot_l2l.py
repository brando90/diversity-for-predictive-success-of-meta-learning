"""
Note:
    - train_samples != k_shots and test_samples != k_eval. Instead train_samples = test_samples = k_shots + k_eval.
"""
import os
import random

import learn2learn as l2l
import numpy as np
import torch
import torchvision
from learn2learn.data import MetaDataset
from learn2learn.data.transforms import TaskTransform
from learn2learn.vision.benchmarks import BenchmarkTasksets
from torch import nn
from torchvision.transforms import Compose

from diversity_src.dataloaders.common import IndexableDataSet, ToRGB, DifferentTaskTransformIndexableForEachDataset

from uutils.torch_uu.dataloaders.cifar100fs_fc100 import get_transform
from uutils.torch_uu.dataloaders.meta_learning.cifarfs_l2l import get_cifarfs_datasets, get_remaining_transforms_cifarfs
from uutils.torch_uu.dataloaders.meta_learning.omniglot_l2l import get_omniglot_datasets, \
    get_remaining_transforms_omniglot


def get_indexable_list_of_datasets_cifarfs_and_omniglot(
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb2',
        device=None,
        **kwargs,
) -> tuple[IndexableDataSet]:
    """
        - omniglot has no img data (i.e. x) data augmentation but cifarfs does (random crop, color jitter, h flip).
        Omniglot's img (i.e. x) transform is only to match size of cifarfs images.
    """
    dataset_list_train = []
    dataset_list_validation = []
    dataset_list_test = []

    train_dataset, validation_dataset, test_dataset = get_cifarfs_datasets(root, data_augmentation, device)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    train_dataset, validation_dataset, test_dataset = get_omniglot_datasets(root, data_transform_option='hdb2',
                                                                            device=device)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)

    train_dataset = IndexableDataSet(dataset_list_train)
    validation_dataset = IndexableDataSet(dataset_list_validation)
    test_dataset = IndexableDataSet(dataset_list_test)
    _datasets = (train_dataset, validation_dataset, test_dataset)
    return _datasets


def hdb2_cifarfs_omniglot_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,
        root='~/data/l2l_data/',
        device=None,
        **kwargs,
):
    root = os.path.expanduser(root)

    #
    _datasets: tuple[IndexableDataSet] = get_indexable_list_of_datasets_cifarfs_and_omniglot(root)
    train_dataset, validation_dataset, test_dataset = _datasets
    assert isinstance(train_dataset[0].dataset, l2l.vision.datasets.cifarfs.CIFARFS)
    assert isinstance(train_dataset[1].dataset.dataset, l2l.vision.datasets.full_omniglot.FullOmniglot)
    assert isinstance(validation_dataset[0].dataset, l2l.vision.datasets.cifarfs.CIFARFS)
    assert isinstance(validation_dataset[1].dataset.dataset, l2l.vision.datasets.full_omniglot.FullOmniglot)
    assert isinstance(test_dataset[0].dataset, l2l.vision.datasets.cifarfs.CIFARFS)
    assert isinstance(test_dataset[1].dataset.dataset, l2l.vision.datasets.full_omniglot.FullOmniglot)
    # TODO addition assert to check its the right split by checking the name

    # - get task transforms
    train_task_transform_cifarfs = lambda dataset: get_remaining_transforms_cifarfs(dataset, train_ways, train_samples)
    test_task_transform_cifarfs = lambda dataset: get_remaining_transforms_cifarfs(dataset, test_ways, test_samples)

    train_task_transform_omni = lambda dataset: get_remaining_transforms_omniglot(dataset, train_ways, train_samples)
    test_task_transform_omni = lambda dataset: get_remaining_transforms_omniglot(dataset, test_ways, test_samples)

    dict_cons_remaining_task_transforms: dict = {
        train_dataset[0].name: train_task_transform_cifarfs,
        train_dataset[1].name: train_task_transform_omni,
        validation_dataset[0].name: test_task_transform_cifarfs,
        validation_dataset[1].name: test_task_transform_omni,
        test_dataset[0].name: test_task_transform_cifarfs,
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

def loop_through_l2l_indexable_benchmark_with_model_hdb2_test():
    from diversity_src.diversity.task2vec_based_metrics.models import get_model
    # - for determinism
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # - options for number of tasks/meta-batch size
    batch_size = 5

    # - get benchmark
    benchmark: BenchmarkTasksets = hdb2_cifarfs_omniglot_tasksets()

    # - get train taskdata set
    splits = ['train', 'validation', 'test']
    tasksets = [getattr(benchmark, split) for split in splits]

    # - loop through tasks
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model = get_model('resnet18', pretrained=True, num_classes=5).to(device)
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


# -- Run experiment

if __name__ == "__main__":
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    loop_through_l2l_indexable_benchmark_with_model_hdb2_test()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
