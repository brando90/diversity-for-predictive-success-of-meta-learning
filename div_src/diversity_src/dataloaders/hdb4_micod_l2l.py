"""
micod = mi + cifarfs + omniglot + delauny
"""

import os
from typing import Callable

from learn2learn.data import MetaDataset, FilteredMetaDataset
from learn2learn.data.transforms import TaskTransform
from learn2learn.vision.benchmarks import BenchmarkTasksets

# -
from torch.utils.data import Dataset

from diversity_src.dataloaders.common import IndexableDataSet, DifferentTaskTransformIndexableForEachDataset
from uutils.torch_uu.dataloaders.meta_learning.cifarfs_l2l import Task_transform_cifarfs
from uutils.torch_uu.dataloaders.meta_learning.delaunay_l2l import Task_transform_delaunay
from uutils.torch_uu.dataloaders.meta_learning.mini_imagenet_mi_l2l import Task_transform_mi
from uutils.torch_uu.dataloaders.meta_learning.omniglot_l2l import Task_transform_omniglot


def get_indexable_list_of_datasets_hdb4_micod(
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb4_micod',
        device=None,
        **kwargs,
) -> tuple[IndexableDataSet]:
    """ Get data sets for the benchmark as usual, but with the indexable datasets."""
    #
    dataset_list_train: list[MetaDataset, MetaDataset, MetaDataset] = []
    dataset_list_validation: list[MetaDataset, MetaDataset, MetaDataset] = []
    dataset_list_test: list[MetaDataset, MetaDataset, MetaDataset] = []

    #
    from uutils.torch_uu.dataloaders.meta_learning.mini_imagenet_mi_l2l import get_mi_datasets
    datasets: tuple[MetaDataset] = get_mi_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    from uutils.torch_uu.dataloaders.meta_learning.omniglot_l2l import get_omniglot_datasets
    datasets: tuple[MetaDataset] = get_omniglot_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    from uutils.torch_uu.dataloaders.meta_learning.cifarfs_l2l import get_cifarfs_datasets
    datasets: tuple[MetaDataset] = get_cifarfs_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    from uutils.torch_uu.dataloaders.meta_learning.delaunay_l2l import get_delaunay_datasets
    datasets: tuple[MetaDataset] = get_delaunay_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    # return dataset_list_train, dataset_list_validation, dataset_list_test
    assert len(dataset_list_train) == 4
    assert len(dataset_list_validation) == 4
    assert len(dataset_list_test) == 4
    #
    train_dataset = IndexableDataSet(dataset_list_train)
    validation_dataset = IndexableDataSet(dataset_list_validation)
    test_dataset = IndexableDataSet(dataset_list_test)
    _datasets = (train_dataset, validation_dataset, test_dataset)
    return _datasets


def get_task_transforms_hdb4_micod(
        _datasets: tuple[IndexableDataSet],
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
) -> tuple[TaskTransform, TaskTransform, TaskTransform]:
    """ Get task transforms for the benchmark as usual, but with the indexable datasets."""
    train_dataset, validation_dataset, test_dataset = _datasets

    train_task_transform_mi: Callable = Task_transform_mi(train_ways, train_samples)
    test_task_transform_mi: Callable = Task_transform_mi(test_ways, test_samples)

    train_task_transform_omni: Callable = Task_transform_omniglot(train_ways, train_samples)
    test_task_transform_omni: Callable = Task_transform_omniglot(test_ways, test_samples)

    train_task_transform_cifarfs: Callable = Task_transform_cifarfs(train_ways, train_samples)
    test_task_transform_cifarfs: Callable = Task_transform_cifarfs(test_ways, test_samples)

    train_task_transform_delaunay: Callable = Task_transform_delaunay(train_ways, train_samples)
    test_task_transform_delaunay: Callable = Task_transform_delaunay(test_ways, test_samples)

    dict_cons_remaining_task_transforms: dict = {
        train_dataset[0].name: train_task_transform_mi,
        train_dataset[1].name: train_task_transform_omni,
        train_dataset[2].name: train_task_transform_cifarfs,
        train_dataset[3].name: train_task_transform_delaunay,

        validation_dataset[0].name: test_task_transform_mi,
        validation_dataset[1].name: test_task_transform_omni,
        validation_dataset[2].name: test_task_transform_cifarfs,
        validation_dataset[3].name: test_task_transform_delaunay,

        test_dataset[0].name: test_task_transform_mi,
        test_dataset[1].name: test_task_transform_omni,
        test_dataset[2].name: test_task_transform_cifarfs,
        test_dataset[3].name: test_task_transform_delaunay,
    }

    train_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(train_dataset,
                                                                                    dict_cons_remaining_task_transforms)
    validation_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(validation_dataset,
                                                                                         dict_cons_remaining_task_transforms)
    test_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(test_dataset,
                                                                                   dict_cons_remaining_task_transforms)
    return train_transforms, validation_transforms, test_transforms


def hdb4_micod_l2l_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,  # let it be -1 for continual tasks https://github.com/learnables/learn2learn/issues/315
        root='~/data/l2l_data/',
        data_augmentation='hdb4_micod',
        # device=None,
        **kwargs,
) -> BenchmarkTasksets:
    """ micod = mi + cifarfs + omniglot + delauny

    Note, the general pattern with l2l is:
        - create a dataset as a meta-dataset
            - data transforms done here
        - create a task-transform
        - create a taskset
        - create a benchmark-taskset
    this applies for indedable datasets as well, so step 1 is indexable and the task transfroms are different for each dataset
    """
    root = os.path.expanduser(root)
    # - get data sets
    from diversity_src.dataloaders.common import IndexableDataSet
    _datasets: tuple[IndexableDataSet] = get_indexable_list_of_datasets_hdb4_micod(root, data_augmentation)
    train_dataset, validation_dataset, test_dataset = _datasets

    # - get task transforms
    _transforms: tuple[TaskTransform, TaskTransform, TaskTransform] = get_task_transforms_hdb4_micod(_datasets,
                                                                                                     train_ways,
                                                                                                     train_samples,
                                                                                                     test_ways,
                                                                                                     test_samples)
    train_transforms, validation_transforms, test_transforms = _transforms

    # Instantiate the tasksets
    import learn2learn as l2l
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


# -


def loop_through_l2l_indexable_benchmark_with_model_test():
    # - for determinism
    import random
    import torch
    import numpy as np
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # - options for number of tasks/meta-batch size
    batch_size = 5

    # - get benchmark
    benchmark: BenchmarkTasksets = hdb4_micod_l2l_tasksets()
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
    from torch import nn
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


# -

if __name__ == "__main__":
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    loop_through_l2l_indexable_benchmark_with_model_test()
    # download_mini_imagenet_brandos_download_from_zenodo()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
