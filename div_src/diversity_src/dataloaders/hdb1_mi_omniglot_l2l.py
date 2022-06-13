"""
HDB1 := MI + Omni

Note:
    - train_samples != k_shots and test_samples != k_eval. Instead train_samples = test_samples = k_shots + k_eval.
ref:
    - https://wandb.ai/brando/entire-diversity-spectrum/reports/Copy-of-brando-s-Div-ala-task2vec-HDB1-mi-omniglot-HDB2-cifarfs-omniglot---VmlldzoyMDc3OTAz

Diversity: (div, ci)﻿=﻿(﻿0.2161356031894684﻿, 0.038439472241579925﻿)
"""
import os
import random
from typing import Callable

import learn2learn as l2l
import numpy as np
import torch
import torchvision
from learn2learn.data import MetaDataset, FilteredMetaDataset, DataDescription
from learn2learn.data.transforms import TaskTransform
from learn2learn.vision.benchmarks import BenchmarkTasksets
from torch import nn
from torchvision.transforms import Compose, Normalize, ToPILImage, RandomCrop, ColorJitter, RandomHorizontalFlip, \
    ToTensor

from diversity_src.dataloaders.common import IndexableDataSet, ToRGB, BenchmarkName

from torchvision import transforms
from PIL.Image import LANCZOS

from models import get_model

def get_remaining_transforms_mi(dataset: MetaDataset, ways:int, samples: int) -> list[TaskTransform]:
    import learn2learn as l2l
    remaining_task_transforms = [
        l2l.data.transforms.NWays(dataset, ways),
        l2l.data.transforms.KShots(dataset, samples),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
    ]
    return remaining_task_transforms

def get_remaining_transforms_omniglot(dataset: MetaDataset, ways: int, shots: int) -> list[TaskTransform]:
    import learn2learn as l2l
    remaining_task_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset, ways, shots),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    return remaining_task_transforms

class TaskTransformIndexableDataset(Callable):
    """
    Transform that samples a data set first (from indexable data set),
    then creates a cls fls task (e.g. n-way, k-shot) and finally
    gets the remaining task transforms for that data set and applies it.
    """

    def __init__(self,
                 indexable_dataset: IndexableDataSet,
                 dict_cons_remaining_task_transforms: dict[BenchmarkName, Callable],
                 ):
        """

        :param cons_remaining_task_transforms: list of constructors to constructs of task transforms for each data set.
        Given a data set you get the remaining task transforms for that data set so to create tasks properly for
        that data set. One the right data set (and thus split) is known from the name, by indexing using the name you
        get a constructor (function) that should be of type MetaDataset -> list[TaskTransforms]
        i.e. given the actualy dataset (not the name) it returns the remaining transforms for it.
        """
        self.indexable_dataset = MetaDataset(indexable_dataset)
        self.dict_cons_remaining_task_transforms = dict_cons_remaining_task_transforms

    def __call__(self, task_description: list):
        """
        idea:
        - receives the index of the dataset to use
        - then use the normal NWays l2l function
        """
        # - this is what I wish could have gone in a seperate callable transform, but idk how since the transforms take apriori (not dynamically) which data set to use.
        i = random.randint(0, len(self.indexable_dataset) - 1)
        task_description = [DataDescription(index=i)]  # using this to follow the l2l convention

        # - get the sampled data set
        dataset_index = task_description[0].index
        dataset = self.indexable_dataset[dataset_index]
        dataset = MetaDataset(dataset) if not isinstance(dataset, MetaDataset) else dataset
        dataset_name = dataset.name
        # print(f'{dataset_name=}')
        # self.assert_right_dataset(dataset)

        # - use the sampled data set to create task
        remaining_task_transforms: list[TaskTransform] = self.dict_cons_remaining_task_transforms[dataset_name](dataset)
        description = None
        for transform in remaining_task_transforms:
            description = transform(description)
        return description

    # def assert_right_dataset(dataset):
    #     if 'mi' in dataset.name:
    #         assert isinstance(dataset.dataset, l2l.vision.datasets.mini_imagenet.MiniImagenet)
    #     else:
    #         assert isinstance(dataset.dataset.dataset, l2l.vision.datasets.full_omniglot.FullOmniglot)


def get_omniglot_datasets(
        root: str = '~/data/l2l_data/',
        data_transform_option: str = 'hdb1',
        device=None,
        **kwargs,
):
    if data_transform_option == 'l2l_original_data_transform':
        data_transforms = transforms.Compose([
            transforms.Resize(28, interpolation=LANCZOS),
            transforms.ToTensor(),
            lambda x: 1.0 - x,
        ])
    elif data_transform_option == 'hdb1':
        data_transforms = transforms.Compose([
            ToRGB(),
            # lambda x: x.convert("RGB")
            # transforms.Resize(84, interpolation=LANCZOS),
            # mimicking RandomCrop(84, padding=8) in mi
            transforms.Resize(84),
            # torchvision.transforms.Pad(8),
            transforms.ToTensor(),
            lambda x: 1.0 - x,  # note: task2vec doesn't have this for mnist, wonder why...
        ])
    elif data_transform_option == 'hdb2':
        data_transforms = transforms.Compose([
            ToRGB(),
            # lambda x: x.convert("RGB")
            # transforms.Resize(84, interpolation=LANCZOS),
            # mimicking RandomCrop(84, padding=8) in mi
            transforms.Resize(32),
            # torchvision.transforms.Pad(8),
            transforms.ToTensor(),
            lambda x: 1.0 - x,  # note: task2vec doesn't have this for mnist, wonder why...
        ])
    else:
        raise ValueError(f'Invalid data transform option for omniglot, got instead {data_transform_option=}')

    dataset = l2l.vision.datasets.FullOmniglot(
        root=root,
        transform=data_transforms,
        download=True,
    )
    if device is not None:
        dataset = l2l.data.OnDeviceDataset(dataset, device=device)  # bug in l2l
    # dataset: MetaDataset = l2l.data.MetaDataset(omniglot)

    classes = list(range(1623))
    random.shuffle(classes)
    train_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[:1100])
    validation_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1100:1200])
    test_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1200:])
    assert isinstance(train_dataset, MetaDataset)
    assert isinstance(validation_dataset, MetaDataset)
    assert isinstance(test_dataset, MetaDataset)

    # - add names to be able to get the right task transform for the indexable dataset
    train_dataset.name = 'train_omniglot'
    validation_dataset.name = 'val_omniglot'
    test_dataset.name = 'test_omniglot'

    _datasets = (train_dataset, validation_dataset, test_dataset)
    return _datasets


def get_mi_datasets(
        root='~/data/l2l_data/',
        data_augmentation='hdb1',
        device=None,
        **kwargs,
):
    if data_augmentation is None:
        train_data_transforms = None
        test_data_transforms = None
    elif data_augmentation == 'normalize':
        train_data_transforms = Compose([
            lambda x: x / 255.0,
        ])
        test_data_transforms = train_data_transforms
    elif data_augmentation == 'lee2019' or data_augmentation == 'hdb1':
        normalize = Normalize(
            mean=[120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
            std=[70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0],
        )
        train_data_transforms = Compose([
            ToPILImage(),
            RandomCrop(84, padding=8),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])
        test_data_transforms = Compose([
            normalize,
        ])
    else:
        raise ('Invalid data_augmentation argument.')

    train_dataset = l2l.vision.datasets.MiniImagenet(
        root=root,
        mode='train',
        download=True,
    )
    valid_dataset = l2l.vision.datasets.MiniImagenet(
        root=root,
        mode='validation',
        download=True,
    )
    test_dataset = l2l.vision.datasets.MiniImagenet(
        root=root,
        mode='test',
        download=True,
    )
    if device is None:
        train_dataset.transform = train_data_transforms
        valid_dataset.transform = test_data_transforms
        test_dataset.transform = test_data_transforms
    else:
        train_dataset = l2l.data.OnDeviceDataset(
            dataset=train_dataset,
            transform=train_data_transforms,
            device=device,
        )
        valid_dataset = l2l.data.OnDeviceDataset(
            dataset=valid_dataset,
            transform=test_data_transforms,
            device=device,
        )
        test_dataset = l2l.data.OnDeviceDataset(
            dataset=test_dataset,
            transform=test_data_transforms,
            device=device,
        )
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    # - add names to be able to get the right task transform for the indexable dataset
    train_dataset.name = 'train_mi'
    valid_dataset.name = 'val_mi'
    test_dataset.name = 'test_mi'

    _datasets = (train_dataset, valid_dataset, test_dataset)
    return _datasets


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
    dataset_list_train = []
    dataset_list_validation = []
    dataset_list_test = []

    train_dataset, validation_dataset, test_dataset = get_mi_datasets(root, data_augmentation, device)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    train_dataset, validation_dataset, test_dataset = get_omniglot_datasets(root, device=device)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)

    train_dataset = IndexableDataSet(dataset_list_train)
    validation_dataset = IndexableDataSet(dataset_list_validation)
    test_dataset = IndexableDataSet(dataset_list_test)
    _datasets = (train_dataset, validation_dataset, test_dataset)
    return _datasets


def hd1_mi_omniglot_tasksets(
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
    _datasets: tuple[IndexableDataSet] = get_indexable_list_of_datasets_mi_and_omniglot(root)
    train_dataset, validation_dataset, test_dataset = _datasets
    assert isinstance(train_dataset[0].dataset, l2l.vision.datasets.mini_imagenet.MiniImagenet)
    assert isinstance(train_dataset[1].dataset.dataset, l2l.vision.datasets.full_omniglot.FullOmniglot)
    assert isinstance(validation_dataset[0].dataset, l2l.vision.datasets.mini_imagenet.MiniImagenet)
    assert isinstance(validation_dataset[1].dataset.dataset, l2l.vision.datasets.full_omniglot.FullOmniglot)
    assert isinstance(test_dataset[0].dataset, l2l.vision.datasets.mini_imagenet.MiniImagenet)
    assert isinstance(test_dataset[1].dataset.dataset, l2l.vision.datasets.full_omniglot.FullOmniglot)
    # TODO addition assert to check its the right split by checking the name

    # - get task transforms
    train_task_transform_mi = lambda dataset: get_remaining_transforms_mi(dataset, train_ways, train_samples)
    test_task_transform_mi = lambda dataset: get_remaining_transforms_mi(dataset, test_ways, test_samples)

    train_task_transform_omni = lambda dataset: get_remaining_transforms_omniglot(dataset, train_ways, train_samples)
    test_task_transform_omni = lambda dataset: get_remaining_transforms_omniglot(dataset, test_ways, test_samples)

    dict_cons_remaining_task_transforms: dict = {
        train_dataset[0].name: train_task_transform_mi,
        train_dataset[1].name: train_task_transform_omni,
        validation_dataset[0].name: test_task_transform_mi,
        validation_dataset[1].name: test_task_transform_omni,
        test_dataset[0].name: test_task_transform_mi,
        test_dataset[1].name: test_task_transform_omni
    }

    train_transforms: TaskTransform = TaskTransformIndexableDataset(train_dataset, dict_cons_remaining_task_transforms)
    validation_transforms: TaskTransform = TaskTransformIndexableDataset(validation_dataset, dict_cons_remaining_task_transforms)
    test_transforms: TaskTransform = TaskTransformIndexableDataset(test_dataset, dict_cons_remaining_task_transforms)

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
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # - options for number of tasks/meta-batch size
    # args TODO
    batch_size = 5

    # - get benchmark
    benchmark: BenchmarkTasksets = hd1_mi_omniglot_tasksets()

    # - get train taskdata set
    splits = ['train', 'validation', 'test']
    tasksets = [getattr(benchmark, split) for split in splits]

    # - loop through tasks
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model = get_model('resnet18', pretrained=False, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    # TODO: model = resnet12
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
    loop_through_l2l_indexable_benchmark_with_model_test()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
