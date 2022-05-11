import os
from random import random

import learn2learn as l2l
import numpy as np
import torch
import torchvision
from learn2learn.data import MetaDataset, FilteredMetaDataset
from learn2learn.vision.benchmarks import BenchmarkTasksets
from torch import nn
from torchvision.transforms import Compose, Normalize, ToPILImage, RandomCrop, ColorJitter, RandomHorizontalFlip, \
    ToTensor

from diversity_src.dataloaders.common import IndexableDataSet, ToRGB

from torchvision import transforms
from PIL.Image import LANCZOS

from models import get_model


def get_remaining_transforms_mi(dataset: MetaDataset, ways:int, shots: int) -> list[TaskTransform]:
    import learn2learn as l2l
    remaining_task_transforms = [
        l2l.data.transforms.NWays(dataset, n=ways),
        l2l.data.transforms.KShots(dataset, k=shots),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
    ]
    return remaining_task_transforms

def get_remaining_transforms_omnigglot(dataset: MetaDataset, ways: int, shots: int) -> list[TaskTransform]:
    import learn2learn as l2l
    remaining_task_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=ways,
                                             k=shots),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    return remaining_task_transforms

class SingleDatasetPerTaskTransform(Callable):
    """
    Transform that samples a data set first, then creates a task (e.g. n-way, k-shot) and finally
    applies the remaining task transforms.
    """

    def __init__(self,
                 indexable_dataset: IndexableDataSet,
                 dict_cons_remaining_task_transforms: dict[str, Callable],
                 ):
        """

        :param: cons_remaining_task_transforms; constructor that builds the remaining task transforms. Cannot be a list
        of transforms because we don't know apriori which is the data set we will use. So this function should be of
        type MetaDataset -> list[TaskTransforms] i.e. given the dataset it returns the transforms for it.
        """
        self.indexable_dataset = MetaDataset(indexable_dataset)
        self.cons_remaining_task_transforms = dict_cons_remaining_task_transforms

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
        dataset = MetaDataset(dataset)
        dataset_type = type(dataset.dataset.dataset.dataset)

        # - use the sampled data set to create task
        remaining_task_transforms: list[TaskTransform] = self.cons_remaining_task_transforms[dataset_type](dataset)
        description = None
        for transform in remaining_task_transforms:
            description = transform(description)
        return description

def get_omniglot_datasets(
        root: str = '~/data/l2l_data/',
        data_transform_option: bool = 'hd1',
        device=None,
        **kwargs,
):
    if data_transform_option == 'l2l_original_data_transform':
        data_transforms = transforms.Compose([
            transforms.Resize(28, interpolation=LANCZOS),
            transforms.ToTensor(),
            lambda x: 1.0 - x,
        ])
    elif data_transform_option == 'hd1':
        data_transforms = transforms.Compose([
            ToRGB(),
            # lambda x: x.convert("RGB")
            # transforms.Resize(84, interpolation=LANCZOS),
            # mimicking RandomCrop(84, padding=8) in mi
            transforms.Resize(84),
            torchvision.transforms.Pad(8),
            transforms.ToTensor(),
            lambda x: 1.0 - x,  # note: task2vec doesn't have this for mnist, wonder why...
        ])
    else:
        raise ValueError(f'Invalid data transform option for omniglot, got {}')

    omniglot = l2l.vision.datasets.FullOmniglot(
        root=root,
        transform=data_transforms,
        download=True,
    )
    if device is not None:
        dataset = l2l.data.OnDeviceDataset(omniglot, device=device)  # bug in l2l
    dataset: MetaDataset = l2l.data.MetaDataset(omniglot)

    classes = list(range(1623))
    random.shuffle(classes)
    train_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[:1100])
    validation_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1100:1200])
    test_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1200:])
    assert isinstance(train_dataset, MetaDataset)

    _datasets = (train_dataset, validation_dataset, test_dataset)
    return _datasets


def get_mi_datasets(
        root='~/data/l2l_data/',
        data_augmentation='hd1',
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
    elif data_augmentation == 'lee2019' or data_augmentation == 'hd1':
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

    _datasets = (train_dataset, valid_dataset, test_dataset)
    return _datasets


def get_indexable_list_of_datasets_mi_and_omniglot(
        root: str = '~/data/l2l_data/',
        data_augmentation='hd1',
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
    train_dataset, validation_dataset, test_dataset = get_omniglot_datasets(root, device)
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
        num_tasks = -1,
        root='~/data/l2l_data/',
        device=None,
        **kwargs,
):
    root = os.path.expanduser(root)

    #
    _datasets: tuple[IndexableDataSet] = get_indexable_list_of_datasets_mi_and_omniglot(root)
    train_dataset, validation_dataset, test_dataset = _datasets

    # - get task transforms
    # "<class 'learn2learn.vision.datasets.full_omniglot.FullOmniglot'>"
    # dataset.dataset.dataset.dataset
    train_task_transforms = None
    test_val_task_transforms = None
    dict_cons_remaining_task_transforms = {}
    train_transforms: TaskTransform = SingleDatasetPerTaskTransform(indexable_dataset, get_remaining_transforms)
    validation_transforms: TaskTransform = SingleDatasetPerTaskTransform(indexable_dataset, get_remaining_transforms)
    test_transforms: TaskTransform = SingleDatasetPerTaskTransform(indexable_dataset, get_remaining_transforms)

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

    # - get benchmark
    benchmark: BenchmarkTasksets = hd1_mi_omniglot_tasksets()

    # - get train taskdata set
    taskset = getattr(benchmark, 'train')

    # - loop through tasks
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model = get_model('resnet18', pretrained=False, num_classes=5).to(device)
    # TODO: model = resnet12
    criterion = nn.CrossEntropyLoss()
    for task_num in range(batch_size):
        print(f'{task_num=}')
        X, y = taskset.sample()
        y_pred = model(X)
        print(f'{X.size()=}')
        print(f'{y.size()=}')
        print(f'{y=}')
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
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
