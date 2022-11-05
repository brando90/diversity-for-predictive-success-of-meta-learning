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

# !/usr/bin/env python3

import os
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.omniglot import Omniglot


class FullOmniglotUU(Dataset):
    """

    [[Source]]()

    **Description**

    This class provides an interface to the Omniglot dataset.

    The Omniglot dataset was introduced by Lake et al., 2015.
    Omniglot consists of 1623 character classes from 50 different alphabets, each containing 20 samples.
    While the original dataset is separated in background and evaluation sets,
    this class concatenates both sets and leaves to the user the choice of classes splitting
    as was done in Ravi and Larochelle, 2017.
    The background and evaluation splits are available in the `torchvision` package.

    **References**

    1. Lake et al. 2015. “Human-Level Concept Learning through Probabilistic Program Induction.” Science.
    2. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **root** (str) - Path to download the data.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.

    **Example**
    ~~~python
    omniglot = l2l.vision.datasets.FullOmniglot(root='./data',
                                                transform=transforms.Compose([
                                                    transforms.Resize(28, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]),
                                                download=True)
    omniglot = l2l.data.MetaDataset(omniglot)
    ~~~

    """

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        # Set up both the background and eval dataset
        omni_background = Omniglot(self.root, background=True, download=download)
        self.len_omni_background_characters = len(omni_background._characters)
        # Eval labels also start from 0.
        # It's important to add 964 to label values in eval so they don't overwrite background dataset.
        omni_evaluation = Omniglot(self.root,
                                   background=False,
                                   download=download,
                                   target_transform=self._target_transform)

        self.dataset = ConcatDataset((omni_background, omni_evaluation))
        self._bookkeeping_path = os.path.join(self.root, 'omniglot-bookkeeping.pkl')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, character_class = self.dataset[item]
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

    def _target_transform(self, x):
        return x + self.len_omni_background_characters


def get_remaining_transforms_mi(dataset: MetaDataset, ways: int, samples: int) -> list[TaskTransform]:
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
    """

    Q: todo, what does RandomClassRotation do? https://github.com/learnables/learn2learn/issues/372
    """
    import learn2learn as l2l
    remaining_task_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset, ways, shots),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    return remaining_task_transforms


def one_minus_x(x):
    return 1.0 - x


def get_omniglot_datasets(
        root: str = '~/data/l2l_data/',
        data_transform_option: str = 'hdb1',
        device=None,
        **kwargs,
) -> tuple[MetaDataset, MetaDataset, MetaDataset]:
    """
    Get omniglot data set with the provided re-size function -- since when combining different data sets they have to
    be re-sized to have the same size.
    For example, `hdb1` uses the re-size size of MI.
    """
    if data_transform_option == 'l2l_original_data_transform':
        data_transforms = transforms.Compose([
            transforms.Resize(28, interpolation=LANCZOS),
            transforms.ToTensor(),
            one_minus_x,
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
            # lambda x: 1.0 - x,  # note: task2vec doesn't have this for mnist, wonder why...
            one_minus_x
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
            one_minus_x,  # note: task2vec doesn't have this for mnist, wonder why...
        ])
    else:
        raise ValueError(f'Invalid data transform option for omniglot, got instead {data_transform_option=}')

    # dataset = l2l.vision.datasets.FullOmniglot(
    #     root=root,
    #     transform=data_transforms,
    #     download=True,
    # )
    dataset: Dataset = FullOmniglotUU(root=root,
                                      transform=data_transforms,
                                      download=True,
                                      )
    if device is not None:
        dataset = l2l.data.OnDeviceDataset(dataset, device=device)  # bug in l2l
    # dataset: MetaDataset = l2l.data.MetaDataset(omniglot)

    classes = list(range(1623))
    # random.shuffle(classes)  # todo: wish I wouldn't have copied l2l here and removed this...idk if shuffling this does anything interesting. Doubt it.
    train_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[:1100])
    validation_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1100:1200])
    test_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1200:])
    assert isinstance(train_dataset, MetaDataset)
    assert isinstance(validation_dataset, MetaDataset)
    assert isinstance(test_dataset, MetaDataset)
    assert len(train_dataset.labels) == 1100
    assert len(validation_dataset.labels) == 100
    # print(f'{len(classes[1200:])=}')
    assert len(test_dataset.labels) == 423, f'Error, got: {len(test_dataset.labels)=}'

    # - add names to be able to get the right task transform for the indexable dataset
    train_dataset.name = 'train_omniglot'
    validation_dataset.name = 'val_omniglot'
    test_dataset.name = 'test_omniglot'

    _datasets = (train_dataset, validation_dataset, test_dataset)
    return _datasets


def mi_img_int_to_img_float(x) -> float:
    return x / 255.0


def get_mi_datasets(
        root='~/data/l2l_data/',
        data_augmentation='hdb1',
        device=None,
        **kwargs,
) -> tuple[MetaDataset, MetaDataset, MetaDataset]:
    """
    Returns MI according to l2l -- note it seems we are favoring the resizing of MI since when unioning datasets they
    have to have the same size.
    """
    if data_augmentation is None:
        train_data_transforms = None
        test_data_transforms = None
    elif data_augmentation == 'normalize':
        train_data_transforms = Compose([
            # lambda x: x / 255.0,
            mi_img_int_to_img_float,
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
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    train_dataset, validation_dataset, test_dataset = get_omniglot_datasets(root, data_augmentation, device)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
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


class Task_transform_mi(Callable):
    def __init__(self, ways, samples):
        self.ways = ways
        self.samples = samples

    def __call__(self, dataset):
        return get_remaining_transforms_mi(dataset, self.ways, self.samples)


class Task_transform_omniglot(Callable):
    def __init__(self, ways, samples):
        self.ways = ways
        self.samples = samples

    def __call__(self, dataset):
        return get_remaining_transforms_omniglot(dataset, self.ways, self.samples)


def hdb1_mi_omniglot_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,
        root='~/data/l2l_data/',
        device=None,
        **kwargs,
) -> BenchmarkTasksets:
    root = os.path.expanduser(root)

    #
    _datasets: tuple[IndexableDataSet] = get_indexable_list_of_datasets_mi_and_omniglot(root)
    train_dataset, validation_dataset, test_dataset = _datasets
    # assert isinstance(train_dataset[0].dataset, l2l.vision.datasets.mini_imagenet.MiniImagenet)
    # assert isinstance(train_dataset[1].dataset.dataset, l2l.vision.datasets.full_omniglot.FullOmniglot)
    # assert isinstance(validation_dataset[0].dataset, l2l.vision.datasets.mini_imagenet.MiniImagenet)
    # assert isinstance(validation_dataset[1].dataset.dataset, l2l.vision.datasets.full_omniglot.FullOmniglot)
    # assert isinstance(test_dataset[0].dataset, l2l.vision.datasets.mini_imagenet.MiniImagenet)
    # assert isinstance(test_dataset[1].dataset.dataset, l2l.vision.datasets.full_omniglot.FullOmniglot)
    # TODO addition assert to check its the right split by checking the name

    # - get task transforms
    train_task_transform_mi = Task_transform_mi(train_ways, train_samples)
    test_task_transform_mi = Task_transform_mi(test_ways, test_samples)

    train_task_transform_omni = Task_transform_omniglot(train_ways, train_samples)
    test_task_transform_omni = Task_transform_omniglot(test_ways, test_samples)

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
    from uutils import report_times

    start = time.time()
    # - run experiment
    next_omniglot_and_mi_normal_dataloader()
    loop_through_l2l_indexable_benchmark_with_model_test()
    check_if_omniglots_labels_are_consistent()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
