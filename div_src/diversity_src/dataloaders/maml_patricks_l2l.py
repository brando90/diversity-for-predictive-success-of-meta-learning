"""
"""

import os
from typing import Callable

from learn2learn.data import MetaDataset, FilteredMetaDataset
from learn2learn.data.transforms import TaskTransform
from learn2learn.vision.benchmarks import BenchmarkTasksets

# -
from torch.utils.data import Dataset

from diversity_src.dataloaders.common import IndexableDataSet, DifferentTaskTransformIndexableForEachDataset
from uutils.torch_uu.dataloaders.meta_learning.mds_l2l import Task_transform_dtd
from uutils.torch_uu.dataloaders.meta_learning.mds_l2l import Task_transform_fc100
from uutils.torch_uu.dataloaders.meta_learning.mds_l2l import Task_transform_cu_birds
from uutils.torch_uu.dataloaders.meta_learning.mds_l2l import Task_transform_fungi
from uutils.torch_uu.dataloaders.meta_learning.omniglot_l2l import Task_transform_omniglot
from uutils.torch_uu.dataloaders.meta_learning.delaunay_l2l import Task_transform_delaunay
from uutils.torch_uu.dataloaders.meta_learning.aircraft_l2l import Task_transform_aircraft
from uutils.torch_uu.dataloaders.meta_learning.flower_l2l import Task_transform_flower

#- dtd
def get_dtd_list_data_set_splits(
        root: str = '~/data/l2l_data/',
        data_augmentation='dtd',
        device=None,
        **kwargs,
) -> tuple[list, list, list]:
    """ Get data sets for the benchmark as usual, but with the indexable datasets."""
    print(f'{data_augmentation=}')
    #
    dataset_list_train: list[MetaDataset] = []
    dataset_list_validation: list[MetaDataset] = []
    dataset_list_test: list[MetaDataset] = []

    #
    from uutils.torch_uu.dataloaders.meta_learning.mds_l2l import get_dtd_datasets

    datasets: tuple[MetaDataset] = get_dtd_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    # return dataset_list_train, dataset_list_validation, dataset_list_test
    assert len(dataset_list_train) == 1
    assert len(dataset_list_validation) == 1
    assert len(dataset_list_test) == 1
    # - print the number of classes for all splits (but only need train for usl model final layer)
    print('-- Printing num classes')
    from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_test, verbose=True)
    return dataset_list_train, dataset_list_validation, dataset_list_test


def get_task_transforms_dtd(
        _datasets: tuple[IndexableDataSet],
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
) -> tuple[TaskTransform, TaskTransform, TaskTransform]:
    """ Get task transforms for the benchmark as usual, but with the indexable datasets."""
    train_dataset, validation_dataset, test_dataset = _datasets

    train_task_transform_dtd: Callable = Task_transform_dtd(train_ways, train_samples)
    test_task_transform_dtd: Callable = Task_transform_dtd(test_ways, test_samples)

    dict_cons_remaining_task_transforms: dict = {
        train_dataset[0].name: train_task_transform_dtd,
        validation_dataset[0].name: test_task_transform_dtd,
        test_dataset[0].name: test_task_transform_dtd,
    }

    train_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(train_dataset,
                                                                                    dict_cons_remaining_task_transforms)
    validation_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(validation_dataset,
                                                                                         dict_cons_remaining_task_transforms)
    test_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(test_dataset,
                                                                                   dict_cons_remaining_task_transforms)
    return train_transforms, validation_transforms, test_transforms


def dtd_l2l_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,  # let it be -1 for continual tasks https://github.com/learnables/learn2learn/issues/315
        root='~/data/l2l_data/',
        data_augmentation='dtd',
        # device=None,
        **kwargs,
) -> BenchmarkTasksets:
    root = os.path.expanduser(root)
    # - get data sets lists
    dataset_list_train, dataset_list_validation, dataset_list_test = get_dtd_list_data_set_splits(root, data_augmentation)

    # - get indexable datasets
    from diversity_src.dataloaders.common import IndexableDataSet
    train_dataset = IndexableDataSet(dataset_list_train)
    validation_dataset = IndexableDataSet(dataset_list_validation)
    test_dataset = IndexableDataSet(dataset_list_test)
    _datasets = (train_dataset, validation_dataset, test_dataset)

    # - get task transforms
    _transforms: tuple[TaskTransform, TaskTransform, TaskTransform] = get_task_transforms_dtd(_datasets,
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


# - fc100
def get_fc100_list_data_set_splits(
        root: str = '~/data/l2l_data/',
        data_augmentation='fc100',
        device=None,
        **kwargs,
) -> tuple[list, list, list]:
    """ Get data sets for the benchmark as usual, but with the indexable datasets."""
    print(f'{data_augmentation=}')
    #
    dataset_list_train: list[MetaDataset] = []
    dataset_list_validation: list[MetaDataset] = []
    dataset_list_test: list[MetaDataset] = []

    #
    from uutils.torch_uu.dataloaders.meta_learning.mds_l2l import get_fc100_datasets
    datasets: tuple[MetaDataset] = get_fc100_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    # return dataset_list_train, dataset_list_validation, dataset_list_test
    assert len(dataset_list_train) == 1
    assert len(dataset_list_validation) == 1
    assert len(dataset_list_test) == 1
    # - print the number of classes for all splits (but only need train for usl model final layer)
    print('-- Printing num classes')
    from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_test, verbose=True)
    return dataset_list_train, dataset_list_validation, dataset_list_test


def get_task_transforms_fc100(
        _datasets: tuple[IndexableDataSet],
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
) -> tuple[TaskTransform, TaskTransform, TaskTransform]:
    """ Get task transforms for the benchmark as usual, but with the indexable datasets."""
    train_dataset, validation_dataset, test_dataset = _datasets

    train_task_transform_fc100: Callable = Task_transform_fc100(train_ways, train_samples)
    test_task_transform_fc100: Callable = Task_transform_fc100(test_ways, test_samples)

    dict_cons_remaining_task_transforms: dict = {
        train_dataset[0].name: train_task_transform_fc100,
        validation_dataset[0].name: test_task_transform_fc100,
        test_dataset[0].name: test_task_transform_fc100,
    }

    train_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(train_dataset,
                                                                                    dict_cons_remaining_task_transforms)
    validation_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(validation_dataset,
                                                                                         dict_cons_remaining_task_transforms)
    test_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(test_dataset,
                                                                                   dict_cons_remaining_task_transforms)
    return train_transforms, validation_transforms, test_transforms


def fc100_l2l_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,  # let it be -1 for continual tasks https://github.com/learnables/learn2learn/issues/315
        root='~/data/l2l_data/',
        data_augmentation='fc100',
        # device=None,
        **kwargs,
) -> BenchmarkTasksets:
    root = os.path.expanduser(root)
    # - get data sets lists
    dataset_list_train, dataset_list_validation, dataset_list_test = get_fc100_list_data_set_splits(root, data_augmentation)

    # - get indexable datasets
    from diversity_src.dataloaders.common import IndexableDataSet
    train_dataset = IndexableDataSet(dataset_list_train)
    validation_dataset = IndexableDataSet(dataset_list_validation)
    test_dataset = IndexableDataSet(dataset_list_test)
    _datasets = (train_dataset, validation_dataset, test_dataset)

    # - get task transforms
    _transforms: tuple[TaskTransform, TaskTransform, TaskTransform] = get_task_transforms_fc100(_datasets,
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


# - cu_birds
def get_cu_birds_list_data_set_splits(
        root: str = '~/data/l2l_data/',
        data_augmentation='cu_birds',
        device=None,
        **kwargs,
) -> tuple[list, list, list]:
    """ Get data sets for the benchmark as usual, but with the indexable datasets."""
    print(f'{data_augmentation=}')
    #
    dataset_list_train: list[MetaDataset] = []
    dataset_list_validation: list[MetaDataset] = []
    dataset_list_test: list[MetaDataset] = []

    #
    from uutils.torch_uu.dataloaders.meta_learning.mds_l2l import get_cu_birds_datasets
    datasets: tuple[MetaDataset] = get_cu_birds_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    # return dataset_list_train, dataset_list_validation, dataset_list_test
    assert len(dataset_list_train) == 1
    assert len(dataset_list_validation) == 1
    assert len(dataset_list_test) == 1
    # - print the number of classes for all splits (but only need train for usl model final layer)
    print('-- Printing num classes')
    from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_test, verbose=True)
    return dataset_list_train, dataset_list_validation, dataset_list_test


def get_task_transforms_cu_birds(
        _datasets: tuple[IndexableDataSet],
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
) -> tuple[TaskTransform, TaskTransform, TaskTransform]:
    """ Get task transforms for the benchmark as usual, but with the indexable datasets."""
    train_dataset, validation_dataset, test_dataset = _datasets

    train_task_transform_cu_birds: Callable = Task_transform_cu_birds(train_ways, train_samples)
    test_task_transform_cu_birds: Callable = Task_transform_cu_birds(test_ways, test_samples)

    dict_cons_remaining_task_transforms: dict = {
        train_dataset[0].name: train_task_transform_cu_birds,
        validation_dataset[0].name: test_task_transform_cu_birds,
        test_dataset[0].name: test_task_transform_cu_birds,
    }

    train_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(train_dataset,
                                                                                    dict_cons_remaining_task_transforms)
    validation_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(validation_dataset,
                                                                                         dict_cons_remaining_task_transforms)
    test_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(test_dataset,
                                                                                   dict_cons_remaining_task_transforms)
    return train_transforms, validation_transforms, test_transforms


def cu_birds_l2l_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,  # let it be -1 for continual tasks https://github.com/learnables/learn2learn/issues/315
        root='~/data/l2l_data/',
        data_augmentation='cu_birds',
        # device=None,
        **kwargs,
) -> BenchmarkTasksets:
    root = os.path.expanduser(root)
    # - get data sets lists
    dataset_list_train, dataset_list_validation, dataset_list_test = get_cu_birds_list_data_set_splits(root, data_augmentation)

    # - get indexable datasets
    from diversity_src.dataloaders.common import IndexableDataSet
    train_dataset = IndexableDataSet(dataset_list_train)
    validation_dataset = IndexableDataSet(dataset_list_validation)
    test_dataset = IndexableDataSet(dataset_list_test)
    _datasets = (train_dataset, validation_dataset, test_dataset)

    # - get task transforms
    _transforms: tuple[TaskTransform, TaskTransform, TaskTransform] = get_task_transforms_cu_birds(_datasets,
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




# - fungi
def get_fungi_list_data_set_splits(
        root: str = '~/data/l2l_data/',
        data_augmentation='fungi',
        device=None,
        **kwargs,
) -> tuple[list, list, list]:
    """ Get data sets for the benchmark as usual, but with the indexable datasets."""
    print(f'{data_augmentation=}')
    #
    dataset_list_train: list[MetaDataset] = []
    dataset_list_validation: list[MetaDataset] = []
    dataset_list_test: list[MetaDataset] = []

    #
    from uutils.torch_uu.dataloaders.meta_learning.mds_l2l import get_fungi_datasets
    datasets: tuple[MetaDataset] = get_fungi_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    # return dataset_list_train, dataset_list_validation, dataset_list_test
    assert len(dataset_list_train) == 1
    assert len(dataset_list_validation) == 1
    assert len(dataset_list_test) == 1
    # - print the number of classes for all splits (but only need train for usl model final layer)
    print('-- Printing num classes')
    from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_test, verbose=True)
    return dataset_list_train, dataset_list_validation, dataset_list_test


def get_task_transforms_fungi(
        _datasets: tuple[IndexableDataSet],
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
) -> tuple[TaskTransform, TaskTransform, TaskTransform]:
    """ Get task transforms for the benchmark as usual, but with the indexable datasets."""
    train_dataset, validation_dataset, test_dataset = _datasets

    train_task_transform_fungi: Callable = Task_transform_fungi(train_ways, train_samples)
    test_task_transform_fungi: Callable = Task_transform_fungi(test_ways, test_samples)

    dict_cons_remaining_task_transforms: dict = {
        train_dataset[0].name: train_task_transform_fungi,
        validation_dataset[0].name: test_task_transform_fungi,
        test_dataset[0].name: test_task_transform_fungi,
    }

    train_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(train_dataset,
                                                                                    dict_cons_remaining_task_transforms)
    validation_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(validation_dataset,
                                                                                         dict_cons_remaining_task_transforms)
    test_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(test_dataset,
                                                                                   dict_cons_remaining_task_transforms)
    return train_transforms, validation_transforms, test_transforms


def fungi_l2l_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,  # let it be -1 for continual tasks https://github.com/learnables/learn2learn/issues/315
        root='~/data/l2l_data/',
        data_augmentation='fungi',
        # device=None,
        **kwargs,
) -> BenchmarkTasksets:
    root = os.path.expanduser(root)
    # - get data sets lists
    dataset_list_train, dataset_list_validation, dataset_list_test = get_fungi_list_data_set_splits(root, data_augmentation)

    # - get indexable datasets
    from diversity_src.dataloaders.common import IndexableDataSet
    train_dataset = IndexableDataSet(dataset_list_train)
    validation_dataset = IndexableDataSet(dataset_list_validation)
    test_dataset = IndexableDataSet(dataset_list_test)
    _datasets = (train_dataset, validation_dataset, test_dataset)

    # - get task transforms
    _transforms: tuple[TaskTransform, TaskTransform, TaskTransform] = get_task_transforms_fungi(_datasets,
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



# - aircraft
def get_aircraft_list_data_set_splits(
        root: str = '~/data/l2l_data/',
        data_augmentation='aircraft',
        device=None,
        **kwargs,
) -> tuple[list, list, list]:
    """ Get data sets for the benchmark as usual, but with the indexable datasets."""
    print(f'{data_augmentation=}')
    #
    dataset_list_train: list[MetaDataset] = []
    dataset_list_validation: list[MetaDataset] = []
    dataset_list_test: list[MetaDataset] = []

    #
    from uutils.torch_uu.dataloaders.meta_learning.aircraft_l2l import get_aircraft_datasets
    datasets: tuple[MetaDataset] = get_aircraft_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    # return dataset_list_train, dataset_list_validation, dataset_list_test
    assert len(dataset_list_train) == 1
    assert len(dataset_list_validation) == 1
    assert len(dataset_list_test) == 1
    # - print the number of classes for all splits (but only need train for usl model final layer)
    print('-- Printing num classes')
    from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_test, verbose=True)
    return dataset_list_train, dataset_list_validation, dataset_list_test


def get_task_transforms_aircraft(
        _datasets: tuple[IndexableDataSet],
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
) -> tuple[TaskTransform, TaskTransform, TaskTransform]:
    """ Get task transforms for the benchmark as usual, but with the indexable datasets."""
    train_dataset, validation_dataset, test_dataset = _datasets

    train_task_transform_aircraft: Callable = Task_transform_aircraft(train_ways, train_samples)
    test_task_transform_aircraft: Callable = Task_transform_aircraft(test_ways, test_samples)

    dict_cons_remaining_task_transforms: dict = {
        train_dataset[0].name: train_task_transform_aircraft,
        validation_dataset[0].name: test_task_transform_aircraft,
        test_dataset[0].name: test_task_transform_aircraft,
    }

    train_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(train_dataset,
                                                                                    dict_cons_remaining_task_transforms)
    validation_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(validation_dataset,
                                                                                         dict_cons_remaining_task_transforms)
    test_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(test_dataset,
                                                                                   dict_cons_remaining_task_transforms)
    return train_transforms, validation_transforms, test_transforms


def aircraft_l2l_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,  # let it be -1 for continual tasks https://github.com/learnables/learn2learn/issues/315
        root='~/data/l2l_data/',
        data_augmentation='hdb5_vggair',
        # device=None,
        **kwargs,
) -> BenchmarkTasksets:
    root = os.path.expanduser(root)
    # - get data sets lists
    dataset_list_train, dataset_list_validation, dataset_list_test = get_aircraft_list_data_set_splits(root, data_augmentation)

    # - get indexable datasets
    from diversity_src.dataloaders.common import IndexableDataSet
    train_dataset = IndexableDataSet(dataset_list_train)
    validation_dataset = IndexableDataSet(dataset_list_validation)
    test_dataset = IndexableDataSet(dataset_list_test)
    _datasets = (train_dataset, validation_dataset, test_dataset)

    # - get task transforms
    _transforms: tuple[TaskTransform, TaskTransform, TaskTransform] = get_task_transforms_aircraft(_datasets,
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



# - flower
def get_flower_list_data_set_splits(
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb5_vggair',
        device=None,
        **kwargs,
) -> tuple[list, list, list]:
    """ Get data sets for the benchmark as usual, but with the indexable datasets."""
    print(f'{data_augmentation=}')
    #
    dataset_list_train: list[MetaDataset] = []
    dataset_list_validation: list[MetaDataset] = []
    dataset_list_test: list[MetaDataset] = []

    #
    from uutils.torch_uu.dataloaders.meta_learning.flower_l2l import get_flower_datasets
    datasets: tuple[MetaDataset] = get_flower_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    # return dataset_list_train, dataset_list_validation, dataset_list_test
    assert len(dataset_list_train) == 1
    assert len(dataset_list_validation) == 1
    assert len(dataset_list_test) == 1
    # - print the number of classes for all splits (but only need train for usl model final layer)
    print('-- Printing num classes')
    from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_test, verbose=True)
    return dataset_list_train, dataset_list_validation, dataset_list_test


def get_task_transforms_flower(
        _datasets: tuple[IndexableDataSet],
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
) -> tuple[TaskTransform, TaskTransform, TaskTransform]:
    """ Get task transforms for the benchmark as usual, but with the indexable datasets."""
    train_dataset, validation_dataset, test_dataset = _datasets

    train_task_transform_flower: Callable = Task_transform_flower(train_ways, train_samples)
    test_task_transform_flower: Callable = Task_transform_flower(test_ways, test_samples)

    dict_cons_remaining_task_transforms: dict = {
        train_dataset[0].name: train_task_transform_flower,
        validation_dataset[0].name: test_task_transform_flower,
        test_dataset[0].name: test_task_transform_flower,
    }

    train_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(train_dataset,
                                                                                    dict_cons_remaining_task_transforms)
    validation_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(validation_dataset,
                                                                                         dict_cons_remaining_task_transforms)
    test_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(test_dataset,
                                                                                   dict_cons_remaining_task_transforms)
    return train_transforms, validation_transforms, test_transforms


def flower_l2l_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,  # let it be -1 for continual tasks https://github.com/learnables/learn2learn/issues/315
        root='~/data/l2l_data/',
        data_augmentation='flower',
        # device=None,
        **kwargs,
) -> BenchmarkTasksets:
    root = os.path.expanduser(root)
    # - get data sets lists
    dataset_list_train, dataset_list_validation, dataset_list_test = get_flower_list_data_set_splits(root, data_augmentation)

    # - get indexable datasets
    from diversity_src.dataloaders.common import IndexableDataSet
    train_dataset = IndexableDataSet(dataset_list_train)
    validation_dataset = IndexableDataSet(dataset_list_validation)
    test_dataset = IndexableDataSet(dataset_list_test)
    _datasets = (train_dataset, validation_dataset, test_dataset)

    # - get task transforms
    _transforms: tuple[TaskTransform, TaskTransform, TaskTransform] = get_task_transforms_flower(_datasets,
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



# - delaunay
def get_delaunay_list_data_set_splits(
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb4_micod',
        device=None,
        **kwargs,
) -> tuple[list, list, list]:
    """ Get data sets for the benchmark as usual, but with the indexable datasets."""
    print(f'{data_augmentation=}')
    #
    dataset_list_train: list[MetaDataset] = []
    dataset_list_validation: list[MetaDataset] = []
    dataset_list_test: list[MetaDataset] = []

    #
    from uutils.torch_uu.dataloaders.meta_learning.delaunay_l2l import get_delaunay_datasets
    datasets: tuple[MetaDataset] = get_delaunay_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    # return dataset_list_train, dataset_list_validation, dataset_list_test
    assert len(dataset_list_train) == 1
    assert len(dataset_list_validation) == 1
    assert len(dataset_list_test) == 1
    # - print the number of classes for all splits (but only need train for usl model final layer)
    print('-- Printing num classes')
    from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_test, verbose=True)
    return dataset_list_train, dataset_list_validation, dataset_list_test


def get_task_transforms_delaunay(
        _datasets: tuple[IndexableDataSet],
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
) -> tuple[TaskTransform, TaskTransform, TaskTransform]:
    """ Get task transforms for the benchmark as usual, but with the indexable datasets."""
    train_dataset, validation_dataset, test_dataset = _datasets

    train_task_transform_delaunay: Callable = Task_transform_delaunay(train_ways, train_samples)
    test_task_transform_delaunay: Callable = Task_transform_delaunay(test_ways, test_samples)

    dict_cons_remaining_task_transforms: dict = {
        train_dataset[0].name: train_task_transform_delaunay,
        validation_dataset[0].name: test_task_transform_delaunay,
        test_dataset[0].name: test_task_transform_delaunay,
    }

    train_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(train_dataset,
                                                                                    dict_cons_remaining_task_transforms)
    validation_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(validation_dataset,
                                                                                         dict_cons_remaining_task_transforms)
    test_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(test_dataset,
                                                                                   dict_cons_remaining_task_transforms)
    return train_transforms, validation_transforms, test_transforms


def delaunay_l2l_tasksets(
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
    root = os.path.expanduser(root)
    # - get data sets lists
    dataset_list_train, dataset_list_validation, dataset_list_test = get_delaunay_list_data_set_splits(root, data_augmentation)

    # - get indexable datasets
    from diversity_src.dataloaders.common import IndexableDataSet
    train_dataset = IndexableDataSet(dataset_list_train)
    validation_dataset = IndexableDataSet(dataset_list_validation)
    test_dataset = IndexableDataSet(dataset_list_test)
    _datasets = (train_dataset, validation_dataset, test_dataset)

    # - get task transforms
    _transforms: tuple[TaskTransform, TaskTransform, TaskTransform] = get_task_transforms_delaunay(_datasets,
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

# - hdb6: pick 4 dataset (aircarft vggflower delunay omni)
# # first do Task2Vec computation THEN go ahead and do the expt!!

# - hdb6
def get_hdb6_list_data_set_splits(
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb4_micod',
        device=None,
        **kwargs,
) -> tuple[list, list, list]:
    """ Get data sets for the benchmark as usual, but with the indexable datasets."""
    print(f'{data_augmentation=}')
    #
    dataset_list_train: list[MetaDataset] = []
    dataset_list_validation: list[MetaDataset] = []
    dataset_list_test: list[MetaDataset] = []

    #hdb6 proposal: aircraft vggflower delunay omniglot
    #
    from uutils.torch_uu.dataloaders.meta_learning.aircraft_l2l import get_aircraft_datasets
    datasets: tuple[MetaDataset] = get_aircraft_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    from uutils.torch_uu.dataloaders.meta_learning.flower_l2l import get_flower_datasets
    datasets: tuple[MetaDataset] = get_flower_datasets(root, data_augmentation, device)
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
    #
    from uutils.torch_uu.dataloaders.meta_learning.omniglot_l2l import get_omniglot_datasets
    datasets: tuple[MetaDataset] = get_omniglot_datasets(root, data_augmentation, device)
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
    # - print the number of classes for all splits (but only need train for usl model final layer)
    print('-- Printing num classes')
    from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_test, verbose=True)
    return dataset_list_train, dataset_list_validation, dataset_list_test


def get_task_transforms_hdb6(
        _datasets: tuple[IndexableDataSet],
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
) -> tuple[TaskTransform, TaskTransform, TaskTransform]:
    """ Get task transforms for the benchmark as usual, but with the indexable datasets."""
    train_dataset, validation_dataset, test_dataset = _datasets

    train_task_transform_aircraft: Callable = Task_transform_aircraft(train_ways, train_samples)
    test_task_transform_aircraft: Callable = Task_transform_aircraft(test_ways, test_samples)

    train_task_transform_flower: Callable = Task_transform_flower(train_ways, train_samples)
    test_task_transform_flower: Callable = Task_transform_flower(test_ways, test_samples)

    train_task_transform_delaunay: Callable = Task_transform_delaunay(train_ways, train_samples)
    test_task_transform_delaunay: Callable = Task_transform_delaunay(test_ways, test_samples)

    train_task_transform_omni: Callable = Task_transform_omniglot(train_ways, train_samples)
    test_task_transform_omni: Callable = Task_transform_omniglot(test_ways, test_samples)

    dict_cons_remaining_task_transforms: dict = {
        train_dataset[0].name: train_task_transform_aircraft,
        train_dataset[1].name: train_task_transform_flower,
        train_dataset[2].name: train_task_transform_delaunay,
        train_dataset[3].name: train_task_transform_omni,

        validation_dataset[0].name: test_task_transform_aircraft,
        validation_dataset[1].name: test_task_transform_flower,
        validation_dataset[2].name: test_task_transform_delaunay,
        validation_dataset[3].name: test_task_transform_omni,

        test_dataset[0].name: test_task_transform_aircraft,
        test_dataset[1].name: test_task_transform_flower,
        test_dataset[2].name: test_task_transform_delaunay,
        test_dataset[3].name: test_task_transform_omni
    }

    train_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(train_dataset,
                                                                                    dict_cons_remaining_task_transforms)
    validation_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(validation_dataset,
                                                                                         dict_cons_remaining_task_transforms)
    test_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(test_dataset,
                                                                                   dict_cons_remaining_task_transforms)
    return train_transforms, validation_transforms, test_transforms


def hdb6_l2l_tasksets(
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
    root = os.path.expanduser(root)
    # - get data sets lists
    dataset_list_train, dataset_list_validation, dataset_list_test = get_hdb6_list_data_set_splits(root, data_augmentation)

    # - get indexable datasets
    from diversity_src.dataloaders.common import IndexableDataSet
    train_dataset = IndexableDataSet(dataset_list_train)
    validation_dataset = IndexableDataSet(dataset_list_validation)
    test_dataset = IndexableDataSet(dataset_list_test)
    _datasets = (train_dataset, validation_dataset, test_dataset)

    # - get task transforms
    _transforms: tuple[TaskTransform, TaskTransform, TaskTransform] = get_task_transforms_hdb6(_datasets,
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



# - hdb7
def get_hdb7_list_data_set_splits(
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb4_micod',
        device=None,
        **kwargs,
) -> tuple[list, list, list]:
    """ Get data sets for the benchmark as usual, but with the indexable datasets."""
    print(f'{data_augmentation=}')
    #
    dataset_list_train: list[MetaDataset] = []
    dataset_list_validation: list[MetaDataset] = []
    dataset_list_test: list[MetaDataset] = []

    #hdb7 proposal: aircraft vggflower dtd omniglot
    #
    from uutils.torch_uu.dataloaders.meta_learning.aircraft_l2l import get_aircraft_datasets
    datasets: tuple[MetaDataset] = get_aircraft_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    from uutils.torch_uu.dataloaders.meta_learning.flower_l2l import get_flower_datasets
    datasets: tuple[MetaDataset] = get_flower_datasets(root, data_augmentation, device)
    train_dataset, validation_dataset, test_dataset = datasets
    assert isinstance(train_dataset, Dataset)
    assert isinstance(train_dataset, MetaDataset)
    dataset_list_train.append(train_dataset)
    dataset_list_validation.append(validation_dataset)
    dataset_list_test.append(test_dataset)
    #
    from uutils.torch_uu.dataloaders.meta_learning.mds_l2l import get_dtd_datasets
    #from uutils.torch_uu.dataloaders.meta_learning.dtd_l2l import get_dtd_datasets
    datasets: tuple[MetaDataset] = get_dtd_datasets(root, data_augmentation, device)
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

    # return dataset_list_train, dataset_list_validation, dataset_list_test
    assert len(dataset_list_train) == 4
    assert len(dataset_list_validation) == 4
    assert len(dataset_list_test) == 4
    # - print the number of classes for all splits (but only need train for usl model final layer)
    print('-- Printing num classes')
    from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    get_num_classes_l2l_list_meta_dataset(dataset_list_test, verbose=True)
    return dataset_list_train, dataset_list_validation, dataset_list_test


def get_task_transforms_hdb7(
        _datasets: tuple[IndexableDataSet],
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
) -> tuple[TaskTransform, TaskTransform, TaskTransform]:
    """ Get task transforms for the benchmark as usual, but with the indexable datasets."""
    train_dataset, validation_dataset, test_dataset = _datasets

    train_task_transform_aircraft: Callable = Task_transform_aircraft(train_ways, train_samples)
    test_task_transform_aircraft: Callable = Task_transform_aircraft(test_ways, test_samples)

    train_task_transform_flower: Callable = Task_transform_flower(train_ways, train_samples)
    test_task_transform_flower: Callable = Task_transform_flower(test_ways, test_samples)

    train_task_transform_dtd: Callable = Task_transform_dtd(train_ways, train_samples)
    test_task_transform_dtd: Callable = Task_transform_dtd(test_ways, test_samples)

    train_task_transform_omni: Callable = Task_transform_omniglot(train_ways, train_samples)
    test_task_transform_omni: Callable = Task_transform_omniglot(test_ways, test_samples)

    dict_cons_remaining_task_transforms: dict = {
        train_dataset[0].name: train_task_transform_aircraft,
        train_dataset[1].name: train_task_transform_flower,
        train_dataset[2].name: train_task_transform_dtd,
        train_dataset[3].name: train_task_transform_omni,

        validation_dataset[0].name: test_task_transform_aircraft,
        validation_dataset[1].name: test_task_transform_flower,
        validation_dataset[2].name: test_task_transform_dtd,
        validation_dataset[3].name: test_task_transform_omni,

        test_dataset[0].name: test_task_transform_aircraft,
        test_dataset[1].name: test_task_transform_flower,
        test_dataset[2].name: test_task_transform_dtd,
        test_dataset[3].name: test_task_transform_omni
    }

    train_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(train_dataset,
                                                                                    dict_cons_remaining_task_transforms)
    validation_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(validation_dataset,
                                                                                         dict_cons_remaining_task_transforms)
    test_transforms: TaskTransform = DifferentTaskTransformIndexableForEachDataset(test_dataset,
                                                                                   dict_cons_remaining_task_transforms)
    return train_transforms, validation_transforms, test_transforms


def hdb7_l2l_tasksets(
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
    root = os.path.expanduser(root)
    # - get data sets lists
    dataset_list_train, dataset_list_validation, dataset_list_test = get_hdb7_list_data_set_splits(root, data_augmentation)

    # - get indexable datasets
    from diversity_src.dataloaders.common import IndexableDataSet
    train_dataset = IndexableDataSet(dataset_list_train)
    validation_dataset = IndexableDataSet(dataset_list_validation)
    test_dataset = IndexableDataSet(dataset_list_test)
    _datasets = (train_dataset, validation_dataset, test_dataset)

    # - get task transforms
    _transforms: tuple[TaskTransform, TaskTransform, TaskTransform] = get_task_transforms_hdb7(_datasets,
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


# - test
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
    #benchmark: BenchmarkTasksets = hdb5_vggair_l2l_tasksets()
    for benchmark in [hdb6_l2l_tasksets()]:#delaunay_l2l_tasksets()]:#[dtd_l2l_tasksets(), cu_birds_l2l_tasksets(), fc100_l2l_tasksets()]:
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
