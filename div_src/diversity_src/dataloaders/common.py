from random import random
from typing import Callable

from learn2learn.data import DataDescription, MetaDataset
from learn2learn.data.transforms import TaskTransform
from torch import nn
from torch.utils.data import Dataset


class ToRGB(nn.Module):

    def __init___(self):
        pass

    def __call__(self, x, *args, **kwargs):
        return x.conver("RGB")

class IndexableDataSet(Dataset):

    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self) -> int:
        return len(self.datasets)

    def __getitem__(self, idx: int):
        return self.datasets[idx]

def get_remaining_transforms(dataset: MetaDataset) -> list[TaskTransform]:
    import learn2learn as l2l
    remaining_task_transforms = [
        l2l.data.transforms.NWays(dataset, n=5),
        l2l.data.transforms.KShots(dataset, k=5),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
    ]
    return remaining_task_transforms

class SingleDatasetPerTaskTransform(Callable):
    """
    Transform that samples a data set first, then creates a task (e.g. n-way, k-shot) and finally
    applies the remaining task transforms.
    """

    def __init__(self, indexable_dataset: IndexableDataSet, cons_remaining_task_transforms: Callable):
        """

        :param: cons_remaining_task_transforms; constructor that builds the remaining task transforms. Cannot be a list
        of transforms because we don't know apriori which is the data set we will use. So this function should be of
        type MetaDataset -> list[TaskTransforms] i.e. given the dataset it returns the transforms for it.
        """
        self.indexable_dataset = MetaDataset(indexable_dataset)
        self.cons_remaining_task_transforms = cons_remaining_task_transforms

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

        # - use the sampled data set to create task
        remaining_task_transforms: list[TaskTransform] = self.cons_remaining_task_transforms(dataset)
        description = None
        for transform in remaining_task_transforms:
            description = transform(description)
        return description