"""
Union of data sets for SL training.

note:
- this is what meta data set does: "The non-episodic baselines are trained to solve the large classification problem that results from ‘concatenating’ the training classes of all datasets." https://arxiv.org/abs/1903.03096
"""

# - tests
import random
from argparse import Namespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import get_mi_and_omniglot_list_data_set_splits
from uutils.torch_uu.dataset.concate_dataset import ConcatDatasetMutuallyExclusiveLabels


def hdb1_mi_omniglot_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb1',
        device=None,
) -> dict:
    """
    
    due to:
        classes = list(range(1623))
        train_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[:1100])
        validation_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1100:1200])
        test_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1200:])
    the assert I wrote follow. 
    train: 1100
    val: 100
    test: 423 ( = 1623 - 1100 - 100)
    """
    dataset_list_train, dataset_list_validation, dataset_list_test = get_mi_and_omniglot_list_data_set_splits(root,
                                                                                                              data_augmentation,
                                                                                                              device)
    assert get_len_labels_list_datasets(dataset_list_train) == 64 + 1100
    assert get_len_labels_list_datasets(dataset_list_validation) == 16 + 100
    assert get_len_labels_list_datasets(dataset_list_test) == 20 + 423
    # -
    train_dataset: Dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train)
    valid_dataset: Dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation)
    test_dataset: Dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test)
    assert len(train_dataset.labels) == 64 + 1100, f'Err:\n{len(train_dataset.labels)=}'
    assert len(valid_dataset.labels) == 16 + 100, f'Err:\n{len(valid_dataset.labels)=}'
    assert len(test_dataset.labels) == 20 + 423, f'Err:\n{len(test_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders


def get_len_labels_list_datasets(datasets: list[Dataset], verbose: bool = False) -> int:
    if verbose:
        print('--- get_len_labels_list_datasets')
        print([len(dataset.labels) for dataset in datasets])
        print([dataset.labels for dataset in datasets])
        print('--- get_len_labels_list_datasets')
    return sum([len(dataset.labels) for dataset in datasets])


# - tests

def loop_through_usl_hdb1_and_pass_data_through_mdl():
    print(f'starting {loop_through_usl_hdb1_and_pass_data_through_mdl=} test')
    # - for determinism
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # - args
    args = Namespace(batch_size=8, batch_size_eval=2, rank=-1, world_size=1)

    # - get data loaders
    dataloaders: dict = hdb1_mi_omniglot_usl_all_splits_dataloaders(args)
    print(dataloaders['train'].dataset.labels)
    print(dataloaders['val'].dataset.labels)
    print(dataloaders['test'].dataset.labels)
    n_train_cls: int = len(dataloaders['train'].dataset.labels)
    print('-- got the usl hdb1 data loaders --')

    # - loop through tasks
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    from models import get_model
    # model = get_model('resnet18', pretrained=False, num_classes=n_train_cls).to(device)
    # model = get_model('resnet18', pretrained=True, num_classes=n_train_cls).to(device)
    from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_mi
    model, _ = get_resnet_rfs_model_mi('resnet12_hdb1_mio', num_classes=n_train_cls)
    criterion = nn.CrossEntropyLoss()
    for split, dataloader in dataloaders.items():
        print(f'-- {split=}')
        # next(iter(dataloaders[split]))
        for it, batch in enumerate(dataloaders[split]):
            print(f'{it=}')

            X, y = batch
            print(f'{X.size()=}')
            print(f'{y.size()=}')
            print(f'{y=}')

            y_pred = model(X)
            loss = criterion(y_pred, y)
            print(f'{loss=}')
            print()
            break
    print('-- end of test --')


if __name__ == '__main__':
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    loop_through_usl_hdb1_and_pass_data_through_mdl()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
