"""
Union of data sets for SL training.

note:
- this is what meta data set does: "The non-episodic baselines are trained to solve the
large classification problem that results from ‘concatenating’ the training classes of all datasets." https://arxiv.org/abs/1903.03096
"""

# - tests
import random
from argparse import Namespace

import numpy as np
import torch
from torch import nn

from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import get_mi_and_omniglot_list_data_set_splits


def hdb1_mi_omniglot_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb1',
        device=None,
) -> dict:
    dataset_list_train, dataset_list_validation, dataset_list_test = get_mi_and_omniglot_list_data_set_splits(root,
                                                                                                              data_augmentation,
                                                                                                              device)
    from learn2learn.data import UnionMetaDataset
    train_dataset = UnionMetaDataset(dataset_list_train)
    valid_dataset = UnionMetaDataset(dataset_list_validation)
    test_dataset = UnionMetaDataset(dataset_list_test)
    assert len(train_dataset.labels) == 64 + 1100, f'mi + omnigloat should be number of labels 1164.'
    assert len(valid_dataset.labels) == 16 + 100, f'mi + omnigloat should be number of labels 116.'
    assert len(test_dataset.labels) == 20 + 423, f'mi + omnigloat should be number of labels 443.'

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

    # args.n_cls = 1164
    next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders


def loop_through_usl_hdb1_and_pass_data_through_mdl():
    # - for determinism
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # - args
    args = Namespace(batch_size=8, batch_size_eval=2, rank=-1, world_size=1)

    # - get data loaders
    dataloaders: dict = hdb1_mi_omniglot_usl_all_splits_dataloaders(args)

    # - loop through tasks
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    from models import get_model
    model = get_model('resnet18', pretrained=False, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    for split, dataloader in dataloaders.items():
        print(f'-- {split=}')
        next(iter(dataloaders[split]))
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
