#TODO - add ability to config list of datasets to use for both train, test, and val splits SEPERATELY
#----START mds imports-----#
#import torch
from diversity_src.dataloaders.pytorch_mds_lib.pytorch_meta_dataset.utils import Split
import diversity_src.dataloaders.pytorch_mds_lib.pytorch_meta_dataset.config as config_lib
import diversity_src.dataloaders.pytorch_mds_lib.pytorch_meta_dataset.dataset_spec as dataset_spec_lib
from torch.utils.data import DataLoader
import os
import argparse
import torch.backends.cudnn as cudnn
import random
import numpy as np
import diversity_src.dataloaders.pytorch_mds_lib.pytorch_meta_dataset.pipeline as pipeline
from diversity_src.dataloaders.pytorch_mds_lib.pytorch_meta_dataset.utils import worker_init_fn_
from functools import partial
#----END mds imports-----#
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from pathlib import Path

from argparse import Namespace
import uutils
from uutils import load_cluster_jobids_to, merge_args
from uutils.logging_uu.wandb_logging.common import setup_wandb
from uutils.torch_uu.distributed import set_devices

def get_mds_loader(args):
    data_config = config_lib.DataConfig(args)
    episod_config = config_lib.EpisodeDescriptionConfig(args)

    # Get the data specifications
    datasets = data_config.sources
    # use_dag_ontology_list = [False] * len(datasets)
    # use_bilevel_ontology_list = [False] * len(datasets)
    seeded_worker_fn = partial(worker_init_fn_, seed=args.seed)

    all_dataset_specs_train = []
    all_dataset_specs_test = []
    all_dataset_specs_val = []
    for dataset_name in datasets:
        dataset_records_path = os.path.join(data_config.path, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)

        if (dataset_name == 'traffic_sign'):
            # traffic sign is test only
            all_dataset_specs_test.append(dataset_spec)
        elif (dataset_name == 'mscoco'):
            # mscoco is val and test only
            all_dataset_specs_val.append(dataset_spec)
            all_dataset_specs_test.append(dataset_spec)
        else:
            # all other datasets have all three splits
            all_dataset_specs_train.append(dataset_spec)
            all_dataset_specs_test.append(dataset_spec)
            all_dataset_specs_val.append(dataset_spec)

    use_dag_ontology_list = [False] * len(datasets)
    use_bilevel_ontology_list = [False] * len(datasets)
    episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list
    episod_config.use_dag_ontology_list = use_dag_ontology_list

    # create the dataloaders, this goes first so you can select the mdl (e.g. final layer) based on task
    train_pipeline = pipeline.make_batch_pipeline(dataset_spec_list=all_dataset_specs_train,
                                                 split=Split['TRAIN'],
                                                 data_config=data_config,
                                                 )

    train_loader = DataLoader(dataset=train_pipeline,
                              batch_size=args.batch_size,
                              num_workers=0,
                              worker_init_fn=seeded_worker_fn)

    test_pipeline = pipeline.make_batch_pipeline(dataset_spec_list=all_dataset_specs_test,
                                                  split=Split['TEST'],
                                                  data_config=data_config,
                                                  )

    test_loader = DataLoader(dataset=test_pipeline,
                              batch_size=args.batch_size_eval,
                              num_workers=0,
                              worker_init_fn=seeded_worker_fn)

    val_pipeline = pipeline.make_batch_pipeline(dataset_spec_list=all_dataset_specs_val,
                                                  split=Split['VALID'],
                                                  data_config=data_config,
                                                  )

    val_loader = DataLoader(dataset=val_pipeline,
                              batch_size=args.batch_size_eval,
                              num_workers=0,
                              worker_init_fn=seeded_worker_fn)


    dls: dict ={'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dls

# - test
def loop_test(args):
    from uutils.torch_uu import process_meta_batch
    args.batch_size = 10
    args.batch_size_eval= 10
    args.data_path = '/shared/rsaas/pzy2/records/'  # or whereever

    dataloader = get_mds_loader(args)

    print(f'{len(dataloader)}')
    for batch_idx, batch in enumerate(dataloader['val']):
        X,y = batch
        #print("finish batch",batch_idx)
        print(X.shape)
        print(y.shape)
        print(y, "max label:", max(y))
        #print(X, y)
        if batch_idx == 1000:
            break


# -- Run experiment

if __name__ == "__main__":
    from diversity_src.dataloaders.metadataset_common import get_mds_base_args

    args: Namespace = get_mds_base_args()
    #args.sources = ['vgg_flower','aircraft']
    #set_devices(args)  # args.device = rank or .device
    args.device = uutils.torch_uu.get_device()

    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    loop_test(args)
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
