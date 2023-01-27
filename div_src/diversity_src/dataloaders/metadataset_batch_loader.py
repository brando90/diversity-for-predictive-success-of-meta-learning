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
#import uutils
#from uutils import load_cluster_jobids_to, merge_args
#from uutils.logging_uu.wandb_logging.common import setup_wandb
#from uutils.torch_uu.distributed import set_devices

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



# - get number of images in our split for mds dataloader (for both USL and MAML)
def get_num_images(args, split: str = 'VALID'):
    # first we want to get the sources to figure out which datasets we use
    data_config = config_lib.DataConfig(args)
    datasets = data_config.sources
    num_images = 0

    for dataset_name in datasets:
        dataset_records_path = os.path.join(data_config.path, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)

        all_class_sizes = dataset_spec.images_per_class

        # let's get only the class sizes of our split
        class_set = dataset_spec.get_classes(Split[split])
        #print(class_set)

        for c in class_set:
            # ignore classes that have less images than needed for a n-way k-shot task
            if (all_class_sizes[c] >= args.min_examples_in_class):
                num_images += all_class_sizes[c]

    return num_images



# - get number of images in our split for mds dataloader (for both USL and MAML)
def get_num_classes(args, split: str = 'VALID'):
    # first we want to get the sources to figure out which datasets we use
    data_config = config_lib.DataConfig(args)
    datasets = data_config.sources
    num_classes = 0

    for dataset_name in datasets:
        dataset_records_path = os.path.join(data_config.path, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)

        all_class_sizes = dataset_spec.images_per_class

        # let's get only the class sizes of our split
        class_set = dataset_spec.get_classes(Split[split])
        #print(class_set)

        for c in class_set:
            # ignore classes that have less images than n-way k-shot teas
            if (all_class_sizes[c] >= args.min_examples_in_class):
                num_classes += 1

    return num_classes

# - test
def loop_test(args):
    #from uutils.torch_uu import process_meta_batch
    args.batch_size = 10
    args.batch_size_eval= 2
    args.min_examples_per_class = 20
    #args.data_path = '/shared/rsaas/pzy2/records/'  # or whereever
    args.sources = ['dtd','cu_birds']

    dataloader = get_mds_loader(args)

    print(get_num_images(args, 'TRAIN'))
    print(get_num_images(args, 'VALID'))
    print(get_num_images(args, 'TEST'))

    print(get_num_classes(args, 'TRAIN'))
    print(get_num_classes(args, 'VALID'))
    print(get_num_classes(args, 'TEST'))


    print(f'{len(dataloader)}')
    for batch_idx, batch in enumerate(dataloader['train']):
        X,y = batch
        print(X.shape)
        print(y.shape)
        print(y)
        print("min label:", min(y))
        print("max label:", max(y))
        #print(X, y)
        if batch_idx == 2:
            break


# -- Run experiment

if __name__ == "__main__":
    from uutils.argparse_uu.supervised_learning import parse_args_standard_sl

    args: Namespace = parse_args_standard_sl()

    #args.sources = ['vgg_flower','aircraft']
    #set_devices(args)  # args.device = rank or .device
    #args.device = uutils.torch_uu.get_device()

    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    loop_test(args)
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
