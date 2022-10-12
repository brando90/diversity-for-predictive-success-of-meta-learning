import torch
from pytorch_meta_dataset_old.pytorch_meta_dataset.utils import Split
import pytorch_meta_dataset_old.pytorch_meta_dataset.config as config_lib
import pytorch_meta_dataset_old.pytorch_meta_dataset.dataset_spec as dataset_spec_lib
from torch.utils.data import DataLoader
import os
import argparse
import torch.backends.cudnn as cudnn
import random
import numpy as np
import pytorch_meta_dataset_old.pytorch_meta_dataset.pipeline as pipeline
from pytorch_meta_dataset_old.pytorch_meta_dataset.utils import worker_init_fn_
from functools import partial


import learn2learn as l2l
from learn2learn.data import MetaDataset, FilteredMetaDataset
from learn2learn.data.transforms import TaskTransform
from learn2learn.vision.benchmarks import BenchmarkTasksets
from torch import nn
from torchvision.transforms import Compose, Normalize, ToPILImage, RandomCrop, ColorJitter, RandomHorizontalFlip, \
    ToTensor

from diversity_src.dataloaders.common import IndexableDataSet, ToRGB, DifferentTaskTransformIndexableForEachDataset

from torchvision import transforms
from PIL.Image import LANCZOS

from models import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Records conversion')

    # Data general config
    parser.add_argument('--data_path', type=str, required=True,
                        help='Root to data')

    parser.add_argument('--image_size', type=int, default=84,
                        help='Images will be resized to this value')
    #TODO: Make sure that images are sampled randomly from different sources!!!
    parser.add_argument('--sources', nargs="+", default=['ilsvrc_2012','aircraft','cu_birds','dtd','fungi','omniglot',
                                                         'quickdraw','vgg_flower'], #Mscoco, traffic_sign are VAL only
                        help='List of datasets to use')

    parser.add_argument('--train_transforms', nargs="+", default=['random_resized_crop', 'random_flip'],
                        help='Transforms applied to training data',)

    parser.add_argument('--test_transforms', nargs="+", default=['resize', 'center_crop'],
                        help='Transforms applied to test data',)

    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Whether or not to shuffle data')

    parser.add_argument('--seed', type=int, default=2020,
                        help='Seed for reproducibility')

    # Episode configuration
    parser.add_argument('--num_ways', type=int, default=None,
                        help='Set it if you want a fixed # of ways per task')

    parser.add_argument('--num_support', type=int, default=None,
                        help='Set it if you want a fixed # of support samples per class')

    parser.add_argument('--num_query', type=int, default=None,
                        help='Set it if you want a fixed # of query samples per class')

    parser.add_argument('--min_ways', type=int, default=2,
                        help='Minimum # of ways per task')

    parser.add_argument('--max_ways_upper_bound', type=int, default=10,
                        help='Maximum # of ways per task')

    parser.add_argument('--max_num_query', type=int, default=10,
                        help='Maximum # of query samples')

    parser.add_argument('--max_support_set_size', type=int, default=500,
                        help='Maximum # of support samples')

    parser.add_argument('--min_examples_in_class', type=int, default=0,
                        help='Classes that have less samples will be skipped')

    parser.add_argument('--max_support_size_contrib_per_class', type=int, default=100,
                        help='Maximum # of support samples per class')

    parser.add_argument('--min_log_weight', type=float, default=-0.69314718055994529,
                        help='Do not touch, used to randomly sample support set')

    parser.add_argument('--max_log_weight', type=float, default=0.69314718055994529,
                        help='Do not touch, used to randomly sample support set')

    # Hierarchy options
    parser.add_argument('--ignore_bilevel_ontology', type=bool, default=False,
                        help='Whether or not to use superclass for BiLevel datasets (e.g Omniglot)')

    parser.add_argument('--ignore_dag_ontology', type=bool, default=False,
                        help='Whether to ignore ImageNet DAG ontology when sampling \
                              classes from it. This has no effect if ImageNet is not  \
                              part of the benchmark.')

    parser.add_argument('--ignore_hierarchy_probability', type=float, default=0.,
                        help='if using a hierarchy, this flag makes the sampler \
                              ignore the hierarchy for this proportion of episodes \
                              and instead sample categories uniformly.')
    args = parser.parse_args()
    return args


def metadataset_tasksets(args):
    data_config = config_lib.DataConfig(args)
    episod_config = config_lib.EpisodeDescriptionConfig(args)
    datasets = data_config.sources
    use_dag_ontology_list = [False] * len(datasets)
    use_bilevel_ontology_list = [False] * len(datasets)
    episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list
    episod_config.use_dag_ontology_list = use_dag_ontology_list

    all_dataset_specs = []
    for dataset_name in datasets:
        dataset_records_path = os.path.join(data_config.path, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        all_dataset_specs.append(dataset_spec)

    # Form an episodic dataset
    torch_train = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                                      split=Split["TRAIN"],
                                                      data_config=data_config,
                                                      episode_descr_config=episod_config)

    torch_test = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                                 split=Split["TEST"],
                                                 data_config=data_config,
                                                 episode_descr_config=episod_config)

    torch_validation = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                                 split=Split["VALID"],
                                                 data_config=data_config,
                                                 episode_descr_config=episod_config)

    
    '''
    train_dataset = l2l.data.MetaDataset(torch_train)
    test_dataset = l2l.data.MetaDataset(torch_test)
    validation_dataset = l2l.data.MetaDataset(torch_validation)
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
    )'''
    return BenchmarkTasksets(train_tasks, validation_tasks, test_tasks)


# - test

def loop_through_l2l_indexable_benchmark_with_model_test(args):
    # - for determinism
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # - options for number of tasks/meta-batch size
    batch_size = 5

    # - get benchmark
    benchmark: BenchmarkTasksets = metadataset_tasksets(args)

    # - get train taskdata set
    splits = ['train', 'validation', 'test']
    tasksets = [getattr(benchmark, split) for split in splits]

    # - loop through tasks
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model = get_model('resnet18', pretrained=False, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    for i, taskset in enumerate(tasksets):
        print(f'-- {splits[i]=}')
        for task_num in range(batch_size):
            print(f'{task_num=}')

            X, y = taskset.sample()
            X,y = X.cuda(),y.cuda()
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
    args = parse_args()

    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    loop_through_l2l_indexable_benchmark_with_model_test(args)
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
