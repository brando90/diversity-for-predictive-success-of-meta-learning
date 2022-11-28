import torch
from pytorch_meta_dataset.utils import Split
import pytorch_meta_dataset.config as config_lib
import pytorch_meta_dataset.dataset_spec as dataset_spec_lib
from torch.utils.data import DataLoader
import os
import argparse
import torch.backends.cudnn as cudnn
import random
import numpy as np
import pytorch_meta_dataset.pipeline as pipeline
from pytorch_meta_dataset.utils import worker_init_fn_
from functools import partial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Records conversion')

    # Data general config
    parser.add_argument('--data_path', type=str, required=True,
                        help='Root to data')

    parser.add_argument('--image_size', type=int, default=84,
                        help='Images will be resized to this value')
    # TODO: Make sure that images are sampled randomly from different sources!!!
    parser.add_argument('--sources', nargs="+",
                        default=['ilsvrc_2012', 'aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot',
                                 'quickdraw', 'vgg_flower'],  # Mscoco, traffic_sign are VAL only
                        help='List of datasets to use')

    parser.add_argument('--train_transforms', nargs="+", default=['random_resized_crop', 'random_flip'],
                        help='Transforms applied to training data', )

    parser.add_argument('--test_transforms', nargs="+", default=['resize', 'center_crop'],
                        help='Transforms applied to test data', )

    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--num_workers', type=int, default=0)

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

    parser.add_argument('--min_examples_in_class', type=int, default=0,  # TODO - changed
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


def main(args: argparse.Namespace) -> None:

    # Define your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        np.random.seed(args.seed)

    # Recovering configurations
    data_config = config_lib.DataConfig(args)
    episod_config = config_lib.EpisodeDescriptionConfig(args)

    # Get the data specifications
    datasets = data_config.sources
    use_dag_ontology_list = [False]*len(datasets)
    use_bilevel_ontology_list = [False]*len(datasets)
    if episod_config.num_ways:
        pass
        #if len(datasets) > 1:
        #    raise ValueError('For fixed episodes, not tested yet on > 1 dataset')
    else:
        # Enable ontology aware sampling for Omniglot and ImageNet.
        if 'omniglot' in datasets:
            use_bilevel_ontology_list[datasets.index('omniglot')] = True
        if 'ilsvrc_2012' in datasets:
            use_dag_ontology_list[datasets.index('ilsvrc_2012')] = True

    episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list
    episod_config.use_dag_ontology_list = use_dag_ontology_list

    all_dataset_specs = []
    for dataset_name in datasets:
        dataset_records_path = os.path.join(data_config.path, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        all_dataset_specs.append(dataset_spec)

    # Form an episodic dataset
    split = Split["TRAIN"]
    episodic_dataset = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                                      split=split,
                                                      data_config=data_config,
                                                      episode_descr_config=episod_config)

    #  If you want to get the total number of classes (i.e from combined datasets)
    num_classes = sum([len(d_spec.get_classes(split=split)) for d_spec in all_dataset_specs])
    print(f"=> There are {num_classes} classes in the combined datasets")

    # Use a standard dataloader
    seeded_worker_fn = partial(worker_init_fn_, seed=args.seed)
    
    episodic_loader = DataLoader(dataset=episodic_dataset,
                                 batch_size=1,
                                 num_workers=data_config.num_workers,
                                 worker_init_fn=seeded_worker_fn)

    # Training or validation loop
    for i, (support, support_labels, query,query_labels) in enumerate(episodic_loader):
        # support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
        # query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)
        # Do some operations
        print("=> Example of episode")
        #print(support_labels)
        print("Number of ways: {}   Support size: {}   Query Size: {} \n".format(
                            support_labels.unique().size(0),
                            list(support.size()),
                            list(query.size())))
        print(f"support labels:", query_labels)
        #print(f"Shape of support labels: {list(support_labels.size())}")
        #print(f"Shape of query labels: {list(query.size())}")

        if i == 5000:
            break

    # Form a batch dataset
    batch_dataset = pipeline.make_batch_pipeline(dataset_spec_list=all_dataset_specs,
                                                 split=split,
                                                 data_config=data_config,
                                                 )

    # Use a standard dataloader
    batch_loader = DataLoader(dataset=batch_dataset,
                              batch_size=data_config.batch_size,
                              num_workers=data_config.num_workers,
                              worker_init_fn=seeded_worker_fn)
    # Training or validation loop
    for i, (input, target) in enumerate(batch_loader):
        # input, target = input.to(device), target.long().to(device, non_blocking=True)
        # Do some operations
        print("=> Example of a batch")
        print(f"Shape of batch: {list(input.size())}")
        #print(f"Shape of target: {list(target.size())}")
        print("Target: ", target)
        if i == 5:
            break


if __name__ == '__main__':
    args = parse_args()
    main(args)
