"""
Goal: compute the diversity using task2vec as the distance between tasks.

Idea: diversity should measure how different **two tasks** are. e.g. the idea could be the distance between distributions
e.g. hellinger, total variation, etc. But in real tasks high dimensionality of images makes it impossible to compute the
integral needed to do this properly. One way is to avoid integrals and use metrics on parameters of tasks -- assuming
a closed for is known for them & the parameters of tasks is known.
For real tasks both are impossible (i.e. the generating distribution for a n-way, k-shot task is unknown and even if it
were known, there is no guarantee a closed form exists).
But FIM estimates something like this. At least in the classical MLE case (linear?), we have
    sqrt(n)(theta^MLE(n) - theta^*) -> N(0, FIM^-1) wrt P_theta*, in distribution and n->infinity
so high FIM approximately means your current params are good.
But I think the authors for task2vec are thinking of the specific FIM (big matrix) as the specific signature for the
task. But when is this matrix (or a function of it) a good representation (embedding) for the task?
Or when is it comparable with other values for the FIM? Well, my guess is that if they are the same arch then the values
between different FIM become (more) comparable.
If the weights are the same twice, then the FIM will always be the same. So I think the authors just meant same network
as same architecture. Otherwise, if we also restrict the weights the FIM would always be the same.
So given two FIM -- if it was in 1D -- then perhaps you'd choose the network with the higher FI(M).
If it's higher than 1D, then you need to consider (all or a function of) the FIM.
The distance between FIM (or something like that seems good) given same arch but different weights.
They fix final layer but I'm not sure if that is really needed.
Make sure to have lot's of data (for us a high support set, what we use to fine tune the model to get FIM for the task).
We use the entire FIM since we can't just say "the FIM is large here" since it's multidimensional.
Is it something like this:
    - a fixed task with a fixed network that predicts well on it has a specific shape/signature of the FIM
    (especially for something really close, e.g. the perfect one would be zero vector)
    - if you have two models, then the more you change the weights the more the FIM changes due to the weights being different, instead of because of the task
    - minimize the source of the change due to the weights but maximize it being due to the **task**
    - so change the fewest amount of weights possible?


Paper says embedding of task wrt fixed network & data set corresponding to a task as:
    task2vec(f, D)
    - fixed network and fixed feature extractor weights
    - train final layer wrt to current task D
    - compute FIM = FIM(f, D)
    - diag = diag(FIM)  # ignore different filter correlations
    - task_emb = aggregate_same_filter_fim_values(FIM)  # average weights in the same filter (since they are usually depedent)

div(f, B) = E_{tau1, tau2} E_{spt_tau1, spt_tau2}[d(task2vec(f, spt_tau1),task2vec(f, spt_tau2)) ]

for div operator on set of distances btw tasks use:
    - expectation
    - symmetric R^2 (later)
    - NED (later)

----
l2l comments:
    args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)
    task_dataset: TaskDataset = task_dataset,  # eg args.tasksets.train

    BenchmarkTasksets = contains the 3 splits which have the tasks for each split.
    e.g. the train split is its own set of "tasks_dataset.train = {task_i}_i"

"""
# ----START mds imports-----#
# import torch
from pytorch_meta_dataset_old.pytorch_meta_dataset.utils import Split
import pytorch_meta_dataset_old.pytorch_meta_dataset.config as config_lib
import pytorch_meta_dataset_old.pytorch_meta_dataset.dataset_spec as dataset_spec_lib
from torch.utils.data import DataLoader
import os
from uutils.plot import save_to
import argparse
import torch.backends.cudnn as cudnn
import random
from datetime import datetime
# import numpy as np
import pytorch_meta_dataset_old.pytorch_meta_dataset.pipeline as pipeline
from pytorch_meta_dataset_old.pytorch_meta_dataset.utils import worker_init_fn_
from functools import partial
# ----END mds imports-----#

from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from uutils.logging_uu.wandb_logging.common import setup_wandb

import learn2learn
import numpy as np
import torch.utils.data
from learn2learn.vision.benchmarks import BenchmarkTasksets
from torch import nn, Tensor
from torch.utils.data import Dataset
from uutils import load_cluster_jobids_to, merge_args
from uutils.torch_uu.distributed import set_devices

import task2vec
import task_similarity
from dataset import TaskDataset
from models import get_model
from task2vec import Embedding, Task2Vec, ProbeNetwork
from uutils.argparse_uu.common import create_default_log_root
import torch


def get_mds_args() -> Namespace:
    import argparse

    # - great terminal argument parser
    parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(description='Records conversion')

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

    # parser.add_argument('--batch_size', type=int, default=16)

    # parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Whether or not to shuffle data')

    # parser.add_argument('--seed', type=int, default=2020,
    #                    help='Seed for reproducibility')

    # Episode configuration
    parser.add_argument('--num_ways', type=int, default=5,
                        help='Set it if you want a fixed # of ways per task')

    parser.add_argument('--num_support', type=int, default=5,
                        help='Set it if you want a fixed # of support samples per class')

    parser.add_argument('--num_query', type=int, default=15,
                        help='Set it if you want a fixed # of query samples per class')

    parser.add_argument('--min_ways', type=int, default=2,
                        help='Minimum # of ways per task')

    parser.add_argument('--max_ways_upper_bound', type=int, default=10,
                        help='Maximum # of ways per task')

    parser.add_argument('--max_num_query', type=int, default=10,
                        help='Maximum # of query samples')

    parser.add_argument('--max_support_set_size', type=int, default=500,
                        help='Maximum # of support samples')

    parser.add_argument('--min_examples_in_class', type=int, default=20,  # TODO - changed
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

    # ========maml config below========#

    # -- create argument options
    parser.add_argument('--debug', action='store_true', help='if debug')
    # parser.add_argument('--serial', action='store_true', help='if running serially')
    parser.add_argument('--parallel', action='store_true', help='if running in parallel')

    # - path to log_root
    parser.add_argument('--log_root', type=str, default=Path('~/data/logs/').expanduser())

    # - training options
    parser.add_argument('--training_mode', type=str, default='epochs_train_convergence',
                        help='valid/possible values: '
                             'fit_single_batch'
                             'iterations'
                             'epochs'
                             'iterations_train_convergence'
                             'epochs_train_convergence'
                             '- Note: since the code checkpoints the best validation model anyway, it is already doing'
                             'early stopping, so early stopping criterion is not implemented. You can kill the job'
                             'if you see from the logs in wanbd that you are done.'
                        )
    parser.add_argument('--num_epochs', type=int, default=-1)
    parser.add_argument('--num_its', type=int, default=-1)
    # parser.add_argument('--no_validation', action='store_true', help='no validation is performed')
    parser.add_argument('--train_convergence_patience', type=int, default=5, help='How long to wait for converge of'
                                                                                  'training. Note this code should'
                                                                                  'be saving the validation ckpt'
                                                                                  'so you are automatically doing '
                                                                                  'early stopping already.')
    # model & loss function options
    parser.add_argument('--model_option',
                        type=str,
                        default="5CNN_opt_as_model_for_few_shot_sl",
                        help="Options: e.g."
                             "5CNN_opt_as_model_for_few_shot_sl"
                             "resnet12_rfs"
                        )
    parser.add_argument('--loss', type=str, help='loss/criterion', default=nn.CrossEntropyLoss())

    # optimization
    parser.add_argument('--opt_option', type=str, default='AdafactorDefaultFair')
    parser.add_argument('--lr', type=float, default=None, help='Warning: use a learning rate according to'
                                                               'how previous work trains your model.'
                                                               'Otherwise, tuning might be needed.'
                                                               'Vision resnets usually use 1e-3'
                                                               'and transformers have a smaller'
                                                               'learning 1e-4 or 1e-5.'
                                                               'It might be a good start to have the'
                                                               'Adafactor optimizer with lr=None and'
                                                               'a its defualt scheduler called'
                                                               'every epoch or every '
                                                               '1(1-beta_2)^-1=2000 iterations.'
                                                               'Doing a hp search with with wanbd'
                                                               'a good idea.')
    parser.add_argument('--grad_clip_mode', type=str, default=None)
    parser.add_argument('--num_warmup_steps', type=int, default=-1)
    parser.add_argument('--scheduler_option', type=str, default='AdafactorSchedule', help='Its strongly recommended')
    parser.add_argument('--log_scheduler_freq', type=int, default=1, help='default is to put the epochs or iterations '
                                                                          'default either log every epoch or log ever '
                                                                          '~100 iterations.')

    # - data set args
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--batch_size_eval', type=int, default=2)
    parser.add_argument('--split', type=str, default='train', help="possible values: "
                                                                   "'train', val', test'")
    # warning: sl name is path_to_data_set here its data_path
    parser.add_argument('--data_option', type=str, default='None')
    # parser.add_argument('--data_path', type=str, default=None)
    # parser.add_argument('--data_path', type=str, default='torchmeta_miniimagenet',
    #                     help='path to data set splits. The code will assume everything is saved in'
    #                          'the uutils standard place in ~/data/, ~/data/logs, etc. see the setup args'
    #                          'setup method and log_root.')
    # parser.add_argument('--path_to_data_set', type=str, default='None')
    parser.add_argument('--data_augmentation', type=str, default=None)
    parser.add_argument('--not_augment_train', action='store_false', default=True)
    parser.add_argument('--augment_val', action='store_true', default=True)
    parser.add_argument('--augment_test', action='store_true', default=False)
    # parser.add_argument('--l2', type=float, default=0.0)

    # - checkpoint options
    parser.add_argument('--path_to_checkpoint', type=str, default=None, help='the path to the model checkpoint to '
                                                                             'resume training.'
                                                                             'e.g. path: '
                                                                             '~/data_folder_fall2020_spring2021/logs/nov_all_mini_imagenet_expts/logs_Nov05_15-44-03_jobid_668/ckpt.pt')
    parser.add_argument('--ckpt_freq', type=int, default=-1)

    # - dist/distributed options
    parser.add_argument('--init_method', type=str, default=None)

    # - miscellaneous arguments
    parser.add_argument('--log_freq', type=int, default=1, help='default is to put the epochs or iterations default'
                                                                'either log every epoch or log ever ~100 iterations')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--always_use_deterministic_algorithms', action='store_true',
                        help='tries to make pytorch fully deterministic')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='the number of data_lib-loading threads (when running serially')
    parser.add_argument('--pin_memory', action='store_true', default=False, help="Using pinning is an"
                                                                                 "advanced tip according to"
                                                                                 "pytorch docs, so will "
                                                                                 "leave it False as default"
                                                                                 "use it at your own risk"
                                                                                 "of further debugging and"
                                                                                 "spending time on none"
                                                                                 "essential, likely over"
                                                                                 "optimizing. See:"
                                                                                 "https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning")
    parser.add_argument('--log_to_tb', action='store_true', help='store to weights and biases')

    # - wandb
    parser.add_argument('--log_to_wandb', action='store_true', help='store to weights and biases')
    parser.add_argument('--wandb_project', type=str, default='Meta-Dataset')
    parser.add_argument('--wandb_entity', type=str, default='brando-uiuc')
    # parser.add_argument('--wandb_project', type=str, default='test-project')
    # parser.add_argument('--wandb_entity', type=str, default='brando-uiuc')
    parser.add_argument('--wandb_group', type=str, default='Task2Vec diversity', help='helps grouping experiment runs')
    # parser.add_argument('--wandb_log_freq', type=int, default=10)
    # parser.add_argument('--wandb_ckpt_freq', type=int, default=100)
    # parser.add_argument('--wanbd_mdl_watch_log_freq', type=int, default=-1)

    # - manual loads
    parser.add_argument('--manual_loads_name', type=str, default='None')

    # - meta-learner specific
    parser.add_argument('--k_shots', type=int, default=5, help="")
    parser.add_argument('--k_eval', type=int, default=15, help="")
    parser.add_argument('--n_cls', type=int, default=5, help="")  # n_ways
    parser.add_argument('--n_aug_support_samples', type=int, default=1,
                        help="The puzzling rfs increase in support examples")

    # - parse arguments
    args = parser.parse_args()
    args.criterion = args.loss
    assert args.criterion is args.loss
    # - load cluster ids so that wandb can use it later for naming runs, experiments, etc.
    load_cluster_jobids_to(args)  # UNCOMMENT LATER
    create_default_log_root(args)
    return args


def get_mds_loader(args):
    data_config = config_lib.DataConfig(args)
    episod_config = config_lib.EpisodeDescriptionConfig(args)

    # Get the data specifications
    datasets = data_config.sources
    # use_dag_ontology_list = [False] * len(datasets)
    # use_bilevel_ontology_list = [False] * len(datasets)
    seeded_worker_fn = partial(worker_init_fn_, seed=args.seed)

    all_dataset_specs = []
    for dataset_name in datasets:
        dataset_records_path = os.path.join(data_config.path, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        all_dataset_specs.append(dataset_spec)

    use_dag_ontology_list = [False] * len(datasets)
    use_bilevel_ontology_list = [False] * len(datasets)
    episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list
    episod_config.use_dag_ontology_list = use_dag_ontology_list

    # create the dataloaders, this goes first so you can select the mdl (e.g. final layer) based on task
    train_pipeline = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                                    split=Split['TRAIN'],
                                                    data_config=data_config,
                                                    episode_descr_config=episod_config)

    # print("NUM workers: ", data_config.num_workers)
    train_loader = DataLoader(dataset=train_pipeline,
                              batch_size=args.batch_size,  # TODO change to meta batch size
                              num_workers=data_config.num_workers,
                              worker_init_fn=seeded_worker_fn)

    test_pipeline = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                                   split=Split['TEST'],
                                                   data_config=data_config,
                                                   episode_descr_config=episod_config)
    test_loader = DataLoader(dataset=test_pipeline,
                             batch_size=args.batch_size_eval,
                             num_workers=data_config.num_workers,
                             worker_init_fn=seeded_worker_fn)

    val_pipeline = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                                  split=Split['VALID'],
                                                  data_config=data_config,
                                                  episode_descr_config=episod_config)

    val_loader = DataLoader(dataset=val_pipeline,
                            batch_size=args.batch_size_eval,
                            num_workers=data_config.num_workers,
                            worker_init_fn=seeded_worker_fn)

    dls: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dls


if __name__ == '__main__':
    """
python ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/metadataset/metadataset_task2vec_div.py --data_path /shared/rsaas/pzy2/records
    """
    plot_distance_matrix_and_div_for_MI_test()
    print('Done! successful!\n')
