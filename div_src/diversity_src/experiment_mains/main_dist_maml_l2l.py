"""
Main script to set up supervised learning experiments
"""
# import os
# os.environ["NCCL_DEBUG"] = "INFO"

import os

# import learn2learn
import torch
import torch.nn as nn
import torch.multiprocessing as mp
# import torch.optim as optim

from argparse import Namespace

from learn2learn.vision.benchmarks import BenchmarkTasksets

from uutils import args_hardcoded_in_script, report_times
from uutils.argparse_uu.common import setup_args_for_experiment, setup_wandb
from uutils.argparse_uu.meta_learning import parse_args_meta_learning, fix_for_backwards_compatibility
from uutils.argparse_uu.supervised_learning import make_args_from_supervised_learning_checkpoint, parse_args_standard_sl
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloader
from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
from uutils.torch_uu.distributed import set_sharing_strategy, print_process_info, set_devices, setup_process, cleanup, \
    print_dist, move_opt_to_cherry_opt_and_sync_params, set_devices_and_seed_ala_l2l, init_process_group_l2l, \
    is_lead_worker, find_free_port, is_running_serially, is_running_parallel, get_local_rank
from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
from uutils.torch_uu.mains.main_sl_with_ddp import train
from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner, MAMLMetaLearnerL2L
from uutils.torch_uu.training.meta_training import meta_train_fixed_iterations, meta_train_agent_fit_single_batch, \
    meta_train_iterations_ala_l2l

from pdb import set_trace as st

from uutils.torch_uu.training.supervised_learning import train_agent_fit_single_batch


# - mi

def l2l_resnet12rfs_mi_rfs_adam_cl_100k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_mi'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)

    # - data
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # args.scheduler_option = 'None'
    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 2_000
    args.T_max = args.num_its // args.log_scheduler_freq  # intended 800K/2k
    args.eta_min = 1e-5  # match MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)
    # assert args.T_max == 400, f'T_max is not expected value, instead it is: {args.T_max=}'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_resnet12rfs_mi_rfs_adam_cl_100k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_5CNNl2l_mi_rfs_adam_cl_70k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls, filter_size=32,
                          levels=None, spp=False, in_channels=3)

    # - data
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 70_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++

    # args.scheduler_option = 'None'
    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 2_000
    args.T_max = args.num_its // args.log_scheduler_freq  # intended 800K/2k
    args.eta_min = 1e-5  # match MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)
    # assert args.T_max == 400, f'T_max is not expected value, instead it is: {args.T_max=}'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 8
    args.batch_size = 2

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 1
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_5CNNl2l_mi_rfs_adam_cl_70k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_5CNNl2l_mi_rfs_sgd_cl_100k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls, filter_size=32,
                          levels=None, spp=False, in_channels=3)

    # - data
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Sgd_rfs'
    args.lr = 5e-2
    args.opt_hps: dict = dict(lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # args.scheduler_option = 'None'
    args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.lr_decay_rate = 1e-1
    # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 4
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_5CNNl2l_mi_rfs_sgd_cl_100k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_resnet12rfs_mi_rfs_sgd_cl_100k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_mi'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)

    # - data
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Sgd_rfs'
    args.lr = 5e-2
    args.opt_hps: dict = dict(lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # args.scheduler_option = 'None'
    args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.lr_decay_rate = 1e-1
    # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_resnet12rfs_mi_rfs_sgd_cl_100k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


# - cifarfs

def l2l_4CNNl2l_cifarfs_rfs_adam_cl_70k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.model_option = '4CNN_l2l_cifarfs'
    args.model_hps = dict(ways=args.n_cls, hidden_size=64, embedding_size=64 * 4)

    # - data
    args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'rfs2020'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 70_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++

    # args.scheduler_option = 'None'
    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 2_000
    args.T_max = args.num_its // args.log_scheduler_freq  # intended 800K/2k
    args.eta_min = 1e-5  # match MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)
    # assert args.T_max == 400, f'T_max is not expected value, instead it is: {args.T_max=}'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 1
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_4CNNl2l_cifarfs_rfs_adam_cl_70k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_cifarfs_fc100'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=args.n_cls)

    # - data
    args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'rfs2020'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # args.scheduler_option = 'None'
    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 2_000
    args.T_max = args.num_its // args.log_scheduler_freq  # intended 800K/2k
    args.eta_min = 1e-5  # match MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)
    # assert args.T_max == 400, f'T_max is not expected value, instead it is: {args.T_max=}'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_5CNNl2l_cifarfs_rfs_sgd_cl_100k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.model_option = '4CNN_l2l_cifarfs'
    args.model_hps = dict(ways=args.n_cls, hidden_size=64, embedding_size=64 * 4)

    # - data
    args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'rfs2020'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Sgd_rfs'
    args.lr = 5e-2
    args.opt_hps: dict = dict(lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # args.scheduler_option = 'None'
    args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.lr_decay_rate = 1e-1
    # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # - dist args
    args.world_size = torch.cuda.device_count()
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_5CNNl2l_cifarfs_rfs_sgd_cl_100k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_resnet12rfs_cifarfs_rfs_sgd_cl_100k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_cifarfs_fc100'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=args.n_cls)

    # - data
    args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'rfs2020'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Sgd_rfs'
    args.lr = 5e-2
    args.opt_hps: dict = dict(lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # args.scheduler_option = 'None'
    args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.lr_decay_rate = 1e-1
    # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_resnet12rfs_cifarfs_rfs_sgd_cl_100k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_4CNNl2l_1024_cifarfs_rfs_adam_cl_100k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.model_option = '4CNN_l2l_cifarfs'
    args.hidden_size = 1024
    args.model_hps = dict(ways=args.n_cls, hidden_size=args.hidden_size, embedding_size=args.hidden_size * 4)

    # - data
    args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'rfs2020'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++

    # args.scheduler_option = 'None'
    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 2_000
    args.T_max = args.num_its // args.log_scheduler_freq  # intended 800K/2k
    args.eta_min = 1e-5  # match MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)
    # assert args.T_max == 400, f'T_max is not expected value, instead it is: {args.T_max=}'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # - dist args
    args.world_size = torch.cuda.device_count()
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_4CNNl2l_1024_cifarfs_rfs_adam_cl_100k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_resnet12rfs_mi_adam_no_scheduler_100k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_mi'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)

    # - data
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # args.scheduler_option = 'None'
    args.scheduler_option = 'None'
    # args.log_scheduler_freq = 2_000
    # args.T_max = args.num_its // args.log_scheduler_freq  # intended 800K/2k
    # args.eta_min = 1e-5  # match MAML++
    # args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)
    # assert args.T_max == 400, f'T_max is not expected value, instead it is: {args.T_max=}'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_resnet12rfs_mi_rfs_adam_cl_100k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    args.parallel = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_5CNN_mi_adam_filter_size(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.filter_size = 32
    args.filter_size = 1024
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
                          filter_size=args.filter_size,
                          levels=None, spp=False, in_channels=3)

    # - data
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++

    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # parser.add_argument('--k_eval', type=int, default=15, help="")
    args.k_eval = 5

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 1
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_5CNN_mi_adam_filter_size'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_5CNN_mi_adam_filter_size_32_filter_size(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.filter_size = 32
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
                          filter_size=args.filter_size,
                          levels=None, spp=False, in_channels=3)

    # - data
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++

    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # parser.add_argument('--k_eval', type=int, default=15, help="")
    args.k_eval = 5

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 1
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_5CNN_mi_adam_filter_size'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_5CNN_mi_adam_filter_size_16_filter_size(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.filter_size = 16
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
                          filter_size=args.filter_size,
                          levels=None, spp=False, in_channels=3)

    # - data
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++

    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # parser.add_argument('--k_eval', type=int, default=15, help="")
    args.k_eval = 5

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 1
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_5CNN_mi_adam_filter_size'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_5CNN_mi_adam_filter_size_8_filter_size(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.filter_size = 8
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
                          filter_size=args.filter_size,
                          levels=None, spp=False, in_channels=3)

    # - data
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++

    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # parser.add_argument('--k_eval', type=int, default=15, help="")
    args.k_eval = 5

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 1
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_5CNN_mi_adam_filter_size'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_5CNN_mi_adam_filter_size_4_filter_size(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.filter_size = 4
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
                          filter_size=args.filter_size,
                          levels=None, spp=False, in_channels=3)

    # - data
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++

    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # parser.add_argument('--k_eval', type=int, default=15, help="")
    args.k_eval = 5

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 1
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_5CNN_mi_adam_filter_size'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


# - hdb1 = MI + Omni = MIO

def l2l_resnet12rfs_hdb1_100k_adam_no_scheduler(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_mi'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)

    # - data
    args.data_option = 'hdb1'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'hdb1'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    # args.batch_size = 32
    args.batch_size = 8
    args.batch_size = 2

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_resnet12rfs_hdb1_100k_adam_no_scheduler'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_mi'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)

    # - data
    args.data_option = 'hdb1'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'hdb1'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    # args.batch_size = 32
    args.batch_size = 8
    # args.batch_size = 2

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler_first_order(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_mi'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)

    # - data
    args.data_option = 'hdb1'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'hdb1'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    args.first_order = True
    # args.first_order = False

    # - outer trainer params
    # args.batch_size = 32
    args.batch_size = 8
    # args.batch_size = 2

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler_first_order'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.first_order=}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_resnet18task2vec_hdb1_100k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet18_random'
    args.model_hps = None
    # args.model_option = 'resnet12_rfs_mi'
    # args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
    #                       num_classes=args.n_cls)
    # args.n_cls = 5
    # args.filter_size = 64
    # args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    # args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                       filter_size=args.filter_size,
    #                       levels=None, spp=False, in_channels=3)

    # - data
    args.data_option = 'hdb1'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'hdb1'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # args.scheduler_option = 'None'
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    # args.parallel = False
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_resnet12rfs_mi_rfs_adam_cl_100k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_5cnn_hdb1_100k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.filter_size = 64
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
                          filter_size=args.filter_size,
                          levels=None, spp=False, in_channels=3)

    # - data
    args.data_option = 'hdb1'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'hdb1'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # args.scheduler_option = 'None'
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    args.first_order = False

    # - outer trainer params
    args.batch_size = 32
    args.batch_size = 8

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    # args.parallel = False
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_resnet12rfs_mi_rfs_adam_cl_100k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


# vit

def vit_mi_fo_maml_rfs_adam_cl_100k(args: Namespace):
    """
    """
    from pathlib import Path
    # - model
    args.model_option = 'vit_mi'
    args.model_hps = dict(num_classes=5, image_size=84)
    args.allow_unused = True  # transformers have lots of tokens & params, so I assume some param is not always being used in the forward pass

    # - data
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 2_000
    args.T_max = args.num_its // args.log_scheduler_freq  # intended 800K/2k
    args.eta_min = 1e-5  # match MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)
    # assert args.T_max == 400, f'T_max is not expected value, instead it is: {args.T_max=}'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    args.first_order = True

    # - outer trainer params
    args.batch_size = 8

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'vit_mi_fo_maml_rfs_adam_cl_100k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    # args.dir_wandb = Path('/shared/rsaas/miranda9/data/logs/')

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


# hbd2 = cifar-fs + Omni

# - load args

def load_args() -> Namespace:
    """
    1. parse args from user's terminal
    2. optionally set remaining args values (e.g. manually, hardcoded, from ckpt etc.)
    3. setup remaining args small details from previous values (e.g. 1 and 2).
    """
    # -- parse args from terminal
    # args: Namespace = parse_args_standard_sl()
    args: Namespace = parse_args_meta_learning()
    args.args_hardcoded_in_script = True  # <- REMOVE to remove manual loads
    # args.manual_loads_name = 'l2l_resnet12rfs_hdb1_100k'  # <- REMOVE to remove manual loads
    # args.manual_loads_name = 'l2l_resnet12rfs_mi_rfs_adam_cl_100k'  # <- REMOVE to remove manual loads

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    print(f'{args.manual_loads_name=}')
    if resume_from_checkpoint(args):
        args: Namespace = make_args_from_supervised_learning_checkpoint(args=args, precedence_to_args_checkpoint=True)
    elif args_hardcoded_in_script(args):
        if args.manual_loads_name == 'l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k':
            args: Namespace = l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k(args)
        elif args.manual_loads_name == 'l2l_4CNNl2l_cifarfs_rfs_adam_cl_70k':
            args: Namespace = l2l_4CNNl2l_cifarfs_rfs_adam_cl_70k(args)
        elif args.manual_loads_name == 'l2l_resnet12rfs_mi_rfs_adam_cl_100k':
            args: Namespace = l2l_resnet12rfs_mi_rfs_adam_cl_100k(args)
        elif args.manual_loads_name == 'l2l_5CNNl2l_mi_rfs_adam_cl_70k':
            args: Namespace = l2l_5CNNl2l_mi_rfs_adam_cl_70k(args)
        elif args.manual_loads_name == 'l2l_5CNNl2l_mi_rfs_sgd_cl_100k':
            args: Namespace = l2l_5CNNl2l_mi_rfs_sgd_cl_100k(args)
        elif args.manual_loads_name == 'l2l_5CNNl2l_cifarfs_rfs_sgd_cl_100k':
            args: Namespace = l2l_5CNNl2l_cifarfs_rfs_sgd_cl_100k(args)
        elif args.manual_loads_name == 'l2l_resnet12rfs_cifarfs_rfs_sgd_cl_100k':
            args: Namespace = l2l_resnet12rfs_cifarfs_rfs_sgd_cl_100k(args)
        elif args.manual_loads_name == 'l2l_resnet12rfs_mi_rfs_sgd_cl_100k':
            args: Namespace = l2l_resnet12rfs_mi_rfs_sgd_cl_100k(args)
        elif args.manual_loads_name == 'l2l_4CNNl2l_1024_cifarfs_rfs_adam_cl_100k':
            args: Namespace = l2l_4CNNl2l_1024_cifarfs_rfs_adam_cl_100k(args)
        elif args.manual_loads_name == 'l2l_resnet12rfs_mi_adam_no_scheduler_100k':
            args: Namespace = l2l_resnet12rfs_mi_adam_no_scheduler_100k(args)
        elif args.manual_loads_name == 'l2l_5CNN_mi_adam_filter_size':
            args: Namespace = l2l_5CNN_mi_adam_filter_size(args)
        elif args.manual_loads_name == 'l2l_5CNN_mi_adam_filter_size_32_filter_size':
            args: Namespace = l2l_5CNN_mi_adam_filter_size_32_filter_size(args)
        elif args.manual_loads_name == 'l2l_5CNN_mi_adam_filter_size_16_filter_size':
            args: Namespace = l2l_5CNN_mi_adam_filter_size_16_filter_size(args)
        elif args.manual_loads_name == 'l2l_5CNN_mi_adam_filter_size_8_filter_size':
            args: Namespace = l2l_5CNN_mi_adam_filter_size_8_filter_size(args)
        elif args.manual_loads_name == 'l2l_5CNN_mi_adam_filter_size_4_filter_size':
            args: Namespace = l2l_5CNN_mi_adam_filter_size_4_filter_size(args)
        elif args.manual_loads_name == 'l2l_resnet12rfs_hdb1_100k_adam_no_scheduler':
            args: Namespace = l2l_resnet12rfs_hdb1_100k_adam_no_scheduler(args)
        elif args.manual_loads_name == 'l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler':
            args: Namespace = l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler(args)
        elif args.manual_loads_name == 'l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler_first_order':
            args: Namespace = l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler_first_order(args)
        elif args.manual_loads_name == 'vit_mi_fo_maml_rfs_adam_cl_100k':
            args: Namespace = vit_mi_fo_maml_rfs_adam_cl_100k(args)
        else:
            raise ValueError(f'Invalid value, got: {args.manual_loads_name=}')
    else:
        # NOP: since we are using args from terminal
        pass
    # -- Setup up remaining stuff for experiment
    args: Namespace = setup_args_for_experiment(args)
    return args


def main():
    """
    Note: end-to-end ddp example on mnist: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
    :return:
    """
    # - load the args from either terminal, ckpt, manual etc.
    args: Namespace = load_args()

    # - parallel train
    if not args.parallel:  # serial
        print('RUNNING SERIALLY')
        args.world_size = 1
        args.rank = -1
        assert args.world_size == 1, f'Running serially but world_size is > 1, see: {args.world_size=}'
        assert args.rank == -1
        train(args=args)
    else:
        # mp.spawn(fn=train, args=(args,), nprocs=args.world_size) what ddp does
        # args.rank = get_local_rank()
        print(f'{torch.cuda.device_count()=}')
        print(f'local rank before train function: {args.rank=}')
        assert args.world_size > 1, f'Running parallel but world_size is <= 1, see: {args.world_size=}'
        # assert args.rank != -1
        train(args=args)


def train(args):
    # if is_running_parallel(args.rank):
    if args.parallel:
        # - set up processes a la l2l
        local_rank: int = get_local_rank()
        print(f'{local_rank=}')
        init_process_group_l2l(args, local_rank=local_rank, world_size=args.world_size, init_method=args.init_method)
        rank: int = torch.distributed.get_rank() if is_running_parallel(local_rank) else -1
        args.rank = rank  # have each process save the rank
        set_devices_and_seed_ala_l2l(args)  # args.device = rank or .device
    print(f'setup process done for rank={args.rank}')

    # - set up wandb only for the lead process
    setup_wandb(args) if is_lead_worker(args.rank) else None

    # create the model, opt & scheduler
    get_and_create_model_opt_scheduler_for_run(args)
    args.opt = move_opt_to_cherry_opt_and_sync_params(args) if is_running_parallel(args.rank) else args.opt

    # create the loaders, note: you might have to change the number of layers in the final layer
    args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)
    args.dataloaders = args.tasksets  # for the sake that eval_sl can detect how to get examples for eval
    assert args.model.cls.out_features == 5
    # assert args.model.classifier.out_features == 5

    # Agent does everything, proving, training, evaluate, meta-learnering, etc.
    args.agent = MAMLMetaLearnerL2L(args, args.model)
    args.meta_learner = args.agent

    # -- Start Training Loop
    print_dist(f"{args.model=}\n{args.opt=}\n{args.scheduler=}", args.rank)  # here to make sure mdl has the right cls
    print_dist('====> about to start train loop', args.rank)
    if args.training_mode == 'meta_train_agent_fit_single_batch':
        # meta_train_agent_fit_single_batch(args, args.agent, args.dataloaders, args.opt, args.scheduler)
        raise NotImplementedError
    elif 'iterations' in args.training_mode:
        meta_train_iterations_ala_l2l(args, args.agent, args.opt, args.scheduler)
    elif 'epochs' in args.training_mode:
        # meta_train_epochs(args, agent, args.dataloaders, args.opt, args.scheduler) not implemented
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid training_mode value, got: {args.training_mode}')

    # -- Clean Up Distributed Processes
    print(f'\n----> about to cleanup worker with rank {rank}')
    cleanup(rank)
    print(f'clean up done successfully! {rank}')


# -- Run experiment

if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    main()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
