"""
Main script to set up supervised learning experiments
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
# import torch.optim as optim

from argparse import Namespace

from uutils import args_hardcoded_in_script, report_times
from uutils.argparse_uu.common import setup_args_for_experiment
from uutils.argparse_uu.supervised_learning import make_args_from_supervised_learning_checkpoint, parse_args_standard_sl
# from uutils.torch_uu.agents.common import Agent
# from uutils.torch_uu.agents.supervised_learning import ClassificationSLAgent
from uutils.torch_uu import count_number_of_parameters
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.agents.supervised_learning import ClassificationSLAgent, UnionClsSLAgent
from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.dataloaders.helpers import get_sl_dataloader
from uutils.torch_uu.distributed import set_sharing_strategy, print_process_info, set_devices, setup_process, cleanup, \
    print_dist
# from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_first_time
# from uutils.torch_uu.training.supervised_learning import train_agent_fit_single_batch, train_agent_iterations, \
#     train_agent_epochs
from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_first_time, \
    get_and_create_model_opt_scheduler_for_run
from uutils.torch_uu.mains.main_sl_with_ddp import train
from uutils.torch_uu.training.supervised_learning import train_agent_fit_single_batch, train_agent_iterations, \
    train_agent_epochs

from socket import gethostname

from pdb import set_trace as st


# -- MI

def sl_mi_rfs_5cnn_adam_cl(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64, filter_size=32, levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 2_000
    args.batch_size = 1024
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    # args.training_mode = 'epochs'
    args.training_mode = 'epochs_train_convergence'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_5cnn_adam_cl'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.training_mode}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_5cnn_adam_cl_200(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64, filter_size=32, levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 200
    args.batch_size = 1024
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_5cnn_adam_cl_200'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_5cnn_adam_cl_600(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64, filter_size=32, levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 600
    args.batch_size = 1024
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_5cnn_adam_cl_600'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_resnet_rfs_mi_adam_cl_200(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_mi'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=64)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 200
    args.batch_size = 512
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_resnet_rfs_mi_adam_cl_200'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_5cnn_sgd_cl_600(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64, filter_size=32, levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()

    # - opt
    args.opt_option = 'Sgd_rfs'
    args.num_epochs = 600
    args.batch_size = 1024
    args.lr = 5e-2
    args.opt_hps: dict = dict(lr=args.lr, momentum=0.9, weight_decay=5e-4)

    args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.lr_decay_rate = 1e-1
    # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_5cnn_sgd_cl_600'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_resnet12rfs_sgd_cl_200(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_mi'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=64)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()

    # - opt
    args.opt_option = 'Sgd_rfs'
    args.num_epochs = 200
    args.batch_size = 512
    args.lr = 5e-2
    args.opt_hps: dict = dict(lr=args.lr, momentum=0.9, weight_decay=5e-4)

    args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.lr_decay_rate = 1e-1
    # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_resnet12rfs_sgd_cl_200'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_5cnn_adam(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.filter_size = 32
    # args.filter_size = 128
    # args.filter_size = 512
    # args.filter_size = 1024
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64, filter_size=args.filter_size,
                          levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 1_000
    args.batch_size = 128
    # args.batch_size = 1024
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'None'

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_5cnn_adam'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_5cnn_adam(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.filter_size = 32
    # args.filter_size = 128
    # args.filter_size = 512
    # args.filter_size = 1024
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64, filter_size=args.filter_size,
                          levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 1_000
    args.batch_size = 128
    # args.batch_size = 1024
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'None'

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_5cnn_adam'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_5cnn_adam_32_filter_size(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.filter_size = 32
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64, filter_size=args.filter_size,
                          levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 400
    args.batch_size = 128
    # args.batch_size = 1024
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'None'

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_5cnn_adam'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_5cnn_adam_16_filter_size(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.filter_size = 16
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64, filter_size=args.filter_size,
                          levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 400
    args.batch_size = 128
    # args.batch_size = 1024
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'None'

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_5cnn_adam'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_5cnn_adam_8_filter_size(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.filter_size = 8
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64, filter_size=args.filter_size,
                          levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 400
    args.batch_size = 128
    # args.batch_size = 1024
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'None'

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_5cnn_adam'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_5cnn_adam_4_filter_size(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.filter_size = 4
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64, filter_size=args.filter_size,
                          levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 400
    args.batch_size = 128
    # args.batch_size = 1024
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'None'

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_5cnn_adam'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


# -- cirfarfs

def sl_cifarfs_rfs_4cnn_adam_cl_200(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.model_hps = dict(ways=64, hidden_size=64, embedding_size=64 * 4)

    # - data
    # args.data_path = Path('~/data/CIFAR-FS/').expanduser()
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 200
    args.batch_size = 1024
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_rfs_4cnn_adam_cl_200'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_rfs_4cnn_adam_cl_600(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.model_hps = dict(ways=64, hidden_size=64, embedding_size=64 * 4)

    # - data
    # args.data_path = Path('~/data/CIFAR-FS/').expanduser()
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 600
    args.batch_size = 1024
    args.lr = 5e-1
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_rfs_4cnn_adam_cl_600'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_rfs_resnet12rfs_adam_cl_200(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_cifarfs_fc100'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=64)

    # - data
    # args.data_path = Path('~/data/CIFAR-FS/').expanduser()
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 200
    args.batch_size = 1024
    # args.batch_size = 2 ** 14  # 2**14
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_rfs_resnet12rfs_adam_cl_200'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_rfs_resnet12rfs_adam_cl_600(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_cifarfs_fc100'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=64)

    # - data
    # args.data_path = Path('~/data/CIFAR-FS/').expanduser()
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 600
    args.batch_size = 1024
    # args.batch_size = 2 ** 14  # 2**14
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_rfs_resnet12rfs_adam_cl_600'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_rfs_4cnn_adam_cl(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.model_hps = dict(ways=64, hidden_size=64, embedding_size=64 * 4)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 2_000
    args.batch_size = 1024
    args.lr = 5e-1
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    # args.training_mode = 'epochs'
    args.training_mode = 'epochs_train_convergence'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_rfs_4cnn_adam_cl'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.training_mode}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_rfs_5cnn_sgd_cl_1000(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.n_cls = 64
    args.model_hps = dict(ways=args.n_cls, hidden_size=64, embedding_size=64 * 4)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Sgd_rfs'
    args.num_epochs = 1_000
    args.batch_size = 1024
    args.lr = 5e-2
    args.opt_hps: dict = dict(lr=args.lr, momentum=0.9, weight_decay=5e-4)

    args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.lr_decay_rate = 1e-1
    # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_rfs_5cnn_sgd_cl_1000'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_resnet12rfs_sgd_cl_200(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_cifarfs_fc100'
    args.n_cls = 64
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=args.n_cls)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Sgd_rfs'
    args.num_epochs = 200
    args.batch_size = 512
    args.lr = 5e-2
    args.opt_hps: dict = dict(lr=args.lr, momentum=0.9, weight_decay=5e-4)

    args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.lr_decay_rate = 1e-1
    # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_resnet12rfs_sgd_cl_200'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_rfs_5cnn_adafactor_1000(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.n_cls = 64
    args.model_hps = dict(ways=args.n_cls, hidden_size=args.hidden_size, embedding_size=args.hidden_size * 4)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'AdafactorDefaultFair'
    args.num_epochs = 1_000
    args.batch_size = 1024

    args.scheduler_option = 'AdafactorSchedule'
    args.log_scheduler_freq = 1

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_rfs_5cnn_adafactor_1000'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_4cnn_hidden_size_128_sgd_cl_rfs_500(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.hidden_size = 128
    args.n_cls = 64
    args.model_hps = dict(ways=args.n_cls, hidden_size=args.hidden_size, embedding_size=args.hidden_size * 4)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Sgd_rfs'
    args.num_epochs = 500
    args.batch_size = 1024
    args.lr = 5e-2
    args.opt_hps: dict = dict(lr=args.lr, momentum=0.9, weight_decay=5e-4)

    args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.lr_decay_rate = 1e-1
    # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_4cnn_hidden_size_128_sgd_cl_rfs_500'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=} {args.hidden_size=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_4cnn_hidden_size_1024_sgd_cl_rfs_500(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.hidden_size = 1024
    args.n_cls = 64
    args.model_hps = dict(ways=args.n_cls, hidden_size=args.hidden_size, embedding_size=args.hidden_size * 4)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Sgd_rfs'
    args.num_epochs = 500
    args.batch_size = 256
    args.lr = 5e-2
    args.opt_hps: dict = dict(lr=args.lr, momentum=0.9, weight_decay=5e-4)

    args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.lr_decay_rate = 1e-1
    # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_4cnn_hidden_size_1024_sgd_cl_rfs_500'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=} {args.hidden_size=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_4cnn_hidden_size_1024_adam_rfs_500(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.hidden_size = 1024
    args.n_cls = 64
    args.model_hps = dict(ways=args.n_cls, hidden_size=args.hidden_size, embedding_size=args.hidden_size * 4)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 500
    args.batch_size = 256  # 256 used to work on titans...
    args.lr = 5e-2
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_4cnn_hidden_size_1024_adam_rfs_500'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=} {args.hidden_size=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_4cnn_hidden_size_1024_adam_rfs_epochs_train_convergence(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.hidden_size = 1024
    args.n_cls = 64
    args.model_hps = dict(ways=args.n_cls, hidden_size=args.hidden_size, embedding_size=args.hidden_size * 4)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 500
    args.batch_size = 256  # 256 used to work on titans...
    args.lr = 5e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'None'

    # - training mode
    args.training_mode = 'epochs_train_convergence'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 10  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_4cnn_hidden_size_1024_adam_rfs_epochs_train_convergence'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=} {args.hidden_size=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_4cnn_hidden_size_1024_adafactor_rfs_epochs_train_convergence(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.hidden_size = 1024
    args.n_cls = 64
    args.model_hps = dict(ways=args.n_cls, hidden_size=args.hidden_size, embedding_size=args.hidden_size * 4)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'AdafactorDefaultFair'
    args.scheduler_option = 'None'

    # - training mode
    args.training_mode = 'epochs_train_convergence'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 10  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_4cnn_hidden_size_1024_adafactor_rfs_epochs_train_convergence'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=} {args.hidden_size=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_4cnn_hidden_size_1024_adafactor_adafactor_scheduler_rfs_epochs_train_convergence(
        args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.hidden_size = 1024
    args.n_cls = 64
    args.model_hps = dict(ways=args.n_cls, hidden_size=args.hidden_size, embedding_size=args.hidden_size * 4)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'AdafactorDefaultFair'
    args.scheduler_option = 'AdafactorSchedule'

    # - training mode
    args.training_mode = 'epochs_train_convergence'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 10  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_4cnn_hidden_size_1024_adafactor_adafactor_scheduler_rfs_epochs_train_convergence'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=} {args.hidden_size=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_4cnn_hidden_size_1024_adam_rfs_1000(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.hidden_size = 1024
    args.n_cls = 64
    args.model_hps = dict(ways=args.n_cls, hidden_size=args.hidden_size, embedding_size=args.hidden_size * 4)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 1000
    args.batch_size = 256  # 256 used to work on titans...
    args.lr = 5e-2
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 10  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_4cnn_hidden_size_1024_adam_rfs_1000'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=} {args.hidden_size=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_4cnn_hidden_size_1024_adam_no_scheduler_1000(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.hidden_size = 1024
    args.n_cls = 64
    args.model_hps = dict(ways=args.n_cls, hidden_size=args.hidden_size, embedding_size=args.hidden_size * 4)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 1000
    args.batch_size = 256  # 256 used to work on titans...
    args.lr = 5e-2
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'None'

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 10  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_4cnn_hidden_size_1024_adam_no_scheduler_1000'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=} {args.hidden_size=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_cifarfs_4cnn_hidden_size_1024_adam_no_scheduler_many_epochs(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'
    args.hidden_size = 1024
    args.n_cls = 64
    args.model_hps = dict(ways=args.n_cls, hidden_size=args.hidden_size, embedding_size=args.hidden_size * 4)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 1_000_000
    args.batch_size = 256  # 256 used to work on titans...
    args.lr = 5e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'None'

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 10  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_cifarfs_4cnn_hidden_size_1024_adam_no_scheduler_many_epochs'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.num_epochs}: {args.jobid=} {args.hidden_size=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_5cnn_adam_cl_32_filter_size(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.filter_size = 32
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64, filter_size=args.filter_size,
                          levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 400
    args.batch_size = 128
    # args.batch_size = 1024
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    # args.scheduler_option = 'None'
    args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.lr_decay_rate = 1e-1
    # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_5cnn_adam'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_5cnn_adam_cl_128_filter_size(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.filter_size = 128
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64, filter_size=args.filter_size,
                          levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()
    args.data_option = 'miniImageNet_rfs'

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 400
    args.batch_size = 128
    # args.batch_size = 1024
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    # args.scheduler_option = 'None'
    args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.lr_decay_rate = 1e-1
    # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_5cnn_adam'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_mi_rfs_5cnn_adam_cl_512_filter_size(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs
        - Opt: ?

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.filter_size = 512
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64, filter_size=args.filter_size,
                          levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()
    args.data_option = 'miniImageNet_rfs'

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 400
    args.batch_size = 128
    # args.batch_size = 1024
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    # args.scheduler_option = 'None'
    args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.lr_decay_rate = 1e-1
    # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # SL, epochs training

    # - wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'sl_mi_rfs_5cnn_adam'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


# - hdb1 = MIO, USL

def sl_hdb1_rfs_resnet12rfs_adam_cl(args: Namespace) -> Namespace:
    """
    goal:
        - model: resnet12-rfs

    Note:
        - you need to use the rfs data loaders because you need to do the union of the labels in the meta-train set.
        If you use the cifar100 directly from pytorch it will see images in the meta-test set and SL will have an unfair
        advantage.
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_hdb1_mio'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=64 + 1100)

    # - data
    # args.data_path = Path('~/data/CIFAR-FS/').expanduser()
    args.data_option = 'hdb1_mio_usl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 1000
    # args.batch_size = 1024
    args.batch_size = 512
    # args.batch_size = 2 ** 14  # 2**14
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batch'

    # -
    # args.debug = True
    args.debug = False

    # -
    args.log_freq = 1  # for SL it is meant to be small e.g. 1 or 2

    # - wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'sl_hdb1_rfs_resnet12rfs_adam_cl'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def sl_hdb1_5cnn_adam_cl_filter_size(args: Namespace):
    from pathlib import Path
    # - model
    assert args.filter_size != -1, f'Err: {args.filter_size=}'
    print(f'--->{args.filter_size=}')
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64 + 1100,
                          filter_size=args.filter_size,
                          levels=None,
                          spp=False, in_channels=3)

    # - data
    args.data_option = 'hdb1_mio_usl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 1000
    # args.batch_size = 1024
    if args.filter_size >= 512:
        args.batch_size = 128
    else:
        args.batch_size = 512
    # args.batch_size = 2 ** 14  # 2**14
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    args.training_mode = 'epochs'

    # - debug flag
    # args.debug = True
    args.debug = False

    # - logging params
    args.log_freq = 1  # for SL it is meant to be small e.g. 1 or 2

    # - wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'sl_hdb1_5cnn_adam_cl_filter_size'
    args.run_name = f'{args.filter_size=} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=} {gethostname()}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


# - hbd4 micod


def sl_hdb4_micod_resnet_rfs_adam_cl(args: Namespace) -> Namespace:
    # - model
    # args.model_option = 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    args.n_cls = 5000
    # bellow seems true for all models, they do use avg pool at the global pool/last pooling layer
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)  # dropbock_size=5 is rfs default for MI, 2 for CIFAR, will assume 5 for mds since it works on imagenet

    # - data
    args.data_option = 'hdb4_micod'
    args.n_classes = args.n_cls

    # - training mode
    args.training_mode = 'epochs'
    args.num_its = 1_000
    # args.training_mode = 'iterations'
    # args.num_its = 1_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.batch_size = 256
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # args.scheduler_option = 'None'
    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 2_000
    args.T_max = args.num_its // args.log_scheduler_freq  # intended 800K/2k
    args.eta_min = 1e-5  # match MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)
    print(f'{args.T_max=}')
    # assert args.T_max == 400, f'T_max is not expected value, instead it is: {args.T_max=}'

    # - logging params
    # args.log_freq = 500
    # args.log_freq = 20
    args.log_freq = 2

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = args.manual_loads_name
    args.run_name = f'{args.data_option} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option}: {args.jobid=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False
    return args


# - mds

def get_hardcoded_full_mds_num_classes(sources: list[str] = ['hardcodedmds']) -> int:
    """
    todo: this is a hack, fix it
    - use the dataset_spec in each data set
```
  "classes_per_split": {
    "TRAIN": 140,
    "VALID": 30,
    "TEST": 30
```
    but for imagenet do something else.
    argh, idk what this means
```
{
  "name": "ilsvrc_2012",
  "split_subgraphs": {
    "TRAIN": [
      {
        "wn_id": "n00001740",
        "words": "entity",
        "children_ids": [
          "n00001930",
          "n00002137"
        ],
        "parents_ids": []
      },
      {
        "wn_id": "n00001930",
        "words": "physical entity",
        "children_ids": [
          "n00002684",
          "n00007347",
          "n00020827"
        ],
```
count the leafs in train & make sure it matches 712 158 130 for each split

ref: https://github.com/google-research/meta-dataset#dataset-summary
    """
    print(f'{sources=}')
    n_cls: int = 0
    if sources[0] == 'hardcodedmds':
        # args.sources = ['ilsvrc_2012', 'aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot', 'quickdraw', 'vgg_flower']
        n_cls = 712 + 70 + 140 + 33 + 994 + 883 + 241 + 71
        n_cls = 3144
    else:
        raise NotImplementedError
    print(f'{n_cls=}')
    assert n_cls != 0
    return n_cls


def mds_resnet_usl_adam_scheduler(args: Namespace) -> Namespace:
    # - model
    # args.model_option = 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    args.n_cls = get_hardcoded_full_mds_num_classes()  # ref: https://github.com/google-research/meta-dataset#dataset-summary
    # bellow seems true for all models, they do use avg pool at the global pool/last pooling layer
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)  # dropbock_size=5 is rfs default for MI, 2 for CIFAR, will assume 5 for mds since it works on imagenet

    # - data
    args.data_option = 'mds'
    args.sources = ['ilsvrc_2012', 'aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot', 'quickdraw', 'vgg_flower',
                    'mscoco', 'traffic_sign']
    args.n_classes = args.n_cls

    # if we set args.min_examples_in_class = args.k_shot + args.k_eval
    # we ensure that our n-way k-shot task has enough samples for both the support and query sets.
    args.min_examples_in_class = 20  # assuming 5-way, 5-shot 15-eval shot

    # - training mode
    args.training_mode = 'iterations'
    # args.num_its = 1_000_000_000  # patrick's default for 2 data sets
    # args.num_its = 50_000  # mds 50000: https://github.com/google-research/meta-dataset/blob/d6574b42c0f501225f682d651c631aef24ad0916/meta_dataset/learn/gin/best/pretrain_imagenet_resnet.gin#L20
    args.num_its = 100_000  # mds 50000 so I feel it should be fine to double it
    # args.num_its = 20*6_000 = 120_000 # based on some estimates for resnet50, to reach 0.99 acc
    # args.num_its = 2*120_000  # times 2 to be safe + increase log freq from 20 to something larger for speed up

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.batch_size = 256
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # args.scheduler_option = 'None'
    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 2_000
    args.T_max = args.num_its // args.log_scheduler_freq  # intended 800K/2k
    args.eta_min = 1e-5  # match MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)
    print(f'{args.T_max=}')
    # assert args.T_max == 400, f'T_max is not expected value, instead it is: {args.T_max=}'

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = args.manual_loads_name
    args.run_name = f'{args.data_option} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def mds_resnet_usl_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    # args.model_option = 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    args.n_cls = get_hardcoded_full_mds_num_classes()  # ref: https://github.com/google-research/meta-dataset#dataset-summary
    # bellow seems true for all models, they do use avg pool at the global pool/last pooling layer
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)  # dropbock_size=5 is rfs default for MI, 2 for CIFAR, will assume 5 for mds since it works on imagenet

    # - data
    args.data_option = 'mds'
    # Mscoco, traffic_sign are VAL only (actually we could put them here, fixed script to be able to do so w/o crashing)
    args.sources = ['ilsvrc_2012', 'aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot', 'quickdraw', 'vgg_flower',
                    'mscoco', 'traffic_sign']
    args.n_classes = args.n_cls

    # if we set args.min_examples_in_class = args.k_shot + args.k_eval
    # we ensure that our n-way k-shot task has enough samples for both the support and query sets.
    args.min_examples_in_class = 20  # assuming 5-way, 5-shot 15-eval shot

    # - training mode
    args.training_mode = 'iterations_train_convergence'

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.batch_size = 256
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = args.manual_loads_name
    args.run_name = f'{args.data_option} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False
    return args


def load_args() -> Namespace:
    """
    1. parse args from user's terminal
    2. optionally set remaining args values (e.g. manually, hardcoded, from ckpt etc.)
    3. setup remaining args small details from previous values (e.g. 1 and 2).
    """
    # -- parse args from terminal
    # todo: maybe later, add a try catch that if there is an mds only flag given at the python cmd line then it will load the mds args otherwise do the meta-leanring args
    # todo: https://stackoverflow.com/questions/75141370/how-does-one-have-python-work-when-multiple-arg-parse-options-are-possible
    from diversity_src.dataloaders.metadataset_common import get_mds_base_args
    args: Namespace = get_mds_base_args()
    # args: Namespace = parse_args_standard_sl()
    args.args_hardcoded_in_script = True  # <- REMOVE to remove manual loads
    # args.manual_loads_name = 'sl_hdb1_5cnn_adam_cl_filter_size'  # <- REMOVE to remove manual loads

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    print(f'{args.manual_loads_name=}')
    if resume_from_checkpoint(args):
        args: Namespace = make_args_from_supervised_learning_checkpoint(args=args, precedence_to_args_checkpoint=True)
    elif args_hardcoded_in_script(args):
        args: Namespace = eval(f'{args.manual_loads_name}(args)')
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
    [print(f'{k, v}') for k, v in vars(args).items()]

    # - parallel train
    if not args.parallel:  # serial
        print('RUNNING SERIALLY')
        args.world_size = 1
        train(rank=-1, args=args)
    else:
        print(f"{torch.cuda.device_count()=}")
        args.world_size = torch.cuda.device_count()
        # args.world_size = mp.cpu_count() - 1  # 1 process is main, the rest are (parallel) trainers
        set_sharing_strategy()
        mp.spawn(fn=train, args=(args,), nprocs=args.world_size)


def train(rank, args):
    print_process_info(rank, flush=True)
    args.rank = rank  # have each process save the rank
    set_devices(args)  # args.device = rank or .device
    setup_process(args, rank, master_port=args.master_port, world_size=args.world_size)
    print(f'setup process done for rank={rank}')

    # create the (ddp) model, opt & scheduler
    get_and_create_model_opt_scheduler_for_run(args)
    args.number_of_trainable_parameters = count_number_of_parameters(args.model)

    # create the dataloaders, this goes first so you can select the mdl (e.g. final layer) based on task
    args.dataloaders: dict = get_sl_dataloader(args)
    assert args.model.cls.out_features > 5, f'Not meta-learning training, so always more than 5 classes but got {args.model.cls.out_features=}'
    # assert args.model.cls.out_features == 64  # mi (cifar-fs?)
    # assert args.model.cls.out_features == 3144  # mds
    print(f'{args.model.cls.out_features=}')

    # Agent does everything, proving, training, evaluate etc.
    args.agent: Agent = UnionClsSLAgent(args, args.model)

    # -- Start Training Loop
    print_dist(f"{args.model=}\n{args.opt=}\n{args.scheduler=}", args.rank)  # here to make sure mdl has the right cls
    print_dist('\n\n====> about to start train loop', args.rank)
    print(f'{args.filter_size=}') if hasattr(args, 'filter_size') else None
    print(f'{args.number_of_trainable_parameters=}')
    if args.training_mode == 'fit_single_batch':
        train_agent_fit_single_batch(args, args.agent, args.dataloaders, args.opt, args.scheduler)
    elif 'iterations' in args.training_mode:
        # note train code will see training mode to determine halting criterion
        train_agent_iterations(args, args.agent, args.dataloaders, args.opt, args.scheduler)
    elif 'epochs' in args.training_mode:
        # note train code will see training mode to determine halting criterion
        train_agent_epochs(args, args.agent, args.dataloaders, args.opt, args.scheduler)
    # note: the other options do not appear directly since they are checked in
    # the halting condition.
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
