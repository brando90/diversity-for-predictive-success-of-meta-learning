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
    args.lr = 1e-1
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
    args.num_epochs = 200
    args.batch_size = 1024
    args.lr = 1e-1
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
    args.lr = 1e-1
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
    args.lr = 1e-1
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
    args.lr = 1e-1
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
    args.lr = 1e-1
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
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=64)

    # - data
    # args.data_path = Path('~/data/CIFAR-FS/').expanduser()
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 600
    args.batch_size = 1024
    # args.batch_size = 2 ** 14  # 2**14
    args.lr = 1e-1
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

def sl_cifarfs_rfs_5cnn_sgd_cl_600(args: Namespace) -> Namespace:
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
    args.model_option = '4CNN_l2l_cifarfs'
    args.model_hps = dict(ways=64, hidden_size=64, embedding_size=64 * 4)

    # - data
    args.data_option = 'cifarfs_l2l_sl'
    args.data_path = Path('~/data/l2l_data/').expanduser()

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
    args.experiment_name = f'sl_cifarfs_rfs_5cnn_sgd_cl_600'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
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
    args: Namespace = parse_args_standard_sl()
    args.args_hardcoded_in_script = True  # <- REMOVE to remove manual loads
    args.manual_loads_name = 'sl_cifarfs_rfs_5cnn_sgd_cl_600'  # <- REMOVE to remove manual loads

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    if resume_from_checkpoint(args):
        args: Namespace = make_args_from_supervised_learning_checkpoint(args=args, precedence_to_args_checkpoint=True)
    elif args_hardcoded_in_script(args):
        if args.manual_loads_name == 'sl_cifarfs_rfs_resnet12rfs_adam_cl_200':
            args: Namespace = sl_cifarfs_rfs_resnet12rfs_adam_cl_200(args)
        elif args.manual_loads_name == 'sl_cifarfs_rfs_4cnn_adam_cl_200':
            args: Namespace = sl_cifarfs_rfs_4cnn_adam_cl_200(args)
        elif args.manual_loads_name == 'sl_mi_rfs_5cnn_adam_cl_200':
            args: Namespace = sl_mi_rfs_5cnn_adam_cl_200(args)
        elif args.manual_loads_name == 'sl_mi_rfs_resnet_rfs_mi_adam_cl_200':
            args: Namespace = sl_mi_rfs_resnet_rfs_mi_adam_cl_200(args)
        elif args.manual_loads_name == 'sl_mi_rfs_5cnn_adam_cl':
            args: Namespace = sl_mi_rfs_5cnn_adam_cl(args)
        elif args.manual_loads_name == 'sl_mi_rfs_5cnn_adam_cl_600':
            args: Namespace = sl_mi_rfs_5cnn_adam_cl_600(args)
        elif args.manual_loads_name == 'sl_cifarfs_rfs_resnet12rfs_adam_cl_600':
            args: Namespace = sl_cifarfs_rfs_resnet12rfs_adam_cl_600(args)
        elif args.manual_loads_name == 'sl_cifarfs_rfs_4cnn_adam_cl_600':
            args: Namespace = sl_cifarfs_rfs_4cnn_adam_cl_600(args)
        elif args.manual_loads_name == 'sl_cifarfs_rfs_4cnn_adam_cl':
            args: Namespace = sl_cifarfs_rfs_4cnn_adam_cl(args)
        elif args.manual_loads_name == 'sl_mi_rfs_5cnn_sgd_cl_600':
            args: Namespace = sl_mi_rfs_5cnn_sgd_cl_600(args)
        elif args.manual_loads_name == 'sl_cifarfs_rfs_5cnn_sgd_cl_600':
            args: Namespace = sl_cifarfs_rfs_5cnn_sgd_cl_600(args)
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

    # create the dataloaders, this goes first so you can select the mdl (e.g. final layer) based on task
    args.dataloaders: dict = get_sl_dataloader(args)
    # replace_final_layer(args, n_classes=args.n_cls)  # for SL this should be done at the dataset/loader level

    # Agent does everything, proving, training, evaluate etc.
    args.agent: Agent = UnionClsSLAgent(args, args.model)

    # -- Start Training Loop
    print_dist(f"{args.model=}\n{args.opt=}\n{args.scheduler=}", args.rank)  # here to make sure mdl has the right cls
    print_dist('====> about to start train loop', args.rank)
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
