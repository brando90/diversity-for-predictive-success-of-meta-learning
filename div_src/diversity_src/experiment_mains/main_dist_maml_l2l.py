"""
Main script to set up supervised learning experiments
"""
# import os
# os.environ["NCCL_DEBUG"] = "INFO"
from socket import gethostname

import os

# import learn2learn
import torch
import torch.nn as nn
import torch.multiprocessing as mp
# import torch.optim as optim

from argparse import Namespace

from learn2learn.vision.benchmarks import BenchmarkTasksets

from uutils import args_hardcoded_in_script
from uutils.argparse_uu.common import setup_args_for_experiment
from uutils.argparse_uu.meta_learning import parse_args_meta_learning, fix_for_backwards_compatibility
from uutils.argparse_uu.supervised_learning import make_args_from_supervised_learning_checkpoint, parse_args_standard_sl
from uutils.logging_uu.wandb_logging.common import setup_wandb, cleanup_wandb
from uutils.torch_uu import count_number_of_parameters
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloaders
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
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
    args.model_option = '5CNN_opt_as_model_for_few_shot'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
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
    args.model_option = '5CNN_opt_as_model_for_few_shot'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
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
    args.model_option = '5CNN_opt_as_model_for_few_shot'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
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
    args.model_option = '5CNN_opt_as_model_for_few_shot'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
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
    args.model_option = '5CNN_opt_as_model_for_few_shot'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_5CNN_mi_adam_filter_size_128_filter_size(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.filter_size = 128
    args.model_option = '5CNN_opt_as_model_for_few_shot'
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
    args.first_order = True
    # args.first_order = False

    # - outer trainer params
    args.batch_size = 8

    # parser.add_argument('--k_eval', type=int, default=15, help="")
    args.k_eval = 5

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 1
    args.parallel = args.world_size > 1
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_5CNN_mi_adam_filter_size_512_filter_size(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.filter_size = 512
    args.model_option = '5CNN_opt_as_model_for_few_shot'
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
    args.first_order = True
    # args.first_order = False

    # - outer trainer params
    args.batch_size = 8

    # parser.add_argument('--k_eval', type=int, default=15, help="")
    args.k_eval = 5

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 1
    args.parallel = args.world_size > 1
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.filter_size}: {args.jobid=}'
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
    # - maml l2l
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'l2l_resnet12rfs_hdb1_100k_adam_no_scheduler'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.first_order=}: {args.jobid=}'
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
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
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
    args.model_option = '5CNN_opt_as_model_for_few_shot'
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
    # args.seed = 42  # I think this might be important due to how tasksets works.
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
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler_first_order_from_ckpt(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_mi'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)
    args.path_to_checkpoint = '~/data/logs/logs_Oct15_18-08-54_jobid_96800/ckpt.pt'  # train_acc 0.986, train_loss 0.0531, val_acc 0.621

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
    args.batch_size = 30

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    # args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 200

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler_first_order_from_ckpt'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} {args.first_order=}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def maml_l2l(args: Namespace):
    # def vit_maml(args: Namespace):
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = 'resnet12_rfs'
    # args.model_option = 'resnet12_rfs_cifarfs_fc100'
    # args.model_option = 'vit_mi'
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    # args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    args.allow_unused = True  # transformers have lots of tokens & params, so I assume some param is not always being used in the forward pass
    if '5CNN_opt_as_model_for_few_shot' in args.model_option:
        args.model_hps = dict(n_classes=args.n_cls, filter_size=args.filter_size)
        args.model_hps['image_size'] = 32 if 'cifarfs' in args.data_option else 84
    else:
        args.model_hps = dict(num_classes=args.n_cls)
    print(f'--> {args.model_option=} {args.model_hps=}')

    # - data
    if args.data_option == 'mini-imagenet':
        args.data_path = Path('~/data/l2l_data/').expanduser()
        args.data_augmentation = 'lee2019'
    elif args.data_option == 'hdb4_micod':
        args.data_path = Path('~/data/l2l_data/').expanduser()
        args.data_augmentation = 'hdb4_micod'
    elif args.data_option == 'cifarfs':  # don't think this is needed!
        args.data_path = Path('~/data/l2l_data/').expanduser()
        args.data_augmentation = 'rfs2020'

    # - training mode
    args.training_mode = 'iterations'
    # note: 60K iterations for original maml 5CNN with adam
    # args.num_its = 300_000  # resnet12rfs conv with 300K
    # args.num_its = 400_000  # resnet12rfs conv with 300K
    # args.num_its = 500_000  # resnet12rfs conv with 300K
    args.num_its = 600_000  # resnet12rfs conv with 300K

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'AdafactorDefaultFair'
    args.opt_hps: dict = dict()
    # args.opt_option = 'Adam_rfs_cifarfs'
    # args.lr = 1e-3  # match MAML++
    # # args.lr = 1e-4  # match MAML++
    # args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # args.scheduler_option = 'None'
    args.scheduler_option = 'AdafactorSchedule'
    # args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    # args.log_scheduler_freq = 2_000
    # args.T_max = args.num_its // args.log_scheduler_freq  # intended 800K/2k
    # args.eta_min = 1e-5  # match MAML++
    # args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)
    # print(f'{args.T_max=}')

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    args.first_order = True

    # - outer trainer params
    args.batch_size = 4
    args.batch_size_eval = 2

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    # args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                                metric_to_use='train_acc',
    #                                threshold=0.9, log_speed_up=10)
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                                log_speed_up=1)

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


# - hdb1 - 5CNN scale experiment

def l2l_5CNN_hdb1_adam_cs_filter_size(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    assert args.filter_size != -1, f'Err: {args.filter_size=}'
    print(f'--->{args.filter_size=}')
    args.n_cls = 5
    args.model_option = '5CNN_opt_as_model_for_few_shot'
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

    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    args.first_order = True
    # args.first_order = False

    # - outer trainer params
    # args.batch_size = 32
    args.batch_size = 8

    # args.k_eval = 5
    # args.k_eval = 10
    args.k_eval = 15

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 1
    args.parallel = args.world_size > 1
    # args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'l2l_5CNN_hdb1_adam_cs_filter_size'
    args.run_name = f'{args.manual_loads_name} {args.filter_size=} {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr} : {args.jobid=} hostname: {gethostname()}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


# - hdb4 = micod = mi + omniglot + omniglot + delauny

def maml_hdb4_micod_resnet_rfs(args: Namespace) -> Namespace:
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)

    # - data
    args.data_option = 'hdb4_micod'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'hdb4_micod'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    # args.num_its = 100_000
    args.num_its = 300_000
    # args.num_its = 600_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
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

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    args.first_order = True
    # args.first_order = False

    # - outer trainer params
    args.batch_size = 3
    args.batch_size_eval = 2

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    # args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                                metric_to_use='train_acc',
    #                                threshold=0.9, log_speed_up=10)
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_acc',
    #                                log_speed_up=5)

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = args.manual_loads_name
    args.run_name = f'{args.manual_loads_name} {args.data_option} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


# -- hdb4 micod

def maml_hdb4_micod(args: Namespace) -> Namespace:
    from pathlib import Path
    # - model
    assert args.filter_size != -1, f'Err: {args.filter_size=}'
    print(f'--->{args.filter_size=}')
    args.n_cls = 5
    if model_option == 'resnet12rfs':
        args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    elif model_option == '5CNN_opt_as_model_for_few_shot':
        args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
                              filter_size=args.filter_size, levels=None, spp=False, in_channels=3)
    # else vit

    # - data
    args.data_option = 'hdb4_micod'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'hdb4_micod'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    # args.num_its = 100_000
    # args.num_its = 300_000
    # args.num_its = 500_000  # but it seems resnet12rfs conv with 300K
    args.num_its = 900_000  # resnet12rfs conv with 300K, lets to 3 times to be safe

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
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

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.first_order = True  # need to create new args that uses first order maml, leaving as is for reproducibility
    args.first_order = False  # seems I did higher order maml by accident, leaving it to not be confusing

    # - outer trainer params
    args.batch_size = 4
    args.batch_size_eval = 2

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    # args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                                metric_to_use='train_acc',
    #                                threshold=0.9, log_speed_up=10)
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                                log_speed_up=1)

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'

    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args



# ====== hdb5, meta-dataset-style learn2learn expts below ======


def hdb5_vggair_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    #args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    assert args.filter_size != -1, f'Err: {args.filter_size=}'
    print(f'--->{args.filter_size=}')
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    # args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
                          filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'hdb5_vggair'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb5_vggair'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # This is needed.
    args.first_order = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    args.wandb_project = 'hdb5_5cnn_filter_expts'  # 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'hdb5_5cnn {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def dtd_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'dtd'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'dtd'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    # ---first order config below--#
    # args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    # args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    # args.fo = True  # This is needed.
    # args.first_order = True

    # second order args
    args.first_order = False
    args.fo = False
    args.copy_initial_weights = False
    # args.track_higher_grads = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def cu_birds_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'cu_birds'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'cu_birds'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True/ckpt.pt'
    args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Apr12_01-45-57_jobid_-1_pid_6238_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 5e-4  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    # ---first order config below--#
    # args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    # args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    # args.fo = True  # This is needed.
    # args.first_order = True

    # second order args
    args.first_order = False
    args.fo = False
    args.copy_initial_weights = False#True
    #args.track_higher_grads = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def fc100_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'fc100'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'fc100'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar21_22-34-32_jobid_-1_pid_14347_wandb_True/ckpt.pt'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar25_12-55-11_jobid_-1_pid_10457_wandb_True/ckpt.pt'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-4  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    #args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = False  # This is needed.
    args.first_order = False

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def ti_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'ti'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb4_micod'
    #foMAML
    args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Apr22_21-28-49_jobid_-1_pid_8544_wandb_True/ckpt.pt'
    #hoMAML
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Apr21_00-26-22_jobid_-1_pid_83237_wandb_True/ckpt.pt'

    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-4  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # This is needed.
    args.first_order = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def omni_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'omni'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb4_micod'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar21_22-34-32_jobid_-1_pid_14347_wandb_True/ckpt.pt'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar25_12-55-11_jobid_-1_pid_10457_wandb_True/ckpt.pt'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-4  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    # args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = False  # This is needed.
    args.first_order = False

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args

def mi_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'mi'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb4_micod'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar21_22-34-32_jobid_-1_pid_14347_wandb_True/ckpt.pt'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar25_12-55-11_jobid_-1_pid_10457_wandb_True/ckpt.pt'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Apr20_23-31-02_jobid_-1_pid_77116_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Apr21_00-19-44_jobid_-1_pid_97456_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-4  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = False#True  # This is needed.
    args.first_order = False#True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def quickdraw_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'quickdraw'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb4_micod'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar21_22-34-32_jobid_-1_pid_14347_wandb_True/ckpt.pt'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar25_12-55-11_jobid_-1_pid_10457_wandb_True/ckpt.pt'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Apr20_23-31-02_jobid_-1_pid_77116_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Apr21_00-19-44_jobid_-1_pid_97456_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-4  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # This is needed.
    args.first_order = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def cifarfs_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'cifarfs'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb4_micod'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar21_22-34-32_jobid_-1_pid_14347_wandb_True/ckpt.pt'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar25_12-55-11_jobid_-1_pid_10457_wandb_True/ckpt.pt'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-4  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    # args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = False  # This is needed.
    args.first_order = False

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args

def delauny_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'delaunay'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb4_micod'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True/ckpt.pt'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar22_17-56-28_jobid_-1_pid_53421_wandb_True/ckpt.pt'
    args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Apr12_01-32-34_jobid_-1_pid_104807_wandb_True/ckpt.pt'
    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 5e-4#4  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5

    #---first order config below--#
    #args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    #args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    #args.fo = True  # This is needed.
    #args.first_order = True

    # second order args
    args.first_order = False
    args.fo = False
    args.copy_initial_weights = False
    # args.track_higher_grads = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args




def fungi_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'fungi'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'fungi'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # This is needed.
    args.first_order = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args



def aircraft_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'aircraft'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb5_vggair'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    # ---first order config below--#
    # args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    # args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    # args.fo = True  # This is needed.
    # args.first_order = True

    # second order args
    args.first_order = False
    args.fo = False
    args.copy_initial_weights = False
    # args.track_higher_grads = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args



def flower_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'flower'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb5_vggair'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    # ---first order config below--#
    # args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    # args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    # args.fo = True  # This is needed.
    # args.first_order = True

    # second order args
    args.first_order = False
    args.fo = False
    args.copy_initial_weights = False
    #args.track_higher_grads = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args



def hdb6_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'hdb6'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb4_micod'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True/ckpt.pt'
    args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar30_09-48-53_jobid_-1_pid_10180_wandb_True/ckpt.pt'
    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-4  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # This is needed.
    args.first_order = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args



def hdb7_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'hdb7'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb4_micod'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True/ckpt.pt'
    args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar30_11-41-49_jobid_-1_pid_18532_wandb_True/ckpt.pt'
    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-4  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # This is needed.
    args.first_order = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args



def hdb8_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'hdb8'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb4_micod'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Apr01_01-38-30_jobid_-1_pid_37233_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-4  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # This is needed.
    args.first_order = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args



def hdb9_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'hdb9'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb4_micod'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Apr01_01-38-34_jobid_-1_pid_59713_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-4  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # This is needed.
    args.first_order = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def hdb10_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'hdb10'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb4_micod'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Apr04_17-38-25_jobid_-1_pid_72413_wandb_True/ckpt.pt'#'/home/pzy2/data/logs/logs_Apr02_00-36-15_jobid_-1_pid_98057_wandb_True/ckpt.pt'

    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    #args.lr = 1e-4  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.lr = 1e-4
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # This is needed.
    args.first_order = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'innerlr_{args.inner_lr} {args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def hdb11_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet18_rfs'  # 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    from pathlib import Path
    # - model
    args.n_cls = 5
    # args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
    #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.wandb_entity = 'brando-uiuc'
    args.data_option = 'hdb11'
    args.data_path = '/home/pzy2/data/l2l_data/'
    args.data_augmentation = 'hdb4_micod'
    # - training mode
    # args.training_mode = 'iterations_train_convergence'
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Feb03_23-2 1-43_jobid_-1_pid_108167_wandb_True/ckpt.pt' #Continue 5CNNN
    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Apr04_17-38-25_jobid_-1_pid_72413_wandb_True/ckpt.pt'#'/home/pzy2/data/logs/logs_Apr02_00-36-15_jobid_-1_pid_98057_wandb_True/ckpt.pt'
    args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Apr18_17-32-52_jobid_-1_pid_22901_wandb_True/ckpt.pt'
    args.training_mode = 'iterations'
    args.num_its = 1_000_000_000
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
    #                               log_speed_up=10)
    #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                               metric_to_use='train_acc',
    #                               threshold=0.9, log_speed_up=10)

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
    args.scheduler_option = 'None'

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    #args.lr = 1e-4
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # This is needed.
    args.first_order = True

    # - outer trainer params
    args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2  # 1

    # - dist args
    args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20
    # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    # args.min_examples_in_class=0
    # args.num_support =None
    # args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'Meta-Dataset'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'innerlr_{args.inner_lr} {args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


#=====dataset hdb12======#
def hdb12_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
        # - model
        args.model_option = 'resnet12_rfs'  # 'resnet12_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
        from pathlib import Path
        # - model
        args.n_cls = 5
        # args.model_option = '5CNN_opt_as_model_for_few_shot'
        args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
        #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
        #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

        # - data
        args.wandb_entity = 'brando-uiuc'
        args.data_option = 'hdb12'
        args.data_path = '/home/pzy2/data/l2l_data/'
        args.data_augmentation = 'hdb4_micod'
        # - training mode
        # args.training_mode = 'iterations_train_convergence'
        #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'

        args.training_mode = 'iterations'
        args.num_its = 1_000_000_000
        # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
        #                               log_speed_up=10)
        #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
        #                               metric_to_use='train_acc',
        #                               threshold=0.9, log_speed_up=10)

        # - debug flag
        # args.debug = True
        args.debug = False

        # - opt
        args.opt_option = 'Adam_rfs_cifarfs'
        args.lr = 1e-3  # match MAML++
        args.opt_hps: dict = dict(lr=args.lr)

        # - scheduler
        # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
        args.scheduler_option = 'None'

        # -- Meta-Learner
        # - maml
        args.meta_learner_name = 'maml_fixed_inner_lr'
        #args.lr = 1e-4
        args.inner_lr = 1e-1
        args.nb_inner_train_steps = 5
        args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
        args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
        args.fo = True  # This is needed.
        args.first_order = True

        # - outer trainer params
        args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
        args.batch_size_eval = 2  # 1

        # - dist args
        args.world_size = torch.cuda.device_count()
        # args.world_size = 8
        args.parallel = args.world_size > 1
        args.seed = 42  # I think this might be important due to how tasksets works.
        args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
        # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

        # - logging params
        args.log_freq = 500
        # args.log_freq = 20
        # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
        # args.min_examples_in_class=0
        # args.num_support =None
        # args.num_query=None
        # args.log_freq = 20

        # -- wandb args
        # args.wandb_project = 'playground'  # needed to log to wandb properly
        args.wandb_entity = 'brando-uiuc'
        args.wandb_project = 'Meta-Dataset'
        # - wandb expt args
        args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
        args.run_name = f'innerlr_{args.inner_lr} {args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
        args.log_to_wandb = True
        # args.log_to_wandb = False

        # - fix for backwards compatibility
        args = fix_for_backwards_compatibility(args)
        return args
#=====end dataset hdb12======#
#=====dataset hdb13======#
def hdb13_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
        # - model
        args.model_option = 'resnet12_rfs'  # 'resnet12_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
        from pathlib import Path
        # - model
        args.n_cls = 5
        # args.model_option = '5CNN_opt_as_model_for_few_shot'
        args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
        #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
        #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

        # - data
        args.wandb_entity = 'brando-uiuc'
        args.data_option = 'hdb13'
        args.data_path = '/home/pzy2/data/l2l_data/'
        args.data_augmentation = 'hdb4_micod'
        # - training mode
        # args.training_mode = 'iterations_train_convergence'
        #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'

        args.training_mode = 'iterations'
        args.num_its = 1_000_000_000
        # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
        #                               log_speed_up=10)
        #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
        #                               metric_to_use='train_acc',
        #                               threshold=0.9, log_speed_up=10)

        # - debug flag
        # args.debug = True
        args.debug = False

        # - opt
        args.opt_option = 'Adam_rfs_cifarfs'
        args.lr = 1e-3  # match MAML++
        args.opt_hps: dict = dict(lr=args.lr)

        # - scheduler
        # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
        args.scheduler_option = 'None'

        # -- Meta-Learner
        # - maml
        args.meta_learner_name = 'maml_fixed_inner_lr'
        #args.lr = 1e-4
        args.inner_lr = 1e-1
        args.nb_inner_train_steps = 5
        args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
        args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
        args.fo = True  # This is needed.
        args.first_order = True

        # - outer trainer params
        args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
        args.batch_size_eval = 2  # 1

        # - dist args
        args.world_size = torch.cuda.device_count()
        # args.world_size = 8
        args.parallel = args.world_size > 1
        args.seed = 42  # I think this might be important due to how tasksets works.
        args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
        # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

        # - logging params
        args.log_freq = 500
        # args.log_freq = 20
        # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
        # args.min_examples_in_class=0
        # args.num_support =None
        # args.num_query=None
        # args.log_freq = 20

        # -- wandb args
        # args.wandb_project = 'playground'  # needed to log to wandb properly
        args.wandb_entity = 'brando-uiuc'
        args.wandb_project = 'Meta-Dataset'
        # - wandb expt args
        args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
        args.run_name = f'innerlr_{args.inner_lr} {args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
        args.log_to_wandb = True
        # args.log_to_wandb = False

        # - fix for backwards compatibility
        args = fix_for_backwards_compatibility(args)
        return args
#=====end dataset hdb13======#
#=====dataset hdb14======#
def hdb14_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
        # - model
        args.model_option = 'resnet12_rfs'  # 'resnet12_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
        from pathlib import Path
        # - model
        args.n_cls = 5
        # args.model_option = '5CNN_opt_as_model_for_few_shot'
        args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
        #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
        #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

        # - data
        args.wandb_entity = 'brando-uiuc'
        args.data_option = 'hdb14'
        args.data_path = '/home/pzy2/data/l2l_data/'
        args.data_augmentation = 'hdb4_micod'
        # - training mode
        # args.training_mode = 'iterations_train_convergence'
        #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'

        args.training_mode = 'iterations'
        args.num_its = 1_000_000_000
        # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
        #                               log_speed_up=10)
        #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
        #                               metric_to_use='train_acc',
        #                               threshold=0.9, log_speed_up=10)

        # - debug flag
        # args.debug = True
        args.debug = False

        # - opt
        args.opt_option = 'Adam_rfs_cifarfs'
        args.lr = 1e-3  # match MAML++
        args.opt_hps: dict = dict(lr=args.lr)

        # - scheduler
        # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
        args.scheduler_option = 'None'

        # -- Meta-Learner
        # - maml
        args.meta_learner_name = 'maml_fixed_inner_lr'
        #args.lr = 1e-4
        args.inner_lr = 1e-1
        args.nb_inner_train_steps = 5
        args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
        args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
        args.fo = True  # This is needed.
        args.first_order = True

        # - outer trainer params
        args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
        args.batch_size_eval = 2  # 1

        # - dist args
        args.world_size = torch.cuda.device_count()
        # args.world_size = 8
        args.parallel = args.world_size > 1
        args.seed = 42  # I think this might be important due to how tasksets works.
        args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
        # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

        # - logging params
        args.log_freq = 500
        # args.log_freq = 20
        # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
        # args.min_examples_in_class=0
        # args.num_support =None
        # args.num_query=None
        # args.log_freq = 20

        # -- wandb args
        # args.wandb_project = 'playground'  # needed to log to wandb properly
        args.wandb_entity = 'brando-uiuc'
        args.wandb_project = 'Meta-Dataset'
        # - wandb expt args
        args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
        args.run_name = f'innerlr_{args.inner_lr} {args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
        args.log_to_wandb = True
        # args.log_to_wandb = False

        # - fix for backwards compatibility
        args = fix_for_backwards_compatibility(args)
        return args
#=====end dataset hdb14======#
#=====dataset hdb15======#
def hdb15_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
        # - model
        args.model_option = 'resnet12_rfs'  # 'resnet12_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
        from pathlib import Path
        # - model
        args.n_cls = 5
        # args.model_option = '5CNN_opt_as_model_for_few_shot'
        args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
        #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
        #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

        # - data
        args.wandb_entity = 'brando-uiuc'
        args.data_option = 'hdb15'
        args.data_path = '/home/pzy2/data/l2l_data/'
        args.data_augmentation = 'hdb4_micod'
        # - training mode
        # args.training_mode = 'iterations_train_convergence'
        #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'

        args.training_mode = 'iterations'
        args.num_its = 1_000_000_000
        # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
        #                               log_speed_up=10)
        #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
        #                               metric_to_use='train_acc',
        #                               threshold=0.9, log_speed_up=10)

        # - debug flag
        # args.debug = True
        args.debug = False

        # - opt
        args.opt_option = 'Adam_rfs_cifarfs'
        args.lr = 1e-3  # match MAML++
        args.opt_hps: dict = dict(lr=args.lr)

        # - scheduler
        # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
        args.scheduler_option = 'None'

        # -- Meta-Learner
        # - maml
        args.meta_learner_name = 'maml_fixed_inner_lr'
        #args.lr = 1e-4
        args.inner_lr = 1e-1
        args.nb_inner_train_steps = 5
        args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
        args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
        args.fo = True  # This is needed.
        args.first_order = True

        # - outer trainer params
        args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
        args.batch_size_eval = 2  # 1

        # - dist args
        args.world_size = torch.cuda.device_count()
        # args.world_size = 8
        args.parallel = args.world_size > 1
        args.seed = 42  # I think this might be important due to how tasksets works.
        args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
        # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

        # - logging params
        args.log_freq = 500
        # args.log_freq = 20
        # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
        # args.min_examples_in_class=0
        # args.num_support =None
        # args.num_query=None
        # args.log_freq = 20

        # -- wandb args
        # args.wandb_project = 'playground'  # needed to log to wandb properly
        args.wandb_entity = 'brando-uiuc'
        args.wandb_project = 'Meta-Dataset'
        # - wandb expt args
        args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
        args.run_name = f'innerlr_{args.inner_lr} {args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
        args.log_to_wandb = True
        # args.log_to_wandb = False

        # - fix for backwards compatibility
        args = fix_for_backwards_compatibility(args)
        return args
#=====end dataset hdb15======#
#=====dataset hdb16======#
def hdb16_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
        # - model
        args.model_option = 'resnet12_rfs'  # 'resnet12_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
        from pathlib import Path
        # - model
        args.n_cls = 5
        # args.model_option = '5CNN_opt_as_model_for_few_shot'
        args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
        #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
        #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

        # - data
        args.wandb_entity = 'brando-uiuc'
        args.data_option = 'hdb16'
        args.data_path = '/home/pzy2/data/l2l_data/'
        args.data_augmentation = 'hdb4_micod'
        # - training mode
        # args.training_mode = 'iterations_train_convergence'
        #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'

        args.training_mode = 'iterations'
        args.num_its = 1_000_000_000
        # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
        #                               log_speed_up=10)
        #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
        #                               metric_to_use='train_acc',
        #                               threshold=0.9, log_speed_up=10)

        # - debug flag
        # args.debug = True
        args.debug = False

        # - opt
        args.opt_option = 'Adam_rfs_cifarfs'
        args.lr = 1e-3  # match MAML++
        args.opt_hps: dict = dict(lr=args.lr)

        # - scheduler
        # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
        args.scheduler_option = 'None'

        # -- Meta-Learner
        # - maml
        args.meta_learner_name = 'maml_fixed_inner_lr'
        #args.lr = 1e-4
        args.inner_lr = 1e-1
        args.nb_inner_train_steps = 5
        args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
        args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
        args.fo = True  # This is needed.
        args.first_order = True

        # - outer trainer params
        args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
        args.batch_size_eval = 2  # 1

        # - dist args
        args.world_size = torch.cuda.device_count()
        # args.world_size = 8
        args.parallel = args.world_size > 1
        args.seed = 42  # I think this might be important due to how tasksets works.
        args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
        # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

        # - logging params
        args.log_freq = 500
        # args.log_freq = 20
        # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
        # args.min_examples_in_class=0
        # args.num_support =None
        # args.num_query=None
        # args.log_freq = 20

        # -- wandb args
        # args.wandb_project = 'playground'  # needed to log to wandb properly
        args.wandb_entity = 'brando-uiuc'
        args.wandb_project = 'Meta-Dataset'
        # - wandb expt args
        args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
        args.run_name = f'innerlr_{args.inner_lr} {args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
        args.log_to_wandb = True
        # args.log_to_wandb = False

        # - fix for backwards compatibility
        args = fix_for_backwards_compatibility(args)
        return args
#=====end dataset hdb16======#
#=====dataset hdb17======#
def hdb17_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
        # - model
        args.model_option = 'resnet12_rfs'  # 'resnet12_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
        from pathlib import Path
        # - model
        args.n_cls = 5
        # args.model_option = '5CNN_opt_as_model_for_few_shot'
        args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
        #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
        #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

        # - data
        args.wandb_entity = 'brando-uiuc'
        args.data_option = 'hdb17'
        args.data_path = '/home/pzy2/data/l2l_data/'
        args.data_augmentation = 'hdb4_micod'
        # - training mode
        # args.training_mode = 'iterations_train_convergence'
        #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'

        args.training_mode = 'iterations'
        args.num_its = 1_000_000_000
        # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
        #                               log_speed_up=10)
        #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
        #                               metric_to_use='train_acc',
        #                               threshold=0.9, log_speed_up=10)

        # - debug flag
        # args.debug = True
        args.debug = False

        # - opt
        args.opt_option = 'Adam_rfs_cifarfs'
        args.lr = 1e-3  # match MAML++
        args.opt_hps: dict = dict(lr=args.lr)

        # - scheduler
        # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
        args.scheduler_option = 'None'

        # -- Meta-Learner
        # - maml
        args.meta_learner_name = 'maml_fixed_inner_lr'
        #args.lr = 1e-4
        args.inner_lr = 1e-1
        args.nb_inner_train_steps = 5
        args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
        args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
        args.fo = True  # This is needed.
        args.first_order = True

        # - outer trainer params
        args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
        args.batch_size_eval = 2  # 1

        # - dist args
        args.world_size = torch.cuda.device_count()
        # args.world_size = 8
        args.parallel = args.world_size > 1
        args.seed = 42  # I think this might be important due to how tasksets works.
        args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
        # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

        # - logging params
        args.log_freq = 500
        # args.log_freq = 20
        # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
        # args.min_examples_in_class=0
        # args.num_support =None
        # args.num_query=None
        # args.log_freq = 20

        # -- wandb args
        # args.wandb_project = 'playground'  # needed to log to wandb properly
        args.wandb_entity = 'brando-uiuc'
        args.wandb_project = 'Meta-Dataset'
        # - wandb expt args
        args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
        args.run_name = f'innerlr_{args.inner_lr} {args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
        args.log_to_wandb = True
        # args.log_to_wandb = False

        # - fix for backwards compatibility
        args = fix_for_backwards_compatibility(args)
        return args
#=====end dataset hdb17======#
#=====dataset hdb18======#
def hdb18_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
        # - model
        args.model_option = 'resnet12_rfs'  # 'resnet12_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
        from pathlib import Path
        # - model
        args.n_cls = 5
        # args.model_option = '5CNN_opt_as_model_for_few_shot'
        args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
        #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
        #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

        # - data
        args.wandb_entity = 'brando-uiuc'
        args.data_option = 'hdb18'
        args.data_path = '/home/pzy2/data/l2l_data/'
        args.data_augmentation = 'hdb4_micod'
        # - training mode
        # args.training_mode = 'iterations_train_convergence'
        #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'

        args.training_mode = 'iterations'
        args.num_its = 1_000_000_000
        # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
        #                               log_speed_up=10)
        #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
        #                               metric_to_use='train_acc',
        #                               threshold=0.9, log_speed_up=10)

        # - debug flag
        # args.debug = True
        args.debug = False

        # - opt
        args.opt_option = 'Adam_rfs_cifarfs'
        args.lr = 1e-3  # match MAML++
        args.opt_hps: dict = dict(lr=args.lr)

        # - scheduler
        # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
        args.scheduler_option = 'None'

        # -- Meta-Learner
        # - maml
        args.meta_learner_name = 'maml_fixed_inner_lr'
        #args.lr = 1e-4
        args.inner_lr = 1e-1
        args.nb_inner_train_steps = 5
        args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
        args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
        args.fo = True  # This is needed.
        args.first_order = True

        # - outer trainer params
        args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
        args.batch_size_eval = 2  # 1

        # - dist args
        args.world_size = torch.cuda.device_count()
        # args.world_size = 8
        args.parallel = args.world_size > 1
        args.seed = 42  # I think this might be important due to how tasksets works.
        args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
        # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

        # - logging params
        args.log_freq = 500
        # args.log_freq = 20
        # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
        # args.min_examples_in_class=0
        # args.num_support =None
        # args.num_query=None
        # args.log_freq = 20

        # -- wandb args
        # args.wandb_project = 'playground'  # needed to log to wandb properly
        args.wandb_entity = 'brando-uiuc'
        args.wandb_project = 'Meta-Dataset'
        # - wandb expt args
        args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
        args.run_name = f'innerlr_{args.inner_lr} {args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
        args.log_to_wandb = True
        # args.log_to_wandb = False

        # - fix for backwards compatibility
        args = fix_for_backwards_compatibility(args)
        return args
#=====end dataset hdb18======#
#=====dataset hdb19======#
def hdb19_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
        # - model
        args.model_option = 'resnet12_rfs'  # 'resnet12_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
        from pathlib import Path
        # - model
        args.n_cls = 5
        # args.model_option = '5CNN_opt_as_model_for_few_shot'
        args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
        #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
        #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

        # - data
        args.wandb_entity = 'brando-uiuc'
        args.data_option = 'hdb19'
        args.data_path = '/home/pzy2/data/l2l_data/'
        args.data_augmentation = 'hdb4_micod'
        # - training mode
        # args.training_mode = 'iterations_train_convergence'
        #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'

        args.training_mode = 'iterations'
        args.num_its = 1_000_000_000
        # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
        #                               log_speed_up=10)
        #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
        #                               metric_to_use='train_acc',
        #                               threshold=0.9, log_speed_up=10)

        # - debug flag
        # args.debug = True
        args.debug = False

        # - opt
        args.opt_option = 'Adam_rfs_cifarfs'
        args.lr = 1e-3  # match MAML++
        args.opt_hps: dict = dict(lr=args.lr)

        # - scheduler
        # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
        args.scheduler_option = 'None'

        # -- Meta-Learner
        # - maml
        args.meta_learner_name = 'maml_fixed_inner_lr'
        #args.lr = 1e-4
        args.inner_lr = 1e-1
        args.nb_inner_train_steps = 5
        args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
        args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
        args.fo = True  # This is needed.
        args.first_order = True

        # - outer trainer params
        args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
        args.batch_size_eval = 2  # 1

        # - dist args
        args.world_size = torch.cuda.device_count()
        # args.world_size = 8
        args.parallel = args.world_size > 1
        args.seed = 42  # I think this might be important due to how tasksets works.
        args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
        # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

        # - logging params
        args.log_freq = 500
        # args.log_freq = 20
        # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
        # args.min_examples_in_class=0
        # args.num_support =None
        # args.num_query=None
        # args.log_freq = 20

        # -- wandb args
        # args.wandb_project = 'playground'  # needed to log to wandb properly
        args.wandb_entity = 'brando-uiuc'
        args.wandb_project = 'Meta-Dataset'
        # - wandb expt args
        args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
        args.run_name = f'innerlr_{args.inner_lr} {args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
        args.log_to_wandb = True
        # args.log_to_wandb = False

        # - fix for backwards compatibility
        args = fix_for_backwards_compatibility(args)
        return args
#=====end dataset hdb19======#
#=====dataset hdb20======#
def hdb20_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
        # - model
        args.model_option = 'resnet12_rfs'  # 'resnet12_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
        from pathlib import Path
        # - model
        args.n_cls = 5
        # args.model_option = '5CNN_opt_as_model_for_few_shot'
        args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
        #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
        #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

        # - data
        args.wandb_entity = 'brando-uiuc'
        args.data_option = 'hdb20'
        args.data_path = '/home/pzy2/data/l2l_data/'
        args.data_augmentation = 'hdb4_micod'
        # - training mode
        # args.training_mode = 'iterations_train_convergence'
        #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'

        args.training_mode = 'iterations'
        args.num_its = 1_000_000_000
        # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
        #                               log_speed_up=10)
        #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
        #                               metric_to_use='train_acc',
        #                               threshold=0.9, log_speed_up=10)

        # - debug flag
        # args.debug = True
        args.debug = False

        # - opt
        args.opt_option = 'Adam_rfs_cifarfs'
        args.lr = 1e-3  # match MAML++
        args.opt_hps: dict = dict(lr=args.lr)

        # - scheduler
        # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
        args.scheduler_option = 'None'

        # -- Meta-Learner
        # - maml
        args.meta_learner_name = 'maml_fixed_inner_lr'
        #args.lr = 1e-4
        args.inner_lr = 1e-1
        args.nb_inner_train_steps = 5
        args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
        args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
        args.fo = True  # This is needed.
        args.first_order = True

        # - outer trainer params
        args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
        args.batch_size_eval = 2  # 1

        # - dist args
        args.world_size = torch.cuda.device_count()
        # args.world_size = 8
        args.parallel = args.world_size > 1
        args.seed = 42  # I think this might be important due to how tasksets works.
        args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
        # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

        # - logging params
        args.log_freq = 500
        # args.log_freq = 20
        # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
        # args.min_examples_in_class=0
        # args.num_support =None
        # args.num_query=None
        # args.log_freq = 20

        # -- wandb args
        # args.wandb_project = 'playground'  # needed to log to wandb properly
        args.wandb_entity = 'brando-uiuc'
        args.wandb_project = 'Meta-Dataset'
        # - wandb expt args
        args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
        args.run_name = f'innerlr_{args.inner_lr} {args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
        args.log_to_wandb = True
        # args.log_to_wandb = False

        # - fix for backwards compatibility
        args = fix_for_backwards_compatibility(args)
        return args
#=====end dataset hdb20======#
#=====dataset hdb21======#
def hdb21_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
        # - model
        args.model_option = 'resnet12_rfs'  # 'resnet12_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
        from pathlib import Path
        # - model
        args.n_cls = 5
        # args.model_option = '5CNN_opt_as_model_for_few_shot'
        args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
        #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
        #                     filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

        # - data
        args.wandb_entity = 'brando-uiuc'
        args.data_option = 'hdb21'
        args.data_path = '/home/pzy2/data/l2l_data/'
        args.data_augmentation = 'hdb4_micod'
        # - training mode
        # args.training_mode = 'iterations_train_convergence'
        #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Mar07_18-58-29_jobid_-1_pid_7122_wandb_True/ckpt.pt'

        args.training_mode = 'iterations'
        args.num_its = 1_000_000_000
        # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_convg_reached', metric_to_use='train_loss',
        #                               log_speed_up=10)
        #args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
        #                               metric_to_use='train_acc',
        #                               threshold=0.9, log_speed_up=10)

        # - debug flag
        # args.debug = True
        args.debug = False

        # - opt
        args.opt_option = 'Adam_rfs_cifarfs'
        args.lr = 1e-3  # match MAML++
        args.opt_hps: dict = dict(lr=args.lr)

        # - scheduler
        # no scheduler since we don't know how many steps to do we can't know how to decay with prev code, maybe something else exists e.g. decay once error is low enough
        args.scheduler_option = 'None'

        # -- Meta-Learner
        # - maml
        args.meta_learner_name = 'maml_fixed_inner_lr'
        #args.lr = 1e-4
        args.inner_lr = 1e-1
        args.nb_inner_train_steps = 5
        args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
        args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
        args.fo = True  # This is needed.
        args.first_order = True

        # - outer trainer params
        args.batch_size = 4  # 1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
        args.batch_size_eval = 2  # 1

        # - dist args
        args.world_size = torch.cuda.device_count()
        # args.world_size = 8
        args.parallel = args.world_size > 1
        args.seed = 42  # I think this might be important due to how tasksets works.
        args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
        # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
        args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

        # - logging params
        args.log_freq = 500
        # args.log_freq = 20
        # args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
        # args.min_examples_in_class=0
        # args.num_support =None
        # args.num_query=None
        # args.log_freq = 20

        # -- wandb args
        # args.wandb_project = 'playground'  # needed to log to wandb properly
        args.wandb_entity = 'brando-uiuc'
        args.wandb_project = 'Meta-Dataset'
        # - wandb expt args
        args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
        args.run_name = f'innerlr_{args.inner_lr} {args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
        args.log_to_wandb = True
        # args.log_to_wandb = False

        # - fix for backwards compatibility
        args = fix_for_backwards_compatibility(args)
        return args
#=====end dataset hdb21======#

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
    # this dist script sets up rank dist etc
    # if is_running_parallel(args.rank):
    if args.parallel:
        # this might be set up cleaner if we setup args in this function and not in main train, but don't want to deal with organizing l2l vs ddp vs etc, just in their own main they can setup args
        # - set up processes a la l2l
        local_rank: int = get_local_rank()
        print(f'{local_rank=}')
        init_process_group_l2l(args, local_rank=local_rank, world_size=args.world_size, init_method=args.init_method)
        rank: int = torch.distributed.get_rank() if is_running_parallel(local_rank) else -1
        args.rank = rank  # have each process save the rank
        set_devices_and_seed_ala_l2l(args)  # args.device = rank or .device
    else:  # serial
        set_devices(args)  # args.device = rank or .device
    print(f'setup process done for rank={args.rank}, device={args.device}')

    # - set up wandb only for the lead process. # this might be set up cleaner if we setup args in this function and not in main train
    print('setting up wandb')
    setup_wandb(args) if is_lead_worker(args.rank) else None

    # create the model, opt & scheduler
    print('creating model, opt, scheduler')
    get_and_create_model_opt_scheduler_for_run(args)
    print(f'got model {type(args.model)=}')
    args.number_of_trainable_parameters = count_number_of_parameters(args.model)
    print(f'{args.number_of_trainable_parameters=}')
    print(f'---> {args.number_of_trainable_parameters=}')
    args.opt = move_opt_to_cherry_opt_and_sync_params(args) if is_running_parallel(args.rank) else args.opt

    # create the loaders, note: you might have to change the number of layers in the final layer
    print('creating dataloaders')
    args.dataloaders: BenchmarkTasksets = get_l2l_tasksets(args)
    assert args.model.cls.out_features == 5
    # assert args.model.classifier.out_features == 5

    # Agent does everything, proving, training, evaluate, meta-learnering, etc.
    print('creating agent')
    args.agent = MAMLMetaLearnerL2L(args, args.model)
    args.meta_learner = args.agent
    print(f'{type(args.agent)=}, {type(args.meta_learner)=}')

    # -- Start Training Loop
    print_dist(f"{args.model=}\n{args.opt=}\n{args.scheduler=}\n{type(args.agent)=}", args.rank)  # here to make sure mdl has the right cls
    print_dist('\n\n====> about to start train loop', args.rank)
    print(f'{args.filter_size=}') if hasattr(args, 'filter_size') else None
    print(f'{args.number_of_trainable_parameters=}')
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
    print(f'\n----> about to cleanup worker with rank {args.rank}')
    cleanup(args.rank)
    print(f'clean up done successfully! {args.rank}')
    from uutils.logging_uu.wandb_logging.common import cleanup_wandb
    # cleanup_wandb(args, delete_wandb_dir=True)
    cleanup_wandb(args, delete_wandb_dir=False)


# -- Run experiment

if __name__ == "__main__":
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    main()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
