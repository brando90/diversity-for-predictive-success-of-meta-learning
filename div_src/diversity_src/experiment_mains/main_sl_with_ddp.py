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
from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.dataloaders.helpers import get_sl_dataloader
from uutils.torch_uu.distributed import set_sharing_strategy, print_process_info, set_devices, setup_process, cleanup, \
    print_dist
# from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_first_time
# from uutils.torch_uu.training.supervised_learning import train_agent_fit_single_batch, train_agent_iterations, \
#     train_agent_epochs
from uutils.torch_uu.mains.main_sl_with_ddp import train


def manual_load_cifar100_resnet12rfs(args) -> Namespace:
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
    args.model_option = 'resnet12_rfs'

    # - data
    # args.path_to_data_set = Path('~/data/miniImageNet_rfs/miniImageNet/').expanduser()
    args.path_to_data_set = Path('~/data/CIFAR-FS/').expanduser()

    # - opt
    # args.opt_option = 'AdafactorDefaultFair'

    #
    # args.training_mode = 'epochs_train_convergence'
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
    args.manual_loads_name = 'cifar100_resnet12rfs'  # <- REMOVE to remove manual loads
    # -
    args.wandb_project = 'playground'  # needed to log to wandb properly

    # - debug args
    args.experiment_name = f'debug'
    args.run_name = f'debug (Adafactor) : {args.jobid=}'
    args.force_log = True
    # args.log_to_wandb = True
    args.log_to_wandb = False

    # - real args
    # args.experiment_name = f'Real experiment name (Real)'
    # args.run_name = f'Real experiment run name: {args.jobid=}'
    # args.force_log = False
    # args.log_to_wandb = True
    # #args.log_to_wandb = False

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    if resume_from_checkpoint(args):
        args: Namespace = make_args_from_supervised_learning_checkpoint(args=args, precedence_to_args_checkpoint=True)
    elif args_hardcoded_in_script(args):
        if args.manual_loads_name == 'cifar100_resnet12rfs':
            args: Namespace = manual_load_cifar100_resnet12rfs(args)
        else:
            raise NotImplementedError
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


# -- Run experiment

if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    main()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
