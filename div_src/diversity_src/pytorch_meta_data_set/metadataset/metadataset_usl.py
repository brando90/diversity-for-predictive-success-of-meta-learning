"""
Main script to set up supervised learning experiments
"""


import torch
import torch.multiprocessing as mp

from argparse import Namespace
from pathlib import Path
from typing import Optional


from uutils import args_hardcoded_in_script, report_times
from uutils.argparse_uu.common import setup_args_for_experiment
from uutils.argparse_uu.supervised_learning import make_args_from_supervised_learning_checkpoint, parse_args_standard_sl
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.agents.supervised_learning import ClassificationSLAgent, UnionClsSLAgent
from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
#from uutils.torch_uu.dataloaders.helpers import get_sl_dataloader
from uutils.torch_uu.distributed import set_sharing_strategy, print_process_info, set_devices, setup_process, cleanup, \
    print_dist
from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
from uutils.torch_uu.training.supervised_learning import train_agent_fit_single_batch, train_agent_iterations, \
    train_agent_epochs

from uutils import load_cluster_jobids_to, merge_args
from torch import nn

from diversity_src.dataloaders.metadataset_batch_loader import get_mds_batch_args, get_mds_loader



def manual_load(args: Namespace) -> Namespace:
    """
    Warning: hardcoding the args can make it harder to reproduce later in a main.sh script with the
    arguments to the experiment.
    """
    raise ValueError(f'Not implemented')

#follow hdb1 https://github.com/brando90/diversity-for-predictive-success-of-meta-learning/blob/c76f1df5c3afaa278079c27703a38b25ac5d4c1d/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py
def load_args() -> Namespace:
    """
    1. parse args from user's terminal
    2. optionally set remaining args values (e.g. manually, hardcoded, from ckpt etc.)
    3. setup remaining args small details from previous values (e.g. 1 and 2).
    """
    # -- parse args from terminal
    args: Namespace = get_mds_batch_args()
    args.data_option = 'MDS'
    args.args_hardcoded_in_script = True  # <- REMOVE to remove manual loads
    # args.manual_loads_name = 'resnet12_rfs_cifarfs'  # <- REMOVE to remove manual loads
    # args.manual_loads_name = 'manual_load_cifarfs_resnet12rfs_train_until_convergence'  # <- REMOVE to remove manual loads
    args.model_option = 'resnet12_rfs'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_classes)

    args.opt_option = 'Adam_rfs_cifarfs'
    args.num_epochs = 1000
    args.batch_size = 32
    args.batch_size_eval = 32

    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)
    #args.model_hps = {'num_classes': 3144}

    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_epochs // args.log_scheduler_freq
    args.eta_min = 1e-5  # coincidentally, matches MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

    # - training mode
    args.training_mode = 'epochs'
    # args.training_mode = 'fit_single_batc

    args.debug=True

    args.log_freq = 1

    args.experiment_name = f'MDS USL'
    # args.run_name = f'debug: {args.jobid=}'
    args.run_name = f'all datasets {args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    # args.log_to_wandb = True
    args.log_to_wandb = True

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    '''f
    if resume_from_checkpoint(args):
        args: Namespace = make_args_from_supervised_learning_checkpoint(args=args, precedence_to_args_checkpoint=True)
    elif args_hardcoded_in_script(args):
        if args.manual_loads_name == 'resnet12_rfs_cifarfs':
            args: Namespace = manual_load_cifarfs_resnet12rfs(args)
        elif args.manual_loads_name == 'manual_load_cifarfs_resnet12rfs_train_until_convergence':
            args: Namespace = manual_load_cifarfs_resnet12rfs_train_until_convergence(args)
        else:
            raise ValueError(f'Invalid value, got: {args.manual_loads_name=}')
    else:
        # NOP: since we are using args from terminal
        pass
    '''
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
    #args.model_hps = {'n_classes' : 4934}
    get_and_create_model_opt_scheduler_for_run(args)
    #args.model.cls.out_features = 4934
    print_dist(f"{args.model=}\n{args.opt=}\n{args.scheduler=}", args.rank)

    args.dataloaders = get_mds_loader(args)

    # Agent does everything, proving, training, evaluate etc.
    args.agent: Agent = UnionClsSLAgent(args, args.model)

    # -- Start Training Loop
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