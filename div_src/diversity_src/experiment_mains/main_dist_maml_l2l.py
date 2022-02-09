"""
Main script to set up supervised learning experiments
"""
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
    is_lead_worker, find_free_port, is_running_serially, is_running_parallel
from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
from uutils.torch_uu.mains.main_sl_with_ddp import train
from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner, MAMLMetaLearnerL2L
from uutils.torch_uu.training.meta_training import meta_train_fixed_iterations, meta_train_agent_fit_single_batch, \
    meta_train_iterations_ala_l2l

from pdb import set_trace as st

from uutils.torch_uu.training.supervised_learning import train_agent_fit_single_batch


def l2l_4CNNl2l_cifarfs_rfs_adam_cl_70k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.model_option = '4CNN_l2l_cifarfs'

    # - data
    args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'rfs2020'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 70_000

    # - debug flag
    args.debug = True
    # args.debug = False

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
    """
python -m torch.distributed.run --nproc_per_node=1 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k
    """
    # args.world_size = torch.cuda.device_count()
    args.world_size = 1
    args.parallel = True
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
    # args.run_name = f'debug: {args.jobid=}'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    args.model_option = 'resnet12_rfs_cifarfs_fc100'

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
    args.parallel = True
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
    # args.run_name = f'debug: {args.jobid=}'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


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
    args.manual_loads_name = 'l2l_4CNNl2l_cifarfs_rfs_adam_cl_70k'  # <- REMOVE to remove manual loads

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    if resume_from_checkpoint(args):
        args: Namespace = make_args_from_supervised_learning_checkpoint(args=args, precedence_to_args_checkpoint=True)
    elif args_hardcoded_in_script(args):
        if args.manual_loads_name == 'l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k':
            args: Namespace = l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k(args)
        elif args.manual_loads_name == 'l2l_4CNNl2l_cifarfs_rfs_adam_cl_70k':
            args: Namespace = l2l_4CNNl2l_cifarfs_rfs_adam_cl_70k(args)
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
        train(args=args)
    else:
        # mp.spawn(fn=train, args=(args,), nprocs=args.world_size) what ddp does
        train(args=args)


def get_local_rank() -> int:
    try:
        local_rank: int = int(os.environ["LOCAL_RANK"])
    except:
        local_rank: int = -1
    return local_rank


def train(args):
    # # set_sharing_strategy()
    local_rank: int = get_local_rank()
    print(f'{local_rank=}')
    init_process_group_l2l(args, local_rank=local_rank, world_size=args.world_size, init_method=args.init_method)
    rank: int = torch.distributed.get_rank() if is_running_parallel(local_rank) else -1
    args.rank = rank  # have each process save the rank
    set_devices_and_seed_ala_l2l(args)  # args.device = rank or .device
    print(f'setup process done for rank={rank}')

    # - set up wandb only for the lead process
    setup_wandb(args) if is_lead_worker(args.rank) else None

    # create the model, opt & scheduler
    get_and_create_model_opt_scheduler_for_run(args)
    args.opt = move_opt_to_cherry_opt_and_sync_params(args)
    print_dist(f"{args.model=}\n{args.opt=}\n{args.scheduler=}", args.rank)

    # create the dataloaders, this goes first so you can select the mdl (e.g. final layer) based on task
    args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)
    args.dataloaders = args.tasksets  # for the sake that eval_sl can detect how to get examples for eval

    # Agent does everything, proving, training, evaluate, meta-learnering, etc.
    args.agent = MAMLMetaLearnerL2L(args, args.model)
    args.meta_learner = args.agent

    # -- Start Training Loop
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