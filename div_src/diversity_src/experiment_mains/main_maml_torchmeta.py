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
from uutils.argparse_uu.meta_learning import parse_args_meta_learning, fix_for_backwards_compatibility
from uutils.argparse_uu.supervised_learning import make_args_from_supervised_learning_checkpoint, parse_args_standard_sl
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloader
from uutils.torch_uu.distributed import set_sharing_strategy, print_process_info, set_devices, setup_process, cleanup, \
    print_dist
from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
from uutils.torch_uu.mains.main_sl_with_ddp import train
from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner
from uutils.torch_uu.training.meta_training import meta_train_fixed_iterations, meta_train_agent_fit_single_batch

from pdb import set_trace as st

from uutils.torch_uu.training.supervised_learning import train_agent_fit_single_batch


def manual_load_mi_5cnn_fo_maml(args: Namespace) -> Namespace:
    from pathlib import Path
    # - model
    args.n_cls = 5
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls, filter_size=32,
                          levels=None, spp=False, in_channels=3)

    # - data
    args.data_option = 'torchmeta_miniimagenet'
    args.data_path = Path('~/data/torchmeta_data/').expanduser()

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
    args.opt_hps: dict = dict(lr=args.lr)

    # args.scheduler_option = 'None'
    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 2_000
    args.T_max = args.num_its // args.log_scheduler_freq  # intended 800K/2k
    args.eta_min = 1e-5  # match MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)
    # assert args.T_max == 400, f'T_max is not expected value, instead it is: {args.T_max=}'

    # -- Meta-Learner
    # - maml with higher lib
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # always True, see: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # real flag that control FO maml

    # - outer trainer params
    args.batch_size = 4
    args.batch_size_eval = 2

    # - dist args
    # args.world_size = torch.cuda.device_count()
    # args.world_size = 1
    # args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    # args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'manual_load_mi_5cnn_fo_maml'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def pat_load_mds_net_fo_maml(args: Namespace) -> Namespace:
    from pathlib import Path
    # - model, TODO, choose a reasonable mdl, the one MDS uses? or our Resnet12? idk
    args.n_cls = 5  # TODO, variable?
    args.model_option = ''
    args.model_hps = dict(image_size='TODO', n_classes=args.n_cls)  # TODO

    # - data
    args.data_option = 'meta_dataset_mds'  # TODO
    args.data_path = Path('~/data/meta_dataset_mds/').expanduser()  #TODO

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 70_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt, TODO, idk use whatever optimizer & scheduler mds uses
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
    # - maml with higher lib, TODO, don't change this, you can remove this todo after reading it
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # always True, see: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # real flag that control FO maml

    # - outer trainer params
    args.batch_size = 4
    args.batch_size_eval = 2

    # - dist args
    # args.world_size = torch.cuda.device_count()
    # args.world_size = 1
    # args.parallel = args.world_size > 1
    args.seed = 42  # I think this might be important due to how tasksets works.
    # args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'pat_load_mds_net_fo_maml'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args



# ---


def load_args() -> Namespace:
    """
    1. parse args from user's terminal
    2. optionally set remaining args values (e.g. manually, hardcoded, from ckpt etc.)
    3. setup remaining args small details from previous values (e.g. 1 and 2).
    """
    # -- parse args from terminal
    # args: Namespace = parse_args_standard_sl()
    args: Namespace = parse_args_meta_learning()
    args.args_hardcoded_in_script = True  # <- REMOVE to remove manual loads TODO remove this if you are passing flag for run config/args through the terminal
    args.manual_loads_name = 'pat_load_mds_net_fo_maml'  # <- REMOVE to remove manual loads, TODO remove this if you are passing flag for run config/args through the terminal

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    if resume_from_checkpoint(args):
        args: Namespace = make_args_from_supervised_learning_checkpoint(args=args, precedence_to_args_checkpoint=True)
    elif args_hardcoded_in_script(args):
        if args.manual_loads_name == 'manual_load_mi_5cnn_fo_maml':
            args: Namespace = manual_load_mi_5cnn_fo_maml(args)
        elif args.manual_loads_name == 'pat_load_mds_net_fo_maml':
            args: Namespace = pat_load_mds_net_fo_maml(args)
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
        # args.world_size = torch.cuda.device_count()
        # args.world_size = mp.cpu_count() - 1  # 1 process is main, the rest are (parallel) trainers
        # set_sharing_strategy()
        # mp.spawn(fn=train, args=(args,), nprocs=args.world_size)
        raise NotImplementedError


def train(rank, args):
    print_process_info(rank, flush=True)
    args.rank = rank  # have each process save the rank
    set_devices(args)  # args.device = rank or .device
    setup_process(args, rank, master_port=args.master_port, world_size=args.world_size)
    print(f'setup process done for rank={rank}')

    # create the (ddp) model, opt & scheduler
    get_and_create_model_opt_scheduler_for_run(args)
    print_dist(f"{args.model=}\n{args.opt=}\n{args.scheduler=}", args.rank)

    # create the dataloaders, this goes first so you can select the mdl (e.g. final layer) based on task
    args.dataloaders: dict = get_meta_learning_dataloader(args)
    # assert args.model.cls.out_features == 5  # TODO, I recommend we have asserts checking the right cls layer, usually for fixed nway I just check it's 5, but we could have assert cls == 5 or cls == varied or something? Nothing is ok too
    # assert args.model.classifier.out_features == 5

    # Agent does everything, proving, training, evaluate, meta-learnering, etc.
    args.agent = MAMLMetaLearner(args, args.model)
    args.meta_learner = args.agent

    # -- Start Training Loop
    print_dist('====> about to start train loop', args.rank)
    if args.training_mode == 'meta_train_agent_fit_single_batch':
        meta_train_agent_fit_single_batch(args, args.agent, args.dataloaders, args.opt, args.scheduler)
    elif 'iterations' in args.training_mode:
        meta_train_fixed_iterations(args, args.agent, args.dataloaders, args.opt, args.scheduler)
        # note train code will see training mode to determine halting criterion
        # meta_train_iterations(args, agent, args.dataloaders, args.opt, args.scheduler) not implemented
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
