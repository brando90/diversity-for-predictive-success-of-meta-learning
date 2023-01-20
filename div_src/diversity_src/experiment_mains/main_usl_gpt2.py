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
from uutils.torch_uu.agents.supervised_learning import ClassificationSLAgent, UnionClsSLAgent, GPT2SLAgent
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


def load_args() -> Namespace:
    """
    1. parse args from user's terminal
    2. optionally set remaining args values (e.g. manually, hardcoded, from ckpt etc.)
    3. setup remaining args small details from previous values (e.g. 1 and 2).
    """
    # -- parse args from terminal
    args: Namespace = parse_args_standard_sl()
    args.model_option = 'gpt2'
    args.manual_loads_name = 'usl_gpt2_webtext2'
    args.args_hardcoded_in_script = True  # <- REMOVE to remove manual loads

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    if resume_from_checkpoint(args):
        args: Namespace = make_args_from_supervised_learning_checkpoint(args=args, precedence_to_args_checkpoint=True)
    elif args_hardcoded_in_script(args):
        from pathlib import Path
        # - model
        # args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64 + 1100,
        #                       filter_size=args.filter_size,
        #                       levels=None,
        #                       spp=False, in_channels=3)

        args.model_hps = {}

        # - data
        args.data_option = 'webtext'
        args.data_path = Path('~/data/webtext/').expanduser()

        # - opt
        args.opt_option = 'Adam_rfs_cifarfs'
        args.num_epochs = 1000
        args.lr = 1e-3
        args.opt_hps: dict = dict(lr=args.lr)

        args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
        args.log_scheduler_freq = 1
        args.T_max = args.num_epochs // args.log_scheduler_freq
        args.eta_min = 1e-5  # coincidentally, matches MAML++
        args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)

        # - training mode
        args.training_mode = 'epochs'

        # -
        args.debug = True
        # args.debug = False

        # -
        args.log_freq = 1  # for SL it is meant to be small e.g. 1 or 2

        # - wandb args
        args.wandb_project = 'entire-diversity-spectrum'
        # - wandb expt args
        args.experiment_name = f'usl_gpt2_webtext2'
        args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=} {gethostname()}'
        args.log_to_wandb = True
        # args.log_to_wandb = False
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
    # assert args.model.cls.out_features > 5, f'Not meta-learning training, so always more than 5 classes but got {args.model.cls.out_features=}'
    # assert args.model.cls.out_features == 64  # mi (cifar-fs?)
    # assert args.model.cls.out_features == 64 + 1100, f'hdb1 expects more classes but got {args.model.cls.out_features=},' \
    #                                                  f'\nfor model {type(args.model)=}'  # hdb1

    # Agent does everything, proving, training, evaluate etc.
    args.agent: Agent = GPT2SLAgent(args, args.model)

    # -- Start Training Loop
    print_dist(f"{args.model=}\n{args.opt=}\n{args.scheduler=}", args.rank)  # here to make sure mdl has the right cls
    print_dist('====> about to start train loop', args.rank)
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


if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    main()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")