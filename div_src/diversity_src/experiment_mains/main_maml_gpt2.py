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

import uutils
from uutils import args_hardcoded_in_script, report_times
from uutils.argparse_uu.common import setup_args_for_experiment, setup_wandb
from uutils.argparse_uu.meta_learning import parse_args_meta_learning, fix_for_backwards_compatibility
from uutils.argparse_uu.supervised_learning import make_args_from_supervised_learning_checkpoint, parse_args_standard_sl
from uutils.logging_uu.wandb_logging.common import cleanup_wandb
from uutils.torch_uu import count_number_of_parameters
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.dataloaders.helpers import get_sl_dataloader
from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
from uutils.torch_uu.distributed import set_sharing_strategy, print_process_info, set_devices, setup_process, cleanup, \
    print_dist, move_opt_to_cherry_opt_and_sync_params, set_devices_and_seed_ala_l2l, init_process_group_l2l, \
    is_lead_worker, find_free_port, is_running_serially, is_running_parallel, get_local_rank
from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
from uutils.torch_uu.mains.main_sl_with_ddp import train
from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner, MAMLMetaLearnerL2L
from uutils.torch_uu.meta_learners.gpt2_meta_learner import GPTMetaLearnerL2L
from uutils.torch_uu.training.meta_training import meta_train_fixed_iterations, meta_train_agent_fit_single_batch, \
    meta_train_iterations_ala_l2l, meta_train_gpt2

from pdb import set_trace as st

from uutils.torch_uu.training.supervised_learning import train_agent_fit_single_batch


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
    args.model_option = 'gpt2'
    args.manual_loads_name = 'maml_gpt2_webtext2'
    args.args_hardcoded_in_script = True  # <- REMOVE to remove manual loads

    # - data
    from pathlib import Path
    args.data_option = 'webtext'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/webtext/').expanduser()
    args.data_augmentation = 'lee2019'
    args.log_root = '/dfs/scratch0/saumg/gpt_logs/maml_logs'


    if resume_from_checkpoint(args):
        args: Namespace = make_args_from_supervised_learning_checkpoint(args=args, precedence_to_args_checkpoint=False)
    else:

        # args.model_hps = dict(block_size = 64, vocab_size = 50257, n_layer = 12, n_head = 12, n_embd = 768, dropout = 0.0)
        args.model_hps = dict(block_size = 32, vocab_size = 50257, n_layer = 4, n_head = 4, n_embd = 192, dropout = 0.0)

        # - training mode
        args.training_mode = 'iterations_train_convergence'
        args.train_convergence_patience = 100000

        args.num_its = 100000

        # - debug flag
        # args.debug = True
        args.debug = False

        # - opt
        # args.opt_option = 'Adam_default'
        # args.opt_option = 'adamw_gpt'
        args.opt_option = 'AdafactorDefaultFair'
        # args.lr = 5e-3
        args.weight_decay = 1e-2
        # args.opt_hps: dict = dict(learning_rate=args.lr, weight_decay = args.weight_decay, betas = (0.9, 0.95))
        # args.opt_hps: dict = dict(lr=args.lr, weight_decay = args.weight_decay)

        # args.scheduler_option = 'None'
        args.scheduler_option = 'AdafactorSchedule'
        # args.log_scheduler_freq = 1
        # args.T_max = args.num_its // args.log_scheduler_freq  # intended 800K/2k
        # args.eta_min = 1e-5  # match MAML++
        # args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)
        # assert args.T_max == 400, f'T_max is not expected value, instead it is: {args.T_max=}'

        # -- Meta-Learner
        # - maml
        args.meta_learner_name = 'maml_fixed_inner_lr'
        args.inner_lr = 1e-3  # same as fast_lr in l2l
        args.nb_inner_train_steps = 5
        # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
        args.first_order = True
        args.k_shots = 1
        args.k_eval = 1
        args.n_cls = 4

        # - outer trainer params
        # args.batch_size = 8

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
        args.log_freq = 100
        args.ckpt_freq = 500

        # -- wandb args
        args.wandb_project = 'entire-diversity-spectrum'
        # - wandb expt args
        args.experiment_name = f'maml_gpt2_adafactor'
        args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
        args.log_to_wandb = True
        args.wandb_entity = 'saumyagoyal01'
        # args.log_to_wandb = False
        # args.dir_wandb = Path('/shared/rsaas/miranda9/data/logs/')

        # - fix for backwards compatibility
        args = fix_for_backwards_compatibility(args)


        # args.manual_loads_name = 'l2l_resnet12rfs_hdb1_100k'  # <- REMOVE to remove manual loads
        # args.manual_loads_name = 'l2l_resnet12rfs_mi_rfs_adam_cl_100k'  # <- REMOVE to remove manual loads

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    print(f'{args.manual_loads_name=}')

    [print(f'{k, v}') for k, v in vars(args).items()]
    
    # -- Setup up remaining stuff for experiment
    args: Namespace = setup_args_for_experiment(args)
    args.best_val_loss = 0
    return args


def main():
    """
    Note: end-to-end ddp example on mnist: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
    :return:
    """
    # - load the args from either terminal, ckpt, manual etc.
    args: Namespace = load_args()

    if resume_from_checkpoint(args) and 'eval' in args.training_mode:
        print("calling eval...")
        eval(args)
        return

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
    if args.opt_option == 'adamw_gpt' and args.scheduler_option == 'None':
        # Use custom scheduler
        args.scheduler_hps: dict = dict(warmup_iters = 2000,
                lr_decay_iters = 600000, # should be ~= max_iters per Chinchilla
                min_lr = 6e-5) # minimum learning rate, should be ~= learning_rate/10 per Chinchilla)
    args.number_of_trainable_parameters = count_number_of_parameters(args.model)
    args.opt = move_opt_to_cherry_opt_and_sync_params(args) if is_running_parallel(args.rank) else args.opt

    # dont use a separate l2l taskset, modifications made in training algorithms

    # # create the loaders, note: you might have to change the number of layers in the final layer
    # args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)
    # args.dataloaders = args.tasksets  # for the sake that eval_sl can detect how to get examples for eval
    args.dataloaders: dict = get_sl_dataloader(args)

    # Agent does everything, proving, training, evaluate, meta-learnering, etc.
    args.agent = GPTMetaLearnerL2L(args, args.model)
    args.meta_learner = args.agent

    # -- Start Training Loop
    print_dist(f"{args.model=}\n{args.opt=}\n{args.scheduler=}", args.rank)  # here to make sure mdl has the right cls
    print_dist('\n\n====> about to start train loop', args.rank)
    print(f'{args.number_of_trainable_parameters=}')
    if args.training_mode == 'meta_train_agent_fit_single_batch':
        # meta_train_agent_fit_single_batch(args, args.agent, args.dataloaders, args.opt, args.scheduler)
        raise NotImplementedError
    elif 'iterations' in args.training_mode:
        # meta_train_iterations_ala_l2l(args, args.agent, args.opt, args.scheduler)
        meta_train_gpt2(args, args.agent, args.dataloaders, args.opt, args.scheduler)
    elif 'epochs' in args.training_mode:
        # meta_train_epochs(args, agent, args.dataloaders, args.opt, args.scheduler) not implemented
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid training_mode value, got: {args.training_mode}')

    # -- Clean Up Distributed Processes
    print(f'\n----> about to cleanup worker with rank {rank}')
    cleanup(rank)
    print(f'clean up done successfully! {rank}')
    # cleanup_wandb(args, delete_wandb_dir=True)
    cleanup_wandb(args, delete_wandb_dir=False)

def eval(args):
    # dont worry about scheduler, since we are not training
    get_and_create_model_opt_scheduler_for_run(args)
    args.number_of_trainable_parameters = count_number_of_parameters(args.model)
    args.opt = move_opt_to_cherry_opt_and_sync_params(args) if is_running_parallel(args.rank) else args.opt
    
    # same dataloader for maml and usl
    args.dataloaders: dict = get_sl_dataloader(args)

    # use meta learner for gpt2 eval
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-3  # same as fast_lr in l2l
    args.nb_inner_train_steps = 10
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    args.first_order = True
    args.agent: Agent = GPTMetaLearnerL2L(args, args.model)

    args.bar = uutils.get_good_progressbar(max_value=100)

    args.agent.eval()

    val_accs = []
    val_losses = []

    [print(f'{k, v}') for k, v in vars(args).items()]

    for i, batch in enumerate(args.dataloaders['val']):
        batch = (batch[0].to(args.device), batch[1].to(args.device))

        # print("calling eval_forward")
        val_loss, val_loss_ci, val_acc, val_acc_ci = args.agent.eval_forward(batch, training = False)
        print(val_loss, val_loss_ci, val_acc, val_acc_ci)
        # print("back")
        val_losses.append(val_loss.item())
        val_accs.append(val_acc.item())

        args.bar.update(i+1)
        if i+1 == 100:
            break

    import statistics
    val_loss_mean = statistics.mean(val_losses)
    val_acc_mean = statistics.mean(val_accs)

    print("Results...")
    print(f'{val_loss_mean=}, {val_acc_mean=}')

# -- Run experiment

if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    main()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
