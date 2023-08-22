import torch
import torch.nn as nn
import torch.multiprocessing as mp
# import torch.optim as optim

from argparse import Namespace

import uutils
from uutils import args_hardcoded_in_script, report_times
from uutils.argparse_uu.common import setup_args_for_experiment
from uutils.argparse_uu.supervised_learning import make_args_from_supervised_learning_checkpoint, parse_args_standard_sl
# from uutils.torch_uu.agents.common import Agent
# from uutils.torch_uu.agents.supervised_learning import ClassificationSLAgent
from uutils.torch_uu import count_number_of_parameters
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.agents.supervised_learning import ClassificationSLAgent, UnionClsSLAgent, GPT2SLAgent
from uutils.torch_uu.meta_learners.gpt2_meta_learner import GPTMetaLearnerL2L
from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.dataloaders.helpers import get_sl_dataloader
from uutils.torch_uu.distributed import set_sharing_strategy, print_process_info, set_devices, setup_process, cleanup, \
    print_dist, is_running_parallel
# from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_first_time
# from uutils.torch_uu.training.supervised_learning import train_agent_fit_single_batch, train_agent_iterations, \
#     train_agent_epochs
from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_first_time, \
    get_and_create_model_opt_scheduler_for_run
from uutils.torch_uu.mains.main_sl_with_ddp import train
from uutils.torch_uu.training.supervised_learning import train_agent_fit_single_batch, train_agent_iterations, \
    train_agent_epochs

from uutils.logging_uu.wandb_logging.common import setup_wandb

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

    # # args for model
    # # default values from https://github.com/karpathy/nanoGPT except block size
    # args.block_size = 1024
    # args.vocab_size = 50257
    # args.n_layer = 12
    # args.n_head = 12
    # args.n_embd = 768
    # args.dropout = 0.1


    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    if resume_from_checkpoint(args):
        args: Namespace = make_args_from_supervised_learning_checkpoint(args=args, precedence_to_args_checkpoint=False)
    elif args_hardcoded_in_script(args):
        from pathlib import Path
        # - model
        # args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=64 + 1100,
        #                       filter_size=args.filter_size,
        #                       levels=None,
        #                       spp=False, in_channels=3)

        # args.model_hps = dict(block_size = 64, vocab_size = 50257, n_layer = 12, n_head = 12, n_embd = 768, dropout = 0.0)
        args.model_hps = dict(block_size = 32, vocab_size = 50257, n_layer = 4, n_head = 4, n_embd = 192, dropout = 0.0)

        # - data
        args.data_option = 'webtext'
        args.data_path = Path('~/data/webtext/').expanduser()
        args.log_root = '/lfs/hyperion/0/saumg/logs/usl_gpt2/'

        # - opt
        # args.opt_option = 'adamw_gpt'
        # args.opt_option = 'Adam_default'
        args.opt_option = 'AdafactorDefaultFair'
        # args.opt_option = 'sophia'
        args.num_its = 100000
        # # Sophia
        # args.lr = 3e-4
        # args.weight_decay = 1e-1
        # args.opt_hps = dict(lr = args.lr, weight_decay = args.weight_decay, betas = (0.9,0.95), rho = 0.1)
        # Adafactor
        args.weight_decay = 1e-2
        # # Adamw
        # args.lr = 1e-3
        # args.weight_decay = 1e-2
        # args.opt_hps: dict = dict(learning_rate=args.lr, weight_decay = args.weight_decay, betas = (0.9, 0.95))

        # args.scheduler_option = 'None'
        args.scheduler_option = 'AdafactorSchedule'
        # args.log_scheduler_freq = 1
        # args.T_max = args.num_its // args.log_scheduler_freq
        # args.eta_min = 1e-5  # coincidentally, matches MAML++
        # args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)


        # - training mode
        args.training_mode = 'iterations_train_convergence'
        args.train_convergence_patience = 100000

        # -
        # args.debug = True
        args.debug = False

        # -
        args.log_freq = 100  # for SL it is meant to be small e.g. 1 or 2
        args.ckpt_freq = 500

        # - wandb args
        args.wandb_project = 'entire-diversity-spectrum'
        # - wandb expt args
        args.experiment_name = f'usl_gpt2_adafactor_init'
        args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=} {gethostname()}'
        args.log_to_wandb = True
        args.wandb_entity = 'saumyagoyal01'
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

    

    if resume_from_checkpoint(args) and 'eval' in args.training_mode:
        print("calling eval...")
        eval(args)
        return

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
    if rank == 0:
        setup_wandb(args)
    print(f'setup process done for rank={rank}')

    # create the (ddp) model, opt & scheduler
    get_and_create_model_opt_scheduler_for_run(args)
    if args.opt_option == 'adamw_gpt' and args.scheduler_option == 'None':
        # Use custom scheduler
        args.scheduler_hps: dict = dict(warmup_iters = 2000,
                lr_decay_iters = 600000, # should be ~= max_iters per Chinchilla
                min_lr = 6e-5) # minimum learning rate, should be ~= learning_rate/10 per Chinchilla)
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
        halt_loss = 'train'
        # note train code will see training mode to determine halting criterion
        if halt_loss == 'val':
            train_agent_iterations(args, args.agent, args.dataloaders, args.opt, args.scheduler, halt_loss = 'val', target_loss = 3.039)
        train_agent_iterations(args, args.agent, args.dataloaders, args.opt, args.scheduler, use_half_loss = False)
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
    args.nb_inner_train_steps = 5
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





if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    main()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")