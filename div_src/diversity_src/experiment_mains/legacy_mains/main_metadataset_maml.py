
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
from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloaders
from uutils.torch_uu.distributed import set_sharing_strategy, print_process_info, set_devices, setup_process, cleanup, \
    print_dist
from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
from uutils.torch_uu.mains.main_sl_with_ddp import train
from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner
from uutils.torch_uu.training.meta_training import meta_train_fixed_iterations, meta_train_agent_fit_single_batch
from uutils import load_cluster_jobids_to, merge_args
from uutils.logging_uu.wandb_logging.common import setup_wandb

from pdb import set_trace as st

from uutils.torch_uu.training.supervised_learning import train_agent_fit_single_batch
from pathlib import Path


from diversity_src.dataloaders.metadataset_episodic_loader import get_mds_args, get_mds_loaders


def manual_load_mds_resnet12_maml_adam_no_scheduler(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model
    #args.model_option = 'resnet12_rfs_mi' #or use the one in the MDS fo-protoMAML paper
    #args.model_option = '5CNN_opt_as_model_for_few_shot_sl' #only sanity checks
    args.model_option = 'resnet12_rfs'#'resnet12_rfs_mi'
    args.n_cls = 5
    #args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
    #                      num_classes=args.n_cls)
    #args.model_option = 'resnet18_random'

    #args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls, filter_size=32,
    #                     levels=None, spp=False, in_channels=3)

    # - data
    args.data_option = 'MDS'
    args.sources = ['vgg_flower','aircraft']
    #args.data_path = Path('//').expanduser()

    # - training mode
    args.training_mode = 'iterations'


    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 800_000

    # - debug flag
    # args.debug = True
    args.debug = True


    args.scheduler_option = 'None'#'Adam_cosine_scheduler_rfs_cifarfs'
    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    #args.track_higher_grads = False  # set to false during meta-testing and for (official) fo maml
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True
    args.fo = True  # this flag shouldn't matter for if track_higher_grads is set to False

    # - outer trainer params
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
    args.opt_hps: dict = dict(lr=args.lr)

    args.batch_size = 8
    args.batch_size_eval = 2

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    #args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = f'aircraft vgg {args.model_option} official fo maml {args.opt_option}, no scheduler'
    # args.run_name = f'debug: {args.jobid=}'
    args.run_name = f'mds aircraft+vgg {args.model_option} {args.opt_option} {args.scheduler_option} {args.inner_lr} {args.lr}: {args.jobid=}'
    # args.log_to_wandb = True
    args.log_to_wandb = True

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
    args: Namespace = get_mds_args()
    args.args_hardcoded_in_script = True  # <- REMOVE to remove manual loads
    #args.manual_loads_name = 'manual_load_cifarfs_resnet12rfs_maml_ho_adam_simple_cosine_annealing'  # <- REMOVE to remove manual loads

    args: Namespace = manual_load_mds_resnet12_maml_adam_no_scheduler(args)
    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)

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
        # set_sharing_strategy()
        mp.spawn(fn=train, args=(args,), nprocs=args.world_size)

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
    args.dataloaders: dict = get_mds_loaders(args)

    # Agent does everything, proving, training, evaluate, meta-learnering, etc.
    args.agent = MAMLMetaLearner(args, args.model)
    args.meta_learner = args.agent

    # -- Start Training Loop
    print_dist('====> about to start train loop', args.rank)

    meta_train_fixed_iterations(args, args.agent, args.dataloaders, args.opt, args.scheduler)

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