import sys

import torch
import torch.nn as nn
import torch.multiprocessing as mp
# import torch.optim as optim

from argparse import Namespace

from uutils import args_hardcoded_in_script, report_times
from uutils.argparse_uu.common import setup_args_for_experiment
from uutils.argparse_uu.meta_learning import parse_args_meta_learning, fix_for_backwards_compatibility
from uutils.argparse_uu.supervised_learning import make_args_from_supervised_learning_checkpoint, parse_args_standard_sl
from uutils.torch_uu import count_number_of_parameters
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
from uutils.logging_uu.wandb_logging.common import setup_wandb, cleanup_wandb

from pdb import set_trace as st

from uutils.torch_uu.training.supervised_learning import train_agent_fit_single_batch
from pathlib import Path


def mds_resnet_maml_adam_scheduler(args: Namespace) -> Namespace:
    """
    todo patrick: what exactly is this running? what type of training etc?

    Looking at original mds hps:
        - https://github.com/google-research/meta-dataset/blob/main/meta_dataset/learn/gin/setups/trainer_config.gin
        - https://github.com/google-research/meta-dataset/blob/main/meta_dataset/learn/gin/best/maml_all_from_scratch.gin
        Main summary:
        ```
        Trainer.num_updates = 75000
        Trainer.batch_size = 256  # Only applicable to non-episodic models.
        Trainer.num_eval_episodes = 600
        ```

    """
    # - model
    # args.model_option = 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    args.n_cls = 5
    # bellow seems true for all models, they do use avg pool at the global pool/last pooling layer
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)  # dropbock_size=5 is rfs default for MI, 2 for CIFAR, will assume 5 for mds since it works on imagenet

    # - data
    args.data_option = 'mds'
    # args.sources = ['vgg_flower', 'aircraft']
    # Mscoco, traffic_sign are VAL only (actually we could put them here, fixed script to be able to do so w/o crashing)
    args.sources = ['ilsvrc_2012', 'aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot', 'quickdraw', 'vgg_flower', 'mscoco', 'traffic_sign']

    # - training mode
    args.training_mode = 'iterations'

    # note: 75_000 used by MAML mds https://github.com/google-research/meta-dataset/blob/main/meta_dataset/learn/gin/setups/trainer_config.gin#L1
    args.num_its = 75_000
    # args.num_its = 2_400
    # args.num_its = 100_000
    # args.num_its = 800_000

    # - debug flag
    args.debug = False
    # args.debug = True

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
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # This is needed.

    # - outer trainer params
    args.batch_size = 4  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = args.manual_loads_name
    args.run_name = f'{args.data_option} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def mds_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    # args.model_option = 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    args.n_cls = 5
    # bellow seems true for all models, they do use avg pool at the global pool/last pooling layer
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)  # dropbock_size=5 is rfs default for MI, 2 for CIFAR, will assume 5 for mds since it works on imagenet

    # - data
    args.data_option = 'mds'
    # Mscoco, traffic_sign are VAL only (actually we could put them here, fixed script to be able to do so w/o crashing)
    args.sources = ['ilsvrc_2012', 'aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot', 'quickdraw', 'vgg_flower',
                    'mscoco', 'traffic_sign']

    # - training mode
    args.training_mode = 'iterations_train_convergence'

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

    # - outer trainer params
    args.batch_size = 4  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2

    # - logging params
    args.log_freq = 500
    # args.log_freq = 20

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = args.manual_loads_name
    args.run_name = f'{args.data_option} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args

def mds_vggaircraft_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    args.n_cls = 5
    # bellow seems true for all models, they do use avg pool at the global pool/last pooling layer
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)  # dropbock_size=5 is rfs default for MI, 2 for CIFAR, will assume 5 for mds since it works on imagenet

    # - data
    args.data_option = 'mds'
    args.sources = ['vgg_flower', 'aircraft']
    # Mscoco, traffic_sign are VAL only (actually we could put them here, fixed script to be able to do so w/o crashing)

    # - training mode
    args.training_mode = 'iterations_train_convergence'

    # - debug flag
    args.debug = False#True
    #args.debug = False

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

    # - outer trainer params
    args.batch_size = 4#1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2#1

    # - logging params
    args.log_freq = 500

    #args.path_to_checkpoint = '/home/pzy2/data/logs/logs_Jan21_13-56-48_jobid_-1/ckpt.pt'
    #args.min_examples_in_class=0
    #args.num_support =None
    #args.num_query=None
    # args.log_freq = 20

    # -- wandb args
    args.wandb_project = 'Meta-Dataset'#'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = args.manual_loads_name
    args.run_name = f'{args.data_option} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args

def mds_birdsdtd_resnet_maml_adam_no_scheduler_train_to_convergence(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    args.n_cls = 5
    # bellow seems true for all models, they do use avg pool at the global pool/last pooling layer
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                          num_classes=args.n_cls)  # dropbock_size=5 is rfs default for MI, 2 for CIFAR, will assume 5 for mds since it works on imagenet

    # - data
    args.data_option = 'mds'
    args.sources = ['cu_birds','dtd']#['vgg_flower', 'aircraft']
    # Mscoco, traffic_sign are VAL only (actually we could put them here, fixed script to be able to do so w/o crashing)

    # - training mode
    args.training_mode = 'iterations_train_convergence'

    # - debug flag
    args.debug = False#True
    #args.debug = False

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

    # - outer trainer params
    args.batch_size = 4#1  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2#1

    # - logging params
    args.log_freq = 500

    # -- wandb args
    args.wandb_project = 'Meta-Dataset'#'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = args.manual_loads_name
    args.run_name = f'{args.data_option} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option}: {args.jobid=}'
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
    # todo: maybe later, add a try catch that if there is an mds only flag given at the python cmd line then it will load the mds args otherwise do the meta-leanring args
    # todo: https://stackoverflow.com/questions/75141370/how-does-one-have-python-work-when-multiple-arg-parse-options-are-possible

    # - uncomment below for MDS args
    #from diversity_src.dataloaders.metadataset_common import get_mds_base_args
    #args: Namespace = get_mds_base_args()

    args: Namespace = parse_args_meta_learning()

    #args.args_hardcoded_in_script = True  # <- REMOVE to remove manual loads
    # args.manual_loads_name = 'manual_load_cifarfs_resnet12rfs_maml_ho_adam_simple_cosine_annealing'  # <- REMOVE to remove manual loads
    # args.manual_loads_name = 'mds_resnet_maml_adam_no_scheduler_train_to_convergence'
    # args.manual_loads_name = 'mds_vggaircraft_resnet_maml_adam_no_scheduler_train_to_convergence' # mds_birdsdtd_resnet_maml_adam_no_scheduler_train_to_convergence

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
    args.number_of_trainable_parameters = count_number_of_parameters(args.model)
    print_dist(f"{args.model=}\n{args.opt=}\n{args.scheduler=}", args.rank)

    # create the dataloaders, this goes first so you can select the mdl (e.g. final layer) based on task
    args.dataloaders: dict = get_meta_learning_dataloaders(args)
    assert args.model.cls.out_features == 5

    # Agent does everything, proving, training, evaluate, meta-learnering, etc.
    args.agent = MAMLMetaLearner(args, args.model)
    args.meta_learner = args.agent

    # -- Start Training Loop
    print_dist(f"{args.model=}\n{args.opt=}\n{args.scheduler=}", args.rank)  # here to make sure mdl has the right cls
    print_dist('\n\n====> about to start train loop', args.rank)
    print(f'{args.filter_size=}') if hasattr(args, 'filter_size') else None
    print(f'{args.number_of_trainable_parameters=}')
    if args.training_mode == 'meta_train_agent_fit_single_batch':
        # meta_train_agent_fit_single_batch(args, args.agent, args.dataloaders, args.opt, args.scheduler)
        raise NotImplementedError
    elif 'iterations' in args.training_mode:
        meta_train_fixed_iterations(args, args.agent, args.dataloaders, args.opt, args.scheduler)
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


# -- Run experiment

if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    main()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
