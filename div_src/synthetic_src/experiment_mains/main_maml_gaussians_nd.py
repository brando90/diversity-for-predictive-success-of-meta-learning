"""
Main script to set up meta-learning 1d gaussian experiments
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
from uutils.argparse_uu.common import setup_args_for_experiment, setup_wandb # modified 4/7/22
from uutils.argparse_uu.meta_learning import parse_args_meta_learning, fix_for_backwards_compatibility
from uutils.argparse_uu.supervised_learning import make_args_from_supervised_learning_checkpoint, parse_args_standard_sl
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.dataloaders.helpers import replace_final_layer
from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloader
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
    args.experiment_name = f'l2l_4CNNl2l_1024_cifarfs_rfs_adam_cl_100k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args

#https://cseweb.ucsd.edu/classes/sp11/cse291-d/hw1.pdf
def Hellinger(mu1,sigma1,mu2,sigma2):
    Sigma1 = torch.diag(torch.square(sigma1)) #square sigmas
    Sigma2 = torch.diag(torch.square(sigma2)) #square sigmas
    Sigmabar = (Sigma1 + Sigma2)/2
    u = (mu1-mu2)
    #print(Sigma1,Sigma2,Sigmabar,u)
    #print((torch.det(Sigma1).pow(0.25)*torch.det(Sigma2).pow(0.25)/(torch.det(Sigmabar).pow(0.25))))
    return (1 - (torch.det(Sigma1).pow(0.25)*torch.det(Sigma2).pow(0.25)/(torch.det(Sigmabar).pow(0.25)))*torch.exp(-0.125*(u.T@torch.inverse(Sigmabar)@u)))
    #return torch.sqrt(2*sigma1*sigma2/(sigma1.pow(2)+sigma2.pow(2)))*torch.exp(-(mu1-mu2).pow(2)/(4*(sigma1.pow(2)+sigma2.pow(2))))


def hellinger_div(mu_m_B, sigma_m_B, mu_s_B, sigma_s_B):
    N = 10000#500000
    dv_H = 0
    for i in range(N):
        # mu1,mu2 ~ N(mu1, mu2 | mu_m_B, sigma_m_B)
        # sigma1,sigma2 ~ N(sigma1,sigma2 | mu_s_B, sigma_s_B)
        mu1 = mu_m_B + sigma_m_B * torch.randn(2)  # b_dist.sample()#torch.normal(mu_b, sigma_b)
        sigma1 = torch.abs(mu_s_B + sigma_s_B * torch.randn(2))  # torch.normal(mu_b, sigma_b)
        mu2 = mu_m_B + sigma_m_B * torch.randn(2)  # b_dist.sample()#torch.normal(mu_b, sigma_b)
        sigma2 = torch.abs(mu_s_B + sigma_s_B * torch.randn(2))  # torch.normal(mu_b, sigma_b)
        #print(mu1,sigma1,mu2, sigma2)
        dv_H += Hellinger(mu1, sigma1, mu2, sigma2)
    dv_H /= N
    return dv_H


def l2l_gaussian_1d(args: Namespace) -> Namespace:
    """
    """
    from pathlib import Path
    # - model - we define a 1 x 15 x 15 x 5 FNN
    # 3 layers, takes 1 input (float) and guesses which of 5 1-hot output classes
    # the gaussian belong to.
    args.n_cls = 5
    args.hidden_layers = [128,128,128,128]#[128,128]#[2048,2048,2048,2048,2048,2048,2048,2048] #[128,128,128,128,128,128]###[128,128,128,128]#[15,15]#[128,128,128]#[128,128,128,128]#[32,32,64,128]#[15,15]#[32,32,64,128,128,128]#[32,32,64,128]#[32,32,64,128,128,128]
    #args.hidden_layer1 = 15
    #args.hidden_layer2 = 15
    args.input_size = 2
    args.dim = 2
    # TODO
    args.model_option = '3FNN_5_gaussian' # '4CNN_l2l_cifarfs'
    # args.hidden_size = 1024
    # args.model_hps = dict(ways=args.n_cls, hidden_size=args.hidden_size, embedding_size=args.hidden_size * 4)  # TODO
    args.model_hps = dict(ways = args.n_cls, hidden_layers = args.hidden_layers, input_size=args.input_size) #TODO: Implement this!

    # - data TODO
    args.data_option = 'n_way_gaussians_nd'#' cifarfs_rfs'  #CIFAR RFS dataset # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.mu_m_B = 0#int(sys.argv[1])  # 0  # doesn't matter
    args.sigma_m_B = 10#int(sys.argv[2])  # 10
    args.mu_s_B = 1000 #int(sys.argv[3])  # 100 #CHANGE THIS
    args.sigma_s_B = 0.01#int(sys.argv[4])  # 0.01
    args.div_H = hellinger_div(args.mu_m_B, args.sigma_m_B, args.mu_s_B, args.sigma_s_B)
    #args.rho = 0.1
    args.k_shots = 10#10#10#10#10
    args.k_eval = 30#30#30#30

    # args.data_path = Path('~/data/l2l_data/').expanduser()
    # args.data_augmentation = 'rfs2020'

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 20_000#100_000

    # - debug flag, # TODO when real maml experiments
    args.debug = True
    # args.debug = False

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
    #args.batch_size = 32
    args.batch_size = 8

    # - dist args
    args.world_size = 1
    # args.world_size = torch.cuda.device_count()
    # args.parallel = args.world_size > 1
    # args.seed = 42  # I think this might be important due to how tasksets works.
    # args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # -
    args.log_freq = 500

    # -- wandb args
    args.wandb_project = 'maml_5_gaussians' #'sl_vs_ml_iclr_workshop_paper'  # TODO
    # - wandb expt args
    # args.experiment_name = f'debug'
    args.experiment_name = 'l2l_gaussian_1d' #f'l2l_4CNNl2l_1024_cifarfs_rfs_adam_cl_100k'  # TODO
    args.run_name = f'{args.div_H} {args.mu_m_B} {args.sigma_m_B} {args.mu_s_B} {args.sigma_s_B} {args.hidden_layers} {args.k_shots} {args.k_eval} {args.batch_size} {args.model_option} {args.opt_option} {args.scheduler_option} {args.input_size} {args.n_cls} {args.lr} {args.inner_lr}: {args.jobid=}'  # TODO
    # args.log_to_wandb = True  # TODO when real
    args.log_to_wandb = False #True#True
    args.wandb_entity ="brando-uiuc"# "patrickyu"

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    return args


# - load args

def load_args() -> Namespace:
    """
    1. parse args from user's terminal
    2. optionally set remaining args values (e.g. manually, hardcoded, from ckpt etc.)
    3. setup remaining args small details from previous values (e.g. 1 and 2).
    """
    # -- parse args from terminal
    args: Namespace = parse_args_meta_learning()
    args.args_hardcoded_in_script = True  # <- REMOVE to remove manual loads
    args.manual_loads_name = 'l2l_gaussian_1d' #'l2l_4CNNl2l_1024_cifarfs_rfs_adam_cl_100k'  # <- REMOVE to remove manual loads

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    if resume_from_checkpoint(args):
        args: Namespace = make_args_from_supervised_learning_checkpoint(args=args, precedence_to_args_checkpoint=True)
        raise NotImplementedError
    elif args_hardcoded_in_script(args):
        if args.manual_loads_name == 'l2l_gaussian_1d': #'l2l_4CNNl2l_1024_cifarfs_rfs_adam_cl_100k':
            args: Namespace = l2l_gaussian_1d(args) #l2l_4CNNl2l_1024_cifarfs_rfs_adam_cl_100k(args)
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
        args.rank = get_local_rank()
        assert args.world_size == 1, f'Running serially but world_size is > 1, see: {args.world_size=}'
        assert args.rank == -1
        train_serial(args=args)
    else:
        # assert args.world_size > 1, f'Running parallel but world_size is <= 1, see: {args.world_size=}'
        # print(f'{torch.cuda.device_count()=}')
        # print(f'local rank before train function: {args.rank=}')
        # assert args.world_size > 1, f'Running parallel but world_size is <= 1, see: {args.world_size=}'
        # train(args=args)
        raise NotImplementedError  # if you want this see my main_dist_maml_l2l.py script, hopefully you don't need it


def train_serial(args):
    # - set up wandb only for the lead process
    setup_wandb(args)

    # create the model, opt & scheduler
    get_and_create_model_opt_scheduler_for_run(args)  # TODO, create your own or extend mine

    # create the loaders, note: you might have to change the number of layers in the final layer
    args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)  # TODO, create your own or extend
    args.dataloaders = args.tasksets  # for the sake that eval_sl can detect how to get examples for eval
    print(args.model.cls)
    assert args.model.cls.out_features == 5

    # Agent does everything, proving, training, evaluate, meta-learnering, etc.
    args.agent = MAMLMetaLearnerL2L(args, args.model)
    args.meta_learner = args.agent

    # -- Start Training Loop
    print_dist(f"{args.model=}\n{args.opt=}\n{args.scheduler=}", args.rank)  # here to make sure mdl has the right cls
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

# -- Run experiment

if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    main()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
