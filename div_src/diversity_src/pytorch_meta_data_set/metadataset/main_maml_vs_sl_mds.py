# %%
"""
main script for computing performances of mdls and dist(f, A(f)) vs the model trained.

IMPORTANT  What you just need to do is to make sure that the mediterrane errors of us cell and mammal are sort of similar when doing before making the test script the test comparison in the plots is it clear?

IMPORTANT Also it has to be q bar graph
"""
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy, copy
from pprint import pprint

import torch

from pathlib import Path

from torch import Tensor

from anatome.helper import compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks, pprint_results, \
    compute_stats_from_distance_per_batch_of_data_sets_per_layer, LayerIdentifier, dist_batch_data_sets_for_all_layer

import uutils

import time

from diversity_src.data_analysis.common import get_sl_learner, get_maml_meta_learner, santity_check_maml_accuracy, \
    comparison_via_performance, setup_args_path_for_ckpt_data_analysis, do_diversity_data_analysis, \
    performance_comparison_with_l2l_end_to_end, get_recommended_batch_size_miniimagenet_5CNN
from diversity_src.diversity.diversity import diversity
from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility, parse_args_meta_learning
from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloader

from uutils.torch_uu import equal_two_few_shot_cnn_models, process_meta_batch, approx_equal, get_device, norm
from uutils.torch_uu.distributed import is_lead_worker
from uutils.torch_uu.meta_learners.maml_differentiable_optimizer import get_maml_inner_optimizer, \
    dist_batch_tasks_for_all_layer_mdl_vs_adapted_mdl, dist_batch_tasks_for_all_layer_different_mdl_vs_adapted_mdl
from uutils.torch_uu.models import reset_all_weights
from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_last_two_layers

from pdb import set_trace as st

from uutils.torch_uu.models.resnet_rfs import get_recommended_batch_size_mi_resnet12rfs_body

start = time.time()

# -- hdb1 mio

def resnet12rfs_mds(args):
    # - model
    args.model_option = 'resnet12_rfs'
    args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5)

    #args.data_option = 'hdb1'
    #args.data_path = Path('~/data/l2l_data/').expanduser()

    # - training mode
    args.training_mode = 'iterations'  # needed so setup_args doesn't error out

    # note: 60K iterations for original maml 5CNN with adam
    # args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    args.first_order = True

    # -- options I am considering to have as flags in the args_parser...later
    # - metric for comparison
    args.metric_comparison_type = 'None'
    args.metric_as_sim_or_dist = 'dist'  # since we are trying to show meta-learning is happening, the more distance btw task & change in model the more meta-leanring is the hypothesis

    args.batch_size = 100#1000
    args.batch_size_eval = args.batch_size

    # - expt option
    args.experiment_option = 'performance_comparison'

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner_default'
    # args.agent_opt = 'MAMLMetaLearnerL2L_default'  # current code doesn't support this, it's fine I created a l2l -> torchmeta dataloader so we can use the MAML meta-learner that works for pytorch dataloaders

    # - ckpt name
    #args.sources = ['vgg_flower']
    #args.path_2_init_sl = '~/data/logs/logs_Dec05_17-53-05_jobid_-1'  # train_acc 0.9996, train_loss 0.001050
    #args.path_2_init_maml = '~/data/logs/logs_Dec05_15-11-12_jobid_-1'  #

    #args.sources = ['dtd']
    #args.path_2_init_sl = '/home/pzy2/data/logs/logs_Dec07_14-33-43_jobid_-1'
    #args.path_2_init_maml ="/home/pzy2/data/logs/logs_Dec06_22-12-41_jobid_-1"

    #args.sources = ['aircraft']
    #args.path_2_init_sl = "/home/pzy2/data/logs/logs_Dec05_17-52-23_jobid_-1"
    #args.path_2_init_maml = "/home/pzy2/data/logs/logs_Dec05_15-08-51_jobid_-1"

    #args.sources = ['vgg_flower','dtd']
    #args.path_2_init_sl ="/home/pzy2/data/logs/logs_Dec06_22-10-32_jobid_-1"
    #args.path_2_init_maml = "/home/pzy2/data/logs/logs_Dec06_22-11-09_jobid_-1"

    #args.sources = ['vgg_flower', 'aircraft']
    #args.path_2_init_sl = "/home/pzy2/data/logs/logs_Dec06_00-06-23_jobid_-1"
    #args.path_2_init_maml ="/home/pzy2/data/logs/logs_Dec06_00-20-03_jobid_-1"

    #args.sources = ['vgg_flower', 'omniglot']
    #args.path_2_init_sl = "/home/pzy2/data/logs/logs_Dec05_17-54-26_jobid_-1"
    #args.path_2_init_maml ="/home/pzy2/data/logs/logs_Dec06_00-20-03_jobid_-1"

    args.sources = ['ilsvrc_2012', 'aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot', 'quickdraw', 'vgg_flower']
    args.path_2_init_sl = "/home/pzy2/data/logs/logs_Dec03_00-44-02_jobid_-1"
    args.path_2_init_maml = "/home/pzy2/data/logs/logs_Dec01_01-50-15_jobid_-1"

    # -- wandb args
    args.wandb_project = 'SL vs MAML MDS Subsets'
    # - wandb expt args
    args.experiment_name = f'{args.experiment_option}_resnet12rfs'
    args.run_name = f'{args.sources} {args.model_option} {args.batch_size} {args.metric_comparison_type}: {args.jobid=} {args.path_2_init_sl} {args.path_2_init_maml}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
    # - fill in the missing things and make sure things make sense for run
    args = uutils.setup_args_for_experiment(args)
    return args


# -- data analysis

def load_args() -> Namespace:
    """
    Get the manual args and replace the missing fields using the args from the ckpt. Then make sure the meta-learner
    has the right args from the data analysis by doing args.meta_learner.args = new_args.
    """
    # - args from terminal
    from diversity_src.dataloaders.metadataset_episodic_loader import get_mds_args
    args = get_mds_args()

    args: Namespace = resnet12rfs_mds(args)

    # - over write my manual args (starting args) using the ckpt_args (updater args)
    args.meta_learner = get_maml_meta_learner(args)
    args = uutils.merge_args(starting_args=args.meta_learner.args, updater_args=args)  # second takes priority
    args.meta_learner.args = args  # to avoid meta learner running with args only from past experiment and not with metric analysis experiment

    uutils.print_args(args)
    return args


def main_data_analyis():
    args: Namespace = load_args()


    # - set base_models to be used for experiments
    print(f'{args.data_path=}')
    args.mdl1 = args.meta_learner.base_model
    args.mdl2 = get_sl_learner(args)
    args.mdl_maml = args.mdl1
    args.mdl_sl = args.mdl2
    args.mdl_rand = deepcopy(args.mdl_maml)
    reset_all_weights(args.mdl_rand)
    assert norm(args.mdl_rand) != norm(args.mdl_maml) != norm(args.mdl_sl), f"Error, norms should be different: " \
                                                                            f"{norm(args.mdl_rand)=} " \
                                                                            f"{args.mdl_sl=}" \
                                                                            f"{args.mdl_rand=}"
    print(f'{args.data_path=}')
    # assert equal_two_few_shot_cnn_models(args.mdl1,
    #                                      args.mdl2), f'Error, models should have same arch but they do not:\n{args.mdl1=}\n{args.mdl2}'
    # - print path to checkpoints
    print(f'{args.path_2_init_sl=}')
    print(f'{args.path_2_init_maml=}')

    # - get dataloaders and overwrites so data analysis runs as we want
    #args.dataloaders: dict = get_meta_learning_dataloader(args)
    from diversity_src.dataloaders.metadataset_episodic_loader import get_mds_loader
    args.dataloaders = get_mds_loader(args)  # implement returning dicts of torchmeta like dl's for mds

    # meta_dataloader = dataloaders['train']
    meta_dataloader = args.dataloaders['val']
    # meta_dataloader = dataloaders['test']

    # - layers to do analysis on
    if hasattr(args, 'layer_names'):
        print(f'{args.layer_names=}')

    # - maml param
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    # args.track_higher_grads = False  # note, I don't think this matters for testing since we aren't doing a backward pass. set to false only during meta-testing, but code sets it automatically only for meta-test
    # decided to use the setting for FO that I have for torchmeta learners, but since there is no training it should not matter.
    args.track_higher_grads = True
    args.fo = True

    # -- start analysis
    print('---------- start analysis ----------')
    print(f'{args.dataloaders=}')
    print(f'{args.num_workers=}')
    print(f'-->{args.meta_learner.args.copy_initial_weights}')
    print(f'-->{args.meta_learner.args.track_higher_grads}')
    print(f'-->{args.meta_learner.fo}')
    print(f'-->{args.meta_batch_size_eval=}')
    print(f'-->{args.num_its=}')
    print(f'-->{args.nb_inner_train_steps=}')
    print(f'-->{args.inner_lr=}')
    print(f'-->{args.metric_comparison_type=}')
    print(f'-->{args.metric_as_sim_or_dist=}')
    print(f'-->{args.n_aug_support_samples=}') if hasattr(args, 'n_aug_support_samples') else None
    print(f'-->{args.k_shots=}')

    # - Checks that maml0 acc is lower
    # santity_check_maml_accuracy(args)

    # -- do data analysis
    if args.experiment_option == 'performance_comparison':
        comparison_via_performance(args)
    else:
        print("not defined")
    # - done!
    print(f'time_passed_msg = {uutils.report_times(start)}')
    # - wandb
    if is_lead_worker(args.rank) and args.log_to_wandb:
        import wandb
        wandb.finish()



if __name__ == '__main__':
    main_data_analyis()
    print('\n--> Success Done!\a\n')