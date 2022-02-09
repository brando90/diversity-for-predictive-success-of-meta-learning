# %%
"""
main script for computing dist(f, A(f)) vs the model trained on a specific synthetic benchmark with given std.

python ~/automl-meta-learning/results_plots_sl_vs_ml/fall2021/_main_distance_sl_vs_maml.py

- If track_running_stats is set to False, this layer then does not keep running estimates, and batch statistics are instead used during evaluation time as well.
- .eval() = during inference (eval/testing) running_mean, running_std is used - that was calculated from training(because they want a deterministic output and to use estimates of the population statistics).
- .train() =  the batch statistics is used but a population statistic is estimated with running averages. I assume the reason batch_stats is used during training is to introduce noise that regularizes training (noise robustness). Assuming track_running_stats is True.
"""
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy, copy
from pprint import pprint

import torch

from pathlib import Path

from anatome.helper import compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks, pprint_results, \
    compute_stats_from_distance_per_batch_of_data_sets_per_layer, LayerIdentifier, dist_batch_data_sets_for_all_layer

import uutils

import time

from diversity_src.data_analysis.common import get_sl_learner, get_maml_meta_learner, santity_check_maml_accuracy, \
    comparison_via_performance, setup_args_path_for_ckpt_data_analysis
from diversity_src.diversity.diversity import diversity
from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility, parse_args_meta_learning
from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloader

from uutils.torch_uu import equal_two_few_shot_cnn_models, process_meta_batch, approx_equal, get_device
from uutils.torch_uu.distributed import is_lead_worker
from uutils.torch_uu.meta_learners.maml_differentiable_optimizer import get_maml_inner_optimizer, \
    dist_batch_tasks_for_all_layer_different_mdl_vs_adapted_mdl
from uutils.torch_uu.models import reset_all_weights
from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_last_two_layers

from pdb import set_trace as st

start = time.time()


def get_args_for_experiment() -> Namespace:
    # - get my default args
    args = uutils.parse_basic_meta_learning_args_from_terminal()
    args.log_to_wandb = False
    # args.log_to_wandb = True
    # args.layer_names = get_feature_extractor_pool_layers()
    # args.layer_names = get_all_layers_minus_cls()
    # args.layer_names = get_feature_extractor_conv_layers()
    # args.layer_names = get_feature_extractor_conv_layers(include_cls=True)
    # args.layer_names: list[str] = get_last_two_layers(layer_type='conv', include_cls=True)
    args.metric_comparison_type = 'svcca'
    # args.metric_comparison_type = 'pwcca'
    # args.metric_comparison_type = 'lincka'
    # args.metric_comparison_type = 'opd'
    args.effective_neuron_type = 'filter'
    args.layer_names: list[str] = get_last_two_layers(layer_type='conv', include_cls=True)
    # args.layer_names: list[str] = get_last_two_layers(layer_type='pool', include_cls=True)
    # args.layer_names = get_head_cls()
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'  # DO NOT UNCOMMENT
    # args.experiment_name = 'debug_experiment'
    # args.run_name = 'run_name [debug]'
    # args.run_name = 'run_name (cls - 20 - pwcca) [debug]'
    # args.experiment_name = 'd(f_sl, f_maml) - (choose metric) using conv'
    # args.run_name = 'd(f_sl, f_maml) - pwcca - filter - safety=10 - conv - rep2'
    # args.experiment_name = 'd(f_maml, A(f_maml)) - (choose metric) using conv'
    # args.run_name = 'd(f_maml, A(f_maml)) - pwcca - filter - safety=10 - conv - rep2'
    # args.experiment_name = 'd(f_sl, A(f_maml)) - (choose metric) using conv'
    # args.run_name = 'd(f_sl, A(f_maml)) - pwcca - filter - safety=10 - conv - rep2'
    # args.experiment_name = 'd(f_sl, A(f_sl)) - (choose metric) using conv'
    # args.run_name = 'd(f_sl, A(f_sl)) - pwcca - filter - safety=10 - conv - rep2'
    # args.experiment_name = 'performance comparison'
    # args.run_name = 'performance comparison - 0.01 - meta-batch = 500 - head of sl using head of maml for maml adaptation'
    # args.run_name = 'performance comparison - 0.01 - meta-batch = 500 - head of sl NOT using head of maml'
    # args.run_name = 'performance comparison - 0.01 - meta-batch = 500 - head of sl NOT using head of maml - run 2'
    # args.run_name = 'performance comparison - 0.01 - meta-batch = 200 - head of sl using head of maml for maml adaptation'
    # args.run_name = 'performance comparison - 0.01 - meta-batch = 200 - head of sl NOT using head of maml for maml adaptation'
    # args.run_name = 'performance comparison - 0.01 - meta-batch = 100 - head of sl using head of maml for maml adaptation'
    # args.run_name = 'performance comparison - 0.01 - meta-batch = 50 - head of sl using head of maml for maml adaptation'
    # args.run_name = 'performance comparison - 0.01 - meta-batch = 5 - head of sl using head of maml for maml adaptation'
    # args.run_name = 'performance comparison - 0.1, head of sl using head of maml for maml adaptation'
    # args.run_name = 'performance comparison - 0.1 - meta-batch = 500 - head of sl using head of maml for maml adaptation'
    # args.run_name = 'performance comparison - (minus) -0.1 - meta-batch = 500 - head of sl using head of maml for maml adaptation'
    # args.run_name = 'performance comparison - (minus) -0.1 - meta-batch = 500 - head of sl using head of maml for maml adaptation2 '
    # args.experiment_name = 'd(LR(f_sl), MAML(f_maml)) - (choose metric) using conv'
    # args.run_name = 'd(LR(f_sl), MAML(f_maml)) - pwcca - filter - safety=10 - conv - only cls/head'
    # args.experiment_name = 'd(LR(f_sl), MAML(f_maml)) - (choose metric) using conv - features + cls/head - k_eval=50'
    # args.run_name = 'd(LR(f_sl), MAML(f_maml)) - pwcca - filter - safety=10 - conv - features + cls/head - k_eval=50'
    # args.experiment_name = 'd(LR(f_sl), MAML(f_maml)) - (choose metric) using conv - features + cls/head - k_eval=13'
    # args.run_name = 'd(LR(f_sl), MAML(f_maml)) - pwcca - filter - safety=10 - conv - features + cls/head - k_eval=13'
    # args.experiment_name = 'debug pwcca'
    # args.run_name = 'd(f_maml, A(f_maml)) - pwcca - filter - safety=10 - conv - rep2'
    # args.run_name = 'd(f_sl, f_maml) - pwcca - filter - safety=10 - conv - rep2'
    # args.run_name = 'd(f_sl, A(f_maml)) - pwcca - filter - safety=10 - conv - rep2'
    # args.run_name = 'd(LR(f_sl), A(f_maml)) - pwcca - filter - safety=10 - conv - rep2'
    # args.run_name = 'd(LR(f_sl), A(f_sl)) - pwcca - filter - safety=10 - conv - rep2'
    args.experiment_name = 'diveristiy on mini-imagenet (MI)'
    # args.run_name = f'dv(B_MI) = dv(B_MI, f) - f=f_rand - {args.metric_comparison_type} - filter - safety=10 - {args.layer_names} [DEBUG]'
    # args.run_name = f'dv(B_MI) = dv(B_MI, f) - f=f_maml - {args.metric_comparison_type} - filter - safety=10 - {args.layer_names}'
    args.run_name = f'dv(B_MI) = dv(B_MI, f) - f=f_sl - {args.metric_comparison_type} - filter - safety=10 - {args.layer_names}'
    args = uutils.setup_args_for_experiment(args)

    # - my args
    args.num_workers = 0
    args.safety_margin = 10
    # args.safety_margin = 20
    args.subsample_effective_num_data_method = 'subsampling_data_to_dims_ratio'
    args.subsample_effective_num_data_param = args.safety_margin
    args.metric_as_sim_or_dist = 'dist'  # since we are trying to show meta-learning is happening, the more distance btw task & change in model the more meta-leanring is the hypothesis
    args.num_its = 1
    # args.meta_batch_size_train = 5
    # args.meta_batch_size_train = 10
    args.meta_batch_size_train = 25
    # args.meta_batch_size_train = 50
    # args.meta_batch_size_train = 100
    # args.meta_batch_size_train = 200
    # args.meta_batch_size_train = 500
    args.meta_batch_size_eval = args.meta_batch_size_train
    # args.k_eval = get_recommended_batch_size_miniimagenet_5CNN(safety_margin=args.safety_margin)
    # args.k_eval = get_recommended_batch_size_miniimagenet_head_5CNN(safety_margin=args.safety_margin)

    # -- checkpoints SL & MAML
    # 5CNN
    # ####ckpt_filename = 'ckpt_file_best_loss.pt'  # idk if the they have the same acc for this one, the goal is to minimize diffs so that only SL & MAML is the one causing the difference
    # path_2_init_sl = '~/data_folder_fall2020_spring2021/logs/mar_all_mini_imagenet_expts/logs_Mar05_17-57-23_jobid_4246'
    # path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/meta_learning_expts/logs_Mar09_12-20-03_jobid_14_pid_183122'
    path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/meta_learning_expts/logs_Mar09_12-17-50_jobid_13_pid_177628/'

    # resnet12rfs
    path_2_init_sl = '~/data/rfs_checkpoints/mini_simple.pt'
    path_2_init_maml = '~/data/logs/logs_Nov05_15-44-03_jobid_668_NEW_CKPT'

    # - path2checkpoint_file
    ckpt_filename_sl = 'ckpt_file.pt'  # this one is the one that has the accs that match, at least when I went through the files, json runs, MI_plots_sl_vs_maml_1st_attempt etc.
    args.path_2_init_sl = (Path(path_2_init_sl) / ckpt_filename_sl).expanduser()
    ckpt_filename_maml = 'ckpt_file.pt'  # this one is the one that has the accs that match, at least when I went through the files, json runs, MI_plots_sl_vs_maml_1st_attempt etc.
    # ckpt_filename_maml = 'ckpt_file_best_loss.pt'  # this one is the one that has the accs that match, at least when I went through the files, json runs, MI_plots_sl_vs_maml_1st_attempt etc.
    args.path_2_init_maml = (Path(path_2_init_maml) / ckpt_filename_maml).expanduser()

    # - other sl, maml params
    # args.dataset_name = 'torchmeta_mini_imagenet'
    args.data_path = Path('~/data/miniimagenet').expanduser()  # for some datasets this is enough
    args.device = uutils.torch_uu.get_device()

    # -- print path to init & path to data
    print(f'{args.path_2_init_sl=}')
    print(f'{args.path_2_init_maml=}')
    print(f'{args.data_path=}')
    return args


def l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k(args: Namespace) -> Namespace:
    """
    """
    from uutils.torch_uu.models.resnet_rfs import get_recommended_batch_size_cifarfs_resnet12rfs_body, \
        get_feature_extractor_conv_layers
    # - model
    args.model_option = 'resnet12_rfs_cifarfs_fc100'

    # - data
    # args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    # args.data_path = Path('~/data/l2l_data/').expanduser()
    # args.data_augmentation = 'rfs2020'
    args.data_option = 'torchmeta_cifarfs'  # no name assumes l2l
    args.data_path = Path('~/data/torchmeta_data/').expanduser()
    args.augment_train = True

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.track_higher_grads = True  # set to false only during meta-testing and unofficial fo, but then args.fo has to be True too. Note code sets it automatically only for meta-test
    # args.first_order = True
    # args.first_order = False

    # - outer trainer params
    # args.batch_size = 32
    # args.batch_size = 8

    # - dist args
    # args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    # args.parallel = True
    # args.seed = 42  # I think this might be important due to how tasksets works.
    # args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # # -
    # args.log_freq = 500

    # -- options I am considering to have as flags in the args_parser...later
    # - metric for comparison
    args.metric_comparison_type = 'svcca'
    # args.metric_comparison_type = 'pwcca'
    # args.metric_comparison_type = 'lincka'
    # args.metric_comparison_type = 'opd'
    args.metric_as_sim_or_dist = 'dist'  # since we are trying to show meta-learning is happening, the more distance btw task & change in model the more meta-leanring is the hypothesis

    # - effective neuron type
    args.effective_neuron_type = 'filter'

    # - layers, this gets the feature layers it seems... unsure why I'm doing this. I thought I was doing a comparison
    # with all the layers up to the final layer...
    # args.layer_names: list[str] = get_last_two_layers(layer_type='conv', include_cls=True)
    # args.layer_names = get_head_cls()
    args.layer_names = get_feature_extractor_conv_layers()

    args.safety_margin = 10
    # args.safety_margin = 20

    args.batch_size = 2
    # args.batch_size = 25
    args.batch_size_eval = args.batch_size

    # - set k_eval (qry set batch_size) to make experiments safe/reliable
    args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_body(safety_margin=args.safety_margin)
    # args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_head(safety_margin=args.safety_margin)

    # - expt option
    args.experiment_option = 'performance_comparison'

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner_default'

    # - ckpt name
    # args.path_2_init_sl = '~/data_folder_fall2020_spring2021/logs/mar_all_mini_imagenet_expts/logs_Mar05_17-57-23_jobid_4246'
    # args.path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/meta_learning_expts/logs_Mar09_12-20-03_jobid_14_pid_183122'
    args.path_2_init_sl = '~/data/logs/logs_Feb07_13-55-06_jobid_14887_pid_58061/'
    args.path_2_init_maml = '~/data/logs/logs_Feb05_19-21-50_jobid_11407/'

    # - device
    # args.device = torch.device('cpu')
    # args.device = get_device()

    #
    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'manual_args_l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k'
    args.run_name = f'{args.model_option} {args.opt_option} {args.scheduler_option} {args.lr}: {args.jobid=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

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
    args: Namespace = parse_args_meta_learning()

    # - get manual args
    args: Namespace = l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k(args)

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
    print(f'{args.data_path=}')
    # assert equal_two_few_shot_cnn_models(args.mdl1,
    #                                      args.mdl2), f'Error, models should have same arch but they do not:\n{args.mdl1=}\n{args.mdl2}'

    # - get dataloaders and overwrites so data analysis runs as we want
    args.dataloaders: dict = get_meta_learning_dataloader(args)
    # meta_dataloader = dataloaders['train']
    meta_dataloader = args.dataloaders['val']
    # meta_dataloader = dataloaders['test']

    # - layers to do analysis on
    print(f'{args.layer_names=}')

    # - maml param
    args.track_higher_grads = False  # set to false only during meta-testing, but code sets it automatically only for meta-test
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.fo = False  # True, dissallows flow of higher order grad while still letting params track gradients.

    # -- start analysis
    print('---------- start analysis ----------')
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
    args.it = 1

    halt: bool = False

    # - Checks that maml0 acc is lower
    santity_check_maml_accuracy(args)

    # -- do data analysis
    if args.experiment_option == 'performance_comparison':
        comparison_via_performance(args)
    elif args.experiment_option == 'diveristiy':
        args.mdl_rand = deepcopy(args.mdl1)
        reset_all_weights(args.mdl_rand)

        print('- Choose model for computing diversity')
        print(f'{args.run_name=}')
        if 'f_rand' in args.run_name:
            args.mdl_for_dv = args.mdl_rand
            print('==> f_rand')
        elif 'f_maml' in args.run_name:
            args.mdl_for_dv = args.mdl_maml
            print('==> f_maml')
        elif 'f_sl' in args.run_name:
            args.mdl_for_dv = args.mdl_sl
            print('==> f_sl')
        else:
            raise ValueError(f'Invalid mdl option: {args.run_name=}')
        # - Compute diversity: sample one batch of tasks and use a random cross product of different tasks to compute diversity.
        args.num_tasks_to_consider = args.meta_batch_size_train
        print(f'{args.num_tasks_to_consider=}')
        for batch_idx, batch_tasks in enumerate(meta_dataloader):
            spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch_tasks)

            # - compute diversity
            div_mu, div_std, distances_for_task_pairs = diversity(
                f1=args.mdl_for_dv, f2=args.mdl_for_dv, X1=qry_x, X2=qry_x,
                layer_names1=args.layer_names, layer_names2=args.layer_names,
                num_tasks_to_consider=args.num_tasks_to_consider)
            print(f'{div_mu, div_std, distances_for_task_pairs=}')

            # -- print results
            print('-- raw results')
            print(f'distances_for_task_pairs=')
            pprint(distances_for_task_pairs)

            print('\n-- dist results')
            div_mu, div_std = compute_stats_from_distance_per_batch_of_data_sets_per_layer(distances_for_task_pairs)
            pprint_results(div_mu, div_std)
            mu, std = compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks(
                distances_for_task_pairs)
            print(f'----entire net result:\n  {mu=}, {std=}\n')

            print('-- sim results')
            div_mu, div_std = compute_stats_from_distance_per_batch_of_data_sets_per_layer(distances_for_task_pairs,
                                                                                           dist2sim=True)
            pprint_results(div_mu, div_std)
            mu, std = compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks(
                distances_for_task_pairs, dist2sim=True)
            print(f'----entire net result:\n  {mu=}, {std=}')
            break
    else:
        while not halt:
            for batch_idx, batch_tasks in enumerate(meta_dataloader):
                print(f'it = {args.it}')
                spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch_tasks)

                # -- get comparison - SL vs ML
                # X: Tensor = qry_x
                # distances_per_data_sets_per_layer: list[
                #     OrderedDict[LayerIdentifier, float]] = dist_batch_data_sets_for_all_layer(args.mdl1, args.mdl2, X, X,
                #                                                                               args.layer_names,
                #                                                                               args.layer_names,
                #                                                                               metric_comparison_type=args.metric_comparison_type,
                #                                                                               effective_neuron_type=args.effective_neuron_type,
                #                                                                               subsample_effective_num_data_method=args.subsample_effective_num_data_method,
                #                                                                               subsample_effective_num_data_param=args.subsample_effective_num_data_param,
                #                                                                               metric_as_sim_or_dist=args.metric_as_sim_or_dist)

                # -- get comparison - ML vs A(ML)
                # inner_opt = get_maml_inner_optimizer(args.mdl1, args.inner_lr)
                # distances_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]] = \
                #     dist_batch_tasks_for_all_layer_mdl_vs_adapted_mdl(
                #         mdl=args.mdl1,
                #         spt_x=spt_x, spt_y=spt_y, qry_x=qry_x, qry_y=qry_y,
                #         layer_names=args.layer_names,
                #         inner_opt=inner_opt,
                #         fo=args.fo,
                #         nb_inner_train_steps=args.nb_inner_train_steps,
                #         criterion=args.criterion,
                #         metric_comparison_type=args.metric_comparison_type,
                #         effective_neuron_type=args.effective_neuron_type,
                #         subsample_effective_num_data_method=args.subsample_effective_num_data_method,
                #         subsample_effective_num_data_param=args.subsample_effective_num_data_param,
                #         metric_as_sim_or_dist=args.metric_as_sim_or_dist,
                #         training=True
                #     )

                # -- get comparison - SL vs A(ML)
                # inner_opt = get_maml_inner_optimizer(args.mdl1, args.inner_lr)
                # distances_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]] = \
                #     dist_batch_tasks_for_all_layer_different_mdl_vs_adapted_mdl(
                #         mdl_fixed=args.mdl2, mdl_ml=args.mdl1,
                #         spt_x=spt_x, spt_y=spt_y, qry_x=qry_x, qry_y=qry_y,
                #         layer_names=args.layer_names,
                #         inner_opt=inner_opt,
                #         fo=args.fo,
                #         nb_inner_train_steps=args.nb_inner_train_steps,
                #         criterion=args.criterion,
                #         metric_comparison_type=args.metric_comparison_type,
                #         effective_neuron_type=args.effective_neuron_type,
                #         subsample_effective_num_data_method=args.subsample_effective_num_data_method,
                #         subsample_effective_num_data_param=args.subsample_effective_num_data_param,
                #         metric_as_sim_or_dist=args.metric_as_sim_or_dist,
                #         training=True
                #     )

                # -- get comparison - LR(SL) vs A(ML)
                # inner_opt = get_maml_inner_optimizer(args.mdl1, args.inner_lr)
                # distances_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]] = \
                #     dist_batch_tasks_for_all_layer_different_mdl_vs_adapted_mdl(
                #         mdl_fixed=args.mdl2, mdl_ml=args.mdl1,
                #         spt_x=spt_x, spt_y=spt_y, qry_x=qry_x, qry_y=qry_y,
                #         layer_names=args.layer_names,
                #         inner_opt=inner_opt,
                #         fo=args.fo,
                #         nb_inner_train_steps=args.nb_inner_train_steps,
                #         criterion=args.criterion,
                #         metric_comparison_type=args.metric_comparison_type,
                #         effective_neuron_type=args.effective_neuron_type,
                #         subsample_effective_num_data_method=args.subsample_effective_num_data_method,
                #         subsample_effective_num_data_param=args.subsample_effective_num_data_param,
                #         metric_as_sim_or_dist=args.metric_as_sim_or_dist,
                #         training=True
                #     )

                # -- get comparison - SL vs MAML(SL)
                args.mdl2.model.cls = deepcopy(args.mdl1.model.cls)
                print(
                    '-> sl_mdl has the head of the maml model to make comparisons using maml better, it does not affect when '
                    'fitting the final layer with LR FFL')
                inner_opt = get_maml_inner_optimizer(args.mdl2, args.inner_lr)
                distances_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]] = \
                    dist_batch_tasks_for_all_layer_different_mdl_vs_adapted_mdl(
                        mdl_fixed=args.mdl2, mdl_ml=args.mdl2,
                        spt_x=spt_x, spt_y=spt_y, qry_x=qry_x, qry_y=qry_y,
                        layer_names=args.layer_names,
                        inner_opt=inner_opt,
                        fo=args.fo,
                        nb_inner_train_steps=args.nb_inner_train_steps,
                        criterion=args.criterion,
                        metric_comparison_type=args.metric_comparison_type,
                        effective_neuron_type=args.effective_neuron_type,
                        subsample_effective_num_data_method=args.subsample_effective_num_data_method,
                        subsample_effective_num_data_param=args.subsample_effective_num_data_param,
                        metric_as_sim_or_dist=args.metric_as_sim_or_dist,
                        training=True
                    )

                # - print raw results
                print('-- raw results')
                print(f'distances_per_data_sets_per_layer=')
                pprint(distances_per_data_sets_per_layer)

                # - print dist results
                print('-- dist results')
                mus, stds = compute_stats_from_distance_per_batch_of_data_sets_per_layer(
                    distances_per_data_sets_per_layer)
                pprint_results(mus, stds)
                mu, std = compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks(
                    distances_per_data_sets_per_layer)
                print(f'----entire net result: {mu=}, {std=}')

                # - print sims results
                print('-- sim results')
                mus, stds = compute_stats_from_distance_per_batch_of_data_sets_per_layer(
                    distances_per_data_sets_per_layer, dist2sim=True)
                pprint_results(mus, stds)
                mu, std = compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks(
                    distances_per_data_sets_per_layer, dist2sim=True)
                print(f'----entire net result: {mu=}, {std=}')

                print('--')
                print(f'{args.metric_comparison_type=}')
                print(f'{args.effective_neuron_type=}')
                print(f'{args.metric_as_sim_or_dist=}')
                print(f'{args.subsample_effective_num_data_method=}')
                print(f'{args.safety_margin=}')
                print(f'{args.k_eval=}')
                print('--')

                # - break
                halt: bool = args.it >= args.num_its - 1
                if halt:
                    break
                args.it += 1

    # - done!
    print(f'time_passed_msg = {uutils.report_times(start)}')
    # - wandb
    if is_lead_worker(args.rank) and args.log_to_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main_data_analyis()
    print('--> Success Done!\a\n')