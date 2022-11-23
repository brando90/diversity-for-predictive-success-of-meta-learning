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

import progressbar
import torch

from pathlib import Path

from torch import Tensor, nn

from anatome.helper import compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks, pprint_results, \
    compute_stats_from_distance_per_batch_of_data_sets_per_layer, LayerIdentifier, dist_batch_data_sets_for_all_layer

import uutils

import time

# from meta_learning.diversity.diversity import diversity

# from meta_learning.meta_learners.pretrain_convergence import FitFinalLayer
from diversity_src.diversity.diversity import diversity
# from uutils.torch_uu.meta_learners.pretrain_convergence import FitFinalLayer

from uutils.torch_uu import equal_two_few_shot_cnn_models, process_meta_batch, approx_equal, norm
from uutils.torch_uu.dataloaders.meta_learning.torchmeta_ml_dataloaders import get_miniimagenet_dataloaders_torchmeta
from uutils.torch_uu.distributed import is_lead_worker
from uutils.torch_uu.meta_learners.maml_differentiable_optimizer import get_maml_inner_optimizer, \
    dist_batch_tasks_for_all_layer_mdl_vs_adapted_mdl, meta_eval_no_context_manager, \
    dist_batch_tasks_for_all_layer_different_mdl_vs_adapted_mdl
from uutils.torch_uu.meta_learners.pretrain_convergence import FitFinalLayer
from uutils.torch_uu.models import reset_all_weights
from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_all_layers_minus_cls, \
    get_feature_extractor_conv_layers, get_head_cls, get_last_two_layers, Learner, get_default_learner

from pdb import set_trace as st

start = time.time()


def test_meta_learner(args):
    """
    To see if it's using the current maml_meta_learner code or not.
    """
    spt_x, spt_y, qry_x, qry_y = get_rand_batch()
    args.meta_learner(spt_x, spt_y, qry_x, qry_y, training=True, debug=True)


def get_rand_batch():
    """
    spt_x, spt_y, qry_x, qry_y = get_rand_batch()
    """
    spt_x: Tensor = torch.randn(2, 5 * 5, 3, 84, 84)
    spt_y: Tensor = torch.randint(low=0, high=5, size=[2, 5 * 5])
    qry_x, qry_y = spt_x, spt_y
    return spt_x, spt_y, qry_x, qry_y


def get_recommended_batch_size_miniimagenet_5CNN(safety_margin: int = 10):
    """
    Loop through all the layers and computing the largest B recommnded. Most likely the H*W that is
    smallest woll win but formally just compute B_l for each layer that your computing sims/dists and then choose
    the largest B_l. That ceil(B_l) satisfies B*H*W >= s*C for all l since it's the largest.

    Note: if the cls is present then we need B >= s*D since the output for it has shape
    [B, n_c] where n_c so we need, B >= 10*5 = 50 for example.
    s being used for B = 13 is
        s_cls = B/n_c = 13/5 = 2.6
        s_cls = B/n_c = 26/5 = 5.2
    """
    if safety_margin == 10:
        # -- satisfies B >= (10*32)/(5**2) = 12.8 for this specific 5CNN model
        return 13
    elif safety_margin == 20:
        # -- satisfies B >= (20*32)/(5**2) = 25.6 for this specific 5CNN model
        return 26
    else:
        raise ValueError(f'Not implemented for value: {safety_margin=}')


def get_recommended_batch_size_miniimagenet_head_5CNN(safety_margin: int = 10):
    """
    The cls/head is present then we need B >= s*D since the output for it has shape
    [B, n_c] where n_c so we need, B >= 10*5 = 50 for example.
    s being used for B = 13 is
        s_cls = B/n_c = 13/5 = 2.6
        s_cls = B/n_c = 26/5 = 5.2
    """
    if safety_margin == 10:
        # -- satisfies B >= (10*32)/(5**2) = 12.8 for this specific 5CNN model
        return 50
    elif safety_margin == 20:
        # -- satisfies B >= (20*32)/(5**2) = 25.6 for this specific 5CNN model
        return 100
    else:
        raise ValueError(f'Not implemented for value: {safety_margin=}')


# --

def get_args_for_experiment() -> Namespace:
    # - get my default args
    args = uutils._parse_basic_meta_learning_args_from_terminal()
    # args.layer_names = get_feature_extractor_pool_layers()
    # args.layer_names = get_all_layers_minus_cls()
    # args.layer_names = get_feature_extractor_conv_layers()
    # args.layer_names = get_feature_extractor_conv_layers(include_cls=True)
    # args.layer_names: list[str] = get_last_two_layers(layer_type='conv', include_cls=True)

    # args.metric_comparison_type = 'svcca'
    # args.metric_comparison_type = 'pwcca'
    # args.metric_comparison_type = 'lincka'
    args.metric_comparison_type = 'opd'
    # args.metric_comparison_type = 'none'

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
    # args.experiment_name = 'diveristiy on mini-imagenet (MI)'
    # args.run_name = f'dv(B_MI) = dv(B_MI, f) - f=f_rand - {args.metric_comparison_type} - filter - safety=10 - {args.layer_names} [DEBUG]'
    # args.run_name = f'dv(B_MI) = dv(B_MI, f) - f=f_maml - {args.metric_comparison_type} - filter - safety=10 - {args.layer_names}'
    # args.run_name = f'dv(B_MI) = dv(B_MI, f) - f=f_sl - {args.metric_comparison_type} - filter - safety=10 - {args.layer_names}'

    # - my args
    args.num_workers = 0
    args.safety_margin = 10
    # args.safety_margin = 20
    args.subsample_effective_num_data_method = 'subsampling_data_to_dims_ratio'
    args.subsample_effective_num_data_param = args.safety_margin
    args.metric_as_sim_or_dist = 'dist'  # since we are trying to show meta-learning is happening, the more distance btw task & change in model the more meta-leanring is the hypothesis
    args.num_its = 1
    args.meta_batch_size_train = 5
    # args.meta_batch_size_train = 10
    # args.meta_batch_size_train = 25
    # args.meta_batch_size_train = 50
    # args.meta_batch_size_train = 100
    # args.meta_batch_size_train = 200
    # args.meta_batch_size_train = 500
    args.meta_batch_size_eval = args.meta_batch_size_train
    args.batch_size = args.meta_batch_size_train
    # args.k_eval = get_recommended_batch_size_miniimagenet_5CNN(safety_margin=args.safety_margin)
    args.k_eval = get_recommended_batch_size_miniimagenet_head_5CNN(safety_margin=args.safety_margin)

    # args.log_to_wandb = False
    # args.log_to_wandb = True
    args.log_to_wandb = False

    # args.experiment_name = 'performance comparison'

    # args.experiment_option = 'diveristiy_f_rand'
    # args.experiment_option = 'diveristiy_f_maml'
    args.experiment_option = 'diveristiy_f_sl'

    args.run_name = f'{args.experiment_option=} {args.meta_batch_size_eval=} (reproduction)'
    args = uutils.setup_args_for_experiment(args)

    # -- checkpoints SL & MAML
    # 5CNN
    # ####ckpt_filename = 'ckpt_file_best_loss.pt'  # idk if the they have the same acc for this one, the goal is to minimize diffs so that only SL & MAML is the one causing the difference
    path_2_init_sl = '/logs/mar_all_mini_imagenet_expts/logs_Mar05_17-57-23_jobid_4246'
    # path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/meta_learning_expts/logs_Mar09_12-20-03_jobid_14_pid_183122'
    path_2_init_maml = '/logs/meta_learning_expts/logs_Mar09_12-17-50_jobid_13_pid_177628/'

    # resnet12rfs
    # path_2_init_sl = '~/data/rfs_checkpoints/mini_simple.pt'
    # path_2_init_maml = '~/data/logs/logs_Nov05_15-44-03_jobid_668_NEW_CKPT'
    # path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/nov_all_mini_imagenet_expts/logs_Nov05_15-44-03_jobid_668'

    # - path2checkpoint_file
    ckpt_filename_sl = 'ckpt_file.pt'  # this one is the one that has the accs that match, at least when I went through the files, json runs, MI_plots_sl_vs_maml_1st_attempt etc.
    args.path_2_init_sl = (Path(path_2_init_sl) / ckpt_filename_sl).expanduser()
    ckpt_filename_maml = 'ckpt_file.pt'  # this one is the one that has the accs that match, at least when I went through the files, json runs, MI_plots_sl_vs_maml_1st_attempt etc.
    # ckpt_filename_maml = 'ckpt_file_best_loss.pt'  # this one is the one that has the accs that match, at least when I went through the files, json runs, MI_plots_sl_vs_maml_1st_attempt etc.
    args.path_2_init_maml = (Path(path_2_init_maml) / ckpt_filename_maml).expanduser()

    # - other sl, maml params
    # args.dataset_name = 'torchmeta_mini_imagenet'
    args.data_path = Path('/miniimagenet').expanduser()  # for some datasets this is enough
    args.device = uutils.torch_uu.get_device()

    # -- print path to init & path to data
    print(f'{args.path_2_init_sl=}')
    print(f'{args.path_2_init_maml=}')
    print(f'{args.data_path=}')
    return args


def get_sl_learner(args: Namespace):
    """
    perhaps useful:
        args_ckpt = ckpt['state_dict']
        state_dict = ckpt['model']
    see: save_check_point_sl
    """
    ckpt: dict = torch.load(args.path_2_init_sl, map_location=torch.device('cpu'))
    model = ckpt['model']
    base_model: nn.Module = get_default_learner(n_classes=64)
    base_model.load_state_dict(model.state_dict())
    assert base_model is not model
    assert norm(base_model) == norm(model)
    model = base_model
    if torch.cuda.is_available():
        model = model.cuda()
    print(f'from ckpt (sl), model type: {model}')
    return model


def get_meta_learner(args: Namespace):
    from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner
    ckpt: dict = torch.load(args.path_2_init_maml, map_location=torch.device('cpu'))

    # - get stuff from ckpt
    meta_learner = ckpt['meta_learner']
    base_model: nn.Module = get_default_learner()
    base_model.load_state_dict(meta_learner.base_model.state_dict())
    assert base_model is not meta_learner.base_model
    meta_learner.base_model = base_model

    # - load it to up-to-date objects
    meta_learner = MAMLMetaLearner(meta_learner.args, meta_learner.base_model)
    args.meta_learner = meta_learner
    args.agent = meta_learner
    assert args.agent is args.meta_learner
    if torch.cuda.is_available():
        meta_learner.base_model = meta_learner.base_model.cuda()
    print(f'from ckpt (maml), model type: {meta_learner.args.base_model_mode=}')
    # args.nb_inner_train_steps = 10  # since ANIL paper used 10 for inference
    return meta_learner


def get_dataloader(args: Namespace) -> dict:
    """
    Returns the dict of data laoders e.g.
        dataloaders = {'train': meta_train_dataloader, 'val': meta_val_dataloader, 'test': meta_test_dataloader}
    """
    print(f'{args.data_path=}')
    if 'miniimagenet' in str(args.data_path):
        args.meta_learner.classification()
        args.training_mode = 'iterations'
        # args.k_eval = 100
        print(f'{args.data_path=}')
        dataloaders: dict = get_miniimagenet_dataloaders_torchmeta(args)
        print(f'{args.data_path=}')
    elif 'rfs_mini_imagenet' in str(args.data_path):
        assert False
        # dataloaders = get_rfs_sl_dataloader(args)
    else:
        raise ValueError(f'Data set path not implemented or incorrect, got value: {args.data_path}')
    print(f'{args.data_path=}')
    return dataloaders


def main_run_expt():
    # - get args & merge them with the args of actual experiment run
    args: Namespace = get_args_for_experiment()
    print(f'{args.data_path=}')
    args.meta_learner = get_meta_learner(args)
    print(f'{args.meta_learner=}')
    # - over write starting with updater args
    args = uutils.merge_args(starting_args=args.meta_learner.args, updater_args=args)
    print(f'{args.data_path=}')
    # args = get_working_args_for_torchmeta_mini_imagenet(main_args=args)  # don't think is needed, maml likely has enough
    args = uutils.merge_args(starting_args=args.meta_learner.args, updater_args=args)
    print(f'{args.data_path=}')
    args.meta_learner.args = args  # to avoid meta learner running with args only from past experiment and not with metric analysis experiment
    uutils.print_args(args)

    # - set base_models to be used for experiments
    print(f'{args.data_path=}')
    args.mdl1 = args.meta_learner.base_model
    args.mdl2 = get_sl_learner(args)
    args.mdl_maml = args.mdl1
    args.mdl_sl = args.mdl2
    args.mdl_rand = deepcopy(args.mdl_maml)
    reset_all_weights(args.mdl_rand)
    assert norm(args.mdl_rand) != norm(args.mdl_maml) != norm(args.mdl_sl)
    print(f'{args.data_path=}')
    assert equal_two_few_shot_cnn_models(args.mdl1,
                                         args.mdl2), f'Error, models should have same arch but they do not:\n{args.mdl1=}\n{args.mdl2}'
    print(f'{args.data_path=}')
    assert hasattr(args.mdl1, 'cls')
    assert hasattr(args.mdl2, 'cls')

    # - get dataloaders and overwrites so data analysis runs as we want
    print(f'{args.data_path=}')
    args.dataloaders: dict = get_dataloader(args)
    print(f'{args.data_path=}')

    # meta_dataloader = dataloaders['train']
    meta_dataloader = args.dataloaders['val']
    # meta_dataloader = dataloaders['test']
    print(f'{meta_dataloader.batch_size=}')

    # - layers to do analysis on
    # print(args.layer_names)  # todo - all layers
    print(f'{args.layer_names=}')

    # - maml param
    args.track_higher_grads = False  # set to false only during meta-testing, but code sets it automatically only for meta-test
    # args.track_higher_grads = True  # set to false only during meta-testing, but code sets it automatically only for meta-test
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    # args.copy_initial_weights = True  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.fo = True  # True, dissallows flow of higher order grad while still letting params track gradients.

    # -- start analysis
    print('---------- start (original) analysis ----------')
    # X: Tensor = torch.randn(16, 3, 84, 84)
    # assert_sim_of_model_with_itself_is_approx_one(args.mdl1, X, layer_name='model.features.conv4')
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
    # bar_it = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    # bar_it = uutils.get_good_progressbar(max_value=args.num_its)
    args.it = 1
    halt: bool = False

    # -- Do data analysis
    from diversity_src.data_analysis.common import santity_check_maml_accuracy
    santity_check_maml_accuracy(args)
    if args.experiment_option == 'performance comparison':
        from diversity_src.data_analysis.common import comparison_via_performance
        comparison_via_performance(args)
    elif args.experiment_option.startswith('diveristiy'):
        from diversity_src.data_analysis.common import do_diversity_data_analysis
        do_diversity_data_analysis(args, meta_dataloader)
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
    main_run_expt()
    # from meta_learning.diversity.task2vec_based_metrics.diversity_ask2vec.diversity import get_data_sets_from_example
    # get_data_sets_from_example()
    print('--> Success Done!\a\n')
