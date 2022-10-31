# %%
"""
main script for computing dist(f, A(f)) vs the model trained on a specific synthetic benchmark with given std.

python ~/automl-meta-learning/results_plots_sl_vs_ml/fall2021/_main_distance_sl_vs_maml.py

- If track_running_stats is set to False, this layer then does not keep running estimates, and batch statistics are instead used during evaluation time as well.
- .eval() = during inference (eval/testing) running_mean, running_std is used - that was calculated from training(because they want a deterministic output and to use estimates of the population statistics).
- .train() =  the batch statistics is used but a population statistic is estimated with running averages. I assume the reason batch_stats is used during training is to introduce noise that regularizes training (noise robustness). Assuming track_running_stats is True.


python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main2_distance_sl_vs_maml.py
"""
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy, copy
from pprint import pprint

import diversity_src.data_analysis.common
import torch

import scipy.stats
import numpy as np

from pathlib import Path

from torch import Tensor

from anatome.helper import compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks, pprint_results, \
    compute_stats_from_distance_per_batch_of_data_sets_per_layer, LayerIdentifier, dist_batch_data_sets_for_all_layer

import uutils
from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
from torchmeta.utils.data import BatchMetaDataLoader
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


# - MI

def args_fnn_gaussian(args: Namespace) -> Namespace:
    """
    """
    from uutils.torch_uu.models.resnet_rfs import get_recommended_batch_size_cifarfs_resnet12rfs_body, \
        get_feature_extractor_conv_layers
    # - model
    args.model_option = '3FNN_5_gaussian'
    args.hidden_layers = [128,128,128,128]
    args.dim = 2
    args.input_size = args.dim

    # - data
    args.data_option = 'n_way_gaussians_nd'  # no name assumes l2l
    args.mu_m_B = 0  # doesn't matter
    args.sigma_m_B = 10
    args.mu_s_B = 1000
    args.sigma_s_B = 0.01
    #args.data_path = Path('~/data/torchmeta_data/').expanduser()
    #args.augment_train = True

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 1

    args.path_2_init_sl ='~/Documents/JunNew/0_10_1000_001_USL' #'~/Documents/logs/0.18USL'
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/ip20v98t/logs?workspace=user-brando
    # args.path_2_init_maml = '~/Documents/data/logs/logs_Apr27_02-22-19_jobid_-1'  # Adam MAML 1dgaussian 1x128x128x5
    args.path_2_init_maml ='~/Documents/JunNew/0_10_1000_001_MAML' #'~/Documents/logggy/0.19MAML'

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
    args.world_size = 1
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
    args.metric_comparison_type = 'None'
    # args.metric_comparison_type = 'svcca'
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

    # args.batch_size = 2
    # args.batch_size = 5
    # args.batch_size = 25
    args.batch_size = 100#300
    # args.batch_size = 500
    args.batch_size_eval = args.batch_size

    # - set k_eval (qry set batch_size) to make experiments safe/reliable
    args.k_shots= 10
    args.k_eval = 30#get_recommended_batch_size_miniimagenet_5CNN(safety_margin=args.safety_margin)
    # args.k_eval = get_recommended_batch_size_miniimagenet_head_5CNN(safety_margin=args.safety_margin)

    # - expt option
    args.experiment_option = 'performance_comparison'

    # args.experiment_option = 'diveristiy_f_rand'
    # args.experiment_option = 'diveristiy_f_maml'
    # args.experiment_option = 'diveristiy_f_sl'

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner_default'

    # - ckpt name
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/33frd31p?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Feb19_10-13-22_jobid_2411_pid_28656'  # good! SGD converged with 0.9994 train acc
    #args.path_2_init_sl = '~/Documents/data/logs/logs_Apr29_09-14-31_jobid_-1'  # Adam, USL 1dgaussian 1x128x128x100

    # - device
    # args.device = torch.device('cpu')
    # args.device = get_device()

    # -- wandb args
    args.wandb_project = 'maml_vs_sl_5_gaussians_nd'
    # - wandb expt args
    args.experiment_name = f'{args.experiment_option}_args_fnn_gaussian'
    args.run_name = f'dim={args.dim} {args.experiment_option} {args.model_option} {args.batch_size} {args.metric_comparison_type}: {args.jobid=}'
    args.log_to_wandb = True
    args.wandb_entity ="brando-uiuc"
    #args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
    # - fill in the missing things and make sure things make sense for run
    args = uutils.setup_args_for_experiment(args)
    return args

# - cifarfs

# -- data analysis

def load_args() -> Namespace:
    """
    Get the manual args and replace the missing fields using the args from the ckpt. Then make sure the meta-learner
    has the right args from the data analysis by doing args.meta_learner.args = new_args.
    """
    # - args from terminal
    args: Namespace = parse_args_meta_learning()

    # - get manual args
    # args: Namespace = args_5cnn_cifarfs(args)
    args: Namespace = args_fnn_gaussian(args)
    # args: Namespace = resnet12rfs_cifarfs(args)
    # args: Namespace = resnet12rfs_mi(args)

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
    from uutils.torch_uu.dataloaders.meta_learning.gaussian_nd_tasksets import get_tasksets
    args.dataloaders: BenchmarkTasksets = get_tasksets(
        args.data_option,
        train_samples=args.k_shots + args.k_eval,  # k shots for meta-train, k eval for meta-validaton/eval
        train_ways=args.n_classes,
        test_samples=args.k_shots + args.k_eval,
        test_ways=args.n_classes,
        mu_m_B=args.mu_m_B,
        sigma_m_B=args.sigma_m_B,
        mu_s_B=args.mu_s_B,
        sigma_s_B=args.sigma_s_B,
        dim = args.dim
        # root=args.data_path, #No need for datafile
        # data_augmentation=args.data_augmentation, #TODO: currently not implemented! Do we need to implement?
    )
    #trainmd : MetaDataset = args.tasksets.train.dataset
    #testmd : MetaDataset = args.tasksets.test.dataset
    #valmd : MetaDataset = args.tasksets.validation.dataset
    #args.dataloaders: dict = l2l_to_torchmeta_collate(args)
    #args.dataloaders = {'train': args.dataloaders.train,
    #                    'test': args.dataloaders.test,
    #                    'val': args.dataloaders.validation}
    #from uutils.torch_uu.dataloaders.meta_learning.gaussian_1d_tasksets import get_train_valid_test_data_loader_1d_gaussian
    #args.dataloaders=get_train_valid_test_data_loader_1d_gaussian(args)
    # Sampling to mimick the miniimagenet dataloader
    #from uutils.torch_uu import process_meta_batch

    # args = get_minimum_args_for_torchmeta_mini_imagenet_dataloader()
    # dataloader = get_miniimagenet_dataloaders_torchmeta(args)
    #dataloader = args.dataloaders
    #print(args.tasksets, "TASKSETTTTT")
    '''
    print(f'{len(dataloader)}')
    for batch_idx, batch in enumerate(dataloader['train']):
        print(f'{batch_idx=}')
        #print(process_meta_batch(args, batch))
        #print(batch)
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)

        spt_x_r = torch.unsqueeze(spt_x,0)
        spt_x_r = torch.reshape(spt_x_r, (2,-1,spt_x_r.size(dim=2),spt_x_r.size(dim=3),spt_x_r.size(dim=4)))
        print(spt_x_r.size())

        print(f'Train inputs shape: {spt_x.size()}')  # (2, 25, 3, 28, 28)
        print(f'Train targets shape: {spt_y.size()}'.format(spt_y.shape))  # (2, 25)

        print(f'Test inputs shape: {qry_x.size()}')  # (2, 75, 3, 28, 28)
        print(f'Test targets shape: {qry_y.size()}')  # (2, 75)
        break
    '''
    #args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)  # TODO, create your own or extend
    #args.dataloaders = args.tasksets
    #args.dataloaders = {'train': args.tasksets.train,'test':args.tasksets.test, 'val':args.tasksets.validation}
    #need to modify a bita
    from uutils.torch_uu.dataloaders.meta_learning.l2l_to_torchmeta_dataloader import get_l2l_torchmeta_dataloaders
    args.dataloaders =  get_l2l_torchmeta_dataloaders(args)



    '''args.dataloaders =  {'train' : BatchMetaDataLoader(trainmd,
                                                batch_size=args.meta_batch_size_train,
                                                num_workers=args.num_workers),
                        'test':BatchMetaDataLoader(testmd,
                                                batch_size=args.meta_batch_size_train,
                                                num_workers=args.num_workers),
                        'val' : BatchMetaDataLoader(valmd,
                                                batch_size=args.meta_batch_size_train,
                                                num_workers=args.num_workers)}
    '''
    #print(args.tasksets)
    print(args.dataloaders)
    # meta_dataloader = dataloaders['train']
    #print(args.dataloaders['val'])
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
        for it in range(args.num_its):
            comparison_via_performance(args)
    elif args.experiment_option.startswith('diveristiy'):
        do_diversity_data_analysis(args, meta_dataloader)
    else:
        batch = next(iter(meta_dataloader))
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)

        # -- get comparison - SL vs ML
        # TODO: fix confidence inervals CI
        X: Tensor = qry_x
        if args.experiment_option == 'SL_vs_ML':
            distances_per_data_sets_per_layer: list[
                OrderedDict[LayerIdentifier, float]] = dist_batch_data_sets_for_all_layer(args.mdl1, args.mdl_sl, X, X,
                                                                                          args.layer_names,
                                                                                          args.layer_names,
                                                                                          metric_comparison_type=args.metric_comparison_type,
                                                                                          effective_neuron_type=args.effective_neuron_type,
                                                                                          subsample_effective_num_data_method=args.subsample_effective_num_data_method,
                                                                                          subsample_effective_num_data_param=args.subsample_effective_num_data_param,
                                                                                          metric_as_sim_or_dist=args.metric_as_sim_or_dist)

        # -- get comparison - ML vs A(ML)
        elif args.experiment_option == 'SL_vs_ML':
            inner_opt = get_maml_inner_optimizer(args.mdl1, args.inner_lr)
            distances_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]] = \
                dist_batch_tasks_for_all_layer_mdl_vs_adapted_mdl(
                    mdl=args.mdl1,
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
        # -- get comparison - SL vs A(ML)
        elif args.experiment_option == 'SL_vs_A(AML)':
            inner_opt = get_maml_inner_optimizer(args.mdl1, args.inner_lr)
            distances_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]] = \
                dist_batch_tasks_for_all_layer_different_mdl_vs_adapted_mdl(
                    mdl_fixed=args.mdl_sl, mdl_ml=args.mdl1,
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
        # -- get comparison - LR(SL) vs A(ML)
        elif args.experiment_option == 'LR(SL)_vs_A(ML)':
            inner_opt = get_maml_inner_optimizer(args.mdl1, args.inner_lr)
            distances_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]] = \
                dist_batch_tasks_for_all_layer_different_mdl_vs_adapted_mdl(
                    mdl_fixed=args.mdl_sl, mdl_ml=args.mdl1,
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
        # -- get comparison - SL vs MAML(SL)
        elif args.experiment_option == 'SL vs MAML(SL)':
            args.mdl_sl.model.cls = deepcopy(args.mdl1.model.cls)  # todo - comment why this
            print(
                '-> sl_mdl has the head of the maml model to make comparisons using maml better, it does not affect when '
                'fitting the final layer with LR FFL')
            inner_opt = get_maml_inner_optimizer(args.mdl_sl, args.inner_lr)
            distances_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]] = \
                dist_batch_tasks_for_all_layer_different_mdl_vs_adapted_mdl(
                    mdl_fixed=args.mdl_sl, mdl_ml=args.mdl_sl,
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
        else:
            raise ValueError(f'Invalid experiment option, got{args.args.experiment_option=}')

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

    # - done!
    print(f'time_passed_msg = {uutils.report_times(start)}')
    # - wandb
    if is_lead_worker(args.rank) and args.log_to_wandb:
        import wandb
        wandb.finish()


def main_data_analyis_check_sl_error():
    args: Namespace = load_args()

    performance_comparison_with_l2l_end_to_end(args)

    # - done!
    print(f'time_passed_msg = {uutils.report_times(start)}')
    # - wandb
    if is_lead_worker(args.rank) and args.log_to_wandb:
        import wandb
        wandb.finish()

def mean_confidence_interval(data, confidence: float = 0.95) -> tuple[float, np.ndarray]:
    """
    Returns (tuple of) the mean and confidence interval for given data.
    Data is a np.arrayable iterable.
    e.g.
        - list of floats [1.0, 1.3, ...]
        - tensor of size [B]

    ref:
        - https://stackoverflow.com/a/15034143/1601580
        - https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/eval/meta_eval.py#L19
    """

    a: np.ndarray = 1.0 * np.array(data)
    n: int = len(a)
    if n == 1:
        import logging
        logging.warning('The first dimension of your data is 1, perhaps you meant to transpose your data? or remove the'
                        'singleton dimension?')
    m, se = a.mean(), scipy.stats.sem(a)
    tp = scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    h = se * tp
    return m, h

if __name__ == '__main__':
    main_data_analyis()
    '''
    print(diversity_src.data_analysis.common.maml5train)
    print(diversity_src.data_analysis.common.maml5test)
    print(diversity_src.data_analysis.common.maml5val)
    print(diversity_src.data_analysis.common.maml10train)
    print(diversity_src.data_analysis.common.maml10test)
    print(diversity_src.data_analysis.common.maml10val)
    print(diversity_src.data_analysis.common.usltrain)
    print(diversity_src.data_analysis.common.usltest)
    print(diversity_src.data_analysis.common.uslval)

    #summary stats
    print("maml5train", mean_confidence_interval(diversity_src.data_analysis.common.maml5train))
    print("maml5test", mean_confidence_interval(diversity_src.data_analysis.common.maml5test))
    print("maml5val", mean_confidence_interval(diversity_src.data_analysis.common.maml5val))
    print("maml10train", mean_confidence_interval(diversity_src.data_analysis.common.maml10train))
    print("maml10test", mean_confidence_interval(diversity_src.data_analysis.common.maml10test))
    print("maml10val", mean_confidence_interval(diversity_src.data_analysis.common.maml10val))
    print("usltrain", mean_confidence_interval(diversity_src.data_analysis.common.usltrain))
    print("usltest", mean_confidence_interval(diversity_src.data_analysis.common.usltest))
    print("uslval", mean_confidence_interval(diversity_src.data_analysis.common.uslval))
    '''



    # main_data_analyis_check_sl_error()
    print('--> Success Done!\a\n')
