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

import torch

from pathlib import Path

from torch import Tensor

from anatome.helper import compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks, pprint_results, \
    compute_stats_from_distance_per_batch_of_data_sets_per_layer, LayerIdentifier, dist_batch_data_sets_for_all_layer

import uutils

import time

from diversity_src.data_analysis.common import get_sl_learner, get_maml_meta_learner, santity_check_maml_accuracy, \
    comparison_via_performance, setup_args_path_for_ckpt_data_analysis, do_diversity_data_analysis
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

start = time.time()


# - MI

def resnet12rfs_mi(args: Namespace) -> Namespace:
    """
    """
    from uutils.torch_uu.models.resnet_rfs import get_recommended_batch_size_cifarfs_resnet12rfs_body, \
        get_feature_extractor_conv_layers
    # - model
    args.model_option = 'resnet12_rfs_mi'

    # - data
    # args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    # args.data_path = Path('~/data/l2l_data/').expanduser()
    # args.data_augmentation = 'rfs2020'
    args.data_option = 'torchmeta_miniimagenet'  # no name assumes l2l
    args.data_path = Path('~/data/torchmeta_data/').expanduser()
    # args.data_option = 'rfs_meta_learning_miniimagenet'  # no name assumes l2l
    # args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()
    args.augment_train = True

    # - training mode
    args.training_mode = 'iterations'

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

    # args.n_aug_support_samples = 1
    # args.n_aug_support_samples = 5
    # args.n_aug_support_samples = 10
    # args.n_aug_support_samples = 15
    # args.k_shots = 1
    # args.k_shots = 5
    # args.k_shots = 10
    # args.k_shots = 15
    # args.k_shots = 30

    args.safety_margin = 10
    # args.safety_margin = 20

    # args.batch_size = 2
    # args.batch_size = 25
    # args.batch_size = 30
    # args.batch_size = 100
    args.batch_size = 200
    # args.batch_size = 600
    args.batch_size_eval = args.batch_size

    # - set k_eval (qry set batch_size) to make experiments safe/reliable
    args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_body(safety_margin=args.safety_margin)
    # args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_head(safety_margin=args.safety_margin)

    # - expt option
    args.experiment_option = 'performance_comparison'

    # args.experiment_option = 'diveristiy_f_rand'
    # args.experiment_option = 'diveristiy_f_maml'
    # args.experiment_option = 'diveristiy_f_sl'

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner_default'

    # - ckpt name
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/t9hpyoms?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Feb25_13-23-05_jobid_32368_pid_112292'  # SL SGD CL to see if it beats maml/has same test acc as in original rfs paper
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/1gxb5uds?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Feb10_13-36-08_jobid_3381_pid_109779/'  # Adam CL
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/qlubpsfi?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Feb10_18-21-11_jobid_18097_pid_229674/'  # Adam CL
    # https://github.com/WangYueFt/rfs
    args.path_2_init_sl = '~/data/rfs_checkpoints/mini_simple.pt'
    # args.path_2_init_sl = '~/data/rfs_checkpoints/mini_distilled.pt'

    # original ckpt (likely not compatible with this code)
    # args.path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/nov_all_mini_imagenet_expts/logs_Nov05_15-44-03_jobid_668'
    # old checkpoint 688 in new format location
    args.path_2_init_maml = '~/data/logs/logs_Nov05_15-44-03_jobid_668_NEW_CKPT/'  # Adam (no CL, old higher ckpt)
    # new ckpt using l2l https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/jakzsyhv?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_Feb17_15-28-58_jobid_8957_pid_206937/'
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2w2iezpb?workspace=user-brando
    # args.path_2_init_maml = '/home/miranda9/data/logs/logs_Feb27_09-11-46_jobid_14483_pid_16068'

    # reproduction in l2l
    # /home/miranda9/data/logs/logs_Feb27_09-11-46_jobid_14483_pid_16068, https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2w2iezpb/overview?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_Feb27_09-11-46_jobid_14483_pid_16068'

    # - device
    # args.device = torch.device('cpu')
    # args.device = get_device()

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'{args.experiment_option}_resnet12rfs_mi_k_shots_1_5_10_15_30'
    args.run_name = f'{args.experiment_option} {args.model_option} {args.batch_size} {args.metric_comparison_type}: {args.jobid=} {args.path_2_init_sl=} {args.path_2_init_maml=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
    # - fill in the missing things and make sure things make sense for run
    args = uutils.setup_args_for_experiment(args)
    return args


def args_5cnn_mi(args: Namespace) -> Namespace:
    """
    """
    from uutils.torch_uu.models.resnet_rfs import get_recommended_batch_size_cifarfs_resnet12rfs_body, \
        get_feature_extractor_conv_layers
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'

    # - data
    args.data_option = 'torchmeta_miniimagenet'  # no name assumes l2l
    args.data_path = Path('~/data/torchmeta_data/').expanduser()
    args.augment_train = True

    # - training mode
    args.training_mode = 'iterations'

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
    # args.batch_size = 25
    args.batch_size = 100
    args.batch_size_eval = args.batch_size

    # - set k_eval (qry set batch_size) to make experiments safe/reliable
    args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_body(safety_margin=args.safety_margin)
    # args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_head(safety_margin=args.safety_margin)

    # - expt option
    args.experiment_option = 'performance_comparison'

    # args.experiment_option = 'diveristiy_f_rand'
    # args.experiment_option = 'diveristiy_f_maml'
    # args.experiment_option = 'diveristiy_f_sl'

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner_default'

    # - ckpt name
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2y7mrwx3/logs?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Feb15_16-38-47_jobid_10063_pid_185552'  # idk, not a fan of my current ckpts...with l2l, seems need to use SGD?
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/33frd31p?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_Feb19_10-13-22_jobid_2411_pid_28656'  # good! SGD converged with 0.9994 train acc
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/29hc25u2/overview?workspace=user-brando
    args.path_2_init_maml = '~/data/logs/logs_Feb16_11-59-55_jobid_29315_pid_102939'

    # path_2_init_sl = '~/data_folder_fall2020_spring2021/logs/mar_all_mini_imagenet_expts/logs_Mar05_17-57-23_jobid_4246'
    # path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/meta_learning_expts/logs_Mar09_12-20-03_jobid_14_pid_183122'
    # path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/meta_learning_expts/logs_Mar09_12-17-50_jobid_13_pid_177628/'

    # - device
    # args.device = torch.device('cpu')
    # args.device = get_device()

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'{args.experiment_option}_args_5cnn_mi'
    args.run_name = f'{args.experiment_option} {args.model_option} {args.batch_size} {args.metric_comparison_type}: {args.jobid=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
    # - fill in the missing things and make sure things make sense for run
    args = uutils.setup_args_for_experiment(args)
    return args


def old_5ccnn():
    # -- checkpoints SL & MAML
    # 5CNN
    # ####ckpt_filename = 'ckpt_file_best_loss.pt'  # idk if the they have the same acc for this one, the goal is to minimize diffs so that only SL & MAML is the one causing the difference
    path_2_init_sl = '~/data_folder_fall2020_spring2021/logs/mar_all_mini_imagenet_expts/logs_Mar05_17-57-23_jobid_4246'
    # path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/meta_learning_expts/logs_Mar09_12-20-03_jobid_14_pid_183122'
    path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/meta_learning_expts/logs_Mar09_12-17-50_jobid_13_pid_177628/'


# - cifarfs


def args_5cnn_cifarfs(args: Namespace) -> Namespace:
    """
    """
    from uutils.torch_uu.models.resnet_rfs import get_recommended_batch_size_cifarfs_resnet12rfs_body, \
        get_feature_extractor_conv_layers
    # - model
    args.model_option = '4CNN_l2l_cifarfs'

    # - data
    args.data_option = 'torchmeta_cifarfs'  # no name assumes l2l
    args.data_path = Path('~/data/torchmeta_data/').expanduser()
    args.augment_train = True

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000
    # args.opt_option = 'Sgd_rfs'
    # args.num_epochs = 600
    # args.batch_size = 1024
    # args.lr = 5e-2
    # args.opt_hps: dict = dict(lr=args.lr, momentum=0.9, weight_decay=5e-4)
    #
    # args.scheduler_option = 'Cosine_scheduler_sgd_rfs'
    # args.log_scheduler_freq = 1
    # args.T_max = args.num_epochs // args.log_scheduler_freq
    # args.lr_decay_rate = 1e-1
    # # lr_decay_rate ** 3 does a smooth version of decaying 3 times, but using cosine annealing
    # # args.eta_min = args.lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    # args.scheduler_hps: dict = dict(T_max=args.T_max, lr=args.lr, lr_decay_rate=args.lr_decay_rate)

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
    args.batch_size = 25
    # args.batch_size = 100
    # args.batch_size = 400
    # args.batch_size = 600
    args.batch_size_eval = args.batch_size

    # - set k_eval (qry set batch_size) to make experiments safe/reliable
    args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_body(safety_margin=args.safety_margin)
    # args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_head(safety_margin=args.safety_margin)

    # - expt option
    args.experiment_option = 'performance_comparison'

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner_default'

    # - ckpt name
    # adam models
    #  https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/1yz87dry?workspace=user-brando 13363
    # args.path_2_init_maml = '~/data/logs/logs_Mar02_18-13-23_jobid_13363'  # 0.966 acc, 0.639
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2ni2m08h/overview?workspace=user-brando 13860
    args.path_2_init_maml = '~/data/logs/logs_Mar24_21-06-59_jobid_13860/'  # 1.0 train acc, 0.56 val
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/ehntkv81/overview?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_Mar28_18-31-34_jobid_15884/'

    # sgd models
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/1u7e0gx6?workspace=user-brando 12915, SL
    # args.path_2_init_sl = '~/data/logs/logs_Feb25_14-36-24_jobid_12915'  # 0.9998 acc, na VAL (since it's SL)
    # args.path_2_init_maml = ''

    # adafactor models
    # args.path_2_init_sl = '~/data/logs/logs_Mar29_05-52-51_jobid_15883/'

    # - device
    # args.device = torch.device('cpu')
    # args.device = get_device()

    #
    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'{args.experiment_option}_args_5cnn_cifarfs'
    args.run_name = f'{args.model_option} {args.batch_size} {args.metric_comparison_type}: {args.jobid=} {args.path_2_init_sl} {args.path_2_init_maml}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
    # - fill in the missing things and make sure things make sense for run
    args = uutils.setup_args_for_experiment(args)
    return args


def resnet12rfs_cifarfs(args: Namespace) -> Namespace:
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
    # args.batch_size = 25
    args.batch_size = 100
    args.batch_size_eval = args.batch_size

    # - set k_eval (qry set batch_size) to make experiments safe/reliable
    args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_body(safety_margin=args.safety_margin)
    # args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_head(safety_margin=args.safety_margin)

    # - expt option
    args.experiment_option = 'performance_comparison'

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner_default'

    # - ckpt name
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2rhe2d04?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_Feb25_13-29-11_jobid_2631_pid_129270'  # SL SGD CL! To see if SGD beats my maml adam
    # args.path_2_init_sl = '~/data/logs/logs_Feb10_15-05-22_jobid_20550_pid_94325/'
    # args.path_2_init_sl = '~/data/logs/logs_Feb10_15-05-54_jobid_12449_pid_111612/'
    # args.path_2_init_sl = '~/data/rfs_checkpoints/mini_simple.pt'
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2hjq1vmu/overview?workspace=user-brando 28881
    args.path_2_init_maml = '~/data/logs/logs_Feb10_15-54-14_jobid_28881_pid_101601/'
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2q39rflm?workspace=user-brando
    args.path_2_init_maml = ''
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/13d0aplr?workspace=user-brando
    args.path_2_init_maml = ''

    # - device
    # args.device = torch.device('cpu')
    # args.device = get_device()

    #
    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'{args.experiment_option}_resnet12rfs_cifarfs_600_meta_batch_size'
    args.run_name = f'{args.model_option} {args.batch_size} {args.metric_comparison_type}: {args.jobid=} {args.path_2_init_sl} {args.path_2_init_maml}'
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
    args: Namespace = parse_args_meta_learning()

    # - get manual args
    args: Namespace = args_5cnn_cifarfs(args)
    # args: Namespace = args_5cnn_mi(args)
    # args: Namespace = resnet12rfs_cifarfs(args)
    # args: Namespace = resnet12rfs_mi(args)

    # - over write my manual args (starting args) using the ckpt_args (updater args)
    args.meta_learner = get_maml_meta_learner(args)
    args = uutils.merge_args(starting_args=args.meta_learner.args, updater_args=args)  # second takes priority
    args.meta_learner.args = args  # to avoid meta learner running with args only from past experiment and not with metric analysis experiment

    uutils.print_args(args)
    args.criterion
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
    print(f'-->{args.n_aug_support_samples=}') if hasattr(args, 'n_aug_support_samples') else None
    print(f'-->{args.k_shots=}')

    # - Checks that maml0 acc is lower
    # santity_check_maml_accuracy(args)

    # -- do data analysis
    if args.experiment_option == 'performance_comparison':
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


if __name__ == '__main__':
    main_data_analyis()
    print('--> Success Done!\a\n')
