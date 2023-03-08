# %%
"""
main script for computing performances of mdls and dist(f, A(f)) vs the model trained.
"""
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from pprint import pprint

from pathlib import Path

import torch
from torch import Tensor

# # from anatome.helper import compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks, pprint_results, \
#     compute_stats_from_distance_per_batch_of_data_sets_per_layer, LayerIdentifier, dist_batch_data_sets_for_all_layer

import uutils

import time

from diversity_src.data_analysis.common import get_sl_learner, get_maml_meta_learner, comparison_via_performance, \
    setup_args_path_for_ckpt_data_analysis, \
    get_recommended_batch_size_miniimagenet_5CNN
from diversity_src.data_analysis.stats_analysis_with_emphasis_on_effect_size import \
    stats_analysis_with_emphasis_on_effect_size

from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility, parse_args_meta_learning
from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloaders

from uutils.torch_uu import process_meta_batch, norm
from uutils.torch_uu.distributed import is_lead_worker
from uutils.torch_uu.meta_learners.maml_differentiable_optimizer import get_maml_inner_optimizer, \
    dist_batch_tasks_for_all_layer_mdl_vs_adapted_mdl, dist_batch_tasks_for_all_layer_different_mdl_vs_adapted_mdl
from uutils.torch_uu.models import reset_all_weights

from uutils.torch_uu.models.resnet_rfs import get_recommended_batch_size_mi_resnet12rfs_body

from uutils.argparse_uu.common import setup_args_for_experiment

from uutils.logging_uu.wandb_logging.common import try_printing_wandb_url

import os

start = time.time()


# - MI

def resnet12rfs_mi(args: Namespace) -> Namespace:
    """
    """
    from uutils.torch_uu.models.resnet_rfs import get_feature_extractor_conv_layers
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
    # args.batch_size = 5
    args.batch_size = 10
    # args.batch_size = 25
    # args.batch_size = 30
    # args.batch_size = 100
    # args.batch_size = 200
    # args.batch_size = 600
    args.batch_size_eval = args.batch_size

    # - set k_eval (qry set batch_size) to make experiments safe/reliable
    args.k_eval = get_recommended_batch_size_mi_resnet12rfs_body(safety_margin=args.safety_margin)
    # args.k_eval = get_recommended_batch_size_mi_resnet12rfs_head(safety_margin=args.safety_margin)

    # - expt option
    args.stats_analysis_option = 'performance_comparison'

    # args.stats_analysis_option = 'diveristiy_f_rand'
    # args.stats_analysis_option = 'diveristiy_f_maml'
    # args.stats_analysis_option = 'diveristiy_f_sl'

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner'

    # - ckpt name
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/qlubpsfi?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_Feb10_18-21-11_jobid_18097_pid_229674/'  # Adam CL (THIS ONE)
    # https://github.com/WangYueFt/rfs
    # args.path_2_init_sl = '~/data/rfs_checkpoints/mini_simple.pt'
    # args.path_2_init_sl = '~/data/rfs_checkpoints/mini_distilled.pt'

    # original old ckpt (likely not compatible with this code) in old format
    # #### (OLD) args.path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/nov_all_mini_imagenet_expts/logs_Nov05_15-44-03_jobid_668'
    # usable old checkpoint 688 in new format location
    args.path_2_init_maml = '~/data/logs/logs_Nov05_15-44-03_jobid_668_NEW_CKPT/'  # Adam (no CL, old higher ckpt)

    # new ckpt using l2l https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/jakzsyhv?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_Feb17_15-28-58_jobid_8957_pid_206937/'  # Adam CL

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
    args.experiment_name = f'{args.stats_analysis_option}_resnet12rfs_mi_k_shots_1_5_10_15_30'
    args.run_name = f'{args.stats_analysis_option} {args.model_option} {args.batch_size} {args.metric_comparison_type}: {args.jobid=} {args.path_2_init_sl=} {args.path_2_init_maml=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
    return args


def args_5cnn_mi(args: Namespace) -> Namespace:
    from uutils.torch_uu.models.resnet_rfs import get_feature_extractor_conv_layers
    # - model
    args.model_option = '5CNN_opt_as_model_for_few_shot_sl'

    # - data
    args.data_option = 'torchmeta_miniimagenet'  # no name assumes l2l
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
    # args.batch_size = 5
    # args.batch_size = 25
    args.batch_size = 100
    # args.batch_size = 500
    args.batch_size_eval = args.batch_size

    # - set k_eval (qry set batch_size) to make experiments safe/reliable
    args.k_eval = get_recommended_batch_size_miniimagenet_5CNN(safety_margin=args.safety_margin)
    # args.k_eval = get_recommended_batch_size_miniimagenet_head_5CNN(safety_margin=args.safety_margin)

    # - expt option
    args.stats_analysis_option = 'performance_comparison'

    # args.stats_analysis_option = 'diveristiy_f_rand'
    # args.stats_analysis_option = 'diveristiy_f_maml'
    # args.stats_analysis_option = 'diveristiy_f_sl'

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner'

    # - ckpt name

    # -- 2

    # -- 4
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/7sfgxss7/logs?workspace=user-brando
    # args.path_2_init_sl = ''  # Adam
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/16id6w11
    # args.path_2_init_sl = ''  # Adam
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/3susfzse
    # args.path_2_init_sl = ''  # Adam
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/hdgn6xfd
    # args.path_2_init_sl = ''  # Adam
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/l9rdcfcr?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_May03_18-05-01_jobid_26093'  # Adam

    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/ip20v98t/logs?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_May05_11-31-02_jobid_27495'  # Adam

    # -- 8
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/35qt9vlj?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_May05_11-44-39_jobid_27496'  # Adam

    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/1ws40w58/logs?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_May05_11-27-28_jobid_27494'  # Adam

    # -- 16
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/3o5rsvne?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_May02_17-15-30_jobid_25765'  # Adam

    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2d06xdie/overview?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_May02_17-22-17_jobid_25766'  # Adam

    # -- 32
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/e86rmved?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Apr11_14-52-47_jobid_29253_pid_21780'  # Adam
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/1tzdyoxp?workspace=
    # args.path_2_init_sl = '~/data/logs/logs_Apr11_14-53-51_jobid_9971_pid_25156'  # Adam
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/1uyz497h?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_May02_17-05-36_jobid_25763'  # Adam THIS ONE

    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/29hc25u2/overview?workspace=user-brando (NOT GOOD)
    # args.path_2_init_maml = '~/data/logs/logs_Feb16_11-59-55_jobid_29315_pid_102939'  # uses scheduler :'(
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/kpujevkp?workspace=user-brando
    args.path_2_init_maml = '~/data/logs/logs_May02_17-11-03_jobid_25764'  # Adam

    # -- 128 version todo, both need to be with adam.
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/9r7q98vz?workspace=user-brando
    # args.path_2_init_sl = ''  #
    #
    # args.path_2_init_maml = ''

    # -- 512 version todo, both need to be with adam.
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2tmzp9p5?workspace=
    # args.path_2_init_sl = ''  #
    #
    # args.path_2_init_maml = ''

    # -- 1024 version todo, both need to be with adam.
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/gdmphf5w/logs?workspace=user-brando
    # args.path_2_init_sl = ''  #
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/wi1whoh0?workspace=user-brando
    # args.path_2_init_sl = ''  #
    #
    # args.path_2_init_maml = ''

    # -- 32 (2nd round)
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/e86rmved?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Apr11_14-52-47_jobid_29253_pid_21780'  # Adam
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/1tzdyoxp?workspace=
    # args.path_2_init_sl = '~/data/logs/logs_Apr11_14-53-51_jobid_9971_pid_25156'  # Adam
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/1uyz497h?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_May02_17-05-36_jobid_25763'  # Adam THIS ONE
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/3n1ryuzu/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_May24_11-45-30_jobid_35317'  # 55
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/3v7fpsie/overview?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_May24_11-47-48_jobid_35318'  # 56
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/yjcun827/overview?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_May24_11-50-06_jobid_35319'  # 54
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/vh1ecgr3/overview?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_May24_11-52-24_jobid_35320'  # 57
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/lhsmwapf/overview?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_May24_11-55-39_jobid_35321'  # 54
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/9s93yjt6/overview?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_May24_11-55-43_jobid_35322'  # 55

    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/1yhb8bqd/overview?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_May24_11-55-44_jobid_35323'  # 54

    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2q6jn6h6/overview?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_May24_11-55-43_jobid_35324'  # 55
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/hqfxsf5r/overview?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_May24_11-55-44_jobid_35325'  # 56

    # actually you need to run _main_dista... old code I think
    # path_2_init_sl = '~/data_folder_fall2020_spring2021/logs/mar_all_mini_imagenet_expts/logs_Mar05_17-57-23_jobid_4246'  # THIS I think
    # path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/meta_learning_expts/logs_Mar09_12-20-03_jobid_14_pid_183122'
    # path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/meta_learning_expts/logs_Mar09_12-17-50_jobid_13_pid_177628/'  # THIS I think

    # - device
    # args.device = torch.device('cpu')
    # args.device = get_device()

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'{args.stats_analysis_option}_args_5cnn_mi'
    args.run_name = f'{args.stats_analysis_option} {args.model_option} {args.batch_size} {args.metric_comparison_type}: {args.jobid=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
    return args


def old_5ccnn():
    # -- checkpoints SL & MAML
    # 5CNN
    # ####ckpt_filename = 'ckpt_file_best_loss.pt'  # idk if the they have the same acc for this one, the goal is to minimize diffs so that only SL & MAML is the one causing the difference
    path_2_init_sl = '~/data_folder_fall2020_spring2021/logs/mar_all_mini_imagenet_expts/logs_Mar05_17-57-23_jobid_4246'
    path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/meta_learning_expts/logs_Mar09_12-20-03_jobid_14_pid_183122'
    path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/meta_learning_expts/logs_Mar09_12-17-50_jobid_13_pid_177628/'


# - cifarfs


def args_5cnn_cifarfs(args: Namespace) -> Namespace:
    """
    """
    from uutils.torch_uu.models.resnet_rfs import get_feature_extractor_conv_layers
    # - model
    args.model_option = '4CNN_l2l_cifarfs'

    # - data
    args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'rfs2020'
    # args.data_option = 'torchmeta_cifarfs'  # no name assumes l2l
    # args.data_path = Path('~/data/torchmeta_data/').expanduser()
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
    args.batch_size = 5
    # args.batch_size = 30
    # args.batch_size = 100
    # args.batch_size = 400
    # args.batch_size = 600
    args.batch_size_eval = args.batch_size

    # - set k_eval (qry set batch_size) to make experiments safe/reliable
    # args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_body(safety_margin=args.safety_margin)
    # args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_head(safety_margin=args.safety_margin)
    args.k_eva = 5

    # - expt option
    args.stats_analysis_option = 'performance_comparison'

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner'

    # - ckpt name
    # adam models
    #  https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/1yz87dry?workspace=user-brando 13363
    # args.path_2_init_maml = '~/data/logs/logs_Mar02_18-13-23_jobid_13363'  # 0.966 acc, 0.639
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2ni2m08h/overview?workspace=user-brando 13860
    args.path_2_init_maml = '~/data/logs/logs_Mar24_21-06-59_jobid_13860/'  # 1.0 train acc, 0.56 val  # THIS ONE FOR RESULTS

    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/tmp1d5u2/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar28_18-31-34_jobid_15883/'  # 0.9899 train acc
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/ehntkv81/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar28_18-31-34_jobid_15884/'  # 0.988 train acc
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2f5m59ys/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar28_18-31-36_jobid_15885/'  # 0.988 train acc
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/378tku4q/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar28_18-31-48_jobid_15886/'  # 0.985 train acc
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/m3qbz1bl/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar28_18-31-35_jobid_15887'  # 0.987 train acc
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/1fzto97d?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_Mar30_08-17-19_jobid_17733_pid_142663'  # 0.993 train acc, #THIS ONE FOR RESULTS
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/3lhh7lry/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar30_08-18-46_jobid_28878_pid_153020'  # 0.993 train acc
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2k9udmd3/overview?workspace=user-brando
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2cmtzxhm/overview?workspace=user-brando
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/1hjhrza6/overview?workspace=user-brando
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/kix252xd/overview?workspace=user-brando
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/7njz49ii/overview?workspace=user-brando
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/1qx0qqgw/overview?workspace=user-brando
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2k1hhcca/overview?workspace=user-brando
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/36vraue2/overview?workspace=user-brando

    # adafactor models
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/21dgxvh9?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar28_18-58-06_jobid_15888/'
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/2l22kzde/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar28_18-59-10_jobid_15889/'
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/39wx0tj3/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar28_18-58-06_jobid_15890/'

    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/5g0zo5ti/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar28_19-03-15_jobid_15891/'
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/19yvqm90/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar28_19-03-15_jobid_15892/'
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/3oqrztad/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar28_19-09-03_jobid_15893/'

    #
    # args.path_2_init_sl = '~/data/logs/logs_Mar29_08-22-38_jobid_15889'

    # - device
    # args.device = torch.device('cpu')
    # args.device = get_device()

    #
    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'{args.stats_analysis_option}_args_5cnn_cifarfs'
    args.run_name = f'{args.model_option} {args.batch_size} {args.metric_comparison_type}: {args.jobid=} {args.path_2_init_sl} {args.path_2_init_maml}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
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
    args.stats_analysis_option = 'performance_comparison'

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner'

    # - ckpt name
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/3dx4c9s9?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_Feb10_15-05-22_jobid_20550_pid_94325/'
    # https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/3uwf7b8g/overview?workspace=user-brando
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
    args.experiment_name = f'{args.stats_analysis_option}_resnet12rfs_cifarfs_600_meta_batch_size'
    args.run_name = f'{args.model_option} {args.batch_size} {args.metric_comparison_type}: {args.jobid=} {args.path_2_init_sl} {args.path_2_init_maml}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
    return args


# -- hdb1 mio

def resnet12rfs_hdb1_mio(args):
    # - model
    args.model_option = 'resnet12_hdb1_mio'

    # - data
    args.data_option = 'hdb1'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'hdb1'

    # - training mode
    args.training_mode = 'iterations'  # needed so setup_args doesn't error out (sorry for confusioning line!)
    args.num_its = 6

    # - debug flag
    # args.debug = True
    args.debug = False

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    args.first_order = True

    # args.batch_size = 2  # useful for debugging!
    # args.batch_size = 5  # useful for debugging!
    # args.batch_size = 30
    # args.batch_size = 100
    # args.batch_size = 500
    # args.batch_size = 1000
    # args.batch_size = 2000
    # args.batch_size = 5000
    args.batch_size = 10_000
    args.batch_size_eval = args.batch_size

    # - expt option
    # args.stats_analysis_option = 'performance_comparison'
    # args.stats_analysis_option = 'stats_analysis_with_emphasis_on_effect_size'
    # args.stats_analysis_option = 'stats_analysis_with_emphasis_on_effect_size_and_full_performance_comp_hist'
    args.stats_analysis_option = 'stats_analysis_with_emphasis_on_effect_size_hist'
    args.acceptable_difference1 = 0.01
    args.acceptable_difference2 = 0.02
    args.alpha = 0.01  # not important, p-values is not being emphasized due to large sample size/batch size

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner'
    # args.agent_opt = 'MAMLMetaLearnerL2L_default'  # current code doesn't support this, it's fine I created a l2l -> torchmeta dataloader so we can use the MAML meta-learner that works for pytorch dataloaders

    # - ckpt name
    # https://wandb.ai/brando/entire-diversity-spectrum/runs/3psfe5hn/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Nov01_21-18-12_jobid_102959'  # train_acc 0.970, train_loss 0.119
    args.path_2_init_sl = '~/data/logs/logs_Nov02_15-43-37_jobid_103052'  # train_acc 0.9996, train_loss 0.001050
    # can't find wandb run for bellow
    # args.path_2_init_maml = '~/data/logs/logs_Oct15_18-08-54_jobid_96800'  # train_acc 0.986, train_loss 0.0531, val_acc 0.621
    # training above ckpt for a little longer: https://wandb.ai/brando/entire-diversity-spectrum/runs/1jqbw2cb?workspace=user-brando
    # args.path_2_init_maml = 'need to download ckpt 103320 from vision cluster'
    # https://wandb.ai/brando/entire-diversity-spectrum/runs/1etjuijm/overview?workspace=user-brando
    args.path_2_init_maml = '~/data/logs/logs_Oct15_18-10-01_jobid_96801'  #
    # doesn't seem the run for the ckpt one bellow is easy to find in wandb runs
    # args.path_2_init_maml = '~/data/logs/logs_Oct15_18-11-20_jobid_96802'  #
    # doesn't seem the run for the ckpt one bellow is easy to find in wandb runs
    # args.path_2_init_maml = '~/data/logs/logs_Oct15_18-12-26_jobid_96803'  #
    # old vision ckpts
    # https://wandb.ai/brando/entire-diversity-spectrum/runs/203j6c16?workspace=user-brando

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    args.experiment_name = args.manual_loads_name
    args.run_name = f'{args.model_option} {args.batch_size} {args.stats_analysis_option}: {args.jobid=} {args.path_2_init_sl} {args.path_2_init_maml}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
    return args


# -- hdb4 micod

def hdb4_micod(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet12_rfs'

    # - data
    args.data_option = 'hdb4_micod'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'hdb4_micod'

    # - training mode
    args.training_mode = 'iterations'  # needed so setup_args doesn't error out (sorry for confusioning line!)
    args.num_its = 6

    # - debug flag
    # args.debug = True
    args.debug = False

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    args.first_order = True

    args.batch_size = 2  # useful for debugging!
    # args.batch_size = 5  # useful for debugging!
    # args.batch_size = 30
    # args.batch_size = 100
    # args.batch_size = 500
    # args.batch_size = 1000
    # args.batch_size = 2000
    # args.batch_size = 5000
    # args.batch_size = 10_000
    args.batch_size_eval = args.batch_size

    # - expt option
    # args.stats_analysis_option = 'performance_comparison'
    # args.stats_analysis_option = 'stats_analysis_with_emphasis_on_effect_size'
    # args.stats_analysis_option = 'stats_analysis_with_emphasis_on_effect_size_and_full_performance_comp_hist'
    args.stats_analysis_option = 'stats_analysis_with_emphasis_on_effect_size_hist'
    args.acceptable_difference1 = 0.01
    args.acceptable_difference2 = 0.02
    args.alpha = 0.01  # not important, p-values is not being emphasized due to large sample size/batch size

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner'
    # args.agent_opt = 'MAMLMetaLearnerL2L_default'  # current code doesn't support this, it's fine I created a l2l -> torchmeta dataloader so we can use the MAML meta-learner that works for pytorch dataloaders

    # - ckpt name
    # super trained: https://wandb.ai/brando/entire-diversity-spectrum/runs/3kod7pdv
    args.path_2_init_sl = ''  #
    # super trained: https://wandb.ai/brando/entire-diversity-spectrum/runs/2lmyr2lk
    args.path_2_init_sl = ''  #
    # train to ~0.90 accs: https://wandb.ai/brando/entire-diversity-spectrum/runs/wxrh4t0s
    args.path_2_init_sl = ''  #
    # trained to ~0.92 accs: https://wandb.ai/brando/entire-diversity-spectrum/runs/26c6m7ed
    # args.path_2_init_sl = '~/data/logs/logs_Jan20_14-47-00_jobid_-1'  # train acc 0.921875, train loss 0.25830933451652527
    # trained to 0.98828125 accs: https://wandb.ai/brando/entire-diversity-spectrum/runs/3kod7pdv?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Jan26_20-35-37_jobid_923629_pid_653526_wandb_True'  # train acc 0.98828125, ampere4
    # 5ccn 2 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/u1ndwad4/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Feb09_19-37-27_jobid_667717_pid_967757_wandb_True' # ampere1
    # 5cnn 4 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/r8xgfx07?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Feb02_14-00-31_jobid_43228_pid_2821217_wandb_True'  # ampere3
    # 5cnn 6 filers: https://wandb.ai/brando/entire-diversity-spectrum/runs/v8wih11u/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Feb09_19-29-40_jobid_486495_pid_934615_wandb_True'  # ampere1
    # 5cnn 8 filers: https://wandb.ai/brando/entire-diversity-spectrum/runs/klzycucu/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Feb04_17-27-39_jobid_191466_pid_101120_wandb_True'  # ampere2
    # 5cnn 12 filter: https://wandb.ai/brando/entire-diversity-spectrum/runs/exjfe0ra/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Feb10_14-41-07_jobid_341648_pid_3896402_wandb_True'  # ampere1, corresponding maml failed
    # 5cnn 16 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/1hmce6w2?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar03_14-38-22_jobid_892425_pid_2905700_wandb_True'  # ampere4
    # 5cnn 32 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/fnmjoy4e/overview?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Feb05_22-46-35_jobid_38937_pid_2768598_wandb_True'  # ampere3
    # 5ccn 64 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/1q25bgx0?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Feb07_00-50-02_jobid_486495_pid_1613676_wandb_True'  # ampere1
    # # 5cnn 256 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/fuxwz30l/overview?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_Feb04_17-38-21_jobid_855372_pid_2723881_wandb_True'  # ampere1
    # 5ccn 512 flters: https://wandb.ai/brando/entire-diversity-spectrum/runs/cstug9f3?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_Feb09_19-33-20_jobid_899282_pid_948111_wandb_True'  # ampere1

    # https://wandb.ai/brando/entire-diversity-spectrum/runs/16fnx8of/overview?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_Jan20_12-40-05_jobid_-1'  # train acc 0.9266666769981384, train loss 0.2417697161436081
    # https://wandb.ai/brando/entire-diversity-spectrum/runs/2rkhpnbx/overview?workspace=user-brando
    # args.path_2_init_maml = ''  # train acc 0.9266666769981384, train loss 0.2417697161436081
    # https://wandb.ai/brando/entire-diversity-spectrum/runs/11od07w0/overview?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_Jan26_20-28-37_jobid_406367_pid_649975_wandb_True'  # train acc 0.9911110997200012, ampere4
    # 5ccn 2 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/bjxl55ul/overview?workspace=
    # args.path_2_init_maml = '~//data/logs/logs_Feb09_20-11-25_jobid_178745_pid_1187212_wandb_True'  # ampere1
    # 5cnn 4 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/sgoiu5tx/overview?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_Feb02_14-00-49_jobid_991923_pid_2822438_wandb_True'  # ampere1,2,3,4
    # 5cnn 4 filers redo: https://wandb.ai/brando/entire-diversity-spectrum/runs/ulcwxwl0/overview?workspace=user-brando
    # args.path_2_init_maml = ''  # ampere1
    # 5cnn 6 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/1npe2tv4?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_Feb09_20-13-39_jobid_64221_pid_1202222_wandb_True'  # ampere1
    # 5cnn 8 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/6qgk090q/overview?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_Feb04_17- 31-05_jobid_28465_pid_102367_wandb_True' # ampere2
    # 5cnn 12 filter: https://wandb.ai/brando/entire-diversity-spectrum/runs/fdnl1d1c/overview?workspace=user-brando
    # args.path_2_init_maml = 'FAILED'
    # 5cnn 16 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/1d4mp962?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_Mar03_14-32-30_jobid_998225_pid_2899341_wandb_True'  # ampere4
    # 5cnn 32 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/esu6l2gi/overview?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_Feb05_22-44-43_jobid_851192_pid_2766216_wandb_True'  # ampere3
    # 5cnn 64 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/nzvm7g44/overview?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_Feb07_00-48-16_jobid_670102_pid_1612658_wandb_True'  # ampere1
    # # 5cnn 256 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/96wo1c43/overview?workspace=user-brando
    args.path_2_init_maml = '~/data/logs/logs_Feb04_17-39-17_jobid_568243_pid_2724751_wandb_True'  # ampere1
    # 5cnn 512 filters: https://wandb.ai/brando/entire-diversity-spectrum/runs/6gte637k?workspace=user-brando
    args.path_2_init_maml = '~/data/logs/logs_Feb09_20-11-20_jobid_77267_pid_1186966_wandb_True'  # ampere1

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    args.experiment_name = f'{args.manual_loads_name} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {args.stats_analysis_option}: {args.jobid=} {args.path_2_init_sl} {args.path_2_init_maml}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
    return args


# -- mds vggflower+aircraft resnet18_rfs

def resnet18rfs_vggaircraft(args) -> Namespace:
    """
        """
    # from uutils.torch_uu.models.resnet_rfs import get_recommended_batch_size_cifarfs_resnet12rfs_body, \
    #     get_feature_extractor_conv_layers
    # - model
    args.model_option = 'resnet18_rfs'

    # - data
    # args.data_option = 'torchmeta_cifarfs'  # no name assumes l2l
    # args.data_path = Path('~/data/torchmeta_data/').expanduser()
    # args.augment_train = True
    args.data_option = 'mds'
    args.sources = ['vgg_flower', 'aircraft']
    # args.data_path = Path('~/data/l2l_data/').expanduser()
    # args.data_augmentation = 'mds'

    # - training mode
    args.training_mode = 'iterations'  # needed so setup_args doesn't error out

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # -- Meta-Learner
    # - maml
    args.lr = 1e-3

    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    args.first_order = True

    # - outer trainer params
    # args.batch_size = 32
    # args.batch_size = 8

    # - dist args
    # args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    # args.parallel = True
    # args.seed = 42  # I think this might be important due to how tasksets works.
    # args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # -
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
    # args.effective_neuron_type = 'filter'

    # - layers, this gets the feature layers it seems... unsure why I'm doing this. I thought I was doing a comparison
    # with all the layers up to the final layer...
    # args.layer_names: list[str] = get_last_two_layers(layer_type='conv', include_cls=True)
    # args.layer_names = get_head_cls()
    # args.layer_names = get_feature_extractor_conv_layers()

    # args.safety_margin = 10
    # args.safety_margin = 20

    args.batch_size = 500
    # args.batch_size = 10_000
    args.batch_size_eval = args.batch_size

    # - set k_eval (qry set batch_size) to make experiments safe/reliable
    # args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_body(safety_margin=args.safety_margin)
    # args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_head(safety_margin=args.safety_margin)

    # - expt option
    args.stats_analysis_option = 'performance_comparison'

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner'
    # args.agent_opt = 'MAMLMetaLearnerL2L_default'  # current code doesn't support this, it's fine I created a l2l -> torchmeta dataloader so we can use the MAML meta-learner that works for pytorch dataloaders

    args.path_2_init_sl = '~/data/logs/logs_Jan21_14-02-12_jobid_-1'  # train_acc 0.9922 loss 0.027
    args.path_2_init_maml = '~/data/logs/logs_Jan23_22-40-05_jobid_-1'  # train acc 0.98 loss 0.05 (this is a "continued" ckpt)
    # '~/data/logs/logs_Jan21_13-56-48_jobid_-1'  # train acc 0.9667 and rising

    # -- wandb args
    args.wandb_project = 'SL vs MAML MDS Subsets'  # 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'{args.stats_analysis_option}_resnet18rfs_mds_vggaircraft'
    args.run_name = f'{args.model_option} {args.batch_size} {args.metric_comparison_type}: {args.jobid=} {args.path_2_init_sl} {args.path_2_init_maml}'
    # args.log_to_wandb = True
    args.log_to_wandb = True

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
    return args


# -- hdb5

def hdb5_vggair(args):
    """
        """
    # from uutils.torch_uu.models.resnet_rfs import get_recommended_batch_size_cifarfs_resnet12rfs_body, \
    #     get_feature_extractor_conv_layers
    # - model
    args.model_option = 'resnet12_rfs'
    # args.model_option = '5CNN_opt_as_model_for_few_shot_sl'
    args.data_path = '/home/pzy2/data/l2l_data'
    # - data
    # args.data_option = 'torchmeta_cifarfs'  # no name assumes l2l
    # args.data_path = Path('~/data/torchmeta_data/').expanduser()
    # args.augment_train = True
    args.data_option = 'hdb5_vggair'
    args.data_augmentation = 'hdb5_vggair'
    # args.sources = ['vgg_flower', 'aircraft']
    # args.data_path = Path('~/data/l2l_data/').expanduser()
    # args.data_augmentation = 'mds'

    # - training mode
    args.training_mode = 'iterations'  # needed so setup_args doesn't error out

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 100_000

    # - debug flag
    # args.debug = True
    args.debug = False

    # -- Meta-Learner
    # - maml
    args.lr = 1e-3

    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    args.first_order = True

    # - outer trainer params
    # args.batch_size = 32
    # args.batch_size = 8

    # - dist args
    # args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    # args.parallel = True
    # args.seed = 42  # I think this might be important due to how tasksets works.
    # args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # -
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
    # args.effective_neuron_type = 'filter'

    # - layers, this gets the feature layers it seems... unsure why I'm doing this. I thought I was doing a comparison
    # with all the layers up to the final layer...
    # args.layer_names: list[str] = get_last_two_layers(layer_type='conv', include_cls=True)
    # args.layer_names = get_head_cls()
    # args.layer_names = get_feature_extractor_conv_layers()

    # args.safety_margin = 10
    # args.safety_margin = 20

    args.batch_size = 500  # 1000#500#500#500#500
    # args.batch_size = 10_000
    args.batch_size_eval = args.batch_size

    # - set k_eval (qry set batch_size) to make experiments safe/reliable
    # args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_body(safety_margin=args.safety_margin)
    # args.k_eval = get_recommended_batch_size_cifarfs_resnet12rfs_head(safety_margin=args.safety_margin)

    # - expt option
    # args.stats_analysis_option = 'performance_comparison'
    args.stats_analysis_option = 'stats_analysis_with_emphasis_on_effect_size_hist'
    args.acceptable_difference1 = 0.01
    args.acceptable_difference2 = 0.02
    args.alpha = 0.01  # not important, p-values is not being emphasized due to large sample size/batch size

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner'
    # args.agent_opt = 'MAMLMetaLearnerL2L_default'  # current code doesn't support this, it's fine I created a l2l -> torchmeta dataloader so we can use the MAML meta-learner that works for pytorch dataloaders

    args.path_2_init_sl = '~/data/logs/logs_Feb05_16-58-24_jobid_-1_pid_72478_wandb_True'  # logs_Feb03_23-08-10_jobid_-1_pid_91763_wandb_True'  #'~/data/logs/logs_Jan21_14-02-12_jobid_-1'  # train_acc 0.9922 loss 0.027
    args.path_2_init_maml = '~/data/logs/logs_Feb03_23-04-38_jobid_-1_pid_7540_wandb_True'  # '~/data/logs/logs_Feb04_22-15-52_jobid_-1_pid_100986_wandb_True' #'~/data/logs/logs_Jan23_22-40-05_jobid_-1'  # train acc 0.98 loss 0.05 (this is a "continued" ckpt)
    # '~/data/logs/logs_Jan21_13-56-48_jobid_-1'  # train acc 0.9667 and rising

    # '~/data/logs/logs_Feb04_13-24-44_jobid_-1_pid_13959_wandb_True'#
    # '~/data/logs/logs_Feb04_13-23-16_jobid_-1_pid_52575_wandb_True'#:

    # -- wandb args
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'hdb5 perf comp'  # 'SL vs MAML MDS Subsets'  # 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'{args.stats_analysis_option}_resnet18rfs_mds_vggaircraft'
    args.run_name = f'{args.model_option} {args.batch_size} {args.metric_comparison_type}: {args.jobid=} {args.path_2_init_sl} {args.path_2_init_maml}'
    # args.log_to_wandb = True
    args.log_to_wandb = True

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
    return args


# -- mds full

def mds_full(args: Namespace) -> Namespace:
    # - model
    args.model_option = 'resnet50_rfs'

    # - data
    args.data_option = 'mds'
    # args.sources = ['vgg_flower', 'aircrddaft']
    # Mscoco, traffic_sign are VAL only (actually we could put them here, fixed script to be able to do so w/o crashing)
    args.sources = ['ilsvrc_2012', 'aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot', 'quickdraw', 'vgg_flower',
                    'mscoco', 'traffic_sign']

    # - training mode
    args.training_mode = 'iterations'  # needed so setup_args doesn't error out (sorry for confusioning line!)
    args.num_its = 6

    # - debug flag
    # args.debug = True
    args.debug = False

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    args.first_order = True

    args.batch_size = 2  # useful for debugging!
    # args.batch_size = 5  # useful for debugging!
    # args.batch_size = 30
    args.batch_size = 100
    # args.batch_size = 500
    # args.batch_size = 1000
    # args.batch_size = 2000
    # args.batch_size = 5000
    # args.batch_size = 10_000
    args.batch_size_eval = args.batch_size

    # - expt option
    # args.stats_analysis_option = 'performance_comparison'
    # args.stats_analysis_option = 'stats_analysis_with_emphasis_on_effect_size'
    # args.stats_analysis_option = 'stats_analysis_with_emphasis_on_effect_size_and_full_performance_comp_hist'
    args.stats_analysis_option = 'stats_analysis_with_emphasis_on_effect_size_hist'
    args.acceptable_difference1 = 0.01
    args.acceptable_difference2 = 0.02
    args.alpha = 0.01  # not important, p-values is not being emphasized due to large sample size/batch size

    # - agent/meta_learner type
    args.agent_opt = 'MAMLMetaLearner'
    # args.agent_opt = 'MAMLMetaLearnerL2L_default'  # current code doesn't support this, it's fine I created a l2l -> torchmeta dataloader so we can use the MAML meta-learner that works for pytorch dataloaders

    # - ckpt name
    # resnet50: https://wandb.ai/brando/entire-diversity-spectrum/runs/1laypoiy?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Feb03_15-02-19_jobid_231971_pid_1421508_wandb_True'  # ampere 4 time of writing 0.820 train acc 0.832 train loss
    # resnet50: https://wandb.ai/brando/entire-diversity-spectrum/runs/1z3mm027?workspace=user-brando
    # args.path_2_init_sl = '~/data/logs/logs_Mar07_12-53-56_jobid_231971_pid_3226323_wandb_True/'  # ampere4
    # resnet50: https://wandb.ai/brando/entire-diversity-spectrum/runs/190osagh?workspace=user-brando
    args.path_2_init_sl = '~/data/logs/logs_Feb03_15-34-50_jobid_343974_pid_1433204_wandb_True'  # ampere 4 time of writing 0.805 train acc 0.934 train loss

    # resnet50: https://wandb.ai/brando/entire-diversity-spectrum/runs/3844zgd4/overview?workspace=user-brando
    # args.path_2_init_maml = '~/data/logs/logs_Feb03_15-23-28_jobid_610011_pid_1428316_wandb_True'  # ampere 4 time of writing 0.964 train acc 0.110 train loss
    # resnet50: https://wandb.ai/brando/entire-diversity-spectrum/runs/30651kln/overview?workspace=user-brando
    args.path_2_init_maml = '~/data/logs/logs_Feb03_14-53-34_jobid_873902_pid_1417320_wandb_True'  # ampere 4 time of writing 0.915 train acc 0.242 train loss

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    args.experiment_name = f'{args.manual_loads_name} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {args.stats_analysis_option}: {args.jobid=} {args.path_2_init_sl} {args.path_2_init_maml}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    # - setup paths to ckpts for data analysis
    args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
    return args


# -- data analysis

def load_args() -> Namespace:
    """
    Get the manual args and replace the missing fields using the args from the ckpt. Then make sure the meta-learner
    has the right args from the data analysis by doing args.meta_learner.args = new_args.
    """
    # -- args from terminal
    args: Namespace = parse_args_meta_learning()

    # -- get manual args
    # - set remaining args values (e.g. hardcoded, checkpoint etc.)
    print(f'{args.manual_loads_name=}')
    args: Namespace = eval(f'{args.manual_loads_name}(args)')
    # do this before we the meta-learner code so to not overwrite the meta-learner stuff accidentally, so yes this must be commented out
    args: Namespace = setup_args_for_experiment(args)

    # -- over write my manual args (starting args) using the ckpt_args (updater args)
    args.meta_learner = get_maml_meta_learner(args)
    args = uutils.merge_args(starting_args=args.meta_learner.args, updater_args=args)  # second takes priority
    args.meta_learner.args = args  # to avoid meta learner running with args only from past experiment and not with metric analysis experiment
    # note, this my overwrite your seed. todo: fix this, don't think I need to actually

    # -- Setup up remaining stuff for experiment
    ### args: Namespace = setup_args_for_experiment(args) # do this before we the meta-learner code so to not overwrite the meta-learner stuff accidentally, so yes this must be commented out
    return args


def main_data_analyis():
    args: Namespace = load_args()
    # - print args
    print(f'{try_printing_wandb_url(args.log_to_wandb)=}')
    uutils.print_args(args)
    print(f'{try_printing_wandb_url(args.log_to_wandb)=}')

    # - set base_models to be used for experiments
    from diversity_src.data_analysis.common import sanity_check_models_usl_maml_and_set_rand_model
    sanity_check_models_usl_maml_and_set_rand_model(args)

    # - print path to checkpoints
    print(f'{args.path_2_init_sl=}')
    print(f'{args.path_2_init_maml=}')

    # - get dataloaders and overwrites so data analysis runs as we want
    torchmeta_dataloaders: dict = get_meta_learning_dataloaders(args)
    # create the dataloaders, this goes first so you can select the mdl (e.g. final layer) based on task
    from uutils.torch_uu.dataloaders.helpers import get_sl_dataloader
    usl_loaders: dict = get_sl_dataloader(args)
    assert args.mdl_sl.cls.out_features != 5, f'{args.mdl_sl.cls.out_features=}'
    # - above getter loaders funcs mutate args so we need to fix their wrong mutations
    args.dataloaders = torchmeta_dataloaders
    args.usl_loaders = usl_loaders
    from uutils.torch_uu.dataloaders.meta_learning.l2l_to_torchmeta_dataloader import TorchMetaDLforL2L
    # assert isinstance(torchmeta_dataloaders['train'], TorchMetaDLforL2L)
    assert isinstance(usl_loaders['train'], torch.utils.data.dataloader.DataLoader)

    # - layers to do analysis on
    if hasattr(args, 'layer_names'):
        print(f'{args.layer_names=}')

    # - maml param
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    # decided to use the setting for FO that I have for torchmeta learners, but since there is no training it should not matter.
    args.track_higher_grads = True
    args.fo = True

    # - Checks that maml0 acc is lower
    # santity_check_maml_accuracy(args)

    # -- do data analysis
    print(f'{try_printing_wandb_url(args.log_to_wandb)=}')
    print('\n\n---------- Start analysis ----------')
    print(f'{args.stats_analysis_option=}')
    if args.stats_analysis_option == 'performance_comparison':
        comparison_via_performance(args)
    elif args.stats_analysis_option == 'stats_analysis_with_emphasis_on_effect_size':
        stats_analysis_with_emphasis_on_effect_size(args)
    elif args.stats_analysis_option == 'stats_analysis_with_emphasis_on_effect_size_hist':
        # more often this analysis
        stats_analysis_with_emphasis_on_effect_size(args, hist=True)
    elif args.stats_analysis_option == 'stats_analysis_with_emphasis_on_effect_size_and_full_performance_comp_hist':
        stats_analysis_with_emphasis_on_effect_size(args, perform_full_performance_comparison=True, hist=True)
    else:
        # meta_dataloader = dataloaders['train']
        meta_dataloader = args.dataloaders['val']
        # meta_dataloader = dataloaders['test']
        batch = next(iter(meta_dataloader))
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)

        # -- get comparison - SL vs ML
        # TODO: fix confidence inervals CI
        X: Tensor = qry_x
        if args.stats_analysis_option == 'SL_vs_ML':
            distances_per_data_sets_per_layer: list[
                OrderedDict[LayerIentifier, float]] = dist_batch_data_sets_for_all_layer(args.mdl1, args.mdl_sl, X, X,
                                                                                         args.layer_names,
                                                                                         args.layer_names,
                                                                                         metric_comparison_type=args.metric_comparison_type,
                                                                                         effective_neuron_type=args.effective_neuron_type,
                                                                                         subsample_effective_num_data_method=args.subsample_effective_num_data_method,
                                                                                         subsample_effective_num_data_param=args.subsample_effective_num_data_param,
                                                                                         metric_as_sim_or_dist=args.metric_as_sim_or_dist)

        # -- get comparison - ML vs A(ML)
        elif args.stats_analysis_option == 'SL_vs_ML':
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
        elif args.stats_analysis_option == 'SL_vs_A(AML)':
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
        elif args.stats_analysis_option == 'LR(SL)_vs_A(ML)':
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
        elif args.stats_analysis_option == 'SL vs MAML(SL)':
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
            raise ValueError(f'Invalid experiment option, got{args.args.stats_analysis_option=}')

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
        print(f'{try_printing_wandb_url(args.log_to_wandb)=}')
        from uutils.logging_uu.wandb_logging.common import cleanup_wandb
        # cleanup_wandb(args, delete_wandb_dir=True)
        cleanup_wandb(args, delete_wandb_dir=False)


if __name__ == '__main__':
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    main_data_analyis()
    print(f"\nSuccess Done!: {report_times(start)}\a\n")
