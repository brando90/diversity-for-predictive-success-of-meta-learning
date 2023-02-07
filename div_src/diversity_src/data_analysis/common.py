import logging
from argparse import Namespace
from copy import deepcopy, copy
from pathlib import Path
from typing import Optional, Any

import torch
from torch import nn

from uutils.torch_uu import norm, process_meta_batch
from uutils.torch_uu.eval.eval import eval_sl, meta_eval
from uutils.torch_uu.mains.common import _get_maml_agent, load_model_optimizer_scheduler_from_ckpt, \
    _get_and_create_model_opt_scheduler, load_model_ckpt
from uutils.torch_uu.meta_learners.maml_differentiable_optimizer import meta_eval_no_context_manager
from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner
from uutils.torch_uu.meta_learners.pretrain_convergence import FitFinalLayer
from uutils.torch_uu.models import reset_all_weights
from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import Learner
from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_cifarfs_fc100

from pdb import set_trace as st


def setup_args_path_for_ckpt_data_analysis(args: Namespace,
                                           ckpt_filename: str = 'hardcoded_already_in_paths',
                                           ) -> Namespace:
    """
    note: you didn't actually need this...if the ckpts pointed to the file already this would be redundant...
    ckpt_filename values:
        'ckpt_file.pt'
        'ckpt_file_best_loss.pt'
        'ckpt.pt'
        'ckpt_best_loss.pt'
    """
    # - legacy file processing name, user specified
    if ckpt_filename == 'ckpt.pt':
        # append given file name
        args.path_2_init_sl = Path(args.path_2_init_sl).expanduser()
        #  Whether this path is a regular file (also True for symlinks pointing to regular files).
        if args.path_2_init_sl.is_file():
            pass
        else:
            ckpt_filename_sl = ckpt_filename  # this one is the one that has the accs that match, at least when I went through the files, json runs, MI_plots_sl_vs_maml_1st_attempt etc.
            args.path_2_init_sl = (Path(args.path_2_init_sl) / ckpt_filename_sl).expanduser()
        args.path_2_init_maml = Path(args.path_2_init_maml).expanduser()
        if args.path_2_init_maml.is_file():
            pass
        else:
            ckpt_filename_maml = ckpt_filename  # this one is the one that has the accs that match, at least when I went through the files, json runs, MI_plots_sl_vs_maml_1st_attempt etc.
            # if you need that name just put t in the path from the beginning
            # ckpt_filename_maml = 'ckpt_file_best_loss.pt'  # this one is the one that has the accs that match, at least when I went through the files, json runs, MI_plots_sl_vs_maml_1st_attempt etc.
            args.path_2_init_maml = (Path(args.path_2_init_maml) / ckpt_filename_maml).expanduser()
    elif ckpt_filename == 'hardcoded_already_in_path':
        # the sl path & maml ckpt paths already contain the path to the ckpt, so just open them as is
        # could be nice to do assert strings ckpt and .pt are present
        pass  # nop
    else:
        args.path_2_init_sl = Path(args.path_2_init_sl).expanduser()
        args.path_2_init_sl = (Path(args.path_2_init_sl) / ckpt_filename).expanduser()
        args.path_2_init_maml = Path(args.path_2_init_maml).expanduser()
        args.path_2_init_maml = (Path(args.path_2_init_maml) / ckpt_filename).expanduser()
    # -
    print(f'{args.path_2_init_sl=}')
    print(f'{args.path_2_init_maml=}')
    return args



def santity_check_maml_accuracy(args: Namespace):
    """
    Checks that maml0 acc is lower than adapted maml and returns the good maml's test, train loss and accuracy.
    """
    # - good maml with proper adaptaiton
    print('\n- Sanity check: MAML vs MAML0 (1st should have better performance but there is an assert to check it too)')
    print(f'{args.meta_learner.inner_lr=}')
    eval_loss, _, eval_acc, _ = eval_sl(args, args.agent, args.dataloaders, split='val', training=True)
    print(f'{eval_loss=}, {eval_acc=}')

    # - with no adaptation
    original_inner_lr = args.meta_learner.inner_lr
    args.meta_learner.inner_lr = 0
    print(f'{args.meta_learner.inner_lr=}')
    eval_loss_maml0, _, eval_acc_maml0, _ = eval_sl(args, args.agent, args.dataloaders, split='val', training=True)
    print(f'{eval_loss_maml0=}, {eval_acc_maml0=}')
    assert eval_acc_maml0 < eval_acc, f'The accuracy of no adaptation should be smaller but got ' \
                                      f'{eval_acc_maml0=}, {eval_acc=}'
    args.meta_learner.inner_lr = original_inner_lr
    print(f'{args.meta_learner.inner_lr=} [should be restored inner_lr]\n')


def get_recommended_batch_size_miniimagenet_5CNN(safety_margin: int = 10):
    """
    Loop through all the layers and computing the largest B recommnded. Most likely the H*W that is
    smallest will win but formally just compute B_l for each layer that your computing sims/dists and then choose
    the largest B_l. That ceil(B_l) satisfies B*H*W >= s*C for all l since it's the largest.
        recommended_meta_batch_size = ceil( max([s*C_l/H_l*W_l for l in 1...L]) )
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

def load_model_cifarfs_fix_model_hps(args, path_to_checkpoint):
    path_to_checkpoint = args.path_to_checkpoint if path_to_checkpoint is None else path_to_checkpoint
    ckpt: dict = torch.load(path_to_checkpoint, map_location=args.device)
    model_option = ckpt['model_option']
    # model_hps = ckpt['model_hps']
    model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=args.n_cls)

    opt_option = ckpt['opt_option']
    opt_hps = ckpt['opt_hps']

    scheduler_option = ckpt['scheduler_option']
    scheduler_hps = ckpt['scheduler_hps']
    _get_and_create_model_opt_scheduler(args,
                                        model_option,
                                        model_hps,

                                        opt_option,
                                        opt_hps,

                                        scheduler_option,
                                        scheduler_hps,
                                        )

    # - load state dicts
    model_state_dict: dict = ckpt['model_state_dict']
    args.model.load_state_dict(model_state_dict)
    return args.model


def load_old_mi_resnet12rfs_ckpt(args: Namespace, path_to_checkpoint: Path) -> nn.Module:
    # from meta_learning.base_models.resnet_rfs import _get_resnet_rfs_model_mi
    from diversity_src.models.resnet_rfs import _get_resnet_rfs_model_mi

    model, _ = _get_resnet_rfs_model_mi(args.model_option)

    # ckpt: dict = torch.load(args.path_to_checkpoint, map_location=torch.device('cpu'))
    path_to_checkpoint = args.path_to_checkpoint if path_to_checkpoint is None else path_to_checkpoint
    ckpt: dict = torch.load(path_to_checkpoint, map_location=args.device)
    model_state_dict = ckpt['model_state_dict']
    # model_state_dict = ckpt['f_model_state_dict']
    model.load_state_dict(model_state_dict)
    args.model = model
    return args.model


def load_4cnn_cifarfs_fix_model_hps_maml(args, path_to_checkpoint):
    path_to_checkpoint = args.path_to_checkpoint if path_to_checkpoint is None else path_to_checkpoint
    ckpt: dict = torch.load(path_to_checkpoint, map_location=args.device)
    from uutils.torch_uu.models.l2l_models import cnn4_cifarsfs
    model, model_hps = cnn4_cifarsfs(ways=5)
    model.cls = model.classifier
    model.load_state_dict(ckpt['model_state_dict'])
    args.model = model
    return model


def load_original_rfs_ckpt(args: Namespace, path_to_checkpoint: str):
    # from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_mi
    # from meta_learning.base_models.resnet_rfs import _get_resnet_rfs_model_mi
    from diversity_src.models.resnet_rfs import _get_resnet_rfs_model_mi

    ckpt = torch.load(path_to_checkpoint, map_location=args.device)

    args.model, _ = _get_resnet_rfs_model_mi(model_opt='resnet12_rfs_mi', num_classes=64)
    norm_before_load: float = float(norm(args.model))
    # ckpt['model']['cls.weight'] = ckpt['model']['classifier.weight']
    # assert ckpt['model']['cls.weight'] is ckpt['model']['classifier.weight']
    # ckpt['model']['cls.bias'] = ckpt['model']['classifier.bias']
    # assert ckpt['model']['cls.bias'] is ckpt['model']['classifier.bias']
    args.model.load_state_dict(ckpt['model'])
    norm_after_load: float = float(norm(args.model))
    assert norm_before_load != norm_after_load, f'Error, ckpt not loaded correctly {norm_before_load=}, ' \
                                                f'{norm_after_load=}. These should be different!'
    args.model.to(args.device)
    return args.model


def get_sl_learner(args: Namespace):
    """
    perhaps useful:
        args_ckpt = ckpt['state_dict']
        state_dict = ckpt['model']
    see: save_check_point_sl
    """
    if '12915' in str(args.path_2_init_sl):
        model = load_model_force_add_cls_layer_as_module(args, path_to_checkpoint=args.path_2_init_sl)
    elif 'rfs_checkpoints' in str(args.path_2_init_sl):  # original rfs ckpt
        model = load_original_rfs_ckpt(args, path_to_checkpoint=args.path_2_init_sl)
    else:
        # model, _, _ = load_model_optimizer_scheduler_from_ckpt(args, path_to_checkpoint=args.path_2_init_sl)
        model = load_model_ckpt(args, path_to_checkpoint=args.path_2_init_sl)
    args.model = model

    # args.meta_learner = _get_agent(args)
    if torch.cuda.is_available():
        args.meta_learner.base_model = args.model.cuda()
    return model


def get_maml_meta_learner(args: Namespace):
    if '668' in str(args.path_2_init_maml):  # hack to have old 668 checkpoint work
        # args.path_2_init_maml = Path('~/data_folder_fall2020_spring2021/logs/nov_all_mini_imagenet_expts/logs_Nov05_15-44-03_jobid_668/ckpt_file.pt').expanduser()
        base_model = load_old_mi_resnet12rfs_ckpt(args, path_to_checkpoint=args.path_2_init_maml)
    elif '101601' in str(args.path_2_init_maml):
        base_model = load_model_cifarfs_fix_model_hps(args, path_to_checkpoint=args.path_2_init_maml)
    # elif '23901' in str(args.path_2_init_maml):
    #     base_model = load_4cnn_cifarfs_fix_model_hps_maml(args, path_to_checkpoint=args.path_2_init_maml)
    else:
        # base_model, _, _ = load_model_optimizer_scheduler_from_ckpt(args, path_to_checkpoint=args.path_2_init_maml)
        base_model = load_model_ckpt(args, path_to_checkpoint=args.path_2_init_maml)
    args.model = base_model

    args.meta_learner = _get_maml_agent(args)
    if torch.cuda.is_available():
        args.meta_learner.base_model = base_model.cuda()
    return args.meta_learner


def set_maml_cls_to_maml_cls(args: Namespace, model: nn.Module):
    from uutils.torch_uu.models.resnet_rfs import ResNet
    if isinstance(model, Learner):
        cls = model.model.cls
        args.mdl_sl.model.cls = deepcopy(cls)
    elif isinstance(model, ResNet):
        cls = model.cls
        args.mdl_sl.cls = deepcopy(cls)
    else:
        raise ValueError(f'Model type not supported {type(model)=}')


def basic_guards_that_maml_usl_and_rand_models_loaded_are_different(args: Namespace,
                                                                    verbose: bool = False,
                                                                    ):
    assert norm(args.mdl1) != norm(args.mdl2)
    assert norm(args.mdl_maml) == norm(args.mdl1)
    assert norm(args.mdl_sl) == norm(args.mdl2)
    assert norm(args.mdl_maml) != norm(args.mdl_sl)
    assert norm(args.mdl_rand) != norm(args.mdl_maml) != norm(args.mdl_sl)
    if verbose:
        print(f'{norm(args.mdl_rand)=}\n{norm(args.mdl_maml)=}\n{norm(args.mdl_sl)=}')


def comparison_via_performance(args: Namespace):
    print('\n---- comparison_via_performance ----\n')
    print(f'{args.dataloaders=}')
    loaders = args.dataloaders
    basic_guards_that_maml_usl_and_rand_models_loaded_are_different(args)
    # - varying inner_lr
    original_inner_lr = args.meta_learner.inner_lr

    args.mdl_sl.cls = deepcopy(args.mdl_maml.cls)
    print('-> sl_mdl has the head of the maml model to make comparisons using maml better, it does not affect when '
          'fitting the final layer with LR FFL')

    # - full table
    print('---- full table ----')
    assert isinstance(args.meta_learner, MAMLMetaLearner)

    # -- Adaptation=MAML 0 (for all models, rand, maml, sl)
    print('\n---- maml0 for rand model')
    print_performance_4_maml(args, args.mdl_rand, loaders, nb_inner_steps=0, inner_lr=0.0)
    print('---- maml0 for maml model')
    print_performance_4_maml(args, args.mdl_maml, loaders, nb_inner_steps=0, inner_lr=0.0)
    print('---- maml0 for sl model')
    print_performance_4_maml(args, args.mdl_sl, loaders, nb_inner_steps=0, inner_lr=0.0)

    # -- Adaptation=MAML 5 (for all models, rand, maml, sl)
    print('\n---- maml5 for rand model')
    print_performance_4_maml(args, args.mdl_rand, loaders, nb_inner_steps=5, inner_lr=original_inner_lr)
    print('---- maml5 for maml model')
    print_performance_4_maml(args, args.mdl_maml, loaders, nb_inner_steps=5, inner_lr=original_inner_lr)
    print('---- maml5 for sl model')
    print_performance_4_maml(args, args.mdl_sl, loaders, nb_inner_steps=5, inner_lr=original_inner_lr)

    # -- Adaptation=MAML 10 (for all models, rand, maml, sl)
    print('\n---- maml10 for rand model')
    print_performance_4_maml(args, args.mdl_rand, loaders, nb_inner_steps=10, inner_lr=original_inner_lr)
    print('---- maml10 for maml model')
    print_performance_4_maml(args, args.mdl_maml, loaders, nb_inner_steps=10, inner_lr=original_inner_lr)
    print('---- maml10 for sl model')
    print_performance_4_maml(args, args.mdl_sl, loaders, nb_inner_steps=10, inner_lr=original_inner_lr)

    # -- Adaptation=FFL (LR) (for all models, rand, maml, sl)
    print('\n---- FFL (LR) for rand model')
    print_performance_4_usl_ffl(args, args.mdl_rand, loaders)
    print('---- FFL (LR) for maml model')
    print_performance_4_usl_ffl(args, args.mdl_maml, loaders)
    print('---- FFL (LR) for sl model')
    print_performance_4_usl_ffl(args, args.mdl_sl, loaders)

    # - quick
    print('---- quick ----')
    assert isinstance(args.meta_learner, MAMLMetaLearner)

    # -- Adaptation=MAML 0 (for all models, rand, maml, sl)
    print('---- maml0 for rand model')
    print_performance_4_maml(args, args.mdl_rand, loaders, nb_inner_steps=0, inner_lr=0.0)

    # -- Adaptation=MAML 0 (for all models, rand, maml, sl)
    print('---- maml0 for maml model')
    print_performance_4_maml(args, args.mdl_maml, loaders, nb_inner_steps=0, inner_lr=0.0)

    # -- Adaptation=MAML 5 (for all models, rand, maml, sl)
    print('---- maml5 for maml model')
    print_performance_4_maml(args, args.mdl_maml, loaders, nb_inner_steps=5, inner_lr=original_inner_lr)

    # -- Adaptation=MAML 10 (for all models, rand, maml, sl)
    print('---- maml10 for maml model')
    print_performance_4_maml(args, args.mdl_maml, loaders, nb_inner_steps=10, inner_lr=original_inner_lr)

    # -- Adaptation=FFL (LR) (for all models, rand, maml, sl)
    print('---- FFL (LR) for sl model')
    print_performance_4_usl_ffl(args, args.mdl_sl, loaders)
    print('---- FFL (LR) for rand model')
    print_performance_4_usl_ffl(args, args.mdl_rand, loaders)
    print('---- FFL (LR) for maml model')
    print_performance_4_usl_ffl(args, args.mdl_maml, loaders)

    print()


def print_performance_4_maml(args: Namespace,
                             model: nn.Module,
                             loaders,
                             nb_inner_steps: int,
                             inner_lr: float,
                             debug_print: bool = False,
                             training: bool = True,
                             # True for ML -- even for USL: https://stats.stackexchange.com/a/551153/28986
                             target_type='classification'
                             ):
    """ Print performance of maml for all splits (train, val, test)."""
    if debug_print:
        print(f'---- maml {nb_inner_steps} {inner_lr=} model')
    # - create a new instance of MAMLMetaLearner (avoids overwriting the args one. Also, it only re-assings pointers/refs)
    meta_learner = MAMLMetaLearner(args, model, inner_debug=False, target_type=target_type)
    meta_learner.nb_inner_train_steps = nb_inner_steps
    meta_learner.inner_lr = inner_lr
    assert isinstance(meta_learner, MAMLMetaLearner)
    print_performance_results_simple(args, meta_learner, loaders, training=training)
    assert isinstance(meta_learner, MAMLMetaLearner)
    assert args.meta_learner is not meta_learner


def print_performance_4_usl_ffl(args: Namespace,
                                model: nn.Module,
                                loaders,
                                target_type='classification',
                                classifier='LR',
                                ):
    meta_learner = FitFinalLayer(args, base_model=model, target_type=target_type, classifier=classifier)
    assert isinstance(meta_learner, FitFinalLayer)
    print_performance_results_simple(args, meta_learner, loaders, training=True)
    assert isinstance(meta_learner, FitFinalLayer)
    assert args.meta_learner is not meta_learner


def print_performance_results_simple(args: Namespace,
                                     agent,
                                     loaders,
                                     training: bool = True,
                                     # True for ML -- even for USL: https://stats.stackexchange.com/a/551153/28986
                                     ):
    meta_loss, meta_loss_ci, meta_acc, meta_acc_ci = meta_eval(args, agent, loaders, split='train', training=training)
    meta_loss, meta_loss_ci, meta_acc, meta_acc_ci = items(meta_loss, meta_loss_ci, meta_acc, meta_acc_ci)
    print(f'train: {(meta_loss, meta_loss_ci, meta_acc, meta_acc_ci)=}')
    meta_loss, meta_loss_ci, meta_acc, meta_acc_ci = meta_eval(args, agent, loaders, split='val', training=training)
    meta_loss, meta_loss_ci, meta_acc, meta_acc_ci = items(meta_loss, meta_loss_ci, meta_acc, meta_acc_ci)
    print(f'val: {(meta_loss, meta_loss_ci, meta_acc, meta_acc_ci)=}')
    meta_loss, meta_loss_ci, meta_acc, meta_acc_ci = meta_eval(args, agent, loaders, split='test', training=training)
    meta_loss, meta_loss_ci, meta_acc, meta_acc_ci = items(meta_loss, meta_loss_ci, meta_acc, meta_acc_ci)
    print(f'test: {(meta_loss, meta_loss_ci, meta_acc, meta_acc_ci)=}')


def items(meta_loss, meta_loss_ci, meta_acc, meta_acc_ci) -> tuple[float, float, float, float]:
    return meta_loss.item(), meta_loss_ci.item(), meta_acc.item(), meta_acc_ci.item()


def load_model_force_add_cls_layer_as_module(args: Namespace,
                                             path_to_checkpoint: Optional[str] = None,

                                             add_cls_layer: bool = True,  # FORCE HACK :(
                                             ) -> nn.Module:
    """
    Load the most important things: model for USL hack.
    Ref:
        - standard way: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    Hack explained:
    - I accidentally added a .cls layer via .cls = nn.Linear(...) only on cls.
    This made the checkpointing and thus loading checkpointing. This made the load of ckpts from MAML vs USL asymmtric
    and this is needed to fix it.
    """
    # - prepare args from ckpt
    path_to_checkpoint = args.path_to_checkpoint if path_to_checkpoint is None else path_to_checkpoint
    ckpt: dict = torch.load(path_to_checkpoint, map_location=args.device)
    model_option = ckpt['model_option']
    model_hps = ckpt['model_hps']

    _get_and_create_model_opt_scheduler(args,
                                        model_option,
                                        model_hps,

                                        opt_option='None',
                                        opt_hps='None',

                                        scheduler_option='None',
                                        scheduler_hps='None',
                                        )

    # - load state dicts
    if add_cls_layer:
        # HACK
        args.model.cls = nn.Linear(args.model.cls.in_features, args.model.cls.out_features).to(args.device)

    model_state_dict: dict = ckpt['model_state_dict']
    print(model_state_dict.keys())
    args.model.load_state_dict(model_state_dict)
    args.model.to(args.device)
    return args.model


def get_meta_learning_dataloaders_for_data_analysis(args: Namespace):
    from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloaders
    # - to load torchmeta
    args.dataloaders: dict = get_meta_learning_dataloaders(args)
    # - to load l2l
    from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
    from learn2learn.vision.benchmarks import BenchmarkTasksets
    try:
        args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)
    except Exception as e:
        # this hack is here so that if it was meant not meant as l2l, to no-op and only use the torchmeta data set
        logging.warning(f'{e}')
        pass
    # todo: not sure if this is the ver we need args.dataloaders = args.tasksets  # for the sake that eval_sl can detect how to get examples for eval
    args.dataloaders['']


# - helper function for new stats analysis based on effect size

def basic_sanity_checks_maml0_does_nothing(args: Namespace,
                                           loaders,
                                           save_time: bool = True,
                                           debug_print: bool = False,
                                           ):
    """ Basic sanity checks that maml0 does nothing and thus performs as random. """
    # - do basic guards that models maml != usl != rand, i.e. models were loaded correctly
    basic_guards_that_maml_usl_and_rand_models_loaded_are_different(args)

    print('---- maml0 for maml model (should be around ~0.2 for 5 ways task its never seen) ----')
    print_performance_4_maml(args, args.mdl_maml, loaders, nb_inner_steps=0, inner_lr=0.0, debug_print=debug_print)
    if not save_time:
        print('\n---- maml0 for rand model')
        print_performance_4_maml(args, args.mdl_rand, loaders, nb_inner_steps=0, inner_lr=0.0, debug_print=debug_print)
        print('---- maml0 for sl model')
        print_performance_4_maml(args, args.mdl_sl, loaders, nb_inner_steps=0, inner_lr=0.0, debug_print=debug_print)


def get_accs_losses_all_splits_maml(args: Namespace,
                                    model: nn.Module,
                                    loader,
                                    nb_inner_steps: int,
                                    inner_lr: float,
                                    training: bool = True,
                                    # False for SL, ML: https://stats.stackexchange.com/a/551153/28986
                                    ) -> dict:
    """
    Note:
        - training = True **always** for meta-leanring. Reason is so to always use the batch statistics **for the current task**.
        This avoids doing mdl.eval() and using running statistics, which uses stats from a different task -- which
        makes model perform bad due to distribution shifts. Increase batch size to decrease noise.
        Details: https://stats.stackexchange.com/a/551153/28986
        - note the old code had all these asserts because it used usl+ffl at the end, so it was for extra safety nothing
        went wrong.
        - note that we use the torchmeta MAML for consistency of data loader but I don't think it's needed since the
        code bellow get_meta_eval_lists_accs_losses detects the type of loader and uses the correct one.
    Warning:
        - alwaus manually specify nb_inner_steps and inner_lr. This function might mutate the meta-learner/agent. Sorry! Wont fix.
    """
    results: dict = dict(train=dict(losses=[], accs=[]),
                         val=dict(losses=[], accs=[]),
                         test=dict(losses=[], accs=[]))
    # - if this guard fails your likely not using the model you expect/wanted
    basic_guards_that_maml_usl_and_rand_models_loaded_are_different(args)
    # - load maml params you wanted for eval (note: args.agent_opt == 'MAMLMetaLearner_default' flag exists).
    assert isinstance(args.meta_learner, MAMLMetaLearner)  # for consistent interface to get loader & extra safety ML
    # original_meta_learner = args.meta_learner
    args.meta_learner.base_model = model
    args.meta_learner.nb_inner_train_steps = nb_inner_steps
    args.meta_learner.inner_lr = inner_lr
    # - get accs and losses for all splits
    assert isinstance(args.meta_learner, MAMLMetaLearner)  # for consistent interface to get loader & extra safety ML
    agent = args.meta_learner
    for split in ['train', 'val', 'test']:
        from uutils.torch_uu.eval.eval import get_meta_eval_lists_accs_losses
        losses, accs = get_meta_eval_lists_accs_losses(args, agent, loader)
        assert isinstance(losses, list), f'losses should be a list of floats, but got {type(losses)=}'
        assert isinstance(losses[0], float), f'losses should be a list of floats, but got {type(losses[0])=}'
        results[split]['losses'] = losses
        results[split]['accs'] = accs
    # - return results
    assert isinstance(args.meta_learner, MAMLMetaLearner)  # for consistent interface to get loader & extra safety ML
    return results


def get_accs_losses_all_splits_usl(args: Namespace,
                                   model: nn.Module,
                                   loader,
                                   training: bool = True,
                                   # False for SL, ML: https://stats.stackexchange.com/a/551153/28986
                                   ) -> dict:
    """
    Note:
        - training = True **always** for meta-leanring. Reason is so to always use the batch statistics **for the current task**.
        This avoids doing mdl.eval() and using running statistics, which uses stats from a different task -- which
        makes model perform bad due to distribution shifts. Increase batch size to decrease noise.
        Details: https://stats.stackexchange.com/a/551153/28986
        - note the old code had all these asserts because it used usl+ffl at the end, so it was for extra safety nothing
        went wrong.
        - note that we use the torchmeta MAML for consistency of data loader but I don't think it's needed since the
        code bellow get_meta_eval_lists_accs_losses detects the type of loader and uses the correct one.
    """
    results: dict = dict(train=dict(losses=[], accs=[]),
                         val=dict(losses=[], accs=[]),
                         test=dict(losses=[], accs=[]))
    # - if this guard fails your likely not using the model you expect/wanted
    basic_guards_that_maml_usl_and_rand_models_loaded_are_different(args)
    # - get accs and losses for all splits
    original_meta_learner = args.meta_learner
    args.mdl_sl.cls = deepcopy(args.mdl_maml.cls)  # final layer is a 5-way cls, since ffl is convex anything is fine
    agent = FitFinalLayer(args, base_model=model)
    assert isinstance(agent, FitFinalLayer)  # leaving this to leave a consistent interface to get loader & extra safety
    for split in ['train', 'val', 'test']:
        from uutils.torch_uu.eval.eval import get_meta_eval_lists_accs_losses
        losses, accs = get_meta_eval_lists_accs_losses(args, agent, loader)
        results[split]['losses'] = losses
        results[split]['accs'] = accs
    # - return results
    assert isinstance(agent, FitFinalLayer)  # leaving this to leave a consistent interface to get loader & extra safety
    return results


# -- print accs & losses

def print_accs_losses_mutates_results(results_maml5: dict,
                                      results_maml10: dict,
                                      results_usl: dict,
                                      results: dict,
                                      split: str,
                                      ):
    loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_maml5, 'train')
    print(f'{split} (maml 5): {(loss, loss_ci, acc, acc_ci)=}')
    results[f'{split}_maml5'] = (loss, loss_ci, acc, acc_ci)
    loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_maml10, 'train')
    print(f'{split} (maml 10): {(loss, loss_ci, acc, acc_ci)=}')
    results[f'{split}_maml10'] = (loss, loss_ci, acc, acc_ci)
    loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_usl, 'train')
    print(f'{split} (usl): {(loss, loss_ci, acc, acc_ci)=}')
    results[f'{split}_usl'] = (loss, loss_ci, acc, acc_ci)


def get_mean_and_ci_from_results(results: dict,
                                 split: str,
                                 ) -> tuple[float, float, float, float]:
    losses, accs = results[split]['losses'], results[split]['accs']
    from uutils.torch_uu.metrics.confidence_intervals import mean_confidence_interval
    loss, loss_ci = mean_confidence_interval(losses)
    acc, acc_ci = mean_confidence_interval(accs)
    return loss, loss_ci, acc, acc_ci


# -

def rfs_ckpts():
    path = '~/data/rfs_checkpoints/mini_simple.pt'
    path = Path(path).expanduser()
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    print(ckpt.keys())
    print(ckpt['model'])


if __name__ == '__main__':
    rfs_ckpts()
    print('Done\a\n')
