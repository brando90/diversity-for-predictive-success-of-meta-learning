from argparse import Namespace
from copy import deepcopy, copy
from pathlib import Path

import torch
from torch import nn

from uutils.torch_uu import norm, process_meta_batch
from uutils.torch_uu.eval.eval import eval_sl
from uutils.torch_uu.mains.common import _get_agent, load_model_optimizer_scheduler_from_ckpt
from uutils.torch_uu.meta_learners.maml_differentiable_optimizer import meta_eval_no_context_manager
from uutils.torch_uu.meta_learners.pretrain_convergence import FitFinalLayer
from uutils.torch_uu.models import reset_all_weights
from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import Learner
from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_cifarfs_fc100


def setup_args_path_for_ckpt_data_analysis(args: Namespace,
                                           ckpt_filename: str,
                                           ) -> Namespace:
    """
    ckpt_filename values:
        'ckpt_file.pt'
        'ckpt_file_best_loss.pt'
        'ckpt.pt'
        'ckpt_best_loss.pt'
    """
    ckpt_filename_sl = ckpt_filename  # this one is the one that has the accs that match, at least when I went through the files, json runs, MI_plots_sl_vs_maml_1st_attempt etc.
    args.path_2_init_sl = (Path(args.path_2_init_sl) / ckpt_filename_sl).expanduser()
    ckpt_filename_maml = ckpt_filename  # this one is the one that has the accs that match, at least when I went through the files, json runs, MI_plots_sl_vs_maml_1st_attempt etc.
    # ckpt_filename_maml = 'ckpt_file_best_loss.pt'  # this one is the one that has the accs that match, at least when I went through the files, json runs, MI_plots_sl_vs_maml_1st_attempt etc.
    args.path_2_init_maml = (Path(args.path_2_init_maml) / ckpt_filename_maml).expanduser()
    return args


def santity_check_maml_accuracy(args: Namespace):
    """
    Checks that maml0 acc is lower than adapted maml and returns the good maml's test, train loss and accuracy.
    """
    # - good maml with proper adaptaiton
    print('\n- Sanity check: MAML vs MAML0 (1st should have better performance but there is an assert to check it too)')
    print(f'{args.meta_learner.lr_inner=}')
    # eval_loss, eval_acc, _, _ = meta_eval_no_context_manager(args, split='val', training=True, save_val_ckpt=False)
    eval_loss, _, eval_acc, _ = eval_sl(args, args.agent, args.dataloaders, split='val', training=True)
    print(f'{eval_loss=}, {eval_acc=}')

    # - with no adaptation
    original_lr_inner = args.meta_learner.lr_inner
    args.meta_learner.lr_inner = 0
    print(f'{args.meta_learner.lr_inner=}')
    # eval_loss_maml0, eval_acc_maml0, _, _ = meta_eval_no_context_manager(args, split='val', training=True,
    #                                                                      save_val_ckpt=False)
    eval_loss_maml0, _, eval_acc_maml0, _ = eval_sl(args, args.agent, args.dataloaders, split='val', training=True)
    print(f'{eval_loss_maml0=}, {eval_acc_maml0=}')
    assert eval_acc_maml0 < eval_acc, f'The accuracy of no adaptation should be smaller but got ' \
                                      f'{eval_acc_maml0=}, {eval_acc=}'
    args.meta_learner.lr_inner = original_lr_inner
    print(f'{args.meta_learner.lr_inner=} [should be restored lr_inner]\n')


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


def get_sl_learner(args: Namespace):
    """
    perhaps useful:
        args_ckpt = ckpt['state_dict']
        state_dict = ckpt['model']
    see: save_check_point_sl
    """
    # ckpt: dict = torch.load(args.path_2_init_maml, map_location=torch.device('cpu'))
    model, _, _ = load_model_optimizer_scheduler_from_ckpt(args, path_to_checkpoint=args.path_2_init_sl)
    # model = args.model

    # args.meta_learner = _get_agent(args)
    # if torch.cuda.is_available():
    #     args.meta_learner.base_model = base_model.cuda()
    return model


def get_maml_meta_learner(args: Namespace):
    # ckpt: dict = torch.load(args.path_2_init_maml, map_location=torch.device('cpu'))
    if '668' in str(args.path_2_init_maml):  # hack to have old 668 checkpoint work
        # args.path_2_init_maml = Path('~/data_folder_fall2020_spring2021/logs/nov_all_mini_imagenet_expts/logs_Nov05_15-44-03_jobid_668/ckpt_file.pt').expanduser()
        base_model = load_old_mi_resnet12rfs_ckpt(args, path_to_checkpoint=args.path_2_init_maml)
    else:
        base_model, _, _ = load_model_optimizer_scheduler_from_ckpt(args, path_to_checkpoint=args.path_2_init_maml)

    args.meta_learner = _get_agent(args)
    if torch.cuda.is_available():
        args.meta_learner.base_model = base_model.cuda()
    return args.meta_learner


def _comparison_via_performance(args: Namespace):
    print('\n---- comparison_via_performance ----\n')
    args.mdl_maml = args.mdl1
    args.mdl_sl = args.mdl2

    #
    # original_lr_inner = args.meta_learner.lr_inner
    # original_lr_inner = 0.5
    # original_lr_inner = 0.1
    # original_lr_inner = 0.01
    original_lr_inner = -0.01

    args.mdl_sl.model.cls = deepcopy(args.mdl_maml.model.cls)
    print('-> sl_mdl has the head of the maml model to make comparisons using maml better, it does not affect when '
          'fitting the final layer with LR FFL')

    # -- maml 0
    print('\n---- maml0 for rand model')
    args_mdl_rand = copy(args)
    args_mdl_rand.meta_learner.base_model = args.mdl_rand
    args_mdl_rand.meta_learner.nb_inner_train_steps = 0
    args_mdl_rand.meta_learner.lr_inner = 0.0
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_rand, split='train',
                                                                                    training=True, save_val_ckpt=False)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_rand, split='val',
                                                                                    training=True, save_val_ckpt=False)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_rand, split='test',
                                                                                    training=True, save_val_ckpt=False)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print('---- maml0 for maml model')
    args.meta_learner.base_model = args.mdl_maml
    args.meta_learner.nb_inner_train_steps = 0
    args.meta_learner.lr_inner = 0.0
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args, split='train', training=True,
                                                                                    save_val_ckpt=False)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args, split='val', training=True,
                                                                                    save_val_ckpt=False)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args, split='test', training=True,
                                                                                    save_val_ckpt=False)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    # commented out since the f_sl final layer model has 64 labels, which don't make sense if there is no adaptation
    print('---- maml0 for sl model')
    args_mdl_sl = copy(args)
    args_mdl_sl.meta_learner.base_model = args.mdl_sl
    args_mdl_sl.meta_learner.nb_inner_train_steps = 0
    args_mdl_sl.meta_learner.lr_inner = 0.0
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_sl, split='train',
                                                                                    training=True, save_val_ckpt=False)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_sl, split='val',
                                                                                    training=True, save_val_ckpt=False)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_sl, split='test',
                                                                                    training=True, save_val_ckpt=False)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    # -- maml 5
    print('\n---- maml5 for rand model')
    args_mdl_rand = copy(args)
    args_mdl_rand.meta_learner.base_model = args.mdl_rand
    args_mdl_rand.meta_learner.nb_inner_train_steps = 5
    args_mdl_rand.meta_learner.lr_inner = original_lr_inner
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_rand, split='train',
                                                                                    training=True, save_val_ckpt=False)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_rand, split='val',
                                                                                    training=True, save_val_ckpt=False)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_rand, split='test',
                                                                                    training=True, save_val_ckpt=False)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print('---- maml5 for maml model')
    args.meta_learner.base_model = args.mdl_maml
    args.meta_learner.nb_inner_train_steps = 5
    args.meta_learner.lr_inner = original_lr_inner
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args, split='train', training=True,
                                                                                    save_val_ckpt=False)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args, split='val', training=True,
                                                                                    save_val_ckpt=False)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args, split='test', training=True,
                                                                                    save_val_ckpt=False)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    # commented out since the f_sl final layer model has 64 labels, which don't make sense if there is no adaptation
    print('---- maml5 for sl model')
    args_mdl_sl = copy(args)
    args_mdl_sl.meta_learner.base_model = args.mdl_sl
    args_mdl_sl.meta_learner.nb_inner_train_steps = 5
    args_mdl_sl.meta_learner.lr_inner = original_lr_inner
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_sl, split='train',
                                                                                    training=True, save_val_ckpt=False)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_sl, split='val',
                                                                                    training=True, save_val_ckpt=False)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_sl, split='test',
                                                                                    training=True, save_val_ckpt=False)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    # -- maml 10
    print('\n---- maml10 for rand model')
    args_mdl_rand = copy(args)
    args_mdl_rand.meta_learner.base_model = args.mdl_rand
    args_mdl_rand.meta_learner.nb_inner_train_steps = 10
    args_mdl_rand.meta_learner.lr_inner = original_lr_inner
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_rand, split='train',
                                                                                    training=True, save_val_ckpt=False)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_rand, split='val',
                                                                                    training=True, save_val_ckpt=False)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_rand, split='test',
                                                                                    training=True, save_val_ckpt=False)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print('---- maml10 for maml model')
    args.meta_learner.base_model = args.mdl_maml
    args.meta_learner.nb_inner_train_steps = 10
    args.meta_learner.lr_inner = original_lr_inner
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args, split='train', training=True,
                                                                                    save_val_ckpt=False)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args, split='val', training=True,
                                                                                    save_val_ckpt=False)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args, split='test', training=True,
                                                                                    save_val_ckpt=False)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print('---- maml10 for sl model')
    args_mdl_sl = copy(args)
    args_mdl_sl.meta_learner.base_model = args.mdl_sl
    args_mdl_sl.meta_learner.nb_inner_train_steps = 10
    args_mdl_sl.meta_learner.lr_inner = original_lr_inner
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_sl, split='train',
                                                                                    training=True, save_val_ckpt=False)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_sl, split='val',
                                                                                    training=True, save_val_ckpt=False)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_sl, split='test',
                                                                                    training=True, save_val_ckpt=False)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    # -- SL
    print('\n---- FFL (LR) for rand model')
    args_mdl_rand = copy(args)
    args_mdl_rand.meta_learner = FitFinalLayer(args, base_model=args.mdl_rand, target_type='classification',
                                               classifier='LR')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_rand, split='train',
                                                                                    training=True, save_val_ckpt=False)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_rand, split='val',
                                                                                    training=True, save_val_ckpt=False)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_rand, split='test',
                                                                                    training=True, save_val_ckpt=False)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print('---- FFL (LR) for maml model')
    args.meta_learner = FitFinalLayer(args, base_model=args.mdl_maml, target_type='classification', classifier='LR')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args, split='train', training=True,
                                                                                    save_val_ckpt=False)
    print(f'train: '
          f'{(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args, split='val', training=True,
                                                                                    save_val_ckpt=False)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args, split='test', training=True,
                                                                                    save_val_ckpt=False)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print('---- FFL (LR) for sl model')
    args_mdl_sl = copy(args)
    args_mdl_sl.meta_learner = FitFinalLayer(args, base_model=args.mdl_maml, target_type='classification',
                                             classifier='LR')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_sl, split='train',
                                                                                    training=True, save_val_ckpt=False)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_sl, split='val',
                                                                                    training=True, save_val_ckpt=False)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = meta_eval_no_context_manager(args_mdl_sl, split='test',
                                                                                    training=True, save_val_ckpt=False)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print()


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


def comparison_via_performance(args: Namespace):
    print('\n---- comparison_via_performance ----\n')
    args.mdl_maml = args.mdl1
    assert norm(args.mdl_maml) == norm(args.mdl1)
    args.mdl_sl = args.mdl2
    assert norm(args.mdl_sl) == norm(args.mdl2)
    assert norm(args.mdl_maml) != norm(args.mdl_sl)

    # args.mdl_rand = deepcopy(args.mdl2)
    # assert norm(args.mdl_rand) == norm(args.mdl2)
    # reset_all_weights(args.mdl_rand)
    assert norm(args.mdl_rand) != norm(args.mdl1)
    print(f'{norm(args.mdl_rand)=}, {norm(args.mdl1)=}')

    #
    original_lr_inner = args.meta_learner.lr_inner
    # original_lr_inner = 0.5
    # original_lr_inner = 0.1
    # original_lr_inner = 0.01
    # original_lr_inner = -0.01

    args.mdl_sl.cls = deepcopy(args.mdl_maml.cls)
    print('-> sl_mdl has the head of the maml model to make comparisons using maml better, it does not affect when '
          'fitting the final layer with LR FFL')

    # -- maml 0
    print('\n---- maml0 for rand model')
    args_mdl_rand = copy(args)
    args_mdl_rand.meta_learner.base_model = args.mdl_rand
    # args_mdl_rand.meta_learner.base_model = args.mdl_maml
    args_mdl_rand.meta_learner.nb_inner_train_steps = 0
    args_mdl_rand.meta_learner.lr_inner = 0.0
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_rand, args_mdl_rand.agent, args.dataloaders,
                                                               split='train',
                                                               training=True)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_rand, args_mdl_rand.agent, args.dataloaders,
                                                               split='val',
                                                               training=True)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_rand, args_mdl_rand.agent, args.dataloaders,
                                                               split='test',
                                                               training=True)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print('---- maml0 for maml model')
    args.meta_learner.base_model = args.mdl_maml
    args.meta_learner.nb_inner_train_steps = 0
    args.meta_learner.lr_inner = 0.0
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args, args.agent, args.dataloaders, split='train',
                                                               training=True)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args, args.agent, args.dataloaders, split='val',
                                                               training=True)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args, args.agent, args.dataloaders, split='test',
                                                               training=True)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    # commented out since the f_sl final layer model has 64 labels, which don't make sense if there is no adaptation
    print('---- maml0 for sl model')
    args_mdl_sl = copy(args)
    args_mdl_sl.meta_learner.base_model = args.mdl_sl
    args_mdl_sl.meta_learner.nb_inner_train_steps = 0
    args_mdl_sl.meta_learner.lr_inner = 0.0
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_sl, args_mdl_sl.agent, args.dataloaders,
                                                               split='train',
                                                               training=True)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_sl, args_mdl_sl.agent, args.dataloaders,
                                                               split='val',
                                                               training=True)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_sl, args_mdl_sl.agent, args.dataloaders,
                                                               split='test',
                                                               training=True)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    # -- maml 5
    print('\n---- maml5 for rand model')
    args_mdl_rand = copy(args)
    args_mdl_rand.meta_learner.base_model = args.mdl_rand
    args_mdl_rand.meta_learner.nb_inner_train_steps = 5
    args_mdl_rand.meta_learner.lr_inner = original_lr_inner
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_rand, args_mdl_rand.agent, args.dataloaders,
                                                               split='train',
                                                               training=True)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_rand, args_mdl_rand.agent, args.dataloaders,
                                                               split='val',
                                                               training=True)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_rand, args_mdl_rand.agent, args.dataloaders,
                                                               split='test',
                                                               training=True)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print('---- maml5 for maml model')
    args.meta_learner.base_model = args.mdl_maml
    args.meta_learner.nb_inner_train_steps = 5
    args.meta_learner.lr_inner = original_lr_inner
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args, args.agent, args.dataloaders, split='train',
                                                               training=True)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args, args.agent, args.dataloaders, split='val',
                                                               training=True)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args, args.agent, args.dataloaders, split='test',
                                                               training=True)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    # commented out since the f_sl final layer model has 64 labels, which don't make sense if there is no adaptation
    print('---- maml5 for sl model')
    args_mdl_sl = copy(args)
    args_mdl_sl.meta_learner.base_model = args.mdl_sl
    args_mdl_sl.meta_learner.nb_inner_train_steps = 5
    args_mdl_sl.meta_learner.lr_inner = original_lr_inner
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_sl, args_mdl_sl.agent, args.dataloaders,
                                                               split='train',
                                                               training=True)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_sl, args_mdl_sl.agent, args.dataloaders,
                                                               split='val',
                                                               training=True)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_sl, args_mdl_sl.agent, args.dataloaders,
                                                               split='test',
                                                               training=True)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    # -- maml 10
    print('\n---- maml10 for rand model')
    args_mdl_rand = copy(args)
    args_mdl_rand.meta_learner.base_model = args.mdl_rand
    args_mdl_rand.meta_learner.nb_inner_train_steps = 10
    args_mdl_rand.meta_learner.lr_inner = original_lr_inner
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_rand, args_mdl_rand.agent, args.dataloaders,
                                                               split='train',
                                                               training=True)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_rand, args_mdl_rand.agent, args.dataloaders,
                                                               split='val',
                                                               training=True)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_rand, args_mdl_rand.agent, args.dataloaders,
                                                               split='test',
                                                               training=True)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print('---- maml10 for maml model')
    args.meta_learner.base_model = args.mdl_maml
    args.meta_learner.nb_inner_train_steps = 10
    args.meta_learner.lr_inner = original_lr_inner
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args, args.agent, args.dataloaders, split='train',
                                                               training=True)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args, args.agent, args.dataloaders, split='val',
                                                               training=True)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args, args.agent, args.dataloaders, split='test',
                                                               training=True)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print('---- maml10 for sl model')
    args_mdl_sl = copy(args)
    args_mdl_sl.meta_learner.base_model = args.mdl_sl
    args_mdl_sl.meta_learner.nb_inner_train_steps = 10
    args_mdl_sl.meta_learner.lr_inner = original_lr_inner
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_sl, args_mdl_sl.agent, args.dataloaders,
                                                               split='train',
                                                               training=True)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_sl, args_mdl_sl.agent, args.dataloaders,
                                                               split='val',
                                                               training=True)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_sl, args_mdl_sl.agent, args.dataloaders,
                                                               split='test',
                                                               training=True)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    # -- SL
    print('\n---- FFL (LR) for rand model')
    args_mdl_rand = copy(args)
    args_mdl_rand.meta_learner = FitFinalLayer(args, base_model=args.mdl_rand, target_type='classification',
                                               classifier='LR')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_rand, args_mdl_rand.agent, args.dataloaders,
                                                               split='train',
                                                               training=True)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_rand, args_mdl_rand.agent, args.dataloaders,
                                                               split='val',
                                                               training=True)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_rand, args_mdl_rand.agent, args.dataloaders,
                                                               split='test',
                                                               training=True)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print('---- FFL (LR) for maml model')
    args.meta_learner = FitFinalLayer(args, base_model=args.mdl_maml, target_type='classification', classifier='LR')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args, args.agent, args.dataloaders, split='train',
                                                               training=True)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args, args.agent, args.dataloaders, split='val',
                                                               training=True)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args, args.agent, args.dataloaders, split='test',
                                                               training=True)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print('---- FFL (LR) for sl model')
    args_mdl_sl = copy(args)
    args_mdl_sl.meta_learner = FitFinalLayer(args, base_model=args.mdl_maml, target_type='classification',
                                             classifier='LR')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_sl, args_mdl_sl.agent, args.dataloaders,
                                                               split='train',
                                                               training=True)
    print(f'train: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_sl, args_mdl_sl.agent, args.dataloaders,
                                                               split='val',
                                                               training=True)
    print(f'val: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')
    meta_loss, meta_loss_std, meta_acc, meta_acc_std = eval_sl(args_mdl_sl, args_mdl_sl.agent, args.dataloaders,
                                                               split='test',
                                                               training=True)
    print(f'test: {(meta_loss, meta_loss_std, meta_acc, meta_acc_std)=}')

    print()


def load_old_mi_resnet12rfs_ckpt(args: Namespace, path_to_checkpoint: Path) -> nn.Module:
    from meta_learning.base_models.resnet_rfs import _get_resnet_rfs_model_mi

    model, _ = _get_resnet_rfs_model_mi(args.model_option)

    # ckpt: dict = torch.load(args.path_to_checkpoint, map_location=torch.device('cpu'))
    path_to_checkpoint = args.path_to_checkpoint if path_to_checkpoint is None else path_to_checkpoint
    ckpt: dict = torch.load(path_to_checkpoint, map_location=args.device)
    model_state_dict = ckpt['model_state_dict']
    # model_state_dict = ckpt['f_model_state_dict']
    model.load_state_dict(model_state_dict)
    args.model = model
    return args.model


def do_diversity_data_analysis(args, meta_dataloader):
    from diversity_src.diversity.diversity import compute_diversity
    from pprint import pprint
    from anatome.helper import compute_stats_from_distance_per_batch_of_data_sets_per_layer
    from anatome.helper import compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks
    from anatome.helper import pprint_results

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
    div_mu, div_std, distances_for_task_pairs = compute_diversity(args, meta_dataloader)
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
