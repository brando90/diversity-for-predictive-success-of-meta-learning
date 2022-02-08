from argparse import Namespace
from copy import deepcopy, copy

import torch

from uutils.torch_uu.meta_learners.maml_differentiable_optimizer import meta_eval_no_context_manager
from uutils.torch_uu.meta_learners.pretrain_convergence import FitFinalLayer
from uutils.torch_uu.models import reset_all_weights


def santity_check_maml_accuracy(args: Namespace):
    """
    Checks that maml0 acc is lower than adapted maml and returns the good maml's test, train loss and accuracy.
    """
    # - good maml with proper adaptaiton
    print(f'{args.meta_learner.lr_inner=}')
    eval_loss, eval_acc, _, _ = meta_eval_no_context_manager(args, split='val', training=True, save_val_ckpt=False)
    print(f'{eval_loss=}, {eval_acc=}')

    # - with no adaptation
    original_lr_inner = args.meta_learner.lr_inner
    args.meta_learner.lr_inner = 0
    print(f'{args.meta_learner.lr_inner=}')
    eval_loss_maml0, eval_acc_maml0, _, _ = meta_eval_no_context_manager(args, split='val', training=True,
                                                                         save_val_ckpt=False)
    print(f'{eval_loss_maml0=}, {eval_acc_maml0=}')
    assert eval_acc_maml0 < eval_acc, f'The accuracy of no adaptation should be smaller but got ' \
                                      f'{eval_acc_maml0=}, {eval_acc=}'
    args.meta_learner.lr_inner = original_lr_inner
    print(f'{args.meta_learner.lr_inner=} [should be restored lr_inner]')


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


def get_sl_learner(args: Namespace):
    """
    perhaps useful:
        args_ckpt = ckpt['state_dict']
        state_dict = ckpt['model']
    see: save_check_point_sl
    """
    ckpt: dict = torch.load(args.path_2_init_sl, map_location=torch.device('cpu'))
    model = ckpt['model']
    if torch.cuda.is_available():
        model = model.cuda()
    print(f'from ckpt (sl), model type: {model}')
    return model


def get_meta_learner(args: Namespace):
    ckpt: dict = torch.load(args.path_2_init_maml, map_location=torch.device('cpu'))
    meta_learner = ckpt['meta_learner']
    if torch.cuda.is_available():
        meta_learner.base_model = meta_learner.base_model.cuda()
    print(f'from ckpt (maml), model type: {meta_learner.args.base_model_mode=}')
    # args.nb_inner_train_steps = 10  # since ANIL paper used 10 for inference
    return meta_learner


def comparison_via_performance(args: Namespace):
    print('\n---- comparison_via_performance ----\n')
    args.mdl_maml = args.mdl1
    args.mdl_sl = args.mdl2
    args.mdl_rand = deepcopy(args.mdl1)
    reset_all_weights(args.mdl_rand)
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
