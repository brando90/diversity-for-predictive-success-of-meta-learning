"""
Script to re-save ckpts not saving objects. This creates issues because you need the code for the unpickling of dill to
work.

Githashes for div & uutils:
- get old githash bfb367a & repo used brando90 committed on Feb 16, 2022
- div: bfb367a,
- uutils: 3e49322 , 3e4932203fc1e3cb2ea87b49b2069e1d9b0a760b
"""
import pickle

from distutils.dir_util import copy_tree
import os
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn

from uutils import merge_args, make_args_pickable
from uutils.torch_uu.checkpointing_uu.meta_learning import save_for_meta_learning
from uutils.torch_uu.distributed import set_devices
from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_mi
from uutils.torch_uu.optim_uu.adam_uu import get_opt_hps_adam_resnet_rfs_old_mi


def get_model_hps(args: Namespace):
    if 'logs_Nov05_15-44-03_jobid_668' in str(args.current_logs_path):
        args.model_option = 'resnet12_rfs_mi'
        _, model_hps = get_resnet_rfs_model_mi(model_opt=args.model_option)
    else:
        raise NotImplementedError
    return model_hps


def get_opt_hps(args: Namespace):
    if 'logs_Nov05_15-44-03_jobid_668' in str(args.current_logs_path):
        args.opt_option = 'adam_mi_old_resnet12_rfs'
        _, opt_hps = get_opt_hps_adam_resnet_rfs_old_mi(args)
    else:
        raise NotImplementedError
    return opt_hps


def get_scheduler_hps(args: Namespace):
    if 'logs_Nov05_15-44-03_jobid_668' in str(args.current_logs_path):
        args.scheduler_option = None
        _, scheduler_hps = None, None
    else:
        raise NotImplementedError
    return scheduler_hps


def save_old_ckpt_to_new_objectless_format(args: Namespace,

                                           it=None,
                                           ):
    """
    Note:
        - since I didn't save the adam state dict for the training
    """
    # -
    ckpt: dict = torch.load(args.path2old_ckpt, map_location=torch.device('cpu'))
    old_args = ckpt['args']

    # - save old checkpoint in new format
    args.training_mode = 'iterations'
    args.it = old_args.train_iters if (it is None) else 0
    args.epoch_num = -1

    args.agent = ckpt['meta_learner']

    args.model = args.agent.base_model
    args.model_hps = get_model_hps(old_args)

    args.opt = old_args.outer_opt
    args.opt_hps = get_opt_hps(old_args)

    args.scheduler = old_args.scheduler if hasattr(old_args, 'scheduler') else None
    args.scheduler_hps = get_scheduler_hps(old_args)

    args.rank = -1
    args = merge_args(starting_args=make_args_pickable(old_args), updater_args=args)

    # - copy the rest of the contents to the new location
    assert args.log_root != old_args.log_root
    from_directory = str(args.path_2_init_maml)
    to_directory = str(args.log_root)
    # copy contents form old to new
    copy_tree(from_directory, to_directory)
    # rename the old checkpoint in the new directory
    os.rename(args.log_root / args.ckpt_filename_maml, args.log_root / 'ckpt_file_old.pt')

    # - make new checkpoint
    args.logger = old_args.logger
    args.logger.args.rank = -1
    save_for_meta_learning(args, ckpt_filename='ckpt.pt', ignore_logger=True)


# -- tests, main, examples, etc

def main():
    args = Namespace()
    args.path_2_init_maml = '~/data_folder_fall2020_spring2021/logs/nov_all_mini_imagenet_expts/logs_Nov05_15-44-03_jobid_668/'
    # args.path_2_init_maml = '~/data/logs/logs_Nov05_15-44-03_jobid_668_NEW_CKPT/'
    args.path_2_init_maml = Path(args.path_2_init_maml).expanduser()
    args.ckpt_filename_maml = 'ckpt_file.pt'
    args.path2old_ckpt = (Path(args.path_2_init_maml) / args.ckpt_filename_maml).expanduser()

    # - save to new format
    # args.log_root = Path('~/data/logs/logs_Nov05_15-44-03_jobid_668_NEW_CKPT/').expanduser()
    args.log_root = Path('~/data/logs/logs_Nov05_15-44-03_jobid_668_NEW_CKPT_2/').expanduser()
    args.new_ckpt_filename = 'ckpt_file.pt'
    args.path2new_ckpt = (Path(args.log_root) / args.new_ckpt_filename).expanduser()
    # save_old_ckpt_to_new_objectless_format(args)

    # - test saved new format words when loading (but need to test in new code with new commit!)
    # load_new_ckpt_test_()


def load_new_ckpt_test_():
    # - get model ckpt path
    args = Namespace()
    args.rank = -1
    args.log_root = Path('~/data/logs/logs_Nov05_15-44-03_jobid_668_NEW_CKPT/').expanduser()
    args.path_to_checkpoint = (Path(args.log_root) / 'ckpt.pt').expanduser()

    # - set device in args
    set_devices(args)
    # ckpt: dict = torch.load(args.path_to_checkpoint, map_location=args.device)
    ckpt: dict = torch.load(args.path_to_checkpoint, pickle_module=pickle, map_location=args.device)
    print(f'{ckpt.keys()=}')

    # - load model
    model_state_dict = ckpt['model_state_dict']
    # model_state_dict = ckpt['f_model_state_dict']
    from diversity_src.models.resnet_rfs import _get_resnet_rfs_model_mi
    model, _ = _get_resnet_rfs_model_mi(args.model_option)
    model.load_state_dict(model_state_dict)


if __name__ == '__main__':
    main()
    print('Done!\a\n')
