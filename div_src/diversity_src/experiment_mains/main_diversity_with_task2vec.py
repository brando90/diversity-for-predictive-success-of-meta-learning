import time
from argparse import Namespace

# import uutils
from pathlib import Path

import numpy as np
import torch
from learn2learn.vision.benchmarks import BenchmarkTasksets
from torch import Tensor

import task2vec
import task_similarity
from diversity_src.diversity.task2vec_based_metrics.diversity_task2vec.diversity_for_few_shot_learning_benchmark import \
    get_task_embeddings_from_few_shot_l2l_benchmark
from models import get_model
from task2vec import ProbeNetwork
from uutils import report_times, args_hardcoded_in_script, print_args, setup_args_for_experiment, setup_wand, save_args

# - args for each experiment
from uutils.argparse_uu.common import create_default_log_root
from uutils.argparse_uu.meta_learning import parse_args_meta_learning, fix_for_backwards_compatibility
from uutils.logging_uu.wandb_logging.common import cleanup_wandb
from uutils.plot import save_to
from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
from uutils.torch_uu.distributed import is_lead_worker
from uutils.torch_uu.models.probe_networks import get_probe_network


def diversity_ala_task2vec_mi_resnet18_pretrained_imagenet(args: Namespace) -> Namespace:
    args.batch_size = 500
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'diversity_ala_task2vec_mi_resnet18'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_option} {args.model_option}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_mi_resnet18_random(args: Namespace) -> Namespace:
    args.batch_size = 500
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - probe_network
    args.model_option = 'resnet18_random'

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'diversity_ala_task2vec_mi_resnet18'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_option} {args.model_option}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_mi_resnet34_imagenet(args: Namespace) -> Namespace:
    args.batch_size = 500
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - probe_network
    args.model_option = 'resnet34_pretrained_imagenet'

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'diversity_ala_task2vec_mi_resnet34'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_option} {args.model_option}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_mi_resnet34_random(args: Namespace) -> Namespace:
    # args.batch_size = 500
    args.batch_size = 2
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - probe_network
    args.model_option = 'resnet34_random'

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'diversity_ala_task2vec_mi_resnet34'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_option} {args.model_option}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


# -

def load_args() -> Namespace:
    """
    1. parse args from user's terminal
    2. optionally set remaining args values (e.g. manually, hardcoded, from ckpt etc.)
    3. setup remaining args small details from previous values (e.g. 1 and 2).
    """
    # -- parse args from terminal
    args: Namespace = parse_args_meta_learning()
    args.args_hardcoded_in_script = True  # <- REMOVE to remove manual loads
    args.manual_loads_name = 'diversity_ala_task2vec_mi_resnet34_random'  # <- REMOVE to remove manual loads

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    if args_hardcoded_in_script(args):
        if args.manual_loads_name == 'diversity_ala_task2vec_mi_resnet18_pretrained_imagenet':
            args: Namespace = diversity_ala_task2vec_mi_resnet18_pretrained_imagenet(args)
        elif args.manual_loads_name == 'diversity_ala_task2vec_mi_resnet18_random':
            args: Namespace = diversity_ala_task2vec_mi_resnet18_random(args)
        elif args.manual_loads_name == 'diversity_ala_task2vec_mi_resnet34_imagenet':
            args: Namespace = diversity_ala_task2vec_mi_resnet34_imagenet(args)
        elif args.manual_loads_name == 'diversity_ala_task2vec_mi_resnet34_random':
            args: Namespace = diversity_ala_task2vec_mi_resnet34_random(args)
        else:
            raise ValueError(f'Invalid value, got: {args.manual_loads_name=}')
    else:
        # NOP: since we are using args from terminal
        pass

    # -- Setup up remaining stuff for experiment
    # args: Namespace = setup_args_for_experiment(args)
    setup_wand(args)
    create_default_log_root(args)
    return args


def main():
    # - load the args from either terminal, ckpt, manual etc.
    args: Namespace = load_args()

    # - real experiment
    compute_div_and_plot_distance_matrix_for_fsl_benchmark(args)

    # - wandb
    cleanup_wandb(args)


def compute_div_and_plot_distance_matrix_for_fsl_benchmark(args: Namespace):
    """
    - sample one batch of tasks and use a random cross product of different tasks to compute diversity.
    """
    # - print args for experiment
    print_args(args)
    save_args(args)

    # - create probe_network
    args.probe_network: ProbeNetwork = get_probe_network(args)
    print(f'{args.probe_network}')

    # create loader
    args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)

    # - compute task embeddings according to task2vec
    print(f'number of tasks to consider: {args.batch_size=}')
    embeddings: list[task2vec.Embedding] = get_task_embeddings_from_few_shot_l2l_benchmark(args.tasksets,
                                                                               args.probe_network,
                                                                               num_tasks_to_consider=args.batch_size)
    print(f'\n {len(embeddings)=}')

    # - compute distance matrix & task2vec based diversity
    # to demo task2vec, this code computes pair-wise distance between task embeddings
    distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
    print(f'{distance_matrix=}')

    div, ci = task_similarity.stats_of_distance_matrix(distance_matrix)
    print(f'Diversity: {(div, ci)=}')

    # - save results
    torch.save(embeddings, args.log_root/'embeddings.pt')  # saving obj version just in case
    results: dict = {'embeddings': [(embed.hessian, embed.scale, embed.meta) for embed in embeddings],
                     'distance_matrix': distance_matrix,
                     'div': div, 'ci': ci,
                     }
    torch.save(results, args.log_root/'results.pt')

    # - show plot, this code is similar to above but put computes the distance matrix internally & then displays it
    task_similarity.plot_distance_matrix(embeddings, labels=list(range(len(embeddings))), distance='cosine',
                                         show_plot=False)
    save_to(args.log_root, plot_name=f'distance_matrix_fsl_{args.data_option}'.replace('-', '_'))
    import matplotlib.pyplot as plt
    plt.show()

    # todo: log plot to wandb https://docs.wandb.ai/guides/track/log/plots, https://stackoverflow.com/questions/72134168/how-does-one-save-a-plot-in-wandb-with-wandb-log?noredirect=1&lq=1
    # import wandb
    # wandb.log({"chart": plt})


if __name__ == '__main__':
    start = time.time()
    main()
    print(f"\nSuccess Done!: {report_times(start)}\a\n")
