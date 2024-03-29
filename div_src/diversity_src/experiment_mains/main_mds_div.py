import time
from argparse import Namespace

# import uutils
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from learn2learn.vision.benchmarks import BenchmarkTasksets

from diversity_src.diversity.task2vec_based_metrics.diversity_task2vec.diversity_for_few_shot_learning_benchmark import \
    get_task_embeddings_from_few_shot_l2l_benchmark, get_task_embeddings_from_few_shot_dataloader
from diversity_src.diversity.task2vec_based_metrics.task2vec import ProbeNetwork
import diversity_src.diversity.task2vec_based_metrics.task2vec as task2vec
import diversity_src.diversity.task2vec_based_metrics.task_similarity as task_similarity
from uutils import report_times, args_hardcoded_in_script, print_args, setup_args_for_experiment, save_args

# - args for each experiment
from uutils.argparse_uu.common import create_default_log_root
from uutils.argparse_uu.meta_learning import parse_args_meta_learning, fix_for_backwards_compatibility
from uutils.logging_uu.wandb_logging.common import cleanup_wandb, setup_wandb
from uutils.plot import save_to
from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
from uutils.torch_uu.distributed import is_lead_worker
from uutils.torch_uu.models.probe_networks import get_probe_network


# import matplotlib.pyplot as plt

# - mds

def diversity_ala_task2vec_mds(args: Namespace) -> Namespace:
    # - data set options
    # args.batch_size = 5
    # args.data_option = 'delauny_uu_l2l_bm_split'
    # args.data_path = Path('~/data/delauny_l2l_bm_splits').expanduser()
    # args.data_augmentation = 'delauny_pad_random_resized_crop'

    # - probe_network
    # args.model_option = 'resnet18_pretrained_imagenet'

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility
    args = fix_for_backwards_compatibility(args)
    return args


# - main

def load_args() -> Namespace:
    """
    1. parse args from user's terminal
    2. optionally set remaining args values (e.g. manually, hardcoded, from ckpt etc.)
    3. setup remaining args small details from previous values (e.g. 1 and 2).
    """
    # -- parse args from terminal
    args: Namespace = parse_args_meta_learning()
    args.args_hardcoded_in_script = True  # <- REMOVE to remove manual loads
    # args.manual_loads_name = 'diversity_ala_task2vec_mds'  # <- REMOVE to remove manual loads

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    if args_hardcoded_in_script(args):
        if args.manual_loads_name == 'diversity_ala_task2vec_mds':
            args: Namespace = diversity_ala_task2vec_mds(args)
        else:
            raise ValueError(f'Invalid value, got: {args.manual_loads_name=}')
    else:
        # NOP: since we are using args from terminal
        pass

    # -- Setup up remaining stuff for experiment
    # args: Namespace = setup_args_for_experiment(args)  # todo, why is this uncomented? :/
    setup_wandb(args)
    create_default_log_root(args)
    return args


def main():
    # - load the args from either terminal, ckpt, manual etc.
    args: Namespace = load_args()

    # - real experiment
    compute_div_and_plot_distance_matrix_for_fsl_benchmark(args)
    # compute_div_and_plot_distance_matrix_for_fsl_benchmark_for_all_splits(args)

    # - wandb
    cleanup_wandb(args)


def compute_div_and_plot_distance_matrix_for_fsl_benchmark_for_all_splits(args: Namespace, show_plots: bool = True):
    splits: list[str] = ['train', 'validation', 'test']
    print_args(args)

    splits_results = {}
    for split in splits:
        print(f'----> div computations for {split=}')
        results = compute_div_and_plot_distance_matrix_for_fsl_benchmark(args, split, show_plots)
        splits_results[split] = results

    # - print summary
    print('---- Summary of results for all splits')
    for split in splits:
        results: dict = splits_results[split]
        div, ci, distance_matrix, split = results['div'], results['ci'], results['distance_matrix'], results['split']
        print(f'\n-> {split=}')
        print(f'Diversity: {(div, ci)=}')
        print(f'{distance_matrix=}')


def compute_div_and_plot_distance_matrix_for_fsl_benchmark(args: Namespace,
                                                           split: str = 'validation',
                                                           show_plots: bool = True,
                                                           ):
    """
    - sample one batch of tasks and use a random cross product of different tasks to compute diversity.
    """
    start = time.time()
    print(f'---- start task2vec analysis: {split=} ')

    # - print args for experiment
    save_args(args)

    # - create probe_network
    args.probe_network: ProbeNetwork = get_probe_network(args)
    print(f'{type(args.probe_network)=}')

    # create loader
    args.dataloaders: dict = get_mds_loader(args)

    # - compute task embeddings according to task2vec
    print(f'number of tasks to consider: {args.batch_size=}')
    embeddings: list[task2vec.Embedding] = get_task_embeddings_from_few_shot_dataloader(args.tasksets,
                                                                                        args.probe_network,
                                                                                        split=split,
                                                                                        num_tasks_to_consider=args.batch_size,
                                                                                        classifier_opts=args.classifier_opts,
                                                                                        )
    print(f'\n {len(embeddings)=}')

    # - compute distance matrix & task2vec based diversity
    # to demo` task2vec, this code computes pair-wise distance between task embeddings
    distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
    print(f'{distance_matrix=}')

    div, ci = task_similarity.stats_of_distance_matrix(distance_matrix)
    print(f'Diversity: {(div, ci)=}')

    # - save results
    torch.save(embeddings, args.log_root / 'embeddings.pt')  # saving obj version just in case
    results: dict = {'embeddings': [(embed.hessian, embed.scale, embed.meta) for embed in embeddings],
                     'distance_matrix': distance_matrix,
                     'div': div, 'ci': ci,
                     'split': split,
                     }
    torch.save(results, args.log_root / f'results_{split}.pt')

    # - show plot, this code is similar to above but put computes the distance matrix internally & then displays it
    # hierchical clustering
    task_similarity.plot_distance_matrix(embeddings, labels=list(range(len(embeddings))), distance='cosine',
                                         show_plot=False)
    save_to(args.log_root, plot_name=f'clustered_distance_matrix_fsl_{args.data_option}_{split}'.replace('-', '_'))
    import matplotlib.pyplot as plt
    if show_plots:
        # plt.show()
        pass
    # heatmap
    task_similarity.plot_distance_matrix_heatmap_only(embeddings, labels=list(range(len(embeddings))),
                                                      distance='cosine',
                                                      show_plot=False)
    save_to(args.log_root, plot_name=f'heatmap_only_distance_matrix_fsl_{args.data_option}_{split}'.replace('-', '_'))
    if show_plots:
        # plt.show()
        pass

    # todo: log plot to wandb https://docs.wandb.ai/guides/track/log/plots, https://stackoverflow.com/questions/72134168/how-does-one-save-a-plot-in-wandb-with-wandb-log?noredirect=1&lq=1
    # import wandb
    # wandb.log({"chart": plt})
    # -
    # print(f"\n---- {report_times(start)}\n")
    return results


if __name__ == '__main__':
    start = time.time()
    # - run experiment
    main()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
