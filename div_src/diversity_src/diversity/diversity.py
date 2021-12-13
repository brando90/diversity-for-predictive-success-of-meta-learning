"""
File with code for computing diveristy:
    dv(B) = E_{tau1, tau2 ~ p(tau1, tau2 | B)}[ d(f1(tau1), f2(tau2) )]
the expected distance between tasks for a given benchmark B.

Example use:
1. Compute diveristy for one single few-shot learning benchmark:
    Example 1:
    - get 1 big (meta) batch of tasks  X e.g. [B, M, C, H, W]. Note X1=X1=X
    - get cross product of tasks we want to use to pair distance per tasks (to ultimately compute diveristy)
    - then for each pair of tasks compute distance for that task
    - then return distances as [N, L] for each layer according to base_models you gave - but the recommended layer to use
    to compute diveristy is L-1 (i.e. the feature extractor laye) and the the final L.
    Using the avg of all the layers in principle should work to but usually results in too high variance to be any use
    because each layer has very different distances as we get deeper in the network.
    - div = mean(distances_for_tasks_pairs[L-1]) or div = mean(distances_for_tasks_pairs[L]) or div = mean(distances_for_tasks_pairs[L-1, L])

    Example 2:
    - another option is to get two meta-batch of tasks and just feed them directly like in 2. Then one wouldn't have
    to worry as much wether to include the diagnoal or not.

2. Compute diveristy for pair of data sets/loaders:
    Example 1:
    - get 1 big batch of images for each data set/loader such that we have many classes (ideally all the classes for
    each data set)
    - treat a class as a task and construct two tensors X1, X2 of shapes [B1, M1, C1, H1, W1],  [B2, M2, C2, H2, W2]
    where B1, B2 are the number of images.
    - repeat what was done for example 1.1 from few-shot learning benchmark i.e. compute pair of pair distances of tasks
    and compute diversity from it.

3. Compute diversity using task2vec
    Example 1:
    - batch = get one very large batch
    - X = create [Concepts, M, C, H, W]
    - X1 = X2 = X
    - compute div with X
    Example 2 (SL data loaders):
    - batch1, batch2 = get two very large (normal) batches
    - X1, X2 = sort them such that the classes are sorted
    - pass them to distance matrix and compute div
"""
from argparse import Namespace
from collections import OrderedDict
import random
from pprint import pprint
from typing import Optional

import torch
from torch import Tensor, nn

from anatome.helper import LayerIdentifier, dist_data_set_per_layer, _dists_per_task_per_layer_to_list, \
    compute_stats_from_distance_per_batch_of_data_sets_per_layer
from uutils.torch_uu import tensorify, process_meta_batch
from uutils.torch_uu.dataloaders import get_miniimagenet_dataloaders_torchmeta, \
    get_minimum_args_for_torchmeta_mini_imagenet_dataloader
from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner, \
    get_feature_extractor_conv_layers, get_last_two_layers


# get_distances_for_task_pair = dist_data_set_per_layer


def select_index(B: int, rand: bool = True):
    """
    Note:
        - probably could have been made an iterator with yield but decided to keep the code simple and not do it.
    :param B:
    :param rand:
    :return:
    """
    if rand:
        # [a, b], including both end points.
        i = random.randint(a=0, b=B - 1)
        return i
    else:
        raise ValueError(f'Not implemented with value: {rand=}')
        # yield i


def select_index_pair(B1, B2, rand: bool = True, consider_diagonal: bool = False) -> tuple[int, int]:
    """

    :param B1:
    :param B2:
    :param rand:
    :param consider_diagonal:
    :return:
    """
    while True:
        i1 = select_index(B1, rand)
        i2 = select_index(B2, rand)
        # - if no diagonal element then keep sampling until i1 != i2, note since there are N diagonals vs N^2 it
        # shouldn't take to long to select valid indices if we want random element with no diagonal.
        if not consider_diagonal:
            # if indicies are different then it's not a diagonal
            if i1 != i2:
                return i1, i2
        else:
            # - just return if we are ok with the diagonal e.g. when X1, X2 are likely very different at top call.
            return i1, i2


def get_list_tasks_tuples(B1: int, B2: int,
                          num_tasks_to_consider: int = 25,
                          rand: bool = True,
                          consider_diagonal: bool = False
                          ) -> list[tuple[int, int]]:
    """

    :param B1:
    :param B2:
    :param num_tasks_to_consider:
    :param rand:
    :param consider_diagonal:
    :return:
    """
    assert (num_tasks_to_consider <= B1 * B2)
    # - generate num_tasks_to_consider task index pairs
    list_task_index_pairs: list[tuple[int, int]] = []
    for i in range(num_tasks_to_consider):
        idx_task1, idx_task2 = select_index_pair(B1, B2, rand, consider_diagonal=consider_diagonal)
        list_task_index_pairs.append((idx_task1, idx_task2))
    assert len(list_task_index_pairs) == num_tasks_to_consider
    return list_task_index_pairs


def get_all_required_distances_for_pairs_of_tasks(f1: nn.Module, f2: nn.Module,
                                                  X1: Tensor, X2: Tensor,
                                                  layer_names1: list[str], layer_names2: list[str],
                                                  metric_comparison_type: str = 'pwcca',
                                                  iters: int = 1,
                                                  effective_neuron_type: str = 'filter',
                                                  downsample_method: Optional[str] = None,
                                                  downsample_size: Optional[int] = None,
                                                  subsample_effective_num_data_method: Optional[str] = None,
                                                  subsample_effective_num_data_param: Optional[int] = None,
                                                  metric_as_sim_or_dist: str = 'dist',
                                                  force_cpu: bool = False,

                                                  num_tasks_to_consider: int = 25,
                                                  consider_diagonal: bool = False
                                                  ) -> list[OrderedDict[LayerIdentifier, float]]:
    """
    Compute the pairwise distances between the collection of tasks in X1 and X2:
        get_distances_for_task_pairs = [d(f1(tau1), tau2)_s]_{s \in [num_tasks_to_consider]}
    used to compute diversity:
        div = mean(get_distances_for_task_pairs)
        std = std(get_distances_for_task_pairs)

    Note:
        - make sure that M is safe i.e. M >= 10*D' e.g. for filter (image patch) based comparison
        D'= ceil(M*H*W) so M*H*W >= s*C for safety margin s say s=10.
        - if cls/head layer is taken into account and small (e.g. n_c <= D' for the other layers) then M >= s*n_c.

    :param f1:
    :param f2:
    :param X1: [B, M, D] or [B, M, C, H, W]
    :param X2: [B, M, D] or [B, M, C, H, W]
    :param layer_names1:
    :param layer_names2:
    :param num_tasks_to_consider: from the cross product of tasks in X1 and X2, how many to consider
        - num_tasks_to_consider = c*sqrt{B^2 -B} or c*B is a good number of c>=1.
        - num_tasks_to_consider >= 25 is likely a good number.
    :param consider_diagonal: if X1 == X2 then we do NOT want to consider the diagonal since that would compare the
    same tasks to each other and that has a distance = 0.0
    :return:
    """
    B1, B2 = X1.size(0), X2.size(0)
    assert num_tasks_to_consider <= B1 * B2, f'You can\'t use more tasks than exist in the cross product of tasks, choose' \
                                             f'{num_tasks_to_consider=} such that is less than or equal to {B1*B2=}.'
    # - get indices for pair of tasks
    indices_task_tuples: list[tuple[int, int]] = get_list_tasks_tuples(B1, B2,
                                                                       num_tasks_to_consider=num_tasks_to_consider,
                                                                       rand=True,
                                                                       consider_diagonal=consider_diagonal
                                                                       )
    assert len(indices_task_tuples) == num_tasks_to_consider

    # - compute required distance pairs of tasks
    distances_for_task_pairs: list[OrderedDict[LayerIdentifier, float]] = []
    for idx_task1, idx_task2 in indices_task_tuples:
        x1, x2 = X1[idx_task1], X2[idx_task2]
        # x1, x2 = x1.unsqueeze(0), x2.unsqueeze(0)
        # assert x1.size() == torch.Size([1, M, C, H, W]) or x1.size() == torch.Size([1, M, D_])

        # - get distances for task pair [L]
        # dists_task_pair: OrderedDict[LayerIdentifier, float] = get_distances_for_task_pair(f1, f2, x1, x2,
        dists_task_pair: OrderedDict[LayerIdentifier, float] = dist_data_set_per_layer(f1, f2, x1, x2,
                                                                                           layer_names1, layer_names2,
                                                                                           metric_comparison_type=metric_comparison_type,
                                                                                           iters=iters,
                                                                                           effective_neuron_type=effective_neuron_type,
                                                                                           downsample_method=downsample_method,
                                                                                           downsample_size=downsample_size,
                                                                                           subsample_effective_num_data_method=subsample_effective_num_data_method,
                                                                                           subsample_effective_num_data_param=subsample_effective_num_data_param,
                                                                                           metric_as_sim_or_dist=metric_as_sim_or_dist,
                                                                                           force_cpu=force_cpu
                                                                                           )
        assert len(dists_task_pair) == len(layer_names1) == len(layer_names2)
        distances_for_task_pairs.append(dists_task_pair)  # [B, L]
    assert len(distances_for_task_pairs) == num_tasks_to_consider
    return distances_for_task_pairs


def diversity(f1: nn.Module, f2: nn.Module,
              X1: Tensor, X2: Tensor,
              layer_names1: list[str], layer_names2: list[str],
              metric_comparison_type: str = 'pwcca',
              iters: int = 1,
              effective_neuron_type: str = 'filter',
              downsample_method: Optional[str] = None,
              downsample_size: Optional[int] = None,
              subsample_effective_num_data_method: Optional[str] = None,
              subsample_effective_num_data_param: Optional[int] = None,
              metric_as_sim_or_dist: str = 'dist',
              force_cpu: bool = False,

              num_tasks_to_consider: int = 25,
              consider_diagonal: bool = False
              ) -> tuple[Tensor, Tensor, list[OrderedDict[LayerIdentifier, float]]]:
    assert len(layer_names1) >= 2, f'For now the final and one before final layer are the way to compute diversity'
    L = len(layer_names1)
    B = num_tasks_to_consider
    distances_for_task_pairs: list[OrderedDict[LayerIdentifier, float]] = get_all_required_distances_for_pairs_of_tasks(
        f1, f2,
        X1, X2,
        layer_names1, layer_names2,
        metric_comparison_type,
        iters,
        effective_neuron_type,
        downsample_method,
        downsample_size,
        subsample_effective_num_data_method,
        subsample_effective_num_data_param,
        metric_as_sim_or_dist,
        force_cpu,

        num_tasks_to_consider,
        consider_diagonal
        )

    # - list(OrderDict([B, L])) -> list([B, L])
    # distances_for_task_pairs: list[list[float]] = _dists_per_task_per_layer_to_list(distances_for_task_pairs)
    # distances_for_task_pairs: Tensor = tensorify(distances_for_task_pairs)
    # assert distances_for_task_pairs.size() == torch.Size([B, L])

    # - compute diversity
    div_mu, div_std = compute_stats_from_distance_per_batch_of_data_sets_per_layer(distances_for_task_pairs)
    # compute div mean
    # div_final: Tensor = distances_for_task_pairs[-1].mean(dim=0)
    # div_feature_extractor: Tensor = distances_for_task_pairs[-2].mean(dim=0)
    # assert div_final.size() == torch.Size([])
    # assert div_feature_extractor.size() == torch.Size([])
    # # compute div std
    # div_final_std: Tensor = distances_for_task_pairs[-1].mean(dim=0)
    # div_feature_extractor_std: Tensor = distances_for_task_pairs[-2].mean(dim=0)
    # assert div_final_std.size() == torch.Size([])
    # assert div_feature_extractor_std.size() == torch.Size([])
    return div_mu, div_std, distances_for_task_pairs


def compute_diversity_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks(distances_for_task_pairs: Tensor,
                                                                                    dist2sim: bool = False):
    if dist2sim:
        distances_for_task_pairs: Tensor = 1.0 - distances_for_task_pairs
    mu, std = distances_for_task_pairs.mean(), distances_for_task_pairs.std()
    return mu, std


# def pprint_div_results(div_final, div_feature_extractor, div_final_std, div_feature_extractor_std):


# - tests

def compute_div_example1_test():
    """
    - sample one batch of tasks and use a random cross product of different tasks to compute diversity.
    """
    mdl: nn.Module = get_default_learner()
    # layer_names: list[str] = get_feature_extractor_conv_layers(include_cls=True)
    layer_names: list[str] = get_last_two_layers(layer_type='conv', include_cls=True)
    args: Namespace = get_minimum_args_for_torchmeta_mini_imagenet_dataloader()
    dataloaders: dict = get_miniimagenet_dataloaders_torchmeta(args)
    for batch_idx, batch_tasks in enumerate(dataloaders['train']):
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch_tasks)
        # - compute diversity
        div_final, div_feature_extractor, div_final_std, div_feature_extractor_std, distances_for_task_pairs = diversity(
            f1=mdl, f2=mdl, X1=qry_x, X2=qry_x, layer_names1=layer_names, layer_names2=layer_names, num_tasks_to_consider=2)
        pprint(distances_for_task_pairs)
        print(f'{div_final, div_feature_extractor, div_final_std, div_feature_extractor_std=}')
        break


def compute_div_example2_test():
    # - wrap data laoder in iterator
    # - call next twice to get X1, X2
    # - run div but inclduing diagonal is fine now
    pass

if __name__ == '__main__':
    compute_div_example1_test()
    print('Done! success!\a')
