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

from torch import Tensor, nn

# from anatome.helper import LayerIdentifier, dist_data_set_per_layer, _dists_per_task_per_layer_to_list, compute_stats_from_distance_per_batch_of_data_sets_per_layer

from uutils.torch_uu import tensorify, process_meta_batch
from uutils.torch_uu.dataloaders.meta_learning.torchmeta_ml_dataloaders import get_miniimagenet_dataloaders_torchmeta, \
    get_minimum_args_for_torchmeta_mini_imagenet_dataloader
from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner, \
    get_feature_extractor_conv_layers, get_last_two_layers

from pdb import set_trace as st


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
                                                  ) -> list[OrderedDict[str, float]]:
    # ) -> list[OrderedDict[LayerIdentifier, float]]:
    """
    [L] x [B, n*k, C,H,W]^2 -> [L] x [B', n*k, C,H,W] -> [B', L]

    Compute the pairwise distances between the collection of tasks in X1 and X2:
        get_distances_for_task_pairs = [d(f(tau_s1), f(tau_s2))]_{s1,s2 \in {num_tasks_to_consider}}
    of size [B', L]. In particular notice how number of tasks used B' gives 1 distance value btw tasks (for a fixed layer).
    used to compute diversity:
        div = mean(get_distances_for_task_pairs)
        ci = ci(get_distances_for_task_pairs)

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
    from anatome.helper import LayerIdentifier, dist_data_set_per_layer, _dists_per_task_per_layer_to_list, compute_stats_from_distance_per_batch_of_data_sets_per_layer
    assert metric_as_sim_or_dist == 'dist'
    L = len(layer_names1)
    B1, B2 = X1.size(0), X2.size(0)
    B_ = num_tasks_to_consider
    assert B_ <= B1 * B2, f'You can\'t use more tasks than exist in the cross product of tasks, choose' \
                          f'{num_tasks_to_consider=} such that is less than or equal to {B1*B2=}.'
    assert L == len(layer_names1) == len(layer_names2)

    # - get indices for pair of tasks
    indices_task_tuples: list[tuple[int, int]] = get_list_tasks_tuples(B1, B2,
                                                                       num_tasks_to_consider=num_tasks_to_consider,
                                                                       rand=True,
                                                                       consider_diagonal=consider_diagonal
                                                                       )
    # [B']
    assert len(indices_task_tuples) == B_

    # - compute required distance pairs of tasks [B', L]
    distances_for_task_pairs: list[OrderedDict[LayerIdentifier, float]] = []
    for b_ in range(B_):
        idx_task1, idx_task2 = indices_task_tuples[b_]
        x1, x2 = X1[idx_task1], X2[idx_task2]
        # x1, x2 = x1.unsqueeze(0), x2.unsqueeze(0)
        # assert x1.size() == torch.Size([1, M, C, H, W]) or x1.size() == torch.Size([1, M, D_])

        # - get distances (index b_) for task pair per layers [1, L]
        print(f'{x1.size()=}, is the input to anatome/dist func [B,M,C,H,W]')
        assert metric_as_sim_or_dist == 'dist'
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
        # st()
        # [1, L]
        assert len(dists_task_pair) == len(layer_names1) == len(layer_names2)
        # [b, L] (+) [1, L] -> [b + 1, L]
        distances_for_task_pairs.append(dists_task_pair)  # end result size [B', L]
    # [B', L]
    assert len(distances_for_task_pairs) == B_
    assert len(distances_for_task_pairs[0]) == L
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
              ) -> tuple[OrderedDict, OrderedDict, list[OrderedDict[str, float]]]:
    # ) -> tuple[OrderedDict, OrderedDict, list[OrderedDict[LayerIdentifier, float]]]:
    """
    Div computes as follows:
        - takes in a set of layers [L]
        - a set of batch of tasks data  [B, n*k, C,H,W] (X1 == X2)
        - uses a [B', n*k, C,H,W] of the cross product where B'=num_tasks_to_consider
        - for each layer compute the div value [L, 1]
    Thus signature is:
        div: [L] x [B, n*k, C,H,W]^2 -> [L] x [B', n*k, C,H,W] -> [L, 1]
    note:
        - IMPORTANT: X1 == X2 size is [B, n*k, C,H,W]
        - layer_names = [L]
        - note: for dist(task1, task2) we get [B'] distances. dist: [L]x[B',n*k, C,H,W] ->

    :return: [L]^2, [B', L]
    """
    from anatome.helper import LayerIdentifier, dist_data_set_per_layer, _dists_per_task_per_layer_to_list, \
        compute_stats_from_distance_per_batch_of_data_sets_per_layer
    assert metric_as_sim_or_dist == 'dist'
    # assert args.args.metric_as_sim_or_dist == 'dist'
    assert len(layer_names1) >= 2, f'For now the final and one before final layer are the way to compute diversity'
    L = len(layer_names1)
    B_ = num_tasks_to_consider

    # - [B', L] compute the distance for each task. So you get B distances (from which to compute div) for each layer [L]
    print(f'{X1.size()=}, is the input to anatome/dist func [B,M,C,H,W]')
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
    # st()
    # [L] x [B, n*k, C,H,W]^2 -> [L] x [B', n*k, C,H,W] -> [B', L]
    assert len(distances_for_task_pairs) == B_
    assert len(distances_for_task_pairs[0]) == L

    # - compute diversity: [B, L] -> [L, 1]^2 (the 2 due to one for div other ci for div)
    div_mu, div_ci = compute_stats_from_distance_per_batch_of_data_sets_per_layer(distances_for_task_pairs)
    print(f'{div_mu=}')
    print(f'{div_ci=}')
    st()
    assert len(div_mu) == L
    assert len(div_ci) == L
    return div_mu, div_ci, distances_for_task_pairs


# def compute_diversity_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks(distances_for_task_pairs: Tensor,
#                                                                                     dist2sim: bool = False):
#     # if dist2sim:
#     #     distances_for_task_pairs: Tensor = 1.0 - distances_for_task_pairs
#     mu, std = distances_for_task_pairs.mean(), distances_for_task_pairs.std()
#     return mu, std


def compute_diversity_fixed_probe_net(args, meta_dataloader):
    """
    Compute diversity: sample one batch of tasks and use a random cross product of different tasks to compute diversity.

    div: [L] x [B', n*K, C,H,W]^2 -> [L, 1]
    for a set of layers & batch of tasks -> computes div for each layer.
    """
    args.num_tasks_to_consider = args.batch_size
    print(f'{args.num_tasks_to_consider=}')

    L: int = len(args.layer_names)
    B_: int = args.num_tasks_to_consider

    batch = next(iter(meta_dataloader))
    # [B, n*k, C,H,W]
    spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
    # print(f'M = n*k = {spt_x.size(1)=}')
    print(f"M = n*k = {qry_x.size(1)=} (before doing [B, n*k*H*W, D] for [B, N, D'] ")

    # - Compute diversity: [L] x [B', n*K, C,H,W]^2 -> [L, 1]^2, [B, L] i.e. output is div for each layer name
    # assert spt_x.size(0) == args.num_tasks_to_consider
    from copy import deepcopy
    f1 = args.mdl_for_dv
    f2 = args.mdl_for_dv
    assert f1 is f2
    f2 = deepcopy(args.mdl_for_dv)
    # assert they are different objects
    assert f1 is not f2
    # assert they are the same model (e.g. compute if the norm is equal, actually, just check they are the same obj before doing deepcopy)
    assert args.metric_as_sim_or_dist == 'dist', f'Diversity is the expected variation/distance of task, ' \
                                                 f'so you need dist but got {args.metric_as_sim_or_dist=}'
    div_mu, div_ci, distances_for_task_pairs = diversity(
        f1=f1, f2=f2, X1=qry_x, X2=qry_x,
        layer_names1=args.layer_names, layer_names2=args.layer_names,
        num_tasks_to_consider=args.num_tasks_to_consider,
        metric_comparison_type=args.metric_comparison_type,
        metric_as_sim_or_dist=args.metric_as_sim_or_dist
    )
    print(f'{args.experiment_option=}')
    print(f'{div_mu=}')
    print(f'{div_ci=}')
    st()
    assert len(div_mu) == L
    assert len(div_ci) == L
    assert len(distances_for_task_pairs) == B_
    assert len(distances_for_task_pairs[0]) == L
    return div_mu, div_ci, distances_for_task_pairs


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
            f1=mdl, f2=mdl, X1=qry_x, X2=qry_x, layer_names1=layer_names, layer_names2=layer_names,
            num_tasks_to_consider=2)
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
