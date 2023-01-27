"""
Stistical analysis to detect difference, emphasizing on effect size.
Since it meta-learning, the errors/accs are known to have large stds.
"""
import sys
from argparse import Namespace

import torch

from diversity_src.data_analysis.common import basic_guards_that_maml_usl_and_rand_models_loaded_are_different, \
    print_performance_4_maml
from uutils import save_to_json_pretty, save_args

from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner
from uutils.torch_uu import norm


def stats_analysis_with_emphasis_on_effect_size(args: Namespace,
                                                perform_full_performance_comparison: bool = False,
                                                save_hist_losses: bool = True,
                                                ):
    """
    Note:
        - you need to make sure the models rand, sl, maml are loaded correctly before using this function.

    Currently ppl believed:
        - USL > MAML
    Definitions:
        y = acc_mu_usl - acc_mu_maml, the RV Y with realization y is the difference in means (but Y, y won't be too formal about it, clear from context)
    Statistical Analysis:
        - detect difference in means with emphasis on effect size
        - H0: no diff btw usl and maml
        - H1: diff btw usl and maml (& USL > MAML, chekcing populat belief)
    """
    # _debug(args)
    # results: dict = {}
    # original_lr_inner = args.meta_learner.lr_inner
    # print(f'{original_lr_inner=}')
    #
    # print('\n---- test 1: only using print_performance_4_maml() ----')
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)
    # print_performance_4_maml(args, args.mdl_maml, 0, 0, debug_print=True)
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)
    #
    # print('\n---- test 2: basic_sanity_checks_maml0_does_nothing ----')
    # from diversity_src.data_analysis.common import basic_sanity_checks_maml0_does_nothing
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)
    # basic_sanity_checks_maml0_does_nothing(args)
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)
    #
    # print('\n---- test 3: get_accs_losses_all_splits_maml ----')
    # loaders: dict = args.dataloaders
    # from diversity_src.data_analysis.common import get_accs_losses_all_splits_maml
    # from diversity_src.data_analysis.common import get_accs_losses_all_splits_usl
    # from diversity_src.data_analysis.common import get_mean_and_ci_from_results
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)
    # basic_sanity_checks_maml0_does_nothing(args)
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)
    # results_maml5 = get_accs_losses_all_splits_maml(args, args.mdl_maml, loaders, 5, original_lr_inner)
    # loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_maml5, 'train')
    # print(f'--> train: {(loss, loss_ci, acc, acc_ci)=}')
    # results_maml10 = get_accs_losses_all_splits_maml(args, args.mdl_maml, loaders, 10, original_lr_inner)
    # loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_maml10, 'train')
    # print(f'--> train: {(loss, loss_ci, acc, acc_ci)=}')
    # results_usl = get_accs_losses_all_splits_usl(args, args.mdl_sl, loaders)
    # loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_usl, 'train')
    # print(f'--> train: {(loss, loss_ci, acc, acc_ci)=}')
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)
    #
    # print('\n---- test 4: does printing results later screw things up?  ----')
    # print('---- Sanity check: meta-train acc & loss for each method (usl & maml) values at the end of training ----')
    # from diversity_src.data_analysis.common import get_mean_and_ci_from_results
    # loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_maml5, 'train')
    # print(f'train: {(loss, loss_ci, acc, acc_ci)=}')
    # results['train_maml5'] = (loss, loss_ci, acc, acc_ci)
    # loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_maml10, 'train')
    # print(f'train: {(loss, loss_ci, acc, acc_ci)=}')
    # results['train_maml10'] = (loss, loss_ci, acc, acc_ci)
    # loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_usl, 'train')
    # print(f'train: {(loss, loss_ci, acc, acc_ci)=}')
    # results['train_usl'] = (loss, loss_ci, acc, acc_ci)
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)

    # # -- Start code for real
    # print('\n---------  Start code for real  ---------')
    # original_lr_inner = args.meta_learner.lr_inner
    # print(f'{original_lr_inner=}')
    # # - get datalaoders from args
    # loaders: dict = args.dataloaders
    # # - results dict
    # results: dict = {}
    # # -- Guard: that models maml != usl != rand, i.e. models were loaded correctly before using this function
    # basic_guards_that_maml_usl_and_rand_models_loaded_are_different(args)
    # assert norm(args.mdl_rand) != norm(args.mdl_maml) != norm(args.mdl_sl), f'You need to load these before using this.'
    #
    # # -- Sanity check: basic checks that meta-train errors & loss are fine (check not to different from final learning curve vals)
    # from diversity_src.data_analysis.common import basic_sanity_checks_maml0_does_nothing
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)
    # basic_sanity_checks_maml0_does_nothing(args)
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)
    #
    # # -- Get the losses & accs for each method (usl & maml) for all splits & save them
    # # - once the model guard has been passed you should be able to get args.mdl_rand, args.mdl_maml, args.mdl_sl safely
    # original_lr_inner = args.meta_learner.lr_inner
    # print(f'{original_lr_inner=}')
    # from diversity_src.data_analysis.common import get_accs_losses_all_splits_maml
    # results_maml5 = get_accs_losses_all_splits_maml(args, args.mdl_maml, loaders, 5, original_lr_inner)
    # results_maml10 = get_accs_losses_all_splits_maml(args, args.mdl_maml, loaders, 10, original_lr_inner)
    # from diversity_src.data_analysis.common import get_accs_losses_all_splits_usl
    # results_usl = get_accs_losses_all_splits_usl(args, args.mdl_sl, loaders)
    #
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)
    # basic_sanity_checks_maml0_does_nothing(args)
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)
    #
    # from diversity_src.data_analysis.common import get_mean_and_ci_from_results
    # loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_maml5, 'train')
    # print(f'train: {(loss, loss_ci, acc, acc_ci)=}')
    # results['train_maml5'] = (loss, loss_ci, acc, acc_ci)
    # loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_maml10, 'train')
    # print(f'train: {(loss, loss_ci, acc, acc_ci)=}')
    # results['train_maml10'] = (loss, loss_ci, acc, acc_ci)
    # loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_usl, 'train')
    # print(f'train: {(loss, loss_ci, acc, acc_ci)=}')
    # results['train_usl'] = (loss, loss_ci, acc, acc_ci)
    #
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)
    # basic_sanity_checks_maml0_does_nothing(args)
    # print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)

    # -- Start code for real
    print('\n---------------------  Start code for real2  ---------------------')
    # - get datalaoders from args
    loaders: dict = args.dataloaders
    # - results dict
    results: dict = {}
    # -- Guard: that models maml != usl != rand, i.e. models were loaded correctly before using this function
    basic_guards_that_maml_usl_and_rand_models_loaded_are_different(args)
    assert norm(args.mdl_rand) != norm(args.mdl_maml) != norm(args.mdl_sl), f'You need to load these before using this.'

    # -- Sanity check: basic checks that meta-train errors & loss are fine (check not to different from final learning curve vals)
    from diversity_src.data_analysis.common import basic_sanity_checks_maml0_does_nothing
    basic_sanity_checks_maml0_does_nothing(args, loaders)

    # -- Get the losses & accs for each method (usl & maml) for all splits & save them
    # - once the model guard has been passed you should be able to get args.mdl_rand, args.mdl_maml, args.mdl_sl safely
    original_lr_inner = args.meta_learner.lr_inner
    print(f'{original_lr_inner=}')
    from diversity_src.data_analysis.common import get_accs_losses_all_splits_maml
    results_maml5 = get_accs_losses_all_splits_maml(args, args.mdl_maml, loaders, 5, original_lr_inner)
    results_maml10 = get_accs_losses_all_splits_maml(args, args.mdl_maml, loaders, 10, original_lr_inner)
    from diversity_src.data_analysis.common import get_accs_losses_all_splits_usl
    results_usl = get_accs_losses_all_splits_usl(args, args.mdl_sl, loaders)
    print_performance_4_maml(args, args.mdl_maml, loaders, 5, original_lr_inner, debug_print=True)

    sys.exit()

    # -- Sanity check: meta-train acc & loss for each method (usl & maml) are close to final loss after training
    print('---- Sanity check: meta-train acc & loss for each method (usl & maml) values at the end of training ----')
    from diversity_src.data_analysis.common import get_mean_and_ci_from_results
    loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_maml5, 'train')
    print(f'train: {(loss, loss_ci, acc, acc_ci)=}')
    results['train_maml5'] = (loss, loss_ci, acc, acc_ci)
    loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_maml10, 'train')
    print(f'train: {(loss, loss_ci, acc, acc_ci)=}')
    results['train_maml10'] = (loss, loss_ci, acc, acc_ci)
    loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_usl, 'train')
    print(f'train: {(loss, loss_ci, acc, acc_ci)=}')
    results['train_usl'] = (loss, loss_ci, acc, acc_ci)

    # -- do statistical analysis based on effect size
    from uutils.stats_uu.effect_size import stat_test_with_effect_size_as_emphasis
    args.acceptable_difference1 = args.acceptable_difference1 if hasattr(args, 'acceptable_difference1') else 0.01
    args.acceptable_difference2 = args.acceptable_difference2 if hasattr(args, 'acceptable_difference2') else 0.02
    args.alpha: float = args.alpha if hasattr(args, 'alpha') else 0.01
    # - maml5 vs usl
    group1: list = results_usl['test']['accs']
    group2: list = results_maml5['test']['accs']
    cohen_d, standardized_acceptable_difference1, standardized_acceptable_difference2 = stat_test_with_effect_size_as_emphasis(
        group1, group2, args.acceptable_difference1, args.acceptable_difference1,
        args.alpha, print_groups_data=True)
    results['maml5_vs_usl'] = (cohen_d, standardized_acceptable_difference1, standardized_acceptable_difference2)
    # - maml10 vs usl
    group1: list = results_usl['test']['accs']
    group2: list = results_maml10['test']['accs']
    cohen_d, standardized_acceptable_difference1, standardized_acceptable_difference2 = stat_test_with_effect_size_as_emphasis(
        group1, group2, args.acceptable_difference1, args.acceptable_difference1,
        args.alpha, print_groups_data=True)
    results['maml10_vs_usl'] = (cohen_d, standardized_acceptable_difference1, standardized_acceptable_difference2)

    # -- Save hist losses
    if save_hist_losses:
        pass
    # -- Save results
    results['results_maml5'] = results_maml5
    results['results_maml10'] = results_maml10
    results['results_usl'] = results_usl
    torch.save(results, args.log_root / f'results.pt')
    save_to_json_pretty(results, args.log_root / f'results.json')
    save_args(args)
    # -- do perform comparison
    if perform_full_performance_comparison:
        print('---- Full performance comparison ----')
        from diversity_src.data_analysis.common import comparison_via_performance
        # comparison_via_performance(args)
        print_performance_4_maml(args, args.mdl_maml, 5, original_lr_inner, debug_print=True)
    return results


def _debug(args: Namespace):
    print('\n---- Start Debug ----')
    from diversity_src.data_analysis.common import basic_sanity_checks_maml0_does_nothing

    from diversity_src.data_analysis.common import get_accs_losses_all_splits_maml
    from diversity_src.data_analysis.common import get_accs_losses_all_splits_usl

    from diversity_src.data_analysis.common import get_mean_and_ci_from_results

    results: dict = {}
    original_lr_inner = args.meta_learner.lr_inner
    print(f'{original_lr_inner=}')
    loaders: dict = args.dataloaders

    basic_guards_that_maml_usl_and_rand_models_loaded_are_different(args)

    print()
    print_performance_4_maml(args, args.mdl_maml, loaders, 5, original_lr_inner, debug_print=True)
    basic_sanity_checks_maml0_does_nothing(args, loaders)
    print_performance_4_maml(args, args.mdl_maml, loaders, 5, original_lr_inner, debug_print=True)

    results_maml5 = get_accs_losses_all_splits_maml(args, args.mdl_maml, loaders, 5, original_lr_inner)
    results_maml10 = get_accs_losses_all_splits_maml(args, args.mdl_maml, loaders, 10, original_lr_inner)
    results_usl = get_accs_losses_all_splits_usl(args, args.mdl_sl, loaders)

    print()
    print_performance_4_maml(args, args.mdl_maml, loaders, 5, original_lr_inner, debug_print=True)
    basic_sanity_checks_maml0_does_nothing(args, loaders)
    print_performance_4_maml(args, args.mdl_maml, loaders, 5, original_lr_inner, debug_print=True)

    print()
    loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_maml5, 'train')
    print(f'----> train: {(loss, loss_ci, acc, acc_ci)=}')
    results['train_maml5'] = (loss, loss_ci, acc, acc_ci)
    loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_maml10, 'train')
    print(f'----> train: {(loss, loss_ci, acc, acc_ci)=}')
    results['train_maml10'] = (loss, loss_ci, acc, acc_ci)
    loss, loss_ci, acc, acc_ci = get_mean_and_ci_from_results(results_usl, 'train')
    print(f'----> train: {(loss, loss_ci, acc, acc_ci)=}')
    results['train_usl'] = (loss, loss_ci, acc, acc_ci)

    print()
    print_performance_4_maml(args, args.mdl_maml, loaders, 5, original_lr_inner, debug_print=True)
    basic_sanity_checks_maml0_does_nothing(args, loaders)
    print_performance_4_maml(args, args.mdl_maml, loaders, 5, original_lr_inner, debug_print=True)
    print('---- End Debug ----\n')

