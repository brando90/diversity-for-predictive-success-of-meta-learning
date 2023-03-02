"""
Stistical analysis to detect difference, emphasizing on effect size.
Since it meta-learning, the errors/accs are known to have large stds.
"""
import sys
from argparse import Namespace

import torch

from diversity_src.data_analysis.common import basic_guards_that_maml_usl_and_rand_models_loaded_are_different, \
    print_performance_4_maml, print_accs_losses_mutates_results, print_usl_accs_losses_mutates_results, \
    sanity_check_models_usl_maml_and_set_rand_model
from uutils import save_to_json_pretty, save_args
from uutils.logger import print_acc_loss_from_training_curve
from uutils.plot import save_to
from uutils.plot.histograms_uu import get_histogram

from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner
from uutils.torch_uu import norm

from pdb import set_trace as st


def stats_analysis_with_emphasis_on_effect_size(args: Namespace,
                                                perform_full_performance_comparison: bool = False,
                                                hist: bool = False,
                                                ):
    """
    Note:
        - you need to make sure the models rand, sl, maml are loaded correctly before using this function.

        group1: list = results_usl['test']['accs']
        group2: list = results_maml5['test']['accs']

        group1: list = results_usl['test']['accs']
        group2: list = results_maml10['test']['accs']

    Currently ppl believed:
        - USL > MAML
    Definitions:
        y = acc_mu_usl - acc_mu_maml, the RV Y with realization y is the difference in means (but Y, y won't be too formal about it, clear from context)
    Statistical Analysis:
        - detect difference in means with emphasis on effect size
        - H0: no diff btw usl and maml
        - H1: diff btw usl & maml (& USL > MAML, chekcing populat belief)
    """
    # -- Start code for real
    print('\n---------------------  Start Stats analysis  ---------------------')
    print(f'{args.batch_size=}')
    # - get original inner learning rate
    original_inner_lr = args.meta_learner.inner_lr
    print(f'{original_inner_lr=}')
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
    from diversity_src.data_analysis.common import get_episodic_accs_losses_all_splits_maml
    results_maml5: dict = get_episodic_accs_losses_all_splits_maml(args, args.mdl_maml, loaders, 5, original_inner_lr)
    results_maml10: dict = get_episodic_accs_losses_all_splits_maml(args, args.mdl_maml, loaders, 10, original_inner_lr)
    from diversity_src.data_analysis.common import get_episodic_accs_losses_all_splits_usl, \
        get_usl_accs_losses_all_splits_usl
    results_usl_usl: dict = get_usl_accs_losses_all_splits_usl(args, args.mdl_sl, args.usl_loaders)
    # goes later since this modifies cls layer for args.mdl_sl (we do have asserts to guard us from bugs)
    results_usl: dict = get_episodic_accs_losses_all_splits_usl(args, args.mdl_sl, loaders)

    # -- Sanity check: meta-train acc & loss for each method (usl & maml) are close to final loss after training
    print('\n---- Sanity check: meta-train acc & loss for each method (usl & maml) values at the end of training ----')
    print_accs_losses_mutates_results(results_maml5, results_maml10, results_usl, results, 'train')
    print('-- Sanity check: maml episodic/meta acc & losses from learning curves --')
    print_acc_loss_from_training_curve(path=args.path_2_init_maml)  # print maml episodic/meta acc & loss from training

    print('\n-- Sanity check: usl on usl loss/acc, does it match the currently computed vs the one from training? --')
    print_usl_accs_losses_mutates_results(results_usl_usl, results)
    print('-- Sanity check: usl on usl acc & loss from learning curves --')
    print_acc_loss_from_training_curve(path=args.path_2_init_sl)  # print usl usl acc & loss from training

    print('\n---- Print meta-test acc & loss for each method (usl & maml) ----')
    print_accs_losses_mutates_results(results_maml5, results_maml10, results_usl, results, 'test')

    # -- Compute generalization gap for each method (usl & maml) -- to estimate (meta) overfitting
    print('\n---- Compute generatlization gap for each method (usl & maml) ----')
    compute_overfitting_analysis_stats_for_all_models_mutate_results(args, results_maml5, results_maml10, results_usl,
                                                                     results)

    # -- do statistical analysis based on effect size
    print('\n\n---- Statistical analysis based on effect size ----')
    from uutils.stats_uu.effect_size import stat_test_with_effect_size_as_emphasis
    args.acceptable_difference1 = args.acceptable_difference1 if hasattr(args, 'acceptable_difference1') else 0.01
    args.acceptable_difference2 = args.acceptable_difference2 if hasattr(args, 'acceptable_difference2') else 0.02
    args.alpha: float = args.alpha if hasattr(args, 'alpha') else 0.01
    # - usl vs maml5
    print(f'\n--- usl vs maml5 ---')
    group1: list = results_usl['test']['accs']
    group2: list = results_maml5['test']['accs']
    cohen_d, standardized_acceptable_difference1, standardized_acceptable_difference2 = stat_test_with_effect_size_as_emphasis(
        group1, group2, args.acceptable_difference1, args.acceptable_difference2,
        args.alpha, print_groups_data=True)
    results['usl_vs_maml5'] = (cohen_d, standardized_acceptable_difference1, standardized_acceptable_difference2)
    # - usl vs maml10
    print(f'\n--- usl vs maml10 ---')
    group1: list = results_usl['test']['accs']
    group2: list = results_maml10['test']['accs']
    cohen_d, standardized_acceptable_difference1, standardized_acceptable_difference2 = stat_test_with_effect_size_as_emphasis(
        group1, group2, args.acceptable_difference1, args.acceptable_difference2,
        args.alpha, print_groups_data=True)
    results['usl_vs_maml10'] = (cohen_d, standardized_acceptable_difference1, standardized_acceptable_difference2)

    # -- Save results
    results['results_maml5'] = results_maml5
    results['results_maml10'] = results_maml10
    results['results_usl'] = results_usl
    results['results_usl_usl'] = results_usl_usl
    torch.save(results, args.log_root / f'results.pt')
    save_to_json_pretty(results, args.log_root / f'results.json')
    save_args(args)

    # -- Save hist losses
    if hist:
        print('\n---- Save hist losses ----')
        save_loss_histogram(args, results)
    # -- do perform comparison
    if perform_full_performance_comparison:
        print('\n---- Full performance comparison ----')
        from diversity_src.data_analysis.common import comparison_via_performance
        comparison_via_performance(args)
    return results


def compute_overfitting_analysis_stats_for_all_models_mutate_results(args: Namespace,
                                                                     results_maml5: dict,
                                                                     results_maml10: dict,
                                                                     results_usl: dict,
                                                                     results: dict,
                                                                     ) -> None:
    """


    If generalization gap of maml models is larger than that of usl model, then maml models are overfitting more than usl model.

    ref:
        - https://en.wikipedia.org/wiki/Generalization_error
    """
    from uutils.stats_uu.overfitting import compute_generalization_gap
    # - compute generalization gap for all methods using accuracy
    gen_maml5_acc = compute_generalization_gap(results_maml5['train']['accs'], results_maml5['test']['accs'],
                                               metric_name='acc')
    gen_maml10_acc = compute_generalization_gap(results_maml10['train']['accs'], results_maml10['test']['accs'],
                                                metric_name='acc')
    gen_usl_acc = compute_generalization_gap(results_usl['train']['accs'], results_usl['test']['accs'],
                                             metric_name='acc')
    results['gen_maml5_acc'] = gen_maml5_acc
    results['gen_maml10_acc'] = gen_maml10_acc
    results['gen_usl_acc'] = gen_usl_acc
    print(f'{gen_maml5_acc=}')
    print(f'{gen_maml10_acc=}')
    print(f'{gen_usl_acc=}')
    if gen_maml5_acc < gen_usl_acc:
        print(f'Maml5 might be overfitting more than usl model: {gen_maml5_acc=} < {gen_usl_acc=}')
    if gen_maml10_acc < gen_usl_acc:
        print(f'Maml10 might be overfitting more than usl model: {gen_maml10_acc=} < {gen_usl_acc=}')
    # - compute generalization gap for all methods using loss using loss
    gen_maml5_loss = compute_generalization_gap(results_maml5['train']['losses'], results_maml5['test']['losses'],
                                                metric_name='loss')
    gen_maml10_loss = compute_generalization_gap(results_maml10['train']['losses'], results_maml10['test']['losses'],
                                                 metric_name='loss')
    gen_usl_loss = compute_generalization_gap(results_usl['train']['losses'], results_usl['test']['losses'],
                                              metric_name='loss')
    results['gen_maml5_loss'] = gen_maml5_loss
    results['gen_maml10_loss'] = gen_maml10_loss
    results['gen_usl_loss'] = gen_usl_loss
    print(f'{gen_maml5_loss=}')
    print(f'{gen_maml10_loss=}')
    print(f'{gen_usl_loss=}')
    if gen_maml5_loss > gen_usl_loss:
        print(f'Maml5 might be overfitting more than usl model: {gen_maml5_loss=} > {gen_usl_loss=}')
    if gen_maml10_loss > gen_usl_loss:
        print(f'Maml10 might be overfitting more than usl model: {gen_maml10_loss=} > {gen_usl_loss=}')
    # -- measure gen error using cohen's d (mu1 - mu2) / pool(std1, std2) ==> test_metric - train_metric
    print("\n---- Compute generalization gap using cohen's d (effect size) ----")
    # - use accs
    from uutils.stats_uu.effect_size import stat_test_with_effect_size_as_emphasis
    print(f'---- test - train (maml5) ----')
    group1, group2 = results_maml5['test']['accs'], results_maml5['train']['accs']
    standardized_gen_gap_acc_maml5, _, _ = stat_test_with_effect_size_as_emphasis(group1, group2)
    print(f'---- test - train (maml10) ----')
    group1, group2 = results_maml10['test']['accs'], results_maml10['train']['accs']
    standardized_gen_gap_acc_maml10, _, _ = stat_test_with_effect_size_as_emphasis(group1, group2)
    print(f'---- test - train (usl) ----')
    group1, group2 = results_usl['test']['accs'], results_usl['train']['accs']
    standardized_gen_gap_acc_usl, _, _ = stat_test_with_effect_size_as_emphasis(group1, group2)
    results['standardized_gen_gap_acc_maml5'] = standardized_gen_gap_acc_maml5
    results['standardized_gen_gap_acc_maml10'] = standardized_gen_gap_acc_maml10
    results['standardized_gen_gap_acc_usl'] = standardized_gen_gap_acc_usl
    print(f'{standardized_gen_gap_acc_maml5=}')
    print(f'{standardized_gen_gap_acc_maml10=}')
    print(f'{standardized_gen_gap_acc_usl=}')
    # - use losses
    group1, group2 = results_maml5['test']['losses'], results_maml5['train']['losses']
    standardized_gen_gap_loss_maml5, _, _ = stat_test_with_effect_size_as_emphasis(group1, group2)
    group1, group2 = results_maml10['test']['losses'], results_maml10['train']['losses']
    standardized_gen_gap_loss_maml10, _, _ = stat_test_with_effect_size_as_emphasis(group1, group2)
    group1, group2 = results_usl['test']['losses'], results_usl['train']['losses']
    standardized_gen_gap_loss_usl, _, _ = stat_test_with_effect_size_as_emphasis(group1, group2)
    results['standardized_gen_gap_loss_maml5'] = standardized_gen_gap_loss_maml5
    results['standardized_gen_gap_loss_maml10'] = standardized_gen_gap_loss_maml10
    results['standardized_gen_gap_loss_usl'] = standardized_gen_gap_loss_usl
    print(f'{standardized_gen_gap_loss_maml5=}')
    print(f'{standardized_gen_gap_loss_maml10=}')
    print(f'{standardized_gen_gap_loss_usl=}')
    return


def aic_bic_gen_gap_analysis():
    # -- note: decided against bic & aic because all methods have the same number of parameters (and data points) -- so this metric ends up just comparing the likelihoods (so train losses). If we could incorporate maml into this metric perhaps it would have some use for me.
    # # - estimate AIC & BIC for methods
    # from uutils.torch_uu import count_number_of_parameters
    # from uutils.stats_uu.overfitting import aic, bic
    # assert norm(args.mdl_maml) == norm(args.mdl1)
    # assert norm(args.mdl_sl) == norm(args.mdl2)
    # num_params_maml: int = count_number_of_parameters(args.mdl_maml)
    # num_params_sl: int = count_number_of_parameters(args.mdl_sl)
    # aic_maml5 = aic(results_maml5['test']['losses'], num_params_maml)
    # aic_maml10 = aic(results_maml10['test']['losses'], num_params_maml)
    # aic_usl = aic(results_usl['test']['losses'], num_params_sl)
    # bic_maml5 = bic(results_maml5['test']['losses'], num_params_maml, len(results_maml5['test']['losses']))
    # bic_maml10 = bic(results_maml10['test']['losses'], num_params_maml, len(results_maml10['test']['losses']))
    # bic_usl = bic(results_usl['test']['losses'], num_params_sl, len(results_usl['test']['losses']))
    # results['aic_maml5'] = aic_maml5
    # results['aic_maml10'] = aic_maml10
    # results['aic_usl'] = aic_usl
    # results['bic_maml5'] = bic_maml5
    # results['bic_maml10'] = bic_maml10
    # results['bic_usl'] = bic_usl
    # print(f'{aic_maml5=}')
    # print(f'{aic_maml10=}')
    # print(f'{aic_usl=}')
    # print(f'{bic_maml5=}')
    # print(f'{bic_maml10=}')
    # print(f'{bic_usl=}')
    pass


# -- tests, examples, debug_code etc.

def _debug(args: Namespace):
    print('\n---- Start Debug ----')
    from diversity_src.data_analysis.common import basic_sanity_checks_maml0_does_nothing

    from diversity_src.data_analysis.common import get_episodic_accs_losses_all_splits_maml
    from diversity_src.data_analysis.common import get_episodic_accs_losses_all_splits_usl

    from diversity_src.data_analysis.common import get_mean_and_ci_from_results

    results: dict = {}
    original_inner_lr = args.meta_learner.inner_lr
    print(f'{original_inner_lr=}')
    loaders: dict = args.dataloaders

    basic_guards_that_maml_usl_and_rand_models_loaded_are_different(args)

    print()
    print_performance_4_maml(args, args.mdl_maml, loaders, 5, original_inner_lr, debug_print=True)
    basic_sanity_checks_maml0_does_nothing(args, loaders)
    print_performance_4_maml(args, args.mdl_maml, loaders, 5, original_inner_lr, debug_print=True)

    results_maml5 = get_episodic_accs_losses_all_splits_maml(args, args.mdl_maml, loaders, 5, original_inner_lr)
    results_maml10 = get_episodic_accs_losses_all_splits_maml(args, args.mdl_maml, loaders, 10, original_inner_lr)
    results_usl = get_episodic_accs_losses_all_splits_usl(args, args.mdl_sl, loaders)

    print()
    print_performance_4_maml(args, args.mdl_maml, loaders, 5, original_inner_lr, debug_print=True)
    basic_sanity_checks_maml0_does_nothing(args, loaders)
    print_performance_4_maml(args, args.mdl_maml, loaders, 5, original_inner_lr, debug_print=True)

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
    print_performance_4_maml(args, args.mdl_maml, loaders, 5, original_inner_lr, debug_print=True)
    basic_sanity_checks_maml0_does_nothing(args, loaders)
    print_performance_4_maml(args, args.mdl_maml, loaders, 5, original_inner_lr, debug_print=True)
    print('---- End Debug ----\n')


def save_loss_histogram(args: Namespace, results: dict):
    # - save histograms maml5
    accs = results['results_maml5']['test']['accs']
    get_histogram(accs, xlabel='accuracy', ylabel='pdf', title='maml5 accuracies', stat='probability')
    save_to(args.log_root, plot_name='maml5_accs_hist')
    # - save histograms maml10
    accs = results['results_maml10']['test']['accs']
    get_histogram(accs, xlabel='accuracy', ylabel='pdf', title='maml10 accuracies', stat='probability')
    save_to(args.log_root, plot_name='maml10_accs_hist')
    # - save histograms usl
    accs = results['results_usl']['test']['accs']
    get_histogram(accs, xlabel='accuracy', ylabel='pdf', title='usl accuracies', stat='probability')
    save_to(args.log_root, plot_name='usl_accs_hist')


def stats_analysis_with_emphasis_on_effect_size_test_():
    """
python ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/data_analysis/stats_analysis_with_emphasis_on_effect_size.py
    """
    # - effect size analysis -- usl vs maml
    import uutils
    from uutils.argparse_uu.meta_learning import get_args_mi_effect_size_analysis_default
    args: Namespace = get_args_mi_effect_size_analysis_default()
    # - get maml checkpoints
    from diversity_src.data_analysis.common import get_maml_meta_learner
    args.meta_learner = get_maml_meta_learner(args)
    args = uutils.merge_args(starting_args=args.meta_learner.args, updater_args=args)  # second takes priority
    args.meta_learner.args = args  # to avoid meta learner running with args only from past experiment and not with metric analysis experiment
    # - rand model
    from diversity_src.data_analysis.common import sanity_check_models_usl_maml_and_set_rand_model
    sanity_check_models_usl_maml_and_set_rand_model(args)
    # - setup data loaders: l2l -> torchmeta transform
    from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloaders
    torchmeta_dataloaders: dict = get_meta_learning_dataloaders(args)
    from uutils.torch_uu.dataloaders.helpers import get_sl_dataloader
    usl_loaders: dict = get_sl_dataloader(args)
    args.dataloaders = torchmeta_dataloaders
    args.usl_loaders = usl_loaders
    from uutils.torch_uu.dataloaders.meta_learning.l2l_to_torchmeta_dataloader import TorchMetaDLforL2L
    assert isinstance(torchmeta_dataloaders['train'], TorchMetaDLforL2L)
    assert isinstance(usl_loaders['train'], torch.utils.data.dataloader.DataLoader)

    # - sanity checks for maml
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    # decided to use the setting for FO that I have for torchmeta learners, but since there is no training it should not matter.
    args.track_higher_grads = True
    args.fo = True
    # - do stats analysis
    stats_analysis_with_emphasis_on_effect_size(args, hist=True)


# - run it

if __name__ == '__main__':
    import time

    start_time = time.time()
    # main()
    stats_analysis_with_emphasis_on_effect_size_test_()
    print(f'Done in {time.time() - start_time} seconds')
