"""
Stistical analysis to detect difference, emphasizing on effect size.
Since it meta-learning, the errors/accs are known to have large stds.
"""
from argparse import Namespace

from diversity_src.data_analysis.common import basic_guards_that_maml_usl_and_rand_models_loaded_are_different
from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner


def stats_analysis_with_emphasis_on_effect_size(args: Namespace,
                                                meta_dataloader,
                                                ):
    """
    Currently ppl believed:
        - USL > MAML
    Definitions:
        y = acc_mu_usl - acc_mu_maml, the RV Y with realization y is the difference in means (but Y, y won't be too formal about it, clear from context)
    Statistical Analysis:
        - detect difference in means with emphasis on effect size
        - H0: no diff btw usl and maml
        - H1: diff btw usl and maml (& USL > MAML, chekcing populat belief)
    """
    results: dict = {}
    # - do basic guards that models maml != usl != rand, i.e. models were loaded correctly
    basic_guards_that_maml_usl_and_rand_models_loaded_are_different(args)

    # - check that basic checks that meta-train errors & loss are fine (check not to different from final learning curve vals)
    from diversity_src.data_analysis.common import basic_sanity_checks_maml0_does_nothing
    basic_sanity_checks_maml0_does_nothing(args)

    # - get the meta-eval accs for each method (usl & maml) & save them
    # assert isinstance(args.meta_learner, MAMLMetaLearner)
    # maml_accs: list[float] = args.get_lists_accs_losses(batch)

    # - get the meta-train acc & loss for each method (usl & maml) and make sure they are close to final loss after training
    # todo: basic checks that meta-train errors are fine

    # - do statistical analysis based on effect size
    from uutils.stats_uu.effect_size import stat_test_with_effect_size_as_emphasis
    group1: list = usl_accs
    group2: list = maml_accs
    acceptable_difference1: float = args.acceptable_difference1 if hasattr(args, 'acceptable_difference1') else 0.01
    acceptable_difference2: float = args.acceptable_difference2 if hasattr(args, 'acceptable_difference2') else 0.02
    alpha: float = args.alpha if hasattr(args, 'alpha') else 0.01
    stat_test_with_effect_size_as_emphasis(group1, group2, acceptable_difference1, acceptable_difference2, alpha,
                                           print_groups_data=True)
    # - save results
    results['usl_accs'] = usl_accs
    results['maml_accs'] = maml_accs
    # save results to json file
    # import json
    # import os
    # from uutils.torch_uu import get_path_to_results_dir
    # path_to_results_dir = get_path_to_results_dir(args)
    # path_to_json_file = os.path.join(path_to_results_dir, 'results.json')
    # with open(path_to_json_file, 'w') as f:
    #     json.dump(results, f)
    return results
