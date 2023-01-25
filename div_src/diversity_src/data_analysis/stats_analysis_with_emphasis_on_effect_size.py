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
    # - do basic guards that models maml != usl != rand, i.e. models were loaded correctly
    basic_guards_that_maml_usl_and_rand_models_loaded_are_different(args)

    # - check that basic checks that meta-train errors & loss are fine (check not to different from final learning curve vals)
    # todo: basic checks that meta-train errors are fine

    # - get the list/iter errors for each method & save them
    assert isinstance(args.meta_learner, MAMLMetaLearner)
    # maml_accs: list[float] = args.get_lists_accs_losses(batch)

    # - do statistical analysis based on effect size
    from uutils.stats_uu.effect_size import stat_test_with_effect_size_as_emphasis
    group1: list
    group2: list
    acceptable_difference1: float = 0.01  # difference/epsilon
    acceptable_difference2: float = 0.02  # difference/epsilon
    stat_test_with_effect_size_as_emphasis(group1, group2, acceptable_difference1, acceptable_difference2, alpha,
                                           print_groups_data=True)
