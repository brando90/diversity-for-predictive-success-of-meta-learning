"""
Stistical analysis to detect difference, emphasizing on effect size.
Since it meta-learning, the errors/accs are known to have large stds.
"""

"""
Module for doing p-value analysis for USL vs MAML.
"""
from argparse import Namespace

from diversity_src.data_analysis.common import basic_guards_that_models_are_fine
from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner


def stats_analysis_with_emphasis_on_effect_size(args: Namespace, meta_dataloader):
    """
    Currently ppl believed:
        - USL > MAML
    Definitions:
        y = acc_mu_maml - acc_mu_usl y is the difference in means
        mu_y = mu_m - mu_u
    Statistical Analysis:
        - detect difference in means with emphasis on effect size
    """
    #
    basic_guards_that_models_are_fine(args)

    # todo: basic checks that meta-train errors are fine

    # get the list/iter errors for each method
    assert isinstance(args.meta_learner, MAMLMetaLearner)
    # maml_accs: list[float] = args.get_lists_accs_losses(batch)

    # do the p-value test two-tailed test
    print('two-sided test -- tests if usl vs maml sample means are different')

    # do the p-value test one-tailed test, right sided
    print('one-sided test -- tests if ')