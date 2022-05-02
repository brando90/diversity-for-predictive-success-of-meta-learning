"""
Report: https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/reports/Is-the-size-of-the-model-playing-a-key-role-in-meta-learning---VmlldzoxOTM0NDk5

mainly to check if diff = acc(maml) - acc(usl) variaes wrt model size/meta-train error.
hypothesis: is that smaller models need feature extractor to be adapted more (meta-learning matters more).
Answer: TODO
"""

#%%
"""
5CNN, on MI
"""

num_hidden_units_per_layer = [32]

test_performance_diff = [-1]

# - x-axis: "model size" e.g. hidden units, number weights, depth, meta-train error
# x_axis = []
x_axis = num_hidden_units_per_layer

# - y-axis: diff = acc(maml) - acc(usl)
y_axis = []

# - plot it


#%%