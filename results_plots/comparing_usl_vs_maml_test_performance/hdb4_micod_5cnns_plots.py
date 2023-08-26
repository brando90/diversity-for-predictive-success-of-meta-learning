# #%%
# """
#
# \begin{table}[h]
# \begin{tabular}{|c|c|c|c|c|}
# \hline
# Filter Size & Number of Params & USL-MAML 5 (Effect Size) & USL-MAML 10 (Effect Size) & 1\% Effect Size \\ \hline
# 2 (micod) & 441 & -0.0533 (H1\_maml) & 0.0624 (H1\_usl) & 0.0435, 0.0449  \\
# 6 (micod) & 1,961 & -0.184 (H1\_maml) & -0.100 (H1\_maml) & 0.0489, 0.0504 \\
# 8 (micod) & 3,045 & 0.0794 (H1\_usl) & -0.00121 (H0 no diff) & 0.0515, 0.0507 \\
# 16 (micod) & 9,541 & -0.131 (H1\_maml) & -0.149 (H1\_maml) & 0.0577, 0.0572 \\
# 32 (micod)  & 32,901 & 0.0401 (H0 no diff) & -0.0689 (H1\_maml) & 0.0581, 0.0574 \\
# 64 (micod) & 121,093 & -0.0588 (H0 no diff) & -0.145 (H1\_maml) & 0.0601, 0.0608 \\
# 256 (micod) & 1,811,461 & 0.0568 (H0 no diff) & 0.0969 (H1 usl) & 0.0578, 0.0593 \\
# 512 (micod) & 7,161,861 & -0.341 (H1\_maml) & -0.376 (H1\_maml) & 0.0525,0.0531 \\
# \hline
# \end{tabular}
# \caption{
# \textbf{Difference between USL vs MAML using effect size.}
# }
# \label{tab:comparison}
# \end{table}
#
# """
#
# import matplotlib.pyplot as plt
#
# data = [
#     (2, 441, -0.0533, 0.0624),
#     (6, 1961, -0.184, -0.100),
#     (8, 3045, 0.0794, -0.00121),
#     (16, 9541, -0.131, -0.149),
#     (32, 32901, 0.0401, -0.0689),
#     (64, 121093, -0.0588, -0.145),
#     (256, 1811461, 0.0568, 0.0969),
#     (512, 7161861, -0.341, -0.376),
# ]
#
# h1_maml_5_count = 0
# h1_usl_5_count = 0
# h0_no_diff_5_count = 0
# h1_maml_10_count = 0
# h1_usl_10_count = 0
# h0_no_diff_10_count = 0
#
# for _, _, effect_size_5, effect_size_10 in data:
#     if effect_size_5 < 0:
#         h1_maml_5_count += 1
#     else:
#         h1_usl_5_count += 1
#
#     if effect_size_10 < 0:
#         h1_maml_10_count += 1
#     elif effect_size_10 > 0:
#         h1_usl_10_count += 1
#     else:
#         h0_no_diff_10_count += 1
#
# x_labels = ['H1 MAML 5', 'H1 USL 5', 'H0 No Diff 5', 'H1 MAML 10', 'H1 USL 10', 'H0 No Diff 10']
# y_values = [h1_maml_5_count, h1_maml_10_count, h1_usl_10_count, h0_no_diff_10_count]
#
# plt.bar(x_labels, y_values)
# plt.xlabel('USL vs MAML using Effect Size')
# plt.ylabel('Frequency/Counts')
# plt.title('Histogram of Effect Sizes')
# plt.grid(axis='y')  # Add grid lines along the y-axis
# plt.show()
