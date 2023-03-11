#%%
"""
from the following latex table, extract the effect size difference of USL vs MAML5 and USL vs MAML10 and plot on the x-axis the number of parameters in python code, but we only want the micod values. Don't forget to save it to ~/brandomiranda/Desktop and put grid. Do it with matplotlib (no pandas), no regex, plot it.

\begin{table}[h]
\begin{tabular}{|c|c|c|c|c|}
\hline
Filter Size & Number of Params & USL-MAML 5 (Effect Size) & USL-MAML 10 (Effect Size) & 1\% Effect Size \\ \hline
2 (micod) & 441 & -0.0533 (H1\_maml) & 0.0624 (H1\_usl) & 0.0435, 0.0449  \\
% % 4 & 1,093 & 0.3493 (H1\_usl) & 0.275 (H1\_usl) & - \\
6 (micod) & 1,961 & -0.184 (H1\_maml) & -0.100 (H1\_maml) & 0.0489, 0.0504 \\
8 (micod) & 3,045 & 0.0794 (H1\_usl) & -0.00121 (H0 no diff) & 0.0515, 0.0507 \\
% 12 (micod) (failed) & 5,861 & - & - & - \\
% 14 (micod) (failed) & 7,593 & - & - & - \\
16 (micod) & 9,541 & -0.131 (H1\_maml) & -0.149 (H1\_maml) & 0.0577, 0.0572 \\
32 (micod)  & 32,901 & 0.0401 (H0 no diff) & -0.0689 (H1\_maml) & 0.0581, 0.0574 \\
64 (micod)& 121,093 & -0.0588 (H0 no diff) & -0.145 (H1\_maml) & 0.0601, 0.0608 \\
% 256 & 1,811,461 & - & - \\
% 512 & 7,161,861 & - & - \\
ResNet12 (micod) & 1,427,525 & -0.0166 (H0 no diff) & -0.0559 (H0 no diff) & 0.0707, 0.0721 \\
\hline
\end{tabular}
\caption{
\textbf{Difference between USL vs MAML using effect size.}
}
\label{tab:comparison}
\end{table}
"""

import matplotlib.pyplot as plt

# Extract data
x = [441, 1961, 3045, 9541, 32901, 121093, 1427525]
y1 = [-0.0533, -0.184, 0.0794, -0.131, 0.0401, -0.0588, -0.0166]
y2 = [0.0624, -0.1, -0.00121, -0.149, -0.0689, -0.145, -0.0559]

# Plot data
plt.plot(x, y1, label='USL-MAML5', marker='x')
plt.plot(x, y2, label='USL-MAML10', marker='x')

# Add labels and grid
plt.xlabel('Number of Parameters (micod)')
plt.ylabel('Effect Size Difference')
plt.title('Difference between USL vs MAML using effect size')
plt.grid(True)

# Show legend
plt.legend()

# Save plot to desktop
plt.savefig('/Users/brandomiranda/Desktop/effect_size_difference.png')

# Show plot
plt.show()

#%%

import matplotlib.pyplot as plt

# Extract data
x = [441, 1961, 3045, 9541, 32901, 121093, 1427525]
y1 = [-0.0533, -0.184, 0.0794, -0.131, 0.0401, -0.0588, -0.0166]
y2 = [0.0624, -0.1, -0.00121, -0.149, -0.0689, -0.145, -0.0559]

# Plot data
plt.plot(x, y1, label='USL-MAML5', marker='x')
plt.plot(x, y2, label='USL-MAML10', marker='x')

# Add labels and grid
plt.xlabel('Number of Parameters (micod)')
plt.ylabel('Effect Size Difference')
plt.title('Difference between USL vs MAML using effect size')
plt.grid(True)

# Set x-axis scale to logarithmic
plt.xscale('log')

# Show legend
plt.legend()

# Save plot to desktop
plt.savefig('/Users/brandomiranda/Desktop/effect_size_difference_log_linear.png')

# Show plot
plt.show()

# %%
"""

\begin{table}[h]
\centering
\begin{tabular}{|c|c|c|c|c|}
\hline
Filter Size & Number of Params & USL (test acc) & MAML 5 (test acc) & MAML 10 (test acc) \\ \hline
2 (micod) & 441 & 0.481 $\pm$ 0.0205 & 0.493 $\pm$ 0.0197  & 0.467 $\pm$ 0.0184 \\
% 4 & 1,093 & 0. $\pm$ 0. & 0. $\pm$ 0.  & 0.  $\pm$ 0. \\
6 (micod) & 1,961 & 0.588 $\pm$ 0.0169 & 0.626 $\pm$ 0.0189  & 0.608  $\pm$ 0.0178 \\
8 (micod) & 3,045 & 0.606 $\pm$ 0.0161 & 0.591 $\pm$ 0.0178 & 0.607 $\pm$ 0.607 \\
% 12 (failed) & 5,861 & - & - & - \\
% 14 (failed) & 7,593 & - & - & - \\
16 (micod) & 9,541 & 0.655 $\pm$ 0.0149 & 0.678 $\pm$ 0.0154 & 0.681 $\pm$ 0.0157 \\
32 (micod) & 32,901 & 0.689 $\pm$ 0.0151 & 0.682 $\pm$ 0.0150 & 0.701 $\pm$ 0.0154 \\
64 (micod) & 121,093 & 0.694 $\pm$ 0.0135 & 0.704 $\pm$ 0.0155 & 0.718 $\pm$ 0.0152 \\
% 256 & 1,811,461 & - & - & - \\
% 512 & 7,161,861 & - & - & - \\
ResNet12 (micod) & 1,427,525 & 0.778 $\pm$ 0.0124 & 0.781 $\pm$ 0.0124 & 0.786 $\pm$ 0.0119 \\ 
ResNet12 (vggair) & 1,427,525 & 0.727 $\pm$ 0.027 & 0.745 $\pm$ 0.019 & 0.760 $\pm$ 0.019 \\
ResNet12 (mio) & 1,427,525 &  0.845 $\pm$ 0.0121 & 0.849 $\pm$ 0.0136 & 0.848 $\pm$ 0.0133 \\
\hline
\end{tabular}
\caption{
\textbf{Meta Test Accuracy of USL vs MAML.}
}
\label{tab:comparison}
\end{table}
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# define the data as a dictionary
data = {
    'Filter Size': [2, 6, 8, 16, 32, 64, 'ResNet12 (micod)', 'ResNet12 (vggair)', 'ResNet12 (mio)'],
    'Number of Params': [441, 1961, 3045, 9541, 32901, 121093, 1427525, 1427525, 1427525],
    'USL': [0.481, 0.588, 0.606, 0.655, 0.689, 0.694, 0.778, 0.727, 0.845],
    'MAML 5': [0.493, 0.626, 0.591, 0.678, 0.682, 0.704, 0.781, 0.745, 0.849],
    'MAML 10': [0.467, 0.608, 0.607, 0.681, 0.701, 0.718, 0.786, 0.760, 0.848],
}

# create a DataFrame from the dictionary
df = pd.DataFrame(data)

# plot the data using Seaborn
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
ax = sns.lineplot(x='Number of Params', y='value', hue='variable', style='variable',
                  markers=True, dashes=False, data=pd.melt(df, id_vars=['Filter Size', 'Number of Params'],
                                                           value_vars=['USL', 'MAML 5', 'MAML 10']))
ax.set(xscale='log')
ax.set_xticks([500, 1000, 5000, 10000, 50000, 1000000, 10000000])
ax.set_xticklabels(['500', '1k', '5k', '10k', '50k', '1M', '10M'])
plt.xlabel('Number of Parameters')
plt.ylabel('Meta Test Accuracy')
plt.title('Comparison of Meta Test Accuracy of USL vs MAML')
plt.ylim(0.4, 1)
plt.legend(title='Algorithm', loc='lower right')
plt.savefig('/Users/brandomiranda/Desktop/meta_test_acc.png', dpi=300)
plt.show()


#%%

import matplotlib.pyplot as plt

usl_accs = [0.481, 0.588, 0.606, 0.655, 0.689, 0.694, 0.778]
usl_errs = [0.0205, 0.0169, 0.0161, 0.0149, 0.0151, 0.0135, 0.0124]
maml5_accs = [0.493, 0.626, 0.591, 0.678, 0.682, 0.704, 0.781]
maml5_errs = [0.0197, 0.0189, 0.0178, 0.0154, 0.0150, 0.0155, 0.0124]
maml10_accs = [0.467, 0.608, 0.607, 0.681, 0.701, 0.718, 0.786]
maml10_errs = [0.0184, 0.0178, 0.0184, 0.0157, 0.0154, 0.0152, 0.0119]
num_params = [441, 1961, 3045, 9541, 32901, 121093, 1427525]

fig, ax = plt.subplots(figsize=(10,6))

# plot USL data with error bars
ax.errorbar(num_params, usl_accs, yerr=usl_errs, label='USL', linestyle='--', marker='o')

# plot MAML5 data with error bars
ax.errorbar(num_params, maml5_accs, yerr=maml5_errs, label='MAML5', linestyle='--', marker='o')

# plot MAML10 data with error bars
ax.errorbar(num_params, maml10_accs, yerr=maml10_errs, label='MAML10', linestyle='--', marker='o')

# fill between the USL and MAML5 error bars
ax.fill_between(num_params, [x - y for x, y in zip(usl_accs, usl_errs)], [x + y for x, y in zip(usl_accs, usl_errs)], alpha=0.2)
ax.fill_between(num_params, [x - y for x, y in zip(maml5_accs, maml5_errs)], [x + y for x, y in zip(maml5_accs, maml5_errs)], alpha=0.2)
ax.fill_between(num_params, [x - y for x, y in zip(maml10_accs, maml10_errs)], [x + y for x, y in zip(maml10_accs, maml10_errs)], alpha=0.2)

ax.set(xscale='log')
# ax.set_xticks([500, 1000, 5000, 10000, 50000, 1000000, 10000000])
# ax.set_xticklabels(['500', '1k', '5k', '10k', '50k', '1M', '10M'])

# set x and y axis labels
ax.set_xlabel('Number of Parameters')
ax.set_ylabel('Meta Test Accuracy')

# set plot title
ax.set_title('Meta Test Accuracy of USL vs MAML')

# add legend to plot
ax.legend()

# add grid
ax.grid(True)

# save plot to file
plt.savefig('/Users/brandomiranda/Desktop/maml_usl_comparison.png')
# plt.savefig('/Users/brandomiranda/Desktop/maml_usl_comparison.png2')

# show plot
plt.show()
