"""
ES analysis.
ld fo
ld ho
hd all
hd 5cnn
"""

# %%
# fo
"""
From the following latex table:
```
\begin{tabular}{|c|c|c|}
\hline
Model (Dataset) & PT-MAML5 (Decision) & PT-MAML10 (Decision)\\ \hline

Resnet12 (cifar-fs) & -0.266 (H1 maml5) & -0.342 (H1 maml10) \\
Resnet12 (fc100) & -0.251 (H1 maml5) & -0.248 (H1 maml10) \\
Resnet12 (mini-imagenet) & 0.413 (H1 pt) & 0.149 (H1 pt) \\
Resnet12 (tiered-imagenet) & 0.218 (H1 pt) & 0.0290 (H0 no diff.) \\
% Resnet12 (omniglot) & 0.00702 (H0 no diff.) & 0.0679 (H0 no diff.) \\

Resnet12 (aircraft) & -0.671 (H1 maml5) & -1.014 (H1 maml10) \\
Resnet12 (flower) & 1.224 (H1 pt) & 1.125 (H1 pt) \\
Resnet12 (dtd) & 1.332 (H1 pt) & 1.147 (H1 pt) \\
Resnet12 (delaunay) & 1.290 (H1 pt) & 1.262 (H1 pt) \\
Resnet12 (cubirds) & -0.572 (H1 maml5) & -0.452 (H1 maml10) \\

ResNet12 (vggair) &  -0.105 (H1 maml5) & -0.195 (H1 maml10) \\
ResNet12 (vggdtd) & 0.200 (H1 pt) & 0.203 (H1 pt) \\
\hline
\end{tabular}
```
From the above latex table extract the numerical values that have the label (H0 no diff),  (H1 maml[0-9]+), (H1 pt) and put them in three python lists:

ChatGPT
Sure, I'll help you extract the values into three different lists in Python. Let's name these lists h0_no_diff, h1_maml, and h1_pt.

python
Copy code
h0_no_diff = [0.0290]  # We didn't include values from the commented line for 'Resnet12 (omniglot)'
h1_maml = [-0.266, -0.342, -0.251, -0.248, -0.671, -1.014, -0.572, -0.452, -0.105, -0.195]
h1_pt = [0.413, 0.149, 0.218, 1.224, 1.125, 1.332, 1.147, 1.290, 1.262, 0.200, 0.203]
Please note that the values are ordered as they appear in the table (from top to bottom), and the PT-MAML5 column comes before the PT-MAML10 column. The lists h1_maml and h1_pt contain values from both the PT-MAML5 and PT-MAML10 columns. If you want to differentiate these values, you might want to split these lists into two, one for each column.
"""
import numpy as np

h0_no_diff = [0.0290]  # We didn't include values from the commented line for 'Resnet12 (omniglot)'
h1_maml = [-0.266, -0.342, -0.251, -0.248, -0.671, -1.014, -0.572, -0.452, -0.105, -0.195]
h1_pt = [0.413, 0.149, 0.218, 1.224, 1.125, 1.332, 1.147, 1.290, 1.262, 0.200, 0.203]

h0_mean_es = np.mean(h0_no_diff)
h1_maml_mean_es = np.mean(h1_maml)
h1_pt_mean_es = np.mean(h1_pt)

print('--- Low Div (ld) fo:')
print(f"Mean ES for H0: {h0_mean_es}")
print(f"Mean ES for H1 MAML: {h1_maml_mean_es}")
print(f"Mean ES for H1 PT: {h1_pt_mean_es}")

#%%
"""
From the following latex table:
```
\begin{tabular}{|c|c|c|c|c|}
\hline
Model (Dataset) & PT-MAML5 (Decision) & PT-MAML10 (Decision)\\ \hline
Resnet12 (cifar-fs) & -0.602 (H1 maml5) & -0.628 (H1 maml10) \\
Resnet12 (fc100) & -0.800 (H1 maml5) & -0.643 (H1 maml10) \\
Resnet12 (mini-imagenet) & 0.205 (H1 pt) & -0.126 (H1 maml10) \\
% Resnet12 (omniglot) & 0.577 (H0 no diff) & 0.468 (H0 no diff) \\
Resnet12 (tiered-imagenet) & 0.236 (H1 pt) & 0.768 (H1 pt) \\

Resnet12 (aircraft) & -0.667 (H1 maml5) & -0.908 (H1 maml10) \\ 
Resnet12 (flower) & 0.382 (H1 pt) & 0.465 (H1 pt) \\
Resnet12 (dtd) & 1.240 (H1 pt) & 1.020 (H1 pt) \\
Resnet12 (delaunay) & 0.912 (H1 pt) & 0.793 (H1 pt) \\
Resnet12 (cubirds) & -1.043 (H1 maml5) & -1.044 (H1 maml10) \\
\hline
\end{tabular}
\caption{
\textbf{Meta-Test accuracy difference between a Pre-trained (PT) solution vs higher-order (ho) MAML solution using Effect Size (ES/Cohen's d) on low diversity few-shot learning vision datasets.}
```
From the above latex table extract the numerical values that have the labels; 
(H0 no diff),  (H1 maml[0-9]+), (H1 pt) .
Then put them in three python lists, then compute the mean of each list, then print each mean and before printing the means 
print a very concise string starting with --- based on the caption of the table. 
Do it:
"""

import numpy as np

# Extracting values into three lists
h0_no_diff = []  # We didn't include values from the commented line for 'Resnet12 (omniglot)'
h1_maml = [-0.602, -0.628, -0.800, -0.643, -0.126, -0.667, -0.908, -1.043, -1.044]
h1_pt = [0.205, 0.236, 0.768, 0.382, 0.465, 1.240, 1.020, 0.912, 0.793]

# Computing means
h0_no_diff_mean = np.mean(h0_no_diff) if h0_no_diff else "No data"
h1_maml_mean = np.mean(h1_maml)
h1_pt_mean = np.mean(h1_pt)

# Printing a concise string based on the caption
print("--- Comparing Pre-trained vs MAML solutions' accuracy difference on low diversity few-shot learning vision datasets.")

# Printing means
print(f"Mean of H0 no diff: {h0_no_diff_mean}")
print(f"Mean of H1 maml: {h1_maml_mean}")
print(f"Mean of H1 pt: {h1_pt_mean}")

#%%
"""
From the following latex table:
```
\hline
Model (Seeds) (Dataset) & PT-MAML5 (Decision) & PT-MAML10 (Decision) \\ \hline

Resnet12 (fo-maml) (omniglot) & 0.00702 (H0 no diff.) & 0.0679 (H0 no diff.) \\
Resnet12 (ho-maml) (omniglot) & 0.577 (H0 no diff.) & 0.468 (H0 no diff.) \\

ResNet12 (fo-maml) (mio) & -0.0197 (H0 no diff.) & -0.0161 (H0 no diff.) \\

ResNet12 (seed1) (hdb4-micod) & -0.0166 (H0 no diff.) & -0.0559 (H0 no diff.) \\

ResNet12 (seed1) (hdb6-afdo) & -0.0919 (H1 maml5) & 0.0242 (H0 no diff.) \\
ResNet12 (seed1) (hdb7-afto) & 0.0528 (H1 pt) & -0.121 (H1 maml10) \\
ResNet12 (seed1) (hdb8-cado) & -0.167 (H1 maml5) & -0.116 (H1 maml10) \\
ResNet12 (seed1) (hdb9-cavdo) & 0.00798 (H0 no diff.) & 0.0552 (H1 pt) \\
ResNet12 (seed1) (hdb10-micova) & -0.287 (H1 maml5) & -0.308 (H1 maml10) \\

Resnet50 (seed1 vs seed1) (mds) & 0.0834 (H1 pt) & 0.0439 (H0 no diff.) \\
Resnet50 (seed1 vs seed2) (mds) & -0.0391 (H0 no diff.) & -0.102 (H1 maml10) \\
Resnet50 (seed2 vs seed1) (mds) & -0.0566 (H0 no diff.) & -0.131 (H1 maml10) \\
Resnet50 (seed2 vs seed1) (mds) & 0.0178 (H0 no diff.) & -0.0680 (H1 maml10) \\
\hline
\end{tabular}
\caption{
\textbf{Meta-Test accuracy difference between a Pre-trained (PT) solution vs MAML solution using Effect Size (ES/Cohen's d) on high diversity few-shot learning vision datasets.}
```
From the above latex table extract the numerical values that have the labels; 
(H0 no diff),  (H1 maml[0-9]+), (H1 pt) .
Then put them in three python lists, then compute the mean of each list, then print each mean and before printing the means 
print a very concise string starting with --- based on the caption of the table. 
Do it:
"""

# Importing necessary library
import numpy as np

# Lists for the data
h0_no_diff = [0.00702, 0.0679, 0.577, 0.468, -0.0197, -0.0161, -0.0166, -0.0559, 0.0242, 0.00798, 0.0439, -0.0391, -0.0566, 0.0178]
h1_maml = [-0.0919, -0.121, -0.167, -0.116, -0.287, -0.308, -0.102, -0.131, -0.0680]
h1_pt = [0.0528, 0.0552, 0.0834]

# Computing the means
mean_h0_no_diff = np.mean(h0_no_diff)
mean_h1_maml = np.mean(h1_maml)
mean_h1_pt = np.mean(h1_pt)

# Concise statement
# print("--- The table shows the meta-test accuracy difference between a pre-trained solution and MAML solution on high diversity vision datasets.")
print("--- High Div (HD) all:")

# Printing the means
print(f'Mean of H0 no diff: {mean_h0_no_diff}')
print(f'Mean of H1 maml: {mean_h1_maml}')
print(f'Mean of H1 pt: {mean_h1_pt}')

#%%
"""
From the following latex table:
'''
\begin{tabular}{|c|c|c|}
\hline
Filter Size (Dataset) & PT-MAML5 (Decision) & PT-MAML10 (Decision) \\ \hline
2 (micod) & -0.0533 (H1 maml) & 0.0624 (H1 pt)  \\
6 (micod) & -0.184 (H1 maml) & -0.100 (H1 maml) \\
8 (micod) & 0.0794 (H1 pt) & -0.00121 (H0 no diff) \\
16 (micod) & -0.131 (H1 maml) & -0.149 (H1 maml) \\
32 (micod)  & 0.0401 (H0 no diff) & -0.0689 (H1 maml) \\
64 (micod) & -0.0588 (H0 no diff) & -0.145 (H1 maml) \\
256 (micod) & 0.0568 (H0 no diff) & 0.0969 (H1 pt) \\
512 (micod) & -0.341 (H1 maml) & -0.376 (H1 maml) \\
% ResNet12 (micod) & -0.0166 (H0 no diff) & -0.0559 (H0 no diff) \\
% ResNet12 (vggair) & 1,427,525 & 0.727 $\pm$ 0.027 & 0.745 $\pm$ 0.019 & 0.760 $\pm$ 0.019 \\
% ResNet12 (vggdtd) & 1,427,525 & 0.701 $\pm$ 0.022 & 0.701 $\pm$ 0.021 & 0.737 $\pm$ 0.019 \\
% ResNet12 (mio) & 1,427,525 &  0.845 $\pm$ 0.0121 & 0.849 $\pm$ 0.0136 & 0.848 $\pm$ 0.0133 \\
\hline
\end{tabular}
\caption{
\textbf{Meta-Test accuracy difference between a Pre-trained (PT) solution vs MAML solution using Effect Size (ES/Cohen's d) on the high diversity MICOD few-shot learning vision data set.}
'''
From the above latex table extract the numerical values that have the labels; 
(H0 no diff),  (H1 maml[0-9]+), (H1 pt) .
Then put them in three python lists, then compute the mean of each list, then print each mean and before printing the means 
print a very concise string starting with --- based on the caption of the table. 
Do it:
"""

# Importing necessary library
import numpy as np

# Lists for the data
h0_no_diff = [-0.00121, 0.0401, -0.0588, 0.0568]
h1_maml = [-0.0533, -0.184, -0.131, -0.0689, -0.341, -0.376]
h1_pt = [0.0624, 0.0794, 0.0969]

# Computing the means
mean_h0_no_diff = np.mean(h0_no_diff)
mean_h1_maml = np.mean(h1_maml)
mean_h1_pt = np.mean(h1_pt)

# Concise statement
# print("--- The table shows the meta-test accuracy difference between a pre-trained solution and MAML solution on the high diversity MICOD vision dataset.")
print('--- High Div (HD) MICOD 5CNN:')

# Printing the means
print(f'Mean of H0 no diff: {mean_h0_no_diff}')
print(f'Mean of H1 maml: {mean_h1_maml}')
print(f'Mean of H1 pt: {mean_h1_pt}')
