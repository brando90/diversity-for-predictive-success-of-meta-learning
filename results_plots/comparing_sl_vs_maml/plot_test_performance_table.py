"""
4. Comparison of models with respect to meta-test accuracy.
Raw data: https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/reports/SL-vs-ML-vs-Random-via-accuracy-performance-comparison--VmlldzoxMjEzOTc4

Format that pandas wants:
    - key == the column name
    - list == the values
        e.g.  key: [x1, x2] creates a table with key as the colum and x1, x2 as the row values for the column

    data = {'first_column':  ['first_value', 'second_value', ...],
         'second_column': ['first_value', 'second_value', ...],
          ....
        }
    df = pd.DataFrame(data)

refs:
    - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_latex.html
    - https://stackoverflow.com/questions/35634238/how-to-save-a-pandas-dataframe-table-as-a-png
    - https://stackoverflow.com/questions/70008992/how-to-print-a-literal-backslash-to-get-pm-in-a-pandas-data-frame-to-generate-a

"""

import pandas as pd

from uutils.plot import put_pm_to_pandas_data, get_latex_table_as_text_nice_default

data = {
    'Meta-train Initialization': ['Random',
                                  'MAML0',
                                  'SL',

                                  'Random',
                                  'MAML5',
                                  'SL',

                                  'Random',
                                  'MAML5',
                                  'SL',

                                  'Random',
                                  'MAML5',
                                  'SL',
                                  ],
    'Adaptation at Inference': ['no adaptation',
                                'no adaptation',
                                'no adaptation',

                                'MAML5 adaptation',
                                'MAML5 adaptation',
                                'MAML5 adaptation',

                                'MAML10 adaptation',
                                'MAML10 adaptation',
                                'MAML10 adaptation',

                                'Adapt Head only (with LR)',
                                'Adapt Head only (with LR)',
                                'Adapt Head only (with LR)',
                                ],

    'Meta-test Accuracy': ['0.200+-0.029',
                           '0.200+-0.0',
                           '0.200+-0.0025',

                           '0.34+-0.058',
                           '0.60+-0.078',
                           '0.38+-0.066',

                           '0.34+-0.059',
                           '0.60+-0.078',
                           '0.38+-0.064',

                           '0.40+-0.068',
                           '0.60+-0.075',
                           '0.60+-0.073',
                           ],
}

# data = {
#     'Initialization': ['Random',
#                        'Random2',
#                        ],
#
#     'Test Accuracy': ['0.200+-0.029',
#                       '0.200+-0.0',
#                       ],
# }

# - to pandas table
df = pd.DataFrame(data)
print(df)

# https://stackoverflow.com/questions/70009242/how-does-one-generate-latex-table-images-with-proper-equations-from-python-panda

# - to latex,
# idea is to have it initially print a table and then custumize it manually
# https://www.overleaf.com/learn/latex/Tables#Creating_a_simple_table_in_LaTeX
data = put_pm_to_pandas_data(data)
df = pd.DataFrame(data)

print()
# column_format = ''.join(['c' for k in data.keys()])
# print(df.to_latex(index=False, escape=False, column_format=column_format))
# print(df.to_latex(index=False, escape=False, caption='caption goes here', label='label_goes_here'))
print(get_latex_table_as_text_nice_default(df))

