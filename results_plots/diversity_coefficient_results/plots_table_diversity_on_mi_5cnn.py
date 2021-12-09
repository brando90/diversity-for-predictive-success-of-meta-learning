"""
Table for paper with the diversity values from my wandb experiments:
    https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/reports/diversity-on-mini-Imagenet--VmlldzoxMjQxMjkx

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

data: dict = {
    'Probe Networks': ['Random (rep layer)',
                       'MAML (rep layer)',
                       'SL (rep layer)',
                       'Random (head)',
                       'MAML (head)',
                       'SL (head)'],
    'Div with SVCCA': [
        '-6.43e-8 +- 1.38e-7',
        '-6.91e-8 +- 1.75e-7',
        '-3.33e-8 +- 1.59e-7',

        '-1.07e-7 +- 2.35e-7',
        '-8.10e-8 +- 1.78e-7',
        '-3.33e-8 +- 1.43e-7'

    ],
    'Div with PWCCA': [
        '-3.09e-8 +- 1.49e-7',
        '-8.82e-8 +- 1.91e-7',
        '-3.81e-8 +- 1.49e-7',

        '9.53e-9 +- 1.54e-7',
        '-1.00e-7 +- 2.72e-7',
        '-2.38e-9 +- 1.78e-7'

    ],
    'Div with LINCKA': [
        '-9.77e-8 +- 1.32e-7',
        '-1.04e-7 +- 1.43e-7',
        '-5.00e-8 +- 1.67e-7',

        '-3.09e-8 +- 1.66e-7',
        '5.24e-8 +- 2.80e-7',
        '-4.76e-9 +- 1.51e-7'

    ],
    'Div with OPD': [
        '-2.62e-8 +- 1.43e-7',
        '1.59e-7 +- 1.29e-7',
        '-9.29e-8 +- 1.68e-7',

        '-4.76e-9 +- 1.74e-7',
        '-1.19e-8 +- 1.90e-7',
        '-7.15e-9 +- 1.44e-7'
    ],
}

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