"""
Table for paper with the diversity values from my wandb experiments:
    https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/reports/Div-ala-task2vec--VmlldzoxOTYwNTQx

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
    'Probe Networks': [
        'Resnet18 (pt)',
        'Resnet18 (rand)',
        'Resnet34 (pt)',
        'Resnet34 (rand)'
    ],
    'Diversity on MI': [
        '0.117 +- 2.098e-5',
        '0.0955 +- 1.29e-5',
        '0.0999 +- 1.95e-5',
        '0.062 +- 8.12e-6'
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