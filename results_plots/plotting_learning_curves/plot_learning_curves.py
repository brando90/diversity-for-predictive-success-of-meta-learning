"""
File to plot learning curves to showcase the fair performance between MAML vs SL.

Note:
    - removing the val acc/loss for plots because we are already re-computing it in the performance table
    - in addition, the ones in the leanring curves will be misleading -- especially for USL, since the details of
    USL will always lead to zero accuracy (bellow chance!).
        - how is it possible to do bellow chance (reason they are excluded). USL has 100 units in classification layer.
        But the meta-eval has 5. So USL will output indicies in that range with extremly low prob. But pytorch can
        process the loss regardless even if predict class 99 is not in the test set (weird!?). So that curve is removed
        on purpose to remove confusion.
            - note: that this is not an issue since this essential quantity is recomputed and reported in the perf
            comparison table (which are the main contributions of the paper anyway).
"""

from pathlib import Path

from matplotlib import pyplot as plt

from uutils import load_json
from uutils.plot import plot, save_to_desktop

def get_learning_curve_usl(path: Path, plot_name: str, title: str):
    data: dict = load_json(path)

    x = data['train']['its']
    y = data['train']['loss']
    assert len(x) == len(y)

    plot(x, y,
         xlabel='Epochs',
         ylabel='USL Train Loss',
         title=title
         )
    save_to_desktop(plot_name=plot_name)
    plt.show()

def get_learning_curve_maml(path: Path, plot_name: str, title: str):
    data: dict = load_json(path)

    x = data['train']['its']
    y = data['train']['loss']
    assert len(x) == len(y)

    plot(x, y,
         xlabel='Iterations',
         ylabel='Meta Train Loss',
         title=title
         )
    save_to_desktop(plot_name=plot_name)
    plt.show()

def _get_learning_curve_maml_hack_for_old_ckpt(path: Path, plot_name: str, title: str, log_freq: int):
    data: dict = load_json(path)

    y = data['train']['loss']
    x = [log_freq*i for i in range(0, len(y))]
    assert len(x) == len(y)

    plot(x, y,
         xlabel='Iterations',
         ylabel='Meta Train Loss',
         title=title
         )
    save_to_desktop(plot_name=plot_name)
    plt.show()

#%%
# -- 5CNN MI

# SL 32
path: Path = Path('~/data/logs/logs_May02_17-05-36_jobid_25763/experiment_stats.json').expanduser()
plot_name: str = 'learning_curve_loss_5cnn_usl_mi'
title: str = '5CNN MI Learning Curve USL'
get_learning_curve_usl(path, plot_name, title)

# MAML 32
path = Path('~/data/logs/logs_May02_17-11-03_jobid_25764/experiment_stats.json').expanduser()
plot_name: str = 'learning_curve_loss_5cnn_maml_mi'
title: str = '5CNN MI Learning Curve MAML'
get_learning_curve_maml(path, plot_name, title)

#%%
# -- Resnet12 MI

path: Path = Path('~/data/logs/logs_May02_17-05-36_jobid_25763/experiment_stats.json').expanduser()
plot_name: str = 'learning_curve_loss_resnet12_usl_mi'
title: str = 'Resnet12 MI Learning Curve USL'
get_learning_curve_usl(path, plot_name, title)

# MAML 32
path = Path('~/data/logs/logs_Nov05_15-44-03_jobid_668_NEW_CKPT/experiment_stats.json').expanduser()
plot_name: str = 'learning_curve_loss_resnet12_maml_mi'
title: str = 'Resnet12 MI Learning Curve MAML'
_get_learning_curve_maml_hack_for_old_ckpt(path, plot_name, title, log_freq=200)

#%%
# -- 5CNN cifarfs



#%%
# -- Resnet12 cifarfs

