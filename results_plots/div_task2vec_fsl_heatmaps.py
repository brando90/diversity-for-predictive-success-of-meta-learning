"""
This file shouldn't exist. But I forgot to close the figures in the task2vec div runs.
Luckily saved the data. But the save plots close the figs now to avoid that bug.
If you don't close the figures it saves the most recent figure.
"""
import os.path
from pathlib import Path

import task_similarity
from uutils import load_json, load_with_torch, dict2namespace
from uutils.plot import save_to, save_to_desktop


def save_corrected_heatmap(path: str):
    path = os.path.expanduser(path)
    args = load_json(Path(path) / 'args.json')
    args = dict2namespace(args)
    embeddings = load_with_torch(Path(path) / 'embeddings.pt')
    # results = load_with_torch(Path(path) / 'results.pt')

    task_similarity.plot_distance_matrix_heatmap_only(embeddings,
                                                      labels=list(range(len(embeddings))),
                                                      distance='cosine',
                                                      show_plot=False)
    save_to_desktop(f'heatmap_corrected_matrix_fsl_{args.data_option}'.replace('-', '_'),
                    save_pdf=False,
                    save_svg=False
                    )
    args.log_root = args.log_root.replace('/home/miranda9/', '~/')
    save_to(args.log_root, plot_name=f'heatmap_corrected_matrix_fsl_{args.data_option}'.replace('-', '_'),
            save_pdf=False,
            save_svg=False
            )


# %%

path = '~/data/logs/logs_May09_15-13-18_jobid_318_pid_15593'
save_corrected_heatmap(path)
path = '~/data/logs/logs_May09_15-13-46_jobid_319_pid_22837'
save_corrected_heatmap(path)
path = '~/data/logs/logs_May09_15-14-21_jobid_320_pid_28675'
save_corrected_heatmap(path)
path = '~/data/logs/logs_May09_15-14-51_jobid_321_pid_43226'
save_corrected_heatmap(path)
path = '~/data/logs/logs_May09_15-15-24_jobid_322_pid_53344'
save_corrected_heatmap(path)
path = '~/data/logs/logs_May09_15-58-47_jobid_325_pid_23957'
save_corrected_heatmap(path)
path = '~/data/logs/logs_May09_17-36-23_jobid_326_pid_164111'
save_corrected_heatmap(path)
path = '~/data/logs/logs_May09_22-17-51_jobid_327_pid_251838'
save_corrected_heatmap(path)
