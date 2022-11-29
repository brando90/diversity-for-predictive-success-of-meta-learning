'''
Goal: compute task2vec diversity of mds
Also we want a histogram of distances, and a heatmap of distances
'''

from diversity_src.dataloaders.metadataset_episodic_loader import get_mds_args
from diversity_src.experiment_mains.main_diversity_with_task2vec import compute_div_and_plot_distance_matrix_for_fsl_benchmark
import uutils
from uutils.logging_uu.wandb_logging.common import cleanup_wandb, setup_wandb
from uutils.argparse_uu.common import create_default_log_root

def plot_histogram_and_div_for_MDS():
    args = get_mds_args()
    args.batch_size = 100
    args.batch_size_eval = 100
    args.data_option = 'mds'
    args.model_option = 'resnet18_pretrained_imagenet'
    args.classifier_opts = None
    args.log_to_wandb = True
    args.wandb_project = 'Meta-Dataset'
    args.experiment_name = 'Task2Vec w/ Histograms'
    args.run_name = 'Batch size 100 MDS Resnet18 Pretrained'
    args.rank = -1
    args.device = uutils.torch_uu.get_device()
    setup_wandb(args)
    create_default_log_root(args)

    compute_div_and_plot_distance_matrix_for_fsl_benchmark(args,split='val')
    cleanup_wandb(args)

if __name__ == '__main__':
    """
python ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/metadataset/metadataset_task2vec_div.py --data_path /shared/rsaas/pzy2/records
    """
    plot_histogram_and_div_for_MDS()
    print('Done! successful!\n')
