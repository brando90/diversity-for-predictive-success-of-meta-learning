'''
Goal: compute task2vec diversity of mds
Also we want a histogram of distances, and a heatmap of distances
'''

from diversity_src.dataloaders.metadataset_episodic_loader import get_mds_args
from diversity_src.experiment_mains.main_diversity_with_task2vec import compute_div_and_plot_distance_matrix_for_fsl_benchmark
import uutils

def plot_histogram_and_div_for_MDS():
    args = get_mds_args()
    args.batch_size = 3
    args.batch_size_eval = 3
    args.data_option = 'mds'
    args.model_option = 'resnet18_pretrained_imagenet'
    args.classifier_opts = None
    args.rank = -1
    args.device = uutils.torch_uu.get_device()
    compute_div_and_plot_distance_matrix_for_fsl_benchmark(args,split='val')


if __name__ == '__main__':
    """
python ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/metadataset/metadataset_task2vec_div.py --data_path /shared/rsaas/pzy2/records
    """
    plot_histogram_and_div_for_MDS()
    print('Done! successful!\n')
