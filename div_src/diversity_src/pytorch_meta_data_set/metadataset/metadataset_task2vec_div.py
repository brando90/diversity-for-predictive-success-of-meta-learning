'''
Goal: compute task2vec diversity of mds
Also we want a histogram of distances, and a heatmap of distances
'''

from diversity_src.dataloaders.metadataset_episodic_loader import get_mds_args
from diversity_src.experiment_mains.main_diversity_with_task2vec import compute_div_and_plot_distance_matrix_for_fsl_benchmark
import uutils
from uutils.logging_uu.wandb_logging.common import cleanup_wandb, setup_wandb
from uutils.argparse_uu.common import create_default_log_root
from itertools import combinations
#TODO change number of bins??
#report mean, variance

def plot_histogram_and_div_for_MDS(sources):
    args = get_mds_args()
    args.batch_size = 500
    args.batch_size_eval = 500

    #args.k_eval = 20
    #args.k_

    args.num_support = 5
    args.num_query = 15
    args.k_shots = 5
    args.k_query = 15
    args.data_option = 'mds'
    args.model_option = 'resnet18_pretrained_imagenet'
    args.classifier_opts = None
    args.log_to_wandb = False
    args.wandb_project = 'Meta-Dataset'
    args.experiment_name = 'Task2Vec w/ Histograms'
    args.sources = sources
    #args.sources = ['omniglot']
    args.run_name = f'Batch size {args.batch_size_eval} {args.data_option} {args.model_option}'
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
    #'ilsvrc_2012', 'fungi','quickdraw',
    #for source1 in ['aircraft', 'cu_birds', 'dtd',  'omniglot',  'vgg_flower']:

    #plot_histogram_and_div_for_MDS(['omniglot'])
    #for source in list(map(list, combinations(['ilsvrc_2012', 'aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot',
    #                                           'quickdraw', 'vgg_flower'],2))):
    source = ['aircraft','vgg_flower']
    print(source)
    print("===========STARTED SOURCE", source, "===========")
    plot_histogram_and_div_for_MDS(source)
    print("===========FINISHED SOURCE", source, "===========")

    print('Done! successful!\n')
