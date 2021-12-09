"""
Main idea:
- compute the distance matrice between tasks X1, X2 of shape [B, M, C, H, W]
- remove diagonal if needed (e.g. X=X1=X2)
- distance_matrix
- compute the diversity as dv(B,X,f) = sum(distance_matrix)/(B**2 - B)

ref:
- inspired from: https://github.com/awslabs/aws-cv-task2vec/blob/master/small_datasets_example.ipynb

Plan:
- do without fitting the final layer (just the pre-trained net as is...) what is the div (distance matrix)
of the small example with the probe network?
- do the [B, M, C, H, W] without fitting the final layer, what is the div of MI?
"""

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch.utils.data
from torch import nn, Tensor

import task_similarity
from models import get_model
from task2vec import Embedding, Task2Vec
from uutils.torch_uu import process_meta_batch
from uutils.torch_uu.dataloaders import get_minimum_args_for_torchmeta_mini_imagenet_dataloader, \
    get_miniimagenet_dataloaders_torchmeta, get_torchmeta_list_of_meta_batches_of_tasks, \
    get_torchmeta_meta_batch_of_tasks_as_list
from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner

def get_data_sets_from_example():
    import datasets
    dataset_names = ('stl10', 'mnist', 'cifar10', 'cifar100', 'letters', 'kmnist')
    print(f'about to get data sets: {dataset_names=}')
    # dataset_list = [datasets.__dict__[name]('./data')[0] for name in dataset_names]
    dataset_list = [datasets.__dict__[name](Path('~/data').expanduser())[0] for name in dataset_names]
    print(dataset_list)

class MetaLearningTask(torch.utils.data.Dataset):

    def __init__(self, meta_learning_task: tuple[Tensor, Tensor, Tensor, Tensor]):
        """
        Note:
            - size of tensors are [M, C, H, W] but remember it comes from a batch of
            tasks of size [B, M, C, H, W]
        """
        meta_learning_task = (data.to('cpu') for data in meta_learning_task)
        spt_x, spt_y, qry_x, qry_y = meta_learning_task
        self.spt_x, self.spt_y, self.qry_x, self.qry_y = spt_x, spt_y, qry_x, qry_y
        self.tensors = (spt_x, spt_y)  # since get_loader does labels = list(trainset.tensors[1].cpu().numpy())
        self.training = True

    def __len__(self):
        # return len(self.spt_x)
        # return len(self.spt_x)
        if self.training:
            return self.spt_x.size(0)  # not 1 since we gave it a single task
        else:
            return self.qry_x.size(0)  # not 1 since we gave it a single task

    def __getitem__(self, idx: int):
        """

        typical implementation:

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        """
        if self.training:
            return self.spt_x[idx], self.spt_y[idx]
        else:
            return self.qry_x[idx], self.qry_y[idx]

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

# - tests

def create_task_embedding_test():
    from task2vec import Task2Vec
    from models import get_model
    from datasets import get_dataset

    dataset = get_dataset('cifar10')
    probe_network = get_model('resnet34', pretrained=True, num_classes=10)
    embedding = Task2Vec(probe_network).embed(dataset)

def plot_distance_matrix_test():
    from task2vec import Task2Vec
    from models import get_model
    import datasets
    import task_similarity

    dataset_names = ('mnist', 'cifar10', 'cifar100', 'letters', 'kmnist')
    dataset_list = [datasets.__dict__[name]('./data')[0] for name in dataset_names]

    embeddings = []
    for name, dataset in zip(dataset_names, dataset_list):
        print(f"Embedding {name}")
        probe_network = get_model('resnet34', pretrained=True, num_classes=int(max(dataset.targets) + 1)).cuda()
        embeddings.append(Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(dataset))
    task_similarity.plot_distance_matrix(embeddings, dataset_names)

def create_embedding_of_a_single_MI_task_test():
    from task2vec import Task2Vec
    from models import get_model

    # - create a MI task as a dataset (since task2vec needs that format)
    args: Namespace = get_minimum_args_for_torchmeta_mini_imagenet_dataloader()
    meta_batch_list: list[tuple[Tensor, Tensor, Tensor, Tensor]] = get_torchmeta_meta_batch_of_tasks_as_list(args)
    spt_x, spt_y, qry_x, qry_y = meta_batch_list[0]
    dataset: MetaLearningTask = MetaLearningTask((spt_x, spt_y, qry_x, qry_y))

    # - create probe_network
    # probe_network: nn.Module = get_default_learner()
    # probe_network = get_model('resnet34', pretrained=True, num_classes=10)
    probe_network: nn.Module = get_model('resnet34', pretrained=True, num_classes=5)
    # probe_network: nn.Module = get_model('resnet18', pretrained=True, num_classes=5)

    # - task2vec: create embedding of task
    task_embedding: Embedding = Task2Vec(probe_network).embed(dataset)
    # task_embedding: Tensor = task2vec.embed(dataset, probe_network, method='variational')
    print(f'{type(task_embedding)=}')
    print(f'{task_embedding=}')
    print(f'{task_embedding.hessian=}')
    print(f'{task_embedding.scale=}')
    print(f'{task_embedding.hessian.shape}')

def plot_distance_matrix_and_div_for_MI_test():
    """
    - sample one batch of tasks and use a random cross product of different tasks to compute diversity.
    """
    # -
    args: Namespace = get_minimum_args_for_torchmeta_mini_imagenet_dataloader()
    # dataloaders: dict = get_miniimagenet_dataloaders_torchmeta(args)
    meta_batch_list: list[tuple[Tensor, Tensor, Tensor, Tensor]] = get_torchmeta_meta_batch_of_tasks_as_list(args)

    # - create probe_network
    # probe_network: nn.Module = get_default_learner()
    # probe_network = get_model('resnet34', pretrained=True, num_classes=10)
    probe_network: nn.Module = get_model('resnet34', pretrained=True, num_classes=5)
    # probe_network: nn.Module = get_model('resnet18', pretrained=True, num_classes=5)

    # - compute task embeddings according to task2vec
    print(f'-- compute task embeddings according to task2vec for number of tasks: {len(meta_batch_list)=}')
    embeddings: list = []
    for spt_x, spt_y, qry_x, qry_y in meta_batch_list:
        dataset: MetaLearningTask = MetaLearningTask((spt_x, spt_y, qry_x, qry_y))
        dataset.eval()
        # probe_network = get_model('resnet34', pretrained=True, num_classes=int(max(dataset.targets) + 1))
        task_embedding: Embedding = Task2Vec(probe_network, max_samples=100, skip_layers=6).embed(dataset)
        # task_embedding: Embedding = Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed2(dataset)
        embeddings.append(task_embedding)

    # - compute distance matrix & task2vec based diversity
    # task_similarity.plot_distance_matrix(embeddings, list(range(len(meta_batch_list))))
    # task_similarity.plot_distance_matrix(embeddings, list(range(len(meta_batch_list))))
    distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
    div_task2vec_mu, div_task2vec_std = task_similarity.stats_of_distance_matrix(distance_matrix, diagonal=False)
    embeddings.append(Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(dataset))
    print(f'{div_task2vec_mu, div_task2vec_std=}')
    print(f'{div_task2vec_mu}+-{div_task2vec_std}')
    task_similarity.plot_distance_matrix_from_distance_matrix(distance_matrix, list(range(len(meta_batch_list))))

def plot_distance_matrix__hd4ml1_test():
    """
    https://www.quora.com/unanswered/What-does-STL-in-the-STL-10-data-set-for-machine-learning-stand-for?ch=10&oid=104775211&share=a3be814f&srid=ovS7&target_type=question
    """
    from task2vec import Task2Vec
    from models import get_model
    import datasets
    import task_similarity

    # dataset_names = ('mnist', 'cifar10', 'cifar100', 'letters', 'kmnist')
    # dataset_names = ('stl10', 'mnist', 'cifar10', 'cifar100', 'letters', 'kmnist')
    # dataset_names = ('stl10', 'letters', 'kmnist')  # hd4ml1
    dataset_names = ('mnist',)
    dataset_list = [datasets.__dict__[name](Path('~/data/').expanduser())[0] for name in dataset_names]

    embeddings = []
    for name, dataset in zip(dataset_names, dataset_list):
        print(f"Embedding {name}")
        # probe_network = get_model('resnet34', pretrained=True, num_classes=int(max(dataset.targets) + 1)).cuda()
        probe_network = get_model('resnet34', pretrained=True, num_classes=int(max(dataset.targets) + 1))
        # embeddings.append(Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(dataset))
        embeddings.append(Task2Vec(probe_network, max_samples=100, skip_layers=6).embed(dataset))
    task_similarity.plot_distance_matrix(embeddings, dataset_names)

if __name__ == '__main__':
    # create_embedding_of_a_single_MI_task_test()
    # plot_distance_matrix_and_div_for_MI_test()
    plot_distance_matrix__hd4ml1_test()
    # get_data_sets_from_example()
    print('Done! successful!\n')
