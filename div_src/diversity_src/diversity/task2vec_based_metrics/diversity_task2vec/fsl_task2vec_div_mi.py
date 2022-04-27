"""

Goal: compute the diversity using task2vec as the distance between tasks.

Idea: diversity should measure how different **two tasks** are. e.g. the idea could be the distance between distributions
e.g. hellinger, total variation, etc. But in real tasks high dimensionality of images makes it impossible to compute the
integral needed to do this properly. One way is to avoid integrals and use metrics on parameters of tasks -- assuming
a closed for is known for them & the parameters of tasks is known.
For real tasks both are impossible (i.e. the generating distribution for a n-way, k-shot task is unknown and even if it
were known, there is no guarantee a closed form exists).
But FIM estimates something like this. At least in the classical MLE case (linear?), we have
    sqrt(n)(theta^MLE(n) - theta^*) -> N(0, FIM^-1) wrt P_theta*, in distribution and n->infinity
so high FIM approximately means your current params are good.
But I think the authors for task2vec are thinking of the specific FIM (big matrix) as the specific signature for the
task. But when is this matrix (or a function of it) a good representation (embedding) for the task?
Or when is it comparable with other values for the FIM? Well, my guess is that if they are the same arch then the values
between different FIM become (more) comparable.
If the weights are the same twice, then the FIM will always be the same. So I think the authors just meant same network
as same architecture. Otherwise, if we also restrict the weights the FIM would always be the same.
So given two FIM -- if it was in 1D -- then perhaps you'd choose the network with the higher FI(M).
If it's higher than 1D, then you need to consider (all or a function of) the FIM.
The distance between FIM (or something like that seems good) given same arch but different weights.
They fix final layer but I'm not sure if that is really needed.
Make sure to have lot's of data (for us a high support set, what we use to fine tune the model to get FIM for the task).
We use the entire FIM since we can't just say "the FIM is large here" since it's multidimensional.
Is it something like this:
    - a fixed task with a fixed network that predicts well on it has a specific shape/signature of the FIM
    (especially for something really close, e.g. the perfect one would be zero vector)
    - if you have two models, then the more you change the weights the more the FIM changes due to the weights being different, instead of because of the task
    - minimize the source of the change due to the weights but maximize it being due to the **task**
    - so change the fewest amount of weights possible?


Paper says embedding of task wrt fixed network & data set corresponding to a task as:
    task2vec(f, D)
    - fixed network and fixed feature extractor weights
    - train final layer wrt to current task D
    - compute FIM = FIM(f, D)
    - diag = diag(FIM)  # ignore different filter correlations
    - task_emb = aggregate_same_filter_fim_values(FIM)  # average weights in the same filter (since they are usually depedent)

div(f, B) = E_{tau1, tau2} E_{spt_tau1, spt_tau2}[d(task2vec(f, spt_tau1),task2vec(f, spt_tau2)) ]

"""
from torch import nn, Tensor


def distance_tasks_task2vec(probe_net: nn.Module, spt1: Tensor, spt2: Tensor) -> Tensor:
    """

    """
    dataset1 = tensor2dataset(spt1)
    dataset2 = tensor2dataset(spt2)
    embedding1 = Task2Vec(probe_net).embed(dataset1)
    embedding2 = Task2Vec(probe_net).embed(dataset2)
    # TODO: think about distance, R2 or NED or something else?
    dist: Tensor = (embedding1 - embedding2).norm(2)
    return dist


def diversity_task2vec(probe_net: nn.Module,
                       fsl_l2l_loader,
                       ):
    """

    """
    for t in range(num_pair_task_to_consider):
        print(t)
        task_data1: list = task_dataset.sample()  # data, labels
        task_data2: list = task_dataset.sample()  # data, labels
        embed: Tensor = distance_tasks_task2vec()


def get_div_task2vec():
    from task2vec import Task2Vec
    # from models import get_model
    from datasets import get_dataset

    dataset = get_dataset('cifar10')
    # probe_network = get_model('resnet34', pretrained=True, num_classes=10)
    embedding = Task2Vec(probe_network).embed(dataset)

    pass

#

if __name__ == '__main__':
    get_div_task2vec()

