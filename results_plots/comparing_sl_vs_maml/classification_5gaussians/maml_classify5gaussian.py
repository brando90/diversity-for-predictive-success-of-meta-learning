#!/usr/bin/env python3

"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load Omniglot, and
    * sample tasks and split them in adaptation and evaluation sets.
"""

import random
import numpy as np
import torch
import learn2learn as l2l
import torch.distributions as dist

from torch import nn, optim


#todo: make sure this properly computes accuracy for 5-hot gaussian
def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


#todo: make sure this properly adapts a 5-hot gaussian
def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets (half-half)
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    #sanity check: make sure adaptation_labels AND evaluation_labels are
    # in the form [class0...class0 (shots of them), class1..class1,.... class_ways...class_ways]
    # and that there are ways classes, each with shots examples of each class.
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

'''
Returns tasksets for the 5-gaussian model.
each containing ~20K tasks.

In each task, we first sample 5 gaussian distributions
[(mu1, sigma1), ...., (mu5, sigma5)] ~ N(mu_B, sigma_B)

Each task is defined by a tuple with shape
([ways*shots*2,1,1,1],[ways*shots*2])

The data for the task should look:
[2*shots examples for way 1, 2*shots examples for way 2,...... 2*shots examples for way_{ways})]

The labels for the task should look (there are 2*shots values for each way):
[way1....way1, way2...way2, ....., way_{ways}...way_{ways}]

In other words, we sample 2*shot examples from N(mu1, sigma1) (way1), then 2*shot examples from N(mu2, sigma2) (way2), so forth

And finally return the array of 20K examples for train/val/test
'''
def get_gaussian_tasksets(mu_B, sigma_B, num_fixed_classes, ways, samples, num_tasks):
    #we want 3 arrays, each of shape [torch[ways*samples, 1,1,1], torch[ways*samples]]
    #torch[ways*samples, 1,1,1] ~ N(mu_B, sigma_B) and torch[ways*samples]

    #Distribution from which we sample tasks is a multivariate Gaussian N(mu_B, sigma_B)
    #with 2 * ways dimensions so the first half is allocated to mu's
    # second half is allocated to sigmas
    #task_dist = dist.Normal(mu_B * torch.ones(2 * ways), sigma_B * torch.ones(2*ways))

    #-----EDIT 2/17 after BRANDO meeting: We want to keep a fixed amount of gaussians sampled from N(mu_B, sigma_B)----#
    task_dist = dist.Normal(mu_B * torch.ones(2 * num_fixed_classes), sigma_B * torch.ones(2*num_fixed_classes))
    task_params = task_dist.sample()
    all_mus_sigmas = list(zip(task_params[:num_fixed_classes], torch.abs(task_params[num_fixed_classes:])))
    #print(all_mus_sigmas)
    #----------#

    #gaussian_xs = []
    #gaussian_ys = []
    all_tasksets = []
    for i in range(num_tasks):
        #sample TRUE data distribution gaussians for way 0...{ways-1}
        #such that mu_i = {mu_ij ~ N(mu_B, sigma_B) | 0 <= j < ways}
        # sigma_i = {sigma_ij ~ abs(N(mu_B, sigma_B)) | 0 <= j < ways}
        #Both will contain ways elements

        gaussian_xs_task = []
        gaussian_ys_task = []
        #print(np.random.choice(np.array(all_mus_sigmas), ways))
        #-----EDIT 2/17 after BRANDO meeting: We want to keep a fixed amount of gaussians sampled from N(mu_B, sigma_B)----#
        #workaround to draw samples from list of tuples: https://stackoverflow.com/questions/30821071/how-to-use-numpy-random-choice-in-a-list-of-tuples
        ways_classes = [all_mus_sigmas[i] for i in np.random.choice(len(all_mus_sigmas), ways)]
        mu_i, sigma_i = zip(*ways_classes) #unzip the list with zip(*x)
        #----------#
        #task_params = task_dist.sample()
        #mu_i, sigma_i = task_params[:ways], torch.abs(task_params[ways:])

        #Next, we will go ahead and sample way 0 ... way {ways - 1}
        # for each way we want to collect {samples} samples
        #TODO: PARALLELIZE, convert to TORCH
        for j in range(ways):
            #Get our TRUE data distribution for task i, way j
            task_i_way_j = dist.Normal(mu_i[j], sigma_i[j])

            for k in range(samples):
                #sample a point from our TRUE data distribution for way j:
                #val_ij ~ N(mu_ij, sigma_ij)
                val_ij = task_i_way_j.sample()
                gaussian_xs_task.append(val_ij.numpy().reshape(-1,1,1)) #datapoint from task i, sampled for way j
                gaussian_ys_task.append(j) #we are in way j


        all_tasksets.append((torch.from_numpy(np.array(gaussian_xs_task)), torch.from_numpy(np.array(gaussian_ys_task))))
        #gaussian_xs.append(gaussian_xs_task)
        #gaussian_ys.append(gaussian_ys_task)


    return all_tasksets#(gaussian_xs, gaussian_ys)

#5-way 10-shot
def main(
        ways=5, #number of classes
        shots=30, #number of samples per train step. change to 10?
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32, #number of tasks per epoch
        adaptation_steps=1,
        num_iterations=60000,
        cuda=False,
        seed=42,
        mu_B = 0.1,
        sigma_B = 0.01,
        num_fixed_classes = 50, #number of fixed classes in each dataset
        num_tasks = 500 #number of tasks we want to GENERATE (each task has 5 classes/ways, or whatever ways is)
):
    #Keep the seed for more "determnisitic behavior"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Load train/validation/test tasksets using the benchmark interface
    """
    tasksets = l2l.vision.benchmarks.get_tasksets('omniglot',
                                                  train_ways=ways,
                                                  train_samples=2*shots,
                                                  test_ways=ways,
                                                  test_samples=2*shots,
                                                  num_tasks=20000,
                                                  root='~/data',
    )"""

    # TODO:  replace with gaussian 1d model
    #-----!!!QUESTION: how do these 20000 tasks get sampled?!!!-------#
    #For now I assume I pregenerate 20000 tasks and then just sample from one of these 20K tasks#

    #!!!ASSUMPTION!!!: I assume valid_ways, valid_sampels = test_ways, test_samples
    train_gaussian = get_gaussian_tasksets(mu_B, sigma_B, num_fixed_classes, ways, samples = 2 * shots, num_tasks = num_tasks)
    validation_gaussian = get_gaussian_tasksets(mu_B, sigma_B, num_fixed_classes, ways, samples = 2 * shots, num_tasks = num_tasks)
    test_gaussian = get_gaussian_tasksets(mu_B, sigma_B, num_fixed_classes, ways, samples = 2 * shots, num_tasks = num_tasks)



    # Create model
    #model = l2l.vision.models.OmniglotFC(28 ** 2, ways) #change this to 1 -> 5
    model = l2l.vision.models.OmniglotFC(1,5,sizes=[15,15]) #[1,15]->[15,10]->[10,5]
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    #-------BEGIN META-TRAINING-------#
    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0

        #"INNER LOOP" - where we adapt to each task
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()

            #----Sample a new task here, sampled from benchmark.-----#


            """
            This returns a dataset with size [ways * shots * 2, 1, width, height] at first index and
            a dataset with [ways * shots * 2] one-hot labels at second index
            In our case the batch sampling should return a
            [ways*shots*2,1,1,1]  x-datapoints (100 1x1x1 gaussian values) and
            [ways*shots*2] y-datapoints (100 one-hot values, between [0,4] inclusive)
            """


            #todo: make sure this samples properly
            # should return torch tuple ([ways*shots*2,1,1,1], [ways*shots*2])
            # with the condition that the data array looks like this:
            # [2*shots examples for way 1, 2*shots examples for way 2,...... 2*shots examples for way ways)]
            # the label should look like this:
            # [way1....way1, way2...way2, ....., way ways...way ways] (there are 2*shots values for each way)

            # !!QUESTION!!: Is sampling train and validation independent? (e.g. two diff tasks?)
            #batch = tasksets.train.sample()
            batch = train_gaussian[random.randint(0,len(train_gaussian)-1)]

            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            #batch = tasksets.validation.sample() #todo: make sure this samples properly
            batch = validation_gaussian[random.randint(0,len(validation_gaussian)-1)]

            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # OUTER LOOP, where we optimize over all the tasks.
        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()
    #-------END META-TRAINING--------#

    #-------BEGIN META-TESTING---------#
    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        #batch =
        #-----sample a task here-----#

        #batch = tasksets.test.sample() #todo: make sure this samples properly
        batch = test_gaussian[random.randint(0,len(test_gaussian)-1)]
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)
    #-------END META TESTING---------#

if __name__ == '__main__':
    main()
