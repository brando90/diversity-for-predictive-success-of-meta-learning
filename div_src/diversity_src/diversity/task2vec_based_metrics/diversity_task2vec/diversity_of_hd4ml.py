"""
File for high diveristy data set.

"""
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