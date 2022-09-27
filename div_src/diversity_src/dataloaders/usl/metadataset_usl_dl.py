"""

refs for using normal data sets and not l2l meta-datasets:
    - https://stackoverflow.com/questions/69792591/combing-two-torchvision-dataset-objects-into-a-single-dataloader-in-pytorch?noredirect=1#comment130421381_69792591
    - https://discuss.pytorch.org/t/does-concatenate-datasets-preserve-class-labels-and-indices/62611
    - http://learn2learn.net/docs/learn2learn.data/#learn2learn.data.meta_dataset.MetaDataset
"""

def meta_data_set_usl_all_splits_dataloaders_using_normal_datasets():
    """
    see: https://discuss.pytorch.org/t/does-concatenate-datasets-preserve-class-labels-and-indices/62611/12?u=brando_miranda

    solutions:
        - use l2l's union data set if possible
        - if that fails use a custom data set that merges the data points, the labels, and either
            - bisect correct to find the offset to add to the label
            - Actually, it's likely easier to preprocess the data points indices to map to the label required label (as you loop through each data set you'd know this value easily and keep a single counter) -- instead of bisecting.
    """
    pass
