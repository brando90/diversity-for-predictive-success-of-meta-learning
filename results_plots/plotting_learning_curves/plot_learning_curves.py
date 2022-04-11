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



#%%
# -- 5CNN MI


#%%
# -- Resnet12 MI

#%%
# -- 5CNN cifarfs



#%%
# -- Resnet12 cifarfs

