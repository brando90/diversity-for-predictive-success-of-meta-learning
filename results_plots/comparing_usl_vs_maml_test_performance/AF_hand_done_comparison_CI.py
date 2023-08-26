# %%
"""
---- maml5 for maml model
train: (meta_loss, meta_loss_ci, meta_acc, meta_acc_ci)=(0.028214871520831367, 0.003274742182851086, 0.9901600080728531, 0.0013802941901992036)
val: (meta_loss, meta_loss_ci, meta_acc, meta_acc_ci)=(0.3598231140174903, 0.021130204718647724, 0.8864533553123474, 0.006392057688268053)
test: (meta_loss, meta_loss_ci, meta_acc, meta_acc_ci)=(0.3489875080175698, 0.02138200479011287, 0.8929866908788681, 0.0053834137143908)

---- FFL (LR) for sl model
train: (meta_loss, meta_loss_ci, meta_acc, meta_acc_ci)=(0.36775601493169296, 0.012619339413773141, 0.9193333333333332, 0.004778493823455438)
val: (meta_loss, meta_loss_ci, meta_acc, meta_acc_ci)=(0.4798475893884928, 0.016811300493261714, 0.8575999999999999, 0.007487154881626354)
test: (meta_loss, meta_loss_ci, meta_acc, meta_acc_ci)=(0.488883008118312, 0.015160930816228066, 0.8552799999999999, 0.007470889108258175)
"""

print('maml 5 maml 5 meta-train acc')
print('0.9901600080728531, 0.0013802941901992036')
print('usl ffl meta-train acc')
print('0.9193333333333332, 0.004778493823455438')


print('maml 5 maml 5 meta-test acc')
print('0.8929866908788681, 0.0053834137143908')
print('usl ffl meta-test acc')
print('0.8552799999999999, 0.007470889108258175')

# check evernote for calculations
"""
https://www.evernote.com/shard/s410/sh/15a155e0-0136-a09b-87ca-06af9bb9d1e8/528f54187bfa4e06a24cb6563d535c65
"""