#diversity-for-predictive-success-of-meta-learning

## Installing

## Manual installation [Development, Recommended]

If you are going to use a gpu then do this first before continuing 
(or check the offical website: https://pytorch.org/get-started/locally/):
```angular2html
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

To use library first gitclone as you'd usually do. 

Then install it in development mode (python >=3.9) in (conda) virtual env:

```
conda create -n meta_learning python=3.9
conda activate meta_learning
```

Then install it in edibable mode and all it's depedencies in the current activated conda env:

```
pip install -e . 
```

since the depedencies have not been written install them:

```
pip install -e ~/ultimate-utils
```

To test that uutils install do:
```
python -c "import uutils; print(uutils); uutils.hello()"
python -c "import meta_learning; print(meta_learning)"
python -c "import meta_learning; print(meta_learning); meta_learning.hello()"
```

output should be something like this:

```
(metalearning) brando~/automl-meta-learning/automl-proj-src ❯ python -c "import uutils; print(uutils); uutils.hello()"
<module 'uutils' from '/Users/brando/ultimate-utils/ultimate-utils-proj-src/uutils/__init__.py'>

hello from uutils __init__.py in:
<module 'uutils' from '/Users/brando/ultimate-utils/ultimate-utils-proj-src/uutils/__init__.py'>

(metalearning) brando~/automl-meta-learning/automl-proj-src ❯ python -c "import meta_learning; print(meta_learning)"
<module 'meta_learning' from '/Users/brando/automl-meta-learning/automl-proj-src/meta_learning/__init__.py'>
(metalearning) brando~/automl-meta-learning/automl-proj-src ❯ python -c "import meta_learning; print(meta_learning); meta_learning.hello()"
<module 'meta_learning' from '/Users/brando/automl-meta-learning/automl-proj-src/meta_learning/__init__.py'>

hello from torch_uu __init__.py in:
<module 'uutils.torch_uu' from '/Users/brando/ultimate-utils/ultimate-utils-proj-src/uutils/torch_uu/__init__.py'>

```
