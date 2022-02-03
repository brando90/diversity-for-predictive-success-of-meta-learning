"""
conda create -n meta_learning_cpu2 python=3.9
conda activate meta_learning_cpu2
conda remove --all --name meta_learning_cpu2
rm -rf /Users/brando/anaconda3/envs/meta_learning_cpu2

conda create -n meta_learning_gpu python=3.9
conda activate meta_learning_gpu
conda remove --all --name meta_learning_gpu
rm -rf /Users/brando/anaconda3/envs/meta_learning_gpu

conda create -n meta_learning python=3.9
conda activate meta_learning
conda remove --all --name meta_learning
rm -rf /Users/brando/anaconda3/envs/meta_learning

PyTorch:
    basing the torch install from the pytorch website as of this writing: https://pytorch.org/get-started/locally/
    pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
    pip3 install torch torchvision torchaudio

Test installation with:
python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"
python -c "import uutils; uutils.torch_uu.gpu_test()"

refs:
    - setup tools: https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#using-find-or-find-packages
    - https://stackoverflow.com/questions/70295885/how-does-one-install-pytorch-and-related-tools-from-within-the-setup-py-install
"""
from setuptools import setup
from setuptools import find_packages
import os

# import pathlib

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='diversity-for-predictive-success-of-meta-learning',  # project name
    version='0.0.1',
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/brando90/diversity-for-predictive-success-of-meta-learning',
    author='Brando Miranda',
    author_email='brandojazz@gmail.com',
    python_requires='>=3.9.0',
    license='MIT',
    package_dir={'': 'div_src'},
    packages=find_packages('div_src'),  # imports all modules/folders with  __init__.py & python files

    # for pytorch see doc string at the top of file
    install_requires=[
        'torchmeta==1.8.0',
        'higher',

        'wandb',

        'learn2learn',
        'cherry-rl',

        # 'fairseq',
    ]  # see readme, we'll fill this when we release
)
