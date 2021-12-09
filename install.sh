## Installation script
# to install do: bash ~/automl-meta-learning/install.sh

#conda update conda

conda create -y -n metalearning_py3.9_torch1.9_cuda11.1 python=3.9
conda activate metalearning_py3.9_torch1.9_cuda11.1
#conda remove --all --name metalearning_gpu

#module load cuda-toolkit/11.1
#module load gcc/9.2.0

# A40, needs cuda at least 11.0, but 1.9 requires 11
#conda activate metalearning_gpu
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
#pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

#conda activate metalearning_cpu
#conda install pytorch torchvision torchaudio cpuonly -c pytorch
#pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# uutils installs
conda install -y dill
conda install -y networkx>=2.5
conda install -y scipy
conda install -y scikit-learn
conda install -y lark-parser -c conda-forge

# due to compatibility with torch=1.7.x, https://stackoverflow.com/questions/65575871/torchtext-importerror-in-colab
#conda install -y torchtext==0.8.0 -c pytorch

conda install -y tensorboard
conda install -y pandas
conda install -y progressbar2
#conda install -y transformers
conda install -y requests
conda install -y aiohttp
conda install -y numpy
conda install -y plotly
conda install -y matplotlib

pip install wandb

# for automl
conda install -y pyyml
#conda install -y torchviz
#conda install -y graphviz

#pip install tensorflow
#pip install learn2learn

#pip install -U git+https://github.com/brando90/pytorch-meta.git
#pip install --no-deps torchmeta==1.6.1
pip install --no-deps torchmeta==1.7.0
#        'torch>=1.4.0,<1.9.0',
#        'torchvision>=0.5.0,<0.10.0',
#pip install -y numpy
pip install Pillow
pip install h5py
#pip install requests
pip install ordered-set

pip install higher
#pip install torch_optimizer
pip install fairseq

#pip install -U git+https://github.com/moskomule/anatome
pip install --no-deps -U git+https://github.com/moskomule/anatome
#    'torch>=1.9.0',
#    'torchvision>=0.10.0',
pip install tqdm

# - using conda develop rather than pip because uutils installs incompatible versions with the vision cluster
## python -c "import sys; [print(p) for p in sys.path]"
conda install conda-build
conda develop ~/ultimate-utils/ultimate-utils-proj-src
conda develop ~/automl-meta-learning/automl-proj-src

# -- extra notes

# local editable installs
# HAL installs, make sure to clone from wmlce 1.7.0 that has h5py ~= 2.9.0 and torch 1.3.1 and torchvision 0.4.2
# pip install torchmeta==1.3.1
