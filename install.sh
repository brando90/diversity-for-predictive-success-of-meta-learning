## Installation script
# to install do: ./~/automl-meta-learning/install.sh
# note that anything else didn't seem to work in my mac for me.

conda update conda
pip install --upgrade pip
pip3 install --upgrade pip

# DO NOT CHANGE THIS, this is needed for the vision cluster & the librarires we are using.
# Only works with python 3.9
# todo - would be nice to have an if statement if we are in the vision cluster...extra work, probably not worth it
# for now the hack is to try cuda, then if fail try normal, to make sure that didn't overwrite the prev if cuda
# succeeded try again, if it fails it won't do anything.
# Overall install cuda if gpu available o.w. it install the normal version.
# any other behaviour is likely unexpected.
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

#host_v=$(hostname)
#if [ $host_v = vision-submit.cs.illinois.edu ]; then
##    echo "Strings are equal."
#  pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#else
##    echo "Strings are not equal."
#  pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#fi

# not sure if needed but leaving here for now
# conda install -y pyyml
# pip install pyyml
# pip install learn2learn

pip install -e ~/ultimate-utils/
pip install -e ~/ultimate-anatome/
pip install -e ~/diversity-for-predictive-success-of-meta-learning/
pip install -e ~/ultimate-aws-cv-task2vec/

# -- extra notes

# - using conda develop rather than pip because uutils installs incompatible versions with the vision cluster
## python -c "import sys; [print(p) for p in sys.path]"
#conda install conda-build
#conda develop ~/ultimate-utils/ultimate-utils-proj-div_src
#conda develop -u ~/ultimate-utils/ultimate-utils-proj-div_src


# -- a100 notes
#conda create -n meta_learning_a100 python=3.9
#conda activate meta_learning_a100
# the above installation seems to work