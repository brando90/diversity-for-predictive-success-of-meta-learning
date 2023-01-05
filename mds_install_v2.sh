# define folders - if not done already
echo $HOME
mkdir -p $HOME/data/mds
export MDS_DATA_PATH=$HOME/data/mds
echo $MDS_DATA_PATH

mkdir -p $MDS_DATA_PATH/records
mkdir -p $MDS_DATA_PATH/splits
export RECORDS=$MDS_DATA_PATH/records
export SPLITS=$MDS_DATA_PATH/splits
echo $RECORDS
echo $SPLITS

# install gsutil if not done already
pip install gsutil 

# cd into orig mds directory
cd $HOME/meta-dataset/
# or
cd $HOME/diversity-for-predictive-success-of-meta-learning/meta-dataset/

# Create new conda environment for our 
conda update -n base -c defaults conda

conda create -n mds_env_gpu python=3.9
conda activate mds_env_gpu

#prevents tmp directory from complaining of overflow
mkdir ~/new_tmp
export TMPDIR=~/new_tmp

# install newer list of requirements (verified that this list works w/ py3.9)
pip install -r $HOME/diversity-for-predictive-success-of-meta-learning/req_mds_essentials.txt

cd $HOME/diversity-for-predictive-success-of-meta-learning

# -- aircraft installation (verified it works, note that we NEED pillow==9.0 to not have error)
wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz -O $MDS_DATA_PATH/fgvc-aircraft-2013b.tar.gz

# 2. Extract it into fgvc-aircraft-2013b
tar xf $MDS_DATA_PATH/fgvc-aircraft-2013b.tar.gz -C $MDS_DATA_PATH/
ls $MDS_DATA_PATH/fgvc-aircraft-2013b

# 3. conversion script
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=aircraft \
  --aircraft_data_root=$MDS_DATA_PATH/fgvc-aircraft-2013b \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

# -- CUB installation (verified working)
#1. download
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1 -O $MDS_DATA_PATH/CUB_200_2011.tgz
#2. extract
tar -xzf $MDS_DATA_PATH/CUB_200_2011.tgz -C $MDS_DATA_PATH/

#3. conversion
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=cu_birds \
  --cu_birds_data_root=$MDS_DATA_PATH/CUB_200_2011 \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

#3. Find the following outputs in $RECORDS/cu_birds/:

#200 tfrecords files named [0-199].tfrecords
#dataset_spec.json (see note 1)
ls $RECORDS/cu_birds/

# -- TODO other datasets, haven't verified they worked


# final step - run make_index_files.sh
cd $HOME/pytorch-meta-dataset/
chmod +x make_index_files.sh
./make_index_files.sh
