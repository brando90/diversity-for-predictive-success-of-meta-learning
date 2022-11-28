data_path=$RECORDS
all_sources="ilsvrc_2012 fungi"

for source in ${all_sources}
do
    source_path=${data_path}/${source}
    find ${source_path} -name '*.tfrecords' -type f \
                        -exec sh -c 'python3 -m pytorch_meta_dataset.tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${source_path} {} \;
done