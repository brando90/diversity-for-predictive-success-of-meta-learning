def download_mi_l2l_now():
    root = '~/data/l2l_data/',
    url: str = 'https://zenodo.org/record/7311663/files/brandoslearn2learnminiimagenet.zip'
    from uutils import download_and_extract
    download_and_extract(url=url,
                         path_used_for_zip=root,
                         path_used_for_dataset=root,
                         rm_zip_file_after_extraction=False,
                         force_rewrite_data_from_url_to_file=True,
                         clean_old_zip_file=True,
                         )


if __name__ == '__main__':
    download_mi_l2l_now()
