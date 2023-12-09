"""Prepare ADE20K dataset"""
import os
import shutil
import argparse
import zipfile
from gluoncv.utils import download, makedirs

def download_ade(path, download_dir, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        ('http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip', '219e1696abb36c8ba3a3afe7fb2f4b4606a897c7'),
        ('http://data.csail.mit.edu/places/ADEchallenge/release_test.zip', 'e05747892219d10e9243933371a497e905a4860c'),]
    
    makedirs(download_dir)
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with zipfile.ZipFile(filename,"r") as zip_ref:
            zip_ref.extractall(path=path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and process ADE20K dataset')
    parser.add_argument('--target_dir', type=str, default='ADEChallengeData2016Processed', help='Target directory for processed data')
    parser.add_argument('--download_dir', type=str, default='W:/COMP473/ADEChallengeData2016', help='Download directory for raw data')
    args = parser.parse_args()

    target_dir = os.path.expanduser(args.target_dir)
    download_dir = args.download_dir

    makedirs(target_dir)

    try:
        if download_dir is not None:
            os.symlink(download_dir, target_dir)
    except:
        pass

    download_ade(target_dir, download_dir, overwrite=False)
