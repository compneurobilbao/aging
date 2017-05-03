from __future__ import absolute_import, division, print_function
import os
import zipfile, tempfile, urllib, shutil
import aging as ag

data_path = os.path.join(ag.__path__[0], 'data')


def fetch_aging_data():
    """
    Function that downloads preprocessed AGING data if it already does not
    exist in the /aging/data/subjects/

    Parameters
    ----------
    None :

    Returns
    -------
    None :
    """
    aging_data_dir = os.path.join(data_path, 'subjects')
    if os.path.exists(aging_data_dir):
        print('\nDataset found in {}\n'.format(aging_data_dir))
    else:
        temp1 = tempfile.mkdtemp()
        # download zip
        url = ''
        (path, _) = urllib.request.urlretrieve(url)
        # unzip
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(temp1, [x for x in z.namelist()])
        # create subjects folder
        # copy the data to subjects folder
        shutil.copytree(temp1, aging_data_dir)
        shutil.rmtree(temp1)
    return
