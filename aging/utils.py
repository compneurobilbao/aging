#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:36:19 2017

@author: asier
"""
from __future__ import absolute_import, division, print_function

import os
from os.path import join as opj
import zipfile
import tempfile
import urllib
import shutil
import aging as ag

DATA_PATH = os.path.join(ag.__path__[0], 'data')


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
    aging_data_dir = opj(DATA_PATH, 'subjects')
    if os.path.exists(aging_data_dir):
        print('\nDataset found in {}\n'.format(aging_data_dir))
    else:
        temp1 = tempfile.mkdtemp()
        # download zip
        url = ''
        (path, _) = urllib.request.urlretrieve(url)
        # unzip
        with zipfile.ZipFile(path, "r") as zipf:
            zipf.extractall(temp1, [x for x in zipf.namelist()])
        # create subjects folder
        # copy the data to subjects folder
        shutil.copytree(temp1, aging_data_dir)
        shutil.rmtree(temp1)
    return
