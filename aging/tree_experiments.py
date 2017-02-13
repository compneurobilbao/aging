#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 09:52:14 2017

@author: asier
"""

from __future__ import absolute_import, division, print_function
import os.path as op
import scipy.io as sio
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
import numpy as np
import pickle

import aging as ag

data_path = op.join(ag.__path__[0], 'data')
hc_data_path = op.join(data_path, 'hc')


def get_newick(node, newick, parentdist, leaf_names):
    if node.is_leaf():
        return '%s:%f%s' % (leaf_names[node.id],
                            parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = '):%f%s' % (parentdist - node.dist, newick)
        else:
            newick = ');'
        newick = get_newick(node.get_left(), newick,
                            node.dist, leaf_names)
        newick = get_newick(node.get_right(), ',%s' % (newick),
                            node.dist, leaf_names)
        newick = '(%s' % (newick)
        return newick


def create_hc_tree():
    from ete3 import Tree

    partition = op.join(hc_data_path, 'average_networks_12.mat')

    data = sio.loadmat(partition)
    func_network = data['func_network']

    Y = pdist(func_network, 'cosine')
    Z = hierarchy.linkage(Y,  method='weighted')

    leaf_names = [str(i) for i in np.arange(1, 2515)]

    tree = hierarchy.to_tree(Z, False)
    newick = get_newick(tree, '', tree.dist, leaf_names)

    hc_tree = Tree(newick)
    pickle.dump(hc_tree, open(op.join(hc_data_path, 'hc_tree.p'), 'wb'))


def root_dist(node):
    return node.get_distance(node.get_tree_root())


def nodes_from_level(node_list, partition_end):
    # TODO: Save and load from disk to avoid recalculating
    if len(node_list) == partition_end:
        return node_list
    else:
        distances = [root_dist(node) for node in node_list]
        min_dist_idx = np.argmin(distances)

        node_list.append(node_list[min_dist_idx].children[0])
        node_list.append(node_list[min_dist_idx].children[1])
        node_list.remove(node_list[min_dist_idx])

        return nodes_from_level(node_list, partition_end)


def get_leaf_from_list(node_list):
    return [node.get_leaf_names() for node in node_list]

# TODO
def get_partition_leaf(partition_level):
    pass


