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
# import aging.proc as pr


def getNewick(node, newick, parentdist, leaf_names):
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = getNewick(node.get_left(), newick, node.dist, leaf_names)
        newick = getNewick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
        newick = "(%s" % (newick)
        return newick


hc_data_path = "/home/asier/Desktop/hc_tools"
partition = op.join(hc_data_path, 'average_networks_12.mat')

data = sio.loadmat(partition)
func_network = data['func_network']

Y = pdist(func_network, 'cosine')
Z = hierarchy.linkage(Y,  method='weighted')

leaf_names = [str(i) for i in np.arange(1, 2515)]

tree = hierarchy.to_tree(Z, False)
newick = getNewick(tree, "", tree.dist, leaf_names)


from ete3 import Tree

t = Tree(newick)
print(t)
t.show()

search_by_size(t, 50)
len(t)
len(t.children)

t.describe()
    
t.children[0].describe()

t.get_descendants()[0].describe()
t.get_leaf_names()


def root_dist(node):
    return node.get_distance(node.get_tree_root())


def nodes_from_level(nodelist, partition_end):
    #print(len(nodelist))
    if len(nodelist) == partition_end:
        return nodelist
    else:
        distances = [root_dist(node) for node in nodelist]
        min_dist_idx = np.argmin(distances)

        nodelist.append(nodelist[min_dist_idx].children[0])
        nodelist.append(nodelist[min_dist_idx].children[1])
        nodelist.remove(nodelist[min_dist_idx])

        return nodes_from_level(nodelist, partition_end)



for i in range(1,20):
    a = nodes_from_level([t], i)
    b = [t.get_leaf_names() for t in a]
    [print(len(c)) for c in b]  
    print("")
     



for node in t.iter_descendants("levelorder"):
    print(node.name, node.get_distance(node.get_tree_root()))

for node in t.iter_descendants("levelorder"):
    print(node.name, node.dist)
  
    
    
    
def search_by_size(node, size):
    "Finds nodes with a given number of leaves"
    matches = []
    for n in node.traverse():
       if len(n) == size:
          matches.append(n)
    return matches

    
t = Tree('((D,F)E,(B,H)B);', format=8)    
print(t)
    