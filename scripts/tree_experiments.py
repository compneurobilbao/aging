#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 09:52:14 2017

@author: asier
"""

from ete3 import Tree


def collapsed_leaf(node):
    if all(i <= 3 for i in node2labels[node]) and len(node2labels[node]) == 1:
        return True
    else:
        return False

t = Tree( '((H:1,I:3)HI:0,(B:2,(C:4,D:5)CD:0)BCD:0);', format=3 )
print(t)

for node in t.iter_descendants("levelorder"):
    print(node, node.name, node.dist)

node2labels = t.get_cached_content(store_attr="dist")
node2labels
t2 = Tree(t.write(is_leaf_fn=collapsed_leaf, format=1), format = 1)
print(t2)
t2.get_cached_content(store_attr="name")





def collapsed_leaf(node):
    if len(node2labels[node]) == 1:
       return True
    else:
       return False

t = Tree("((((a,a,a)a,a)aa, (b,b)b)ab, (c, (d,d)d)cd);", format=1)
# We create a cache with every node content
node2labels = t.get_cached_content(store_attr="name")
# We can even load the collapsed version as a new tree
t2 = Tree( t.write(is_leaf_fn=collapsed_leaf) )
t2.get_cached_content(store_attr="name")







def processable_node(node):
    if node.dist < 3:
       return True
    else:
       return False

t = Tree( '((H:1,I:3)HI:0,(B:2,(C:4,D:5)CD:0)BCD:0);', format=1 )
print(t)



t2 =  Tree(t.write(is_leaf_fn=processable_node, format=1))
print(t2)
t2.get_cached_content(store_attr="name")





t = Tree( '(((1:1,3:1)0:2,5:3)0:3),(((2:2,4:2)0:2,6:4)0:1,7:5);', format=1 )
print(t)

def processable_node(node):
    if node.name != 0 and node.get_distance(node.get_tree_root()) > 2:
       return True
    else:
       return False

t2 =  Tree(t.write(is_leaf_fn=processable_node), format=1)
print(t2)


t = Tree( '(((1:1,3:1)1:2,5:3)1:3),(((2:2,4:2)1:2,(6:4,7:4)1:1)1:1);', format=1 )
print(t)
t.get_cached_content(store_attr="dist")

for leaf in t:
    print(leaf.get_distance(leaf.get_tree_root()))


for leaf in t:
    print(leaf.dist)
    
    
for leaf in t:
    print(leaf.get_distance(leaf.get_tree_root(), leaf.get_ancestors()[0]))    
        
    
for node in t.iter_leaves():
    print(node.name)
    
    
    
    
    
    
    
t = Tree('((D,F)E,(B,H)B);', format=8)    
print(t)
    
    