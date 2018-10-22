#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:23:00 2018

@author: ivalera
"""
def order_edges(edge_list, node_map):
    ordered_list = []
    for e1, e2 in edge_list:
        if isinstance(e1, int):
           ordered_list.append((e1, node_map[e2]))
        else:
            ordered_list.append((e2, node_map[e1]))
    return ordered_list


def order_idx(edge_list, node_map):
    rowI = []
    colI = []
    for e1 in edge_list:
    #for e1, e2 in enumerate(edge_list):
        if isinstance(e1, int):
            rowI.append(e1)
            #colI.append(node_map[e2])
        else:
            #rowI.append(e2)
            #colI.append(node_map[e1])
            colI.append(node_map[e1])
    return rowI, colI
            
