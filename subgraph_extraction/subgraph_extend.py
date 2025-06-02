
import os
import math
import struct
import logging
import random
import pickle as pkl
import pdb
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import sys
import torch
from scipy.special import softmax
from utils.dgl_utils import _get_neighbors
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize, get_edge_count, diameter, radius
import networkx as nx

def a_bfs_relational(adj, roots, ano_root, max_nodes_per_hop=None):
    """
    BFS for graphs.
    Modified from dgl.contrib.data.knowledge_graph to accomodate node sampling
    """
    visited = set(ano_root)
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)

def a_get_neighbor_nodes(roots, ano_root, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = a_bfs_relational(adj, roots, ano_root, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extend(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    root1_nei = a_get_neighbor_nodes(set([ind[0]]), set([ind[1]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = a_get_neighbor_nodes(set([ind[1]]), set([ind[0]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = b_node_label(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
    # pruned_subgraph_nodes = subgraph_nodes
    # pruned_labels = labels

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)

    return pruned_subgraph_nodes, pruned_labels, subgraph_size

def b_node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_root = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 4) 
                    for r, sg in enumerate(sgs_single_root)]
    dist_to_root01 = dist_to_root[0][0]%4
    dist_to_root02 = dist_to_root[1][0]%4
    dist_to_roots = np.array(list(zip(dist_to_root01, dist_to_root02)), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels
    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]

    dist_to_roots = np.array(list(zip(dist_to_root[0][0], dist_to_root[1][0])), dtype=int)
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels
    
    return labels, enclosing_subgraph_nodes

def a_node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_root = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 4) 
                    for r, sg in enumerate(sgs_single_root)]
    
    dist_to_root01 = dist_to_root[0][0]
    dist_to_root02 = dist_to_root[1][0]
    dis_nodes_sub = np.where(dist_to_root01 == 4)[0]
    dis_nodes_obj = np.where(dist_to_root02 == 4)[0]
    dist_to_root01[dis_nodes_sub] = -1
    dist_to_root02[dis_nodes_obj] = -1

    dist_to_roots = np.array(list(zip(dist_to_root01, dist_to_root02)), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels
    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]

    return labels, enclosing_subgraph_nodes

