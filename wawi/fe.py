# -*- coding: utf-8 -*-
import numpy as np
from .general import blkdiag, transform_unit

'''
FE-related tools.
'''

def intpoints_from_elements(nodes, elements, sort_axis=0):
    '''
    Get integration points from elements.

    Parameters
    ------------
    nodes : 
    elements :
    sort_axis : 0, optional
    '''
    nodeix = nodeix_from_elements(elements, nodes).astype('int')

    intpoints = (nodes[nodeix[:,0], 1:4]+nodes[nodeix[:,1], 1:4])/2
    sortix = np.argsort(intpoints[:,sort_axis])
    intpoints = intpoints[sortix, :]

    x = intpoints[:, 0]
    y = intpoints[:, 1]
    z = intpoints[:, 2]

    return x, y, z


def nodeix_from_elements(element_matrix, node_matrix, return_as_flat=False):
    nodeix = [None]*len(element_matrix[:,0])
    for element_ix, __ in enumerate(element_matrix[:,0]):
        node1 = element_matrix[element_ix, 1]
        node2 = element_matrix[element_ix, 2]

        nodeix1 = np.where(node_matrix[:,0]==node1)[0][0]
        nodeix2 = np.where(node_matrix[:,0]==node2)[0][0]
        nodeix[element_ix] = [nodeix1, nodeix2]

    nodeix = np.array(nodeix)

    if return_as_flat:
        nodeix = np.unique(nodeix.flatten())

    return nodeix


def create_node_dict(element_dict, node_labels, x_nodes, y_nodes, z_nodes):
    node_dict = dict()
    node_matrix = np.vstack([node_labels, x_nodes, y_nodes, z_nodes]).T

    for key in element_dict.keys():
        node_ix = nodeix_from_elements(element_dict[key], node_matrix, return_as_flat=True)
        node_dict[key] = node_matrix[node_ix, :]

    return node_dict


def elements_with_node(element_matrix, node_label):
    element_ixs1 = np.array(np.where(element_matrix[:,1]==node_label)).flatten()
    element_ixs2 = np.array(np.where(element_matrix[:,2]==node_label)).flatten()

    element_ixs = np.hstack([element_ixs1, element_ixs2])
    element_labels = element_matrix[element_ixs, 0]

    local_node_ix = np.zeros(np.shape(element_ixs))
    local_node_ix[np.arange(0,len(element_ixs1))] = 0
    local_node_ix[np.arange(len(element_ixs1), len(element_ixs1) + len(element_ixs2))] = 1

    return element_labels, element_ixs, local_node_ix


def nodes_to_beam_element_matrix(node_labels, first_element_label=1):
    n_nodes = len(node_labels)
    n_elements = n_nodes-1
    
    element_matrix = np.empty([n_elements, 3])
    element_matrix[:, 0] = np.arange(first_element_label,first_element_label+n_elements)
    element_matrix[:, 1] = node_labels[0:-1]
    element_matrix[:, 2] = node_labels[1:]

    return element_matrix


def node_ix_to_dof_ix(node_ix, n_dofs=6):
    start = node_ix*n_dofs
    stop = node_ix*n_dofs+n_dofs
    dof_ix = []
    for (i,j) in zip(start,stop):
        dof_ix.append(np.arange(i,j))

    dof_ix = np.array(dof_ix).flatten()

    return dof_ix


def dof_sel(arr, dof_sel, n_dofs=6, axis=0):
    N = np.shape(arr)[axis]
    all_ix = [range(dof_sel_i, N, n_dofs) for dof_sel_i in dof_sel]
    sel_ix = np.array(all_ix).T.flatten()
    
    # Select the elements
    arr_sel = np.take(arr, sel_ix, axis=axis)
    
    return arr_sel
    


def elements_from_common_nodes(element_matrix, selected_nodes):
    sel_ix = np.where(np.logical_and([selnode in element_matrix[:,1] for selnode in selected_nodes], [selnode in element_matrix[:,2] for selnode in selected_nodes]))[0]
    selected_element_matrix = element_matrix[sel_ix, :]

    return selected_element_matrix, sel_ix


def transform_elements(node_matrix, element_matrix, e2p, repeats=1):

    n_elements = np.shape(element_matrix)[0]
    T_g2el = [None]*n_elements

    for el in range(0, n_elements):
        n1_ix = np.where(node_matrix[:,0]==element_matrix[el, 1])
        n2_ix = np.where(node_matrix[:,0]==element_matrix[el, 2])

        X1 = node_matrix[n1_ix, 1:]
        X2 = node_matrix[n2_ix, 1:]
        dx = X2-X1
        e1 = dx/np.linalg.norm(dx)

        T_g2el[el] = blkdiag(transform_unit(e1, e2p), repeats)   # Transform from global to local coordinates VLocal = T*VGlobal

    return T_g2el