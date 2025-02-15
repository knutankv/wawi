import numpy as np
import numpy as np
import pdb

from abaqus import *
from abaqus import session
from abaqusConstants import *
import __main__
import section
import regionToolset
import displayGroupMdbToolset as dgm
import step
import part
import material
import assembly
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import symbolicConstants
import odbAccess
import shutil

import regionToolset

import csv
from copy import deepcopy


def get_db(db_type):
    """
    Return the current database (either a model or an odb object).

    If a model db is wanted and no model is active, the model in the mdb is selected regardless,
    as long as there is only one model open in the mdb. If no database fits the requirements, None is returned.

    Args:
        db_type: 'odb' or 'model'
    Returns:
        db: database

    NTNU / Knut Andreas Kvaale, 2018
    """
    if db_type is 'model' or db_type is 'mdb':
        if not session_is_odb():
            db = mdb.models[session.viewports['Viewport: 1'].displayedObject.modelName]
        elif len(mdb.models.keys()) is 1:
            db = mdb.models[mdb.models.keys()[0]]
        elif len(mdb.models.keys()) > 1:
            raise AttributeError('No model is not active, and more than one model is available in model database. Impossible to select correct.')
        else:
            db = None
    else:
        if session_is_odb():
            db = session.viewports[session.currentViewportName].displayedObject
        else:
            db = None

    return db

def session_is_odb():
    """
    Check if current session is ODB.

    Returns:
        is_odb: boolean indicating if the session is odb or not
    """    
    is_odb =(('session' in locals() or 'session' in globals()) and
        session.viewports['Viewport: 1'].displayedObject is not None and
        hasattr(session.viewports['Viewport: 1'].displayedObject, 'jobData'))

    return is_odb

def sort_nodes(nodes, sort_nodes_fun=lambda node: node.label):
    if sort_nodes_fun is not None:
        sort_val = [sort_nodes_fun(node) for node in nodes]
        sort_ix = np.argsort(sort_val)
        nodes_new = [None] * len(nodes)
        for node,ix in zip(nodes,sort_ix):
            nodes_new[ix] = node

        return nodes_new
    else:
        return nodes

def get_element_matrices(region, obj=None, sort_nodes_fun=lambda node: node.label):
    
    if region.nodes is None:    #element region
        if obj is None:
            raise ValueError('If region is of element type, `obj` must be given as input. This is either an `instance` object or a `rootAssembly`')

        elements = region.elements
        element_labels = [el.label for el in elements]
        element_matrix = np.zeros([len(element_labels), 3])

        node_labels = []
        for ix,el in enumerate(elements):
            element_matrix[ix,0] = el.label
            nodes = list(el.connectivity)
            if len(nodes)==3:
                __ = nodes.pop(1)    #remove middle node
            element_matrix[ix, 1:] = nodes

            for node in nodes:
                if node not in node_labels:
                    node_labels.append(node)

        nodes = [obj.getNodeFromLabel(node) for node in node_labels]  
        nodes = sort_nodes(nodes, sort_nodes_fun=sort_nodes_fun)
        node_matrix = create_node_matrix(nodes)
        return node_matrix, element_matrix
        
    else: # node region
        if type(region.nodes) is tuple:
            nodes = region.nodes[0]
        else:
            nodes = region.nodes
        nodes = sort_nodes(nodes, sort_nodes_fun=sort_nodes_fun)
        node_matrix = create_node_matrix(nodes)

        return node_matrix


def create_list_of_nodes(region, node_labels):
    return [region.getNodeFromLabel(int(label)) for label in node_labels]

def create_list_of_elements(region, element_labels):
    return [region.getElementFromLabel(int(label)) for label in element_labels]

def create_node_matrix(nodes):
    labels = np.hstack([node.label for node in nodes])[:,np.newaxis]
    coordinates = np.vstack([node.coordinates for node in nodes])
    node_matrix = np.hstack([labels, coordinates])

    return node_matrix

def get_nodal_phi(step_obj, nodes, field_outputs=['U', 'UR'], flatten_components=True, return_base_disp=False):

    if step_obj.domain != MODAL:   #MODAL is a variable in abaqusConstants
        raise TypeError('Type of input step is not modal!')

    n_nodes = len(nodes)
    n_modes = len(step_obj.frames) - 1

    phi = dict()
    basedisp = dict()
    n_dofs = dict()

    for iout, field_output in enumerate(field_outputs):
        foobj0 = step_obj.frames[0].fieldOutputs[field_output]
        
        n_dofs[field_output] = len(foobj0.values[0].data)
        phio = np.zeros([n_dofs[field_output]*n_nodes, n_modes])

        # Get correct data indices to get correct order (as given in node_labels)
        basedisp[field_output] = np.array([foobj0.getSubset(region=node).values[0].data for node in nodes]).flatten()

        for mode in range(0, n_modes):
            foobj = step_obj.frames[mode+1].fieldOutputs[field_output]
            phio[:, mode] = np.array([foobj.getSubset(region=node).values[0].data for node in nodes]).flatten()

        phi[field_output] = phio*1
    
    if flatten_components:
        phi_merged = [None]*n_modes
        
        for mode in range(n_modes):
            phi_merged[mode] = np.hstack([np.array(phi[key][:, mode]).reshape([-1,n_dofs[key]]) for key in field_outputs]).flatten()
        
        basedisp = np.hstack([np.array(basedisp[key]).reshape([-1,n_dofs[key]]) for key in field_outputs]).flatten()
        phi = np.vstack(phi_merged).T

    if return_base_disp:
        return phi, basedisp
    else:
        return phi


def get_element_phi(step_obj, elements, field_outputs=['SF', 'SM'], flatten_components=True, return_integration_points=False):
    if step_obj.domain != MODAL:   #MODAL is a variable in abaqusConstants
        raise TypeError('Type of input step is not modal!')

    n_modes = len(step_obj.frames) - 1
    n_dofs = dict()
    phi = dict()
    integration_points = dict()

    for iout, field_output in enumerate(field_outputs):
        foobj0 = step_obj.frames[0].fieldOutputs[field_output]
        n_dofs[field_output] = len(foobj0.values[0].data)

        # Get correct data indices to get correct order (as given in node_labels)
        all_integration_points = [value.integrationPoint for value in foobj0.values]

        n_int = len(elements) # number of integration points (same element label might appear multiple times if multiple integration points in element)
        phio = np.zeros([n_dofs[field_output]*n_int, n_modes])        

        data_indices = [None]*n_int
        
        for mode in range(0, n_modes):
            foobj = step_obj.frames[mode+1].fieldOutputs[field_output]
            phio[:, mode] = np.array([foobj.getSubset(region=el)[0].data for el in elements]).flatten()

        integration_points[field_output] = [all_integration_points[ix] for ix in data_indices]
        phi[field_output] = phio*1
    
    if flatten_components:
        phi_merged = [None]*n_modes
        
        for mode in range(n_modes):
            phi_merged[mode] = np.hstack([np.array(phi[key][:, mode]).reshape([-1,n_dofs[key]]) for key in field_outputs]).flatten()
            
        integration_points = np.hstack([np.array(integration_points[key]).reshape([-1,n_dofs[key]]) for key in field_outputs]).flatten()
        phi = np.vstack(phi_merged).T


    if return_integration_points:
        return phi, integration_points
    else:
        return phi

    return phi, integration_points


def get_modal_parameters(frequency_step):
    '''
    Output the modal parameters from frequency step of current output database.

    Parameters
    -------------
    frequency_step : str
        name of step containing the modal results (frequency step)

    Returns
    --------------
    f : float
        numpy array with undamped natural frequencies in Hz of all modes computed
    m : float
        numpy array with modal mass for all modes computed
    '''

    odb = get_db('odb')
    history_region_key = odb.steps[frequency_step].historyRegions.keys()[0]

    ftemp = odb.steps[frequency_step].historyRegions[history_region_key].historyOutputs['EIGFREQ'].data
    f = np.array([x[1] for x in ftemp])

    if 'GM' in odb.steps[frequency_step].historyRegions[history_region_key].historyOutputs.keys():
        mtemp = odb.steps[frequency_step].historyRegions[history_region_key].historyOutputs['GM'].data
        m = np.array([x[1] for x in mtemp])
    else:
        m = np.ones(np.shape(f))    #if no GM field is available, mass normalization is assumed used on eigenvalues
    return f, m