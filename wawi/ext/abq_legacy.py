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

import numpy as np
import os

from wawi.general import merge_tr_phi


def export_wawi(odb_path, path):
    pass

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


def modeshapes_from_region(regionobjs, frequency_step, field_outputs):
    """
    Get modes (shape, frequency and modal mass) from "Frequency step" (eigenvalue analysis) in active Abaqus ODB.

    """
    odb = get_db('odb')

    if odb.steps[frequency_step].domain != MODAL:   #MODAL is a variable in abaqusConstants
        raise TypeError('Type of input step is not modal!')

    Nmodes = len(odb.steps[frequency_step].frames)-1
    phi = [None]*len(field_outputs)

    for iout, field_output in enumerate(field_outputs):
        Ndofs, point_ranges, dof_ranges = count_region(regionobjs, field_output, odb.steps[frequency_step].frames[0])
        phio = np.zeros([np.sum(Ndofs), Nmodes])
        foobj0 = odb.steps[frequency_step].frames[0].fieldOutputs[field_output]

        for ix, regionobj in enumerate(regionobjs):
            current_dof_range = np.arange(dof_ranges[ix], dof_ranges[ix+1])

            for mode in range(0, Nmodes):
                foobj = odb.steps[frequency_step].frames[mode+1].fieldOutputs[field_output]
                phio[:, mode] = np.reshape((np.array([v.data for v in foobj.getSubset(region=regionobj).values])), [np.sum(Ndofs)])

        phi[iout] = phio

    return phi

def str2region(instance_name, setnames, region_type, db_type, *args):
    """
    Construct a region object from a string defining the set name or a region object.

    Args:
        instance_name: string defining the set name (either node or element set) or a region object
        setnames: name of set asked for
        region_type: type of set ('elements' or 'nodes')
        db_type: 'odb' or 'model'
    Optional args:
        db: database object, either mdb.model[...] or session.openOdb(...) - will get from viewport 1 if not given
    Returns:
        regionobjs: region objects

    """

    is_assembly = instance_name is None

    set_type = settype(region_type, db_type)
    standard_sets = {'nodes': [' ALL NODES'], 'elements': [' ALL ELEMENTS']}

    if setnames is None:
        setnames = standard_sets[region_type]

    if len(args)==1:    # a db has been input
        db = args[0]
        isodb = hasattr(db,'jobData')   #check if the input db is reffering to result/odb or model

    else:
        db = get_db(db_type)

    if db is None:
        raise TypeError('The database is empty. Please input a database object, or input parameters that matches one. Remember that odbs have to be active to get the db automatically!')

    if is_assembly:       # Instance name is given
        regroot = db.rootAssembly
    else:
        regroot = db.rootAssembly.instances[instance_name]

    regionobjs = [None] * np.size(setnames)

    for ix,thisname in enumerate(setnames):
        regionobjs[ix] = getattr(regroot, set_type)[thisname]

    return regionobjs


def settype(region_type, db_type):
    """
    Define the string used to get set based on region type and database type.

    Args:
        region_type: 'element' or 'node'
        db_type: 'odb' or 'mdb'
    Returns:
        set_string: string used to obtain set data from database object (odb or mdb)

    """
    if db_type is 'odb':
        if 'element' in region_type.lower():
            set_string = 'elementSets'
        elif 'node' in region_type.lower():
            set_string = 'nodeSets'
        else:
            raise TypeError('Wrong input!')
    elif db_type == 'mdb' or db_type == 'model':
        set_string = 'sets'

    return set_string

def count_region(regionobjs, field_output, frame):
    """
    Count the number of DOFs and points in the specified region objects for given field output and frame object.

    Args:
        regionobjs: list of region objects to query
        field_output: string specifying field output
        frame: frame object (from where fieldOutputs field is accessible)
    Returns:
        Ndofs: number of DOFs for each region (list)
        point_ranges: point/node ranges for each region (list of lists)
        dof_ranges: dof ranges for each region (list of lists)
    """   
    odb = get_db('odb')

    Npoints = [len(frame.fieldOutputs[field_output].getSubset(region=regionobj).values) for regionobj in regionobjs]
    Ndofs = np.dot(Npoints, len(frame.fieldOutputs[field_output].componentLabels))

    dof_ranges = np.cumsum(np.append([0], Ndofs))
    point_ranges = np.cumsum(np.append([0], Npoints))

    return Ndofs, point_ranges, dof_ranges


def wind_set_data(set_strings, frequency_step, instance, db_type, field_outputs, mode_type='nodes', use_node_region_acronym=False):
    # use_node_region_acronym: if True, a node set with identical name as the element set given in set_strings is picked and the nodes assumed to correspond to the element. If not the case, the element set is used to establish the nodes (and thus phi)
    wind_element_regions = str2region(instance, set_strings, 'elements', db_type)  # index 0 is girder, index 1 is columns

    if use_node_region_acronym:
        wind_node_regions = str2region(instance, set_strings, 'nodes', db_type)

    element_labels = [None]*len(set_strings)
    element_node_indices = [None]*len(set_strings)
    node_labels = [None]*len(set_strings)
    node_coordinates = [None]*len(set_strings)
    phi_ae = [None]*len(set_strings)

    for set_ix, set_string in enumerate(set_strings):
        element_labels[set_ix], element_node_indices[set_ix], nl, nc = region2elnodes_legacy(wind_element_regions[set_ix])
        if use_node_region_acronym:
            nl, nc = region2nodes(wind_node_regions[set_ix]) 
            
        node_labels[set_ix] = nl
        node_coordinates[set_ix] = nc

    # Establish modal transformation matrix, phi
    if mode_type=='nodes':
        for set_ix, set_string in enumerate(set_strings):
            phi_ae_temp = modeshapes_from_nodelist(node_labels[set_ix], frequency_step, field_outputs)
            phi_ae[set_ix] = merge_tr_phi(phi_ae_temp[0][0], phi_ae_temp[0][1])
    elif mode_type=='elements':
        for set_ix, set_string in enumerate(set_strings):
            phi_ae_temp, integration_points = modeshapes_from_elementlist(element_labels[set_ix], frequency_step, field_outputs)
            phi_ae[set_ix] = merge_tr_phi(phi_ae_temp[0], phi_ae_temp[1])   

    return element_labels, element_node_indices, node_labels, node_coordinates, phi_ae


def region2elnodes_legacy(regionobjs, avoid_central_nodes=True):
    """
    Give node labels (indices) for each node in specified element set.

    Args:
        regionobjs: region objects to query for node labels

    Returns:
        element_labels: the labels (indices) of the elements in list
        element_node_indices: the labels (indices) of the ndoes in each element; list of lists
        node_labels: all the nodes labels (indices) in a flattened list
        node_coordinates: node coordinates for each element (list of lists)

    """

    objstr = regionobjs.__repr__()
    instance_name = objstr.split(".instances['")[1].split("'].")[0]

    if '.odb' in objstr:
        db = get_db('odb')
        dbtype = 'odb'
    else:
        db = get_db('mdb')
        dbtype = 'mdb'

    # Get the elements object root
    if len(np.shape(regionobjs.elements))>1:
        elements = regionobjs.elements[0]
    else:
        elements = regionobjs.elements

    # Get all element labels and corresponding connectivity (node labels)
    element_labels = np.array([element.label for element in elements])

    # Instance object
    instance = db.rootAssembly.instances[instance_name]

    # Full arrays labels and coordinates
    all_node_labels = np.array([node.label for node in instance.nodes]).flatten([-1])
    all_node_coords = np.array([node.coordinates for node in instance.nodes])

    # Nodes belonging to all the elements
    if dbtype is 'odb':
        element_node_labels = [element.connectivity for element in elements]
    else:
        element_node_labels = [[all_node_labels[ix] for ix in element.connectivity] for element in elements]
    
    if avoid_central_nodes:
        element_node_labels = [[node_lb[0], node_lb[-1]] for node_lb in element_node_labels]

    node_labels = np.unique(np.array(element_node_labels).flatten())

    nodeixs = np.array([np.where(all_node_labels==node)[0] for node in node_labels]).flatten()
    node_coordinates = all_node_coords[nodeixs, :]
    element_node_indices = np.array([np.array([np.where(node_labels==node_label) for node_label in node_labels_for_element]).flatten() for node_labels_for_element in element_node_labels])

    return element_labels, element_node_indices, node_labels, node_coordinates


def region2nodes(regionobj, sortfun=None):
    """
    Give node labels (indices) of nodes in specified node set(s).

    Args:
        regionobj: region object to query for node labels

    Optional args:
        sortfun: function with three inputs (1: x, 2: y, 3:z) to sort nodes by
                 examples: sortfun = lambda x, y, z: -np.arctan2(y,x)
                           sortfun = lambda x, y, z: x

    Returns:
        node_labels: list with nodelabels

    NTNU / Knut Andreas Kvaale, 2018
    """

    set_name = regionobj.__repr__().split("ets[")[1].split("'")[1]

    if len(np.shape(regionobj.nodes))>1:
        nodes = regionobj.nodes[0]
    else:
        nodes = regionobj.nodes

    node_labels = np.array([node.label for node in nodes])
    node_coordinates = np.array([node.coordinates for node in nodes])

    if sortfun != None:
        vals = sortfun(x=node_coordinates[:,0], y=node_coordinates[:,1], z=node_coordinates[:,2])
        sort_ix = np.argsort(vals)
        node_labels = node_labels[:, sort_ix]
        node_coordinates = node_coordinates[sort_ix, :]

    return node_labels, node_coordinates

def modeshapes_from_nodelist(node_labels, frequency_step, field_outputs):
    """
    Get mode shapes from "Frequency step" (eigenvalue analysis) in active Abaqus ODB.

    Args:
        node_labels:
        frequency_step:
        field_outputs:
    Returns:
        phi: mode shape transformation matrix, ordered as NumPy matrices in list for each specified outputs

    """
    odb = get_db('odb')

    if odb.steps[frequency_step].domain != MODAL:   #MODAL is a variable in abaqusConstants
        raise TypeError('Type of input step is not modal!')

    Nnodes = len(node_labels)
    Nmodes = len(odb.steps[frequency_step].frames) - 1
    phi = [None]*len(field_outputs)
    basedisp = [None]*len(field_outputs)

    for iout, field_output in enumerate(field_outputs):
        foobj0 = odb.steps[frequency_step].frames[0].fieldOutputs[field_output]
        
        Ndofs = len(foobj0.values[0].data)
        phio = np.zeros([Ndofs*Nnodes, Nmodes])

        # Get correct data indices to get correct order (as given in node_labels)
        all_nodes = [value.nodeLabel for value in foobj0.values]
        data_indices = [None]*Nnodes

        for ix, node in enumerate(node_labels):
            data_indices[ix] = all_nodes.index(node)

        basedisp[iout] = np.array([foobj0.values[data_ix].data for data_ix in data_indices]).flatten()

        for mode in range(0, Nmodes):
            foobj = odb.steps[frequency_step].frames[mode+1].fieldOutputs[field_output]
            phio[:, mode] = np.array([foobj.values[data_ix].data for data_ix in data_indices]).flatten()

        phi[iout] = phio

    return phi, basedisp


def modeshapes_from_elementlist(element_labels, frequency_step, field_outputs):
    """
    Get mode shape from "Frequency step" (eigenvalue analysis) in active Abaqus ODB.

    Args:
        node_labels:
        frequency_step:
        field_outputs:
    Returns:
        phi: mode shape transformation matrix, ordered as NumPy matrices in list for each specified outputs

    """
    odb = get_db('odb')

    if odb.steps[frequency_step].domain != MODAL:   #MODAL is a variable in abaqusConstants
        raise TypeError('Type of input step is not modal!')

    
    Nmodes = len(odb.steps[frequency_step].frames) - 1
    phi = [None]*len(field_outputs)
    integration_points = [None]*len(field_outputs)

    for iout, field_output in enumerate(field_outputs):
        foobj0 = odb.steps[frequency_step].frames[0].fieldOutputs[field_output]
        Ndofs = len(foobj0.values[0].data)

        # Get correct data indices to get correct order (as given in node_labels)
        all_elements = [value.elementLabel for value in foobj0.values]
        all_integration_points = [value.integrationPoint for value in foobj0.values]

        Nintpoints = len(element_labels) # number of integration points (same element label might appear multiple times if multiple integration points in element)
        phio = np.zeros([Ndofs*Nintpoints, Nmodes])        

        data_indices = [None]*Nintpoints

        for ix, element in enumerate(element_labels):
            data_indices[ix] = all_elements.index(element)
        
        for mode in range(0, Nmodes):
            foobj = odb.steps[frequency_step].frames[mode+1].fieldOutputs[field_output]
            phio[:, mode] = np.array([foobj.values[data_ix].data for data_ix in data_indices]).flatten()

        integration_points[iout] = [all_integration_points[ix] for ix in data_indices]
        phi[iout] = phio
        

    return phi, integration_points
