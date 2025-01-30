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

import csv
from copy import deepcopy

import numpy as np
import os

from .general import merge_tr_phi

'''
Abaqus interaction module
'''

## Functions to retrieve data from ODB

def modalparameters(frequency_step):
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


def modeshapes_from_region(regionobjs, frequency_step, field_outputs):
    """
    Get modes (shape, frequency and modal mass) from "Frequency step" (eigenvalue analysis) in active Abaqus ODB.

    Args:
        regionobjs: Abaqus region objects in list
        frequency_step: name of frequency step
        field_outputs: list of strings with field output quantities, e.g., ['U', 'UR']
    Returns:
        phi: mode shape transformation matrix, ordered as NumPy matrices in list for each specified outputs
        f: undamped natural frequencies
        m: modal mass
        output_dict: dictionary to access correct index in output phi

    AAJ / Knut Andreas Kvaale, 2017
    Further developed NTNU / Knut Andreas Kvaale, 2018
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


def modeshapes_from_nodelist(node_labels, frequency_step, field_outputs):
    """
    Get mode shapes from "Frequency step" (eigenvalue analysis) in active Abaqus ODB.

    Args:
        node_labels:
        frequency_step:
        field_outputs:
    Returns:
        phi: mode shape transformation matrix, ordered as NumPy matrices in list for each specified outputs

    NTNU / Knut Andreas Kvaale, 2018
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

    NTNU / Knut Andreas Kvaale, 2018
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


def modeshapes_from_set_xydata(field_output, components, output_position, instance_name, set_name, region_type):
    """
    Get mode shapes from "Frequency step" (eigenvalue analysis) in active Abaqus ODB from specified sets.

    Args: NOT FINISHED
        field_output:
        components:
        data_position:
        output_position:
        set_name:
        region_type:
    Returns:
        phi: mode shape transformation matrix (Numpy array)

    NTNU / Knut Andreas Kvaale, 2018
    """

    set_names = [(instance_name + '.' +set_name)]

    odb = get_db('odb')
    n_components = len(components)
    xy_data = [None]*n_components

    if region_type == 'element':
        data_position = INTEGRATION_POINT
    elif region_type == 'node':
        data_position = NODAL

    if output_position == 'element':
        output_position = ELEMENT_NODAL

    for ix, component in enumerate(components):
        refinement = [[COMPONENT, component]]
        variable = [[field_output, data_position, refinement]]
        
        if region_type == 'element':
            xy_data[ix] = session.xyDataListFromField(odb=odb, outputPosition=output_position, variable=variable, elementSets=set_names) 
        else:
            xy_data[ix] = session.xyDataListFromField(odb=odb, outputPosition=output_position, variable=variable, nodeSets=set_names)

    n_elements = len(xy_data[0])
    n_modes = len(xy_data[0][0])

    phi = np.zeros([n_components*n_elements, n_modes])
    for compix, component in enumerate(xy_data):
        for elix, element in enumerate(component):
            for mode in range(0, n_modes):
                phi[elix*n_components + compix, mode] = element[mode][1]


    return phi, xy_data

## MODIFY ODB OR MDB
def set_view_variable(var, component):
    """
    Set a new view variable and component in current odb session.
    
    Args:
        var: variable name
        component: component to display

    NTNU / Knut Andreas Kvaale, 2018
    """   
    position = {NODAL}
    session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(variableLabel=var, outputPosition=NODAL, refinement=(COMPONENT, component),)


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



## MODIFY ODB
def unlock_odb():
    """
    Unlock current ODB file.

    Returns:
        odb: database (odb) object

    NTNU / Knut Andreas Kvaale, 2018
    """   
    odb = session.viewports[session.currentViewportName].displayedObject

    if odb.isReadOnly:
        load_path = odb.path
        odb.close()
        odb = odbAccess.openOdb(load_path, readOnly=False)
        session.viewports['Viewport: 1'].setValues(displayedObject=session.odbs[load_path])

    return odb


def copy_and_unlock_odb():
    """
    Copy and unlock current ODB file.

    Returns:
        odb: database (odb) object

    NTNU / Knut Andreas Kvaale, 2018
    """   
    odb = session.viewports[session.currentViewportName].displayedObject
    old_file_path = odb.path
    new_file_path = odb.path.split('.odb')[0]+'_org.odb'

    shutil.copyfile(old_file_path, new_file_path)   #copy the old file

    odb.close()
    odb = odbAccess.openOdb(old_file_path, readOnly=False)
    session.viewports['Viewport: 1'].setValues(displayedObject=session.odbs[old_file_path])

    return odb


def session_is_odb():
    """
    Check if current session is ODB.

    Returns:
        is_odb: boolean indicating if the session is odb or not

    NTNU / Knut Andreas Kvaale, 2018
    """    
    is_odb =(('session' in locals() or 'session' in globals()) and
        session.viewports['Viewport: 1'].displayedObject is not None and
        hasattr(session.viewports['Viewport: 1'].displayedObject, 'jobData'))

    return is_odb


def save_and_reopen_odb():
    """
    Save and reopen database (odb) as read-only.

    Returns:
        odb: odb object

    NTNU / Knut Andreas Kvaale, 2018
    """
    odb = get_db('odb')
    odb.save()
    load_path = odb.path
    odb.close()

    odb = odbAccess.openOdb(load_path, readOnly=True)

    return odb


def add_response_step_from_modal(phi_response, field_outputs, modal_var, frequency_step, step_name, region_strings, region_type, instance_name, description):
    """
    Add an artificial step in Abaqus ODB for response data.

    Args:
        phi_response: phi of the requested response quantities (list with one matrix for each response quantities)
        field_outputs: names of field output variables
        modal_var: covariance matrix for the generalized (modal) DOFs
        frequency_step: name of the new artificial step_name
        step_name: node set name or region object that define what nodes / DOFs phi refers to
        regionobjs: Abaqus region objects in list
        instance_name: name of the instance
        description: frame description

    NTNU / Knut Andreas Kvaale, 2018
    """

    odb = copy_and_unlock_odb()
    regionobjs = str2region(instance_name, region_strings, region_type, 'odb')
    instance = odb.rootAssembly.instances[instance_name]

    step_data = odb.Step(name=step_name, description='Response step', domain=TIME, timePeriod=0)
    frame = step_data.Frame(incrementNumber=0, description='Response', frameValue=0)

    type_dict = {'SF': [TENSOR_3D_SURFACE, INTEGRATION_POINT, 'Section forces'], 'SM': [TENSOR_3D_SURFACE, INTEGRATION_POINT, 'Section moments'], 'U': [VECTOR, NODAL, 'Spatial displacement'], 'UR': [VECTOR, NODAL, 'Rotational displacement'] }

    for ix, field_output in enumerate(field_outputs):
        foobj_ref = odb.steps[frequency_step].frames[0].fieldOutputs[field_output]
        phi = phi_response[ix]
        region_type = type_dict[field_output][1]
        comps = len(odb.steps[frequency_step].frames[0].fieldOutputs[field_output].componentLabels)

        sigma = np.sqrt(np.sum((np.dot(phi, modal_var) * phi), axis=1))   # Calculate sigma (square root of covariance matrix) from modal coordinates
        sigma_comp = np.reshape(sigma, [-1, 3]).astype('float')
        data = [list(this) for this in sigma_comp]

        foobj = frame.FieldOutput(name=field_output, description=type_dict[field_output][2], type=type_dict[field_output][0], validInvariants=())

        N = len(odb.steps[frequency_step].frames[0].fieldOutputs[field_output].values)
        Ndofs, point_ranges, dof_ranges = count_region(regionobjs, field_output, odb.steps[frequency_step].frames[0])

        for regix,regionobj in enumerate(regionobjs):
            good_ix, good_entries = good_element_ix(foobj_ref, regionobj)
            point_range = range(point_ranges[regix],point_ranges[regix+1])

            foobj.addData(position=region_type, instance=instance, labels=good_entries, data=data)

        step_data.setDefaultField(foobj)

    odb = save_and_reopen_odb()

    return odb


def add_std_to_frame(odb, frame, instance_name, modal_var, phi, regionobj, field_output, reference_step):
    '''
    Under development. Not verified.
    '''
    if odb.isReadOnly:
        raise TypeError('ODB is read only. Unable to add data.')

    type_dict = {'SF': [TENSOR_3D_SURFACE, INTEGRATION_POINT, 'Section forces'], 'SM': [TENSOR_3D_SURFACE, INTEGRATION_POINT, 'Section moments'], 'U': [VECTOR, NODAL, 'Spatial displacement'], 'UR': [VECTOR, NODAL, 'Rotational displacement'] }
    foobj_ref = odb.steps[reference_step].frames[0].fieldOutputs[field_output]

    region_type = type_dict[field_output][1]
    comps = len(odb.steps[reference_step].frames[0].fieldOutputs[field_output].componentLabels)

    sigma = np.sqrt(np.sum((np.dot(phi, modal_var) * phi), axis=1))   # Calculate sigma (square root of covariance matrix) from modal coordinates
    sigma_comp = np.reshape(sigma, [-1, comps]).astype('float')
    data = [list(this) for this in sigma_comp]

    # If already exists, don't create new, but assign to that.
    if field_output not in frame.fieldOutputs.keys():
        foobj = frame.FieldOutput(name=field_output, description=type_dict[field_output][2], type=type_dict[field_output][0], validInvariants=())
    else:
        foobj = frame.fieldOutputs[field_output]

    N = len(odb.steps[reference_step].frames[0].fieldOutputs[field_output].values)
    good_ix, good_entries = good_element_ix(foobj_ref, regionobj)
    instance = odb.rootAssembly.instances[instance_name]

    foobj.addData(position=region_type, instance=instance, labels=good_entries, data=data)
    step_data.setDefaultField(foobj)



def add_complex_mode_step(phi, eigvals, instance_name, step_name, region):
    """
    Add an artificial step in Abaqus ODB for complex modes.

    Args:
        phi: complex eigenvector matrix
        eigvals: complex eigenvalues
        instance_name: name of the instance
        step_name: name of the new artificial step_name
        regionobj: Abaqus region object

    Knut Andreas Kvaale, 2018
    """

    odb = unlock_odb()
    complex_step = odb.Step(name=step_name, description='Complex modes', domain=MODAL)
    frame0 = complex_step.Frame(incrementNumber=0, description='Base state', frameValue=0)
    frame_data = frame0.FieldOutput(name='U', description='Spatial displacement', type=VECTOR, validInvariants=(MAGNITUDE,))
    instance = odb.rootAssembly.instances[instance_name]

    for m, lambdam in enumerate(eigvals):
        phim = np.reshape(phi[:, m], (-1, 3)).astype('float')
        xim = -np.real(lambdam)/abs(lambdam)

        freqm_ud_rad = abs(lambdam)
        freqm_d_rad = abs(np.imag(lambdam))
        freqm_ud_Hz = freqm_ud_rad/(2*np.pi)

        periodm_ud = 2*np.pi/abs(lambdam)

        description_m = 'Mode ' + str(m+1) + ':  f = ' + str(freqm_ud_Hz) + 'Hz  |  om = ' + str(freqm_ud_rad) + 'rad/s   |   T = ' + str(periodm_ud) + 's  |  xi = ' + str(xim*100) + '%'

        frame_m = complex_step.Frame(incrementNumber=m+1, description=description_m, frameValue=freqm_ud_Hz)
        frame_data = frame_m.FieldOutput(name='U', description='Spatial displacement', type=VECTOR, validInvariants=(MAGNITUDE,))
        nodelabels = np.array([node.label for node in regionobj.nodes[0]]).astype('int')

        frame_data.addData(position=NODAL, instance=instance, labels=nodelabels, data=np.real(phim), conjugateData=np.imag(phim))

    odb.save()
    load_path = odb.path
    odb.close()
    odb = odbAccess.openOdb(load_path, readOnly=True)

## MODIFY MDB
def mass_and_stiffness_input(stiffness,mass,pontoon_set_names,pont_nodes,trans_mats,filename):
    pontoons = len(pontoon_set_names)
    if len(pont_nodes) != pontoons or len(trans_mats)!=pontoons:
        raise ValueError('Mismatch between dimensions for input variables: pontoon_set_names, pont_nodes and trans_mats')
    f = open(filename, 'w')

    for pontoon in range(0,pontoons):
        f.write('********************************PONTOON NUMBER {0} ************************************* \n'.format(str(pontoon+1)))
        f.write('*USER ELEMENT, LINEAR, NODES=1, UNSYM, TYPE=U{0}00 \n'.format(str(pontoon+1))) 	# Defines a linear user element
        f.write('1, 2, 3, 4, 5, 6 \n')                                               # The element has one node with 6 DOFS

        T = trans_mats[pontoon]

        K = np.dot(np.dot(T.transpose(), stiffness),T)
        M = np.dot(np.dot(T.transpose(), mass),T)

        f.write('*MATRIX, TYPE=MASS \n')                                            # Defines the mass matrix in GLOBAL coordinate system
        for n in range(0,6):
            string1 = ','.join(map(str, M[n, 0:4]))
            string2 = ','.join(map(str, M[n, 4:6])) 
            f.write(string1 + '\n' + string2 +'\n')
            
        f.write('*MATRIX, TYPE=STIFFNESS \n')
        for n in range(0,6):
            string1 = ','.join(map(str, K[n, 0:4]))
            string2 = ','.join(map(str, K[n, 4:6]))
            f.write(string1 + '\n' + string2 +'\n')

        f.write('*ELEMENT, TYPE=U{0}00, ELSET={1} \n'.format(str(pontoon+1),pontoon_set_names[pontoon])) 				#Introduce one user element into the FE model
        f.write('800{0}, {1} \n'.format(str(pontoon+1),pont_nodes[pontoon]))     					#Numbering elements as 8001,8002,...,8007, followed by first node number forming the element
        f.write('*UEL PROPERTY, ELSET={0} \n'.format(pontoon_set_names[pontoon]))


def update_input(freq,wadam_file,input_file,pontoon_set_names,pont_nodes,trans_mats):
    from .io import import_wadam_mat
    static_mass, stiffness, added_mass, damping, frequency = import_wadam_mat(wadam_file)
    mass = freq_sysmat(added_mass,frequency,freq)+static_mass
    mass_and_stiffness_input(stiffness,mass,pontoon_set_names,pont_nodes,trans_mats,input_file)
    print('Input file '+ input_file + ' is modified to correspond to added mass at f = ' + str(freq) + ' Hz.')


def imperfection_input(node_labels, displacement_vector, input_file=None, rotations=False):
    
    d = np.array(displacement_vector)

    if rotations is True:
        n_nodes = len(d)/6
        d_trans = np.zeros([n_nodes*3])
        for node in range(0, n_nodes):
            d_trans[node*3:node*3+3] = d[node*6:node*6+3]

        d = d_trans
    else: 
        n_nodes = len(d)/3

    mat = np.hstack([np.array(node_labels)[:, np.newaxis], d.reshape(-1, 3)])

    if input_file != None:
        open(input_file, 'w').close()

        with open(input_file, 'a') as f:
            f.write('*IMPERFECTION \n')
            np.savetxt(f, mat, delimiter=',', fmt='%i,%.8e,%.8e,%.8e')

    return mat


def add_input_file(model, input_file_path, pos, target_string=None, relative_pos=0):

    if target_string != None:
        pos = model.keywordBlock.sieBlocks.index(target_string)

    model.keywordBlock.insert(pos+relative_pos, '*INCLUDE, INPUT={0}'.format(input_file_path))


def add_springs(assem, Kh, region, name):
    
    ON = symbolicConstants.AbaqusBoolean(1)

    Kpos = (Kh+abs(Kh))/2
    Krem = (Kh-abs(Kh))/2

    for dof in range(2, 5):
        if Kpos[dof, dof] != 0:
            assem.engineeringFeatures.SpringDashpotToGround(name=name+'_K%i%i' % (dof+1, dof+1), region=region, orientation=None, dof=dof+1, springBehavior=ON, springStiffness=Kpos[dof, dof])

    return Krem


def add_inertia(assem, M0, region, name, specify_rot=False):
    if specify_rot is True:
        assem.engineeringFeatures.PointMassInertia(alpha=0.0, composite=0.0, i11=M0[3, 3], i12=M0[3, 4], i13=M0[3, 5], i22=M0[4, 4], i23=M0[4, 5],
                                                   i33=M0[5, 5], mass1=M0[0, 0], mass2=M0[1, 1], mass3=M0[2, 2], name=name+'_M0', region=region)
        comps = range(0, 6)
    elif specify_rot is False:
        assem.engineeringFeatures.PointMassInertia(alpha=0.0, composite=0.0, mass1=M0[0, 0], mass2=M0[1, 1], mass3=M0[2, 2], name=name+'_M0', region=region)
        comps = range(0, 3)

    Mrem = deepcopy(M0)

    for comp in comps:
        Mrem[comp, comp] = 0

    return Mrem


#%% USEFUL FUNCTIONS FOR DEALING WITH REGIONS IN DATABASE
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

    NTNU / Knut Andreas Kvaale, 2018
    """   
    odb = get_db('odb')

    Npoints = [len(frame.fieldOutputs[field_output].getSubset(region=regionobj).values) for regionobj in regionobjs]
    Ndofs = np.dot(Npoints, len(frame.fieldOutputs[field_output].componentLabels))

    dof_ranges = np.cumsum(np.append([0], Ndofs))
    point_ranges = np.cumsum(np.append([0], Npoints))

    return Ndofs, point_ranges, dof_ranges


def good_element_ix(foobj, regionobj):
    """
    Get the indices of the good (??) elements.

    Args:
        foobj: field object
        regionobj: region object
    Returns:
        good_ix: ?
        good_entries: ?
                
    NTNU / Knut Andreas Kvaale, 2018
    """   
    foobj_values = foobj.getSubset(region=regionobj).values
    region_type = obtain_region_types([regionobj])[0]

    if region_type is 'elements':
        rootobj = regionobj.elements
        label_string = 'elementLabel'
    elif region_type is 'nodes':
        rootobj = regionobj.nodes
        label_string = 'nodeLabel'

    if type(rootobj) is tuple:
        rootobj = rootobj[0]

    good_entries = [getattr(val, label_string) for val in foobj_values]
    all_entries = [obj.label for obj in rootobj]

    good_ix = [all_entries.index(this_entry) for this_entry in good_entries]

    return good_ix, good_entries


def obtain_region_types(regionobjs):
    """
    Get the region types of list of region objects.

    Args:
        regionobjs: list of region objects
    Returns:
        region_type: list of region types
                
    NTNU / Knut Andreas Kvaale, 2018
    """   
    elementsets = [regionobj.nodes is None for regionobj in regionobjs]    # true if regionobjects are element sets
    settypedict = {False: 'nodes', True: 'elements'}
    region_type = [settypedict[elementset] for elementset in elementsets]

    return region_type


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

    AAJ / Knut Andreas Kvaale, 2017
    Further developed NTNU / Knut Andreas Kvaale, 2018
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

def region2elnodes(regionobj, avoid_central_nodes=True, db_type='odb'):
    """
    Give node labels (indices) for each node in specified element set.

    Args:
        regionobj: region object to query for labels

    Returns:
        element_labels: the labels (indices) of the elements in list
        element_node_indices: the labels (indices) of the ndoes in each element; list of lists
        node_labels: all the nodes labels (indices) in a flattened list
        node_coordinates: node coordinates for each element (list of lists)

    NTNU / Knut Andreas Kvaale, 2018
    """
    
    db = get_db(db_type)
    objstr = regionobj.__repr__()
    if 'instances' in objstr:
        instance_name = objstr.split(".instances['")[1].split("'].")[0]
    else:
        instance_name = None

    if instance_name is None:
        instance = db.rootAssembly
    else:
        instance = db.rootAssembly.instances[instance_name]

    # Get the elements object root
    if len(np.shape(regionobj.elements))>1:
        elements = regionobj.elements[0]
    else:
        elements = regionobj.elements

    # Get all element labels and corresponding connectivity (node labels)
    element_labels = np.array([element.label for element in elements])
    node_labels = [el.connectivity for el in elements]

    if avoid_central_nodes:
        node_labels = np.unique([item for sublist in node_labels for item in sublist[:1]+sublist[-1:]])
    else:
        node_labels = [item for sublist in node_labels for item in sublist]

    element_matrix = None
    
    return element_labels, node_labels, element_matrix


def get_element_matrix(element_labels=None):    #if None is specified, full model is exported
    pass

def get_node_matrix(node_labels=None):    #if None is specified, full model is exported
    pass

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

    NTNU / Knut Andreas Kvaale, 2018
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


#%% RETRIEVE THINGS FROM DATABASE
def element_orientations(element_labels, instance_name):
    """
    Provide transformation matrices describing the three unit vectors of the local CSYS of all elements in element_labels.

    Args:
        element_labels: element labels to query
        instance_name: name of instance to find beam orientations

    Returns:
        element_orientations: array of numpy 2d-arrays with transformation matrices of all elements in element_labels

    NTNU / Knut Andreas Kvaale, 2018
    """
    db_type = 'odb' # may consider mdb option later
    db = get_db(db_type)
    
    all_elements = db.rootAssembly.elementSets[' ALL ELEMENTS'].elements[0]
    all_nodes = db.rootAssembly.nodeSets[' ALL NODES'].nodes[0]
    all_element_labels = [value.label for value in all_elements]   
    all_node_labels = [value.label for value in all_nodes]   
    element_orientations = [None]*len(element_labels)
     
    beam_orientations = db.rootAssembly.instances[instance_name].beamOrientations

    for beam_orientation in beam_orientations:
        bo_elements = [value.label for value in beam_orientation.region.elements]
        for this_element_label in bo_elements:
            if this_element_label in element_labels:
                n1_temp = np.array(beam_orientation.vector)
                node_labels = all_elements[all_element_labels.index(this_element_label)].connectivity

                node_start_coor = all_nodes[all_node_labels.index(node_labels[0])].coordinates
                node_end_coor = all_nodes[all_node_labels.index(node_labels[-1])].coordinates
                t = (node_end_coor-node_start_coor) 
                t = t/np.linalg.norm(t)

                n2 = np.cross(t, n1_temp)
                n2 = n2/np.linalg.norm(n2)

                n1 = np.cross(n2, t)        #does this actually work?

                element_orientations[np.where(element_labels == this_element_label)[0]] = np.array([t,n1,n2])

    return element_orientations


def freq_sysmat(mat,freqs,freq):
    """
    Interpolate frequency dependent matrix, for given frequency value. !! Deprecated - use numpy functions directly instead !!

    Args:
        mat: 3D matrix (Numpy array)
        freqs: frequency axis (Numpy array)
        freq: selected frequency value (scalar)
    Returns:
        mat_sel: 2D matrix corresponding to queried frequency value (Numpy array)

    NTNU / AAJ / Knut Andreas Kvaale, 2018
    """   
    from .general import interp1z
    
    if freq == []:
        mat_sel = 0
    else:
        mat_sel = interp1z(freqs[:,0,0],mat,freq)
    return mat_sel


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



def settype(region_type, db_type):
    """
    Define the string used to get set based on region type and database type.

    Args:
        region_type: 'element' or 'node'
        db_type: 'odb' or 'mdb'
    Returns:
        set_string: string used to obtain set data from database object (odb or mdb)

    NTNU / Knut Andreas Kvaale, 2018
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

#%% EXPORT THINGS
def save_nodes_and_elements(folder, element_labels, element_node_indices, node_labels, node_coordinates, element_orientations=None, set_strings=None):
    for ix, element_labels_i in enumerate(element_labels):
        element_info = np.column_stack([element_labels[ix], element_node_indices[ix]])
        node_info = np.column_stack([node_labels[ix], node_coordinates[ix]])
        np.savetxt(os.path.join(folder, 'node_info_%i.dat' % (ix)), node_info)
        np.savetxt(os.path.join(folder, 'element_info_%i.dat' % (ix)), element_info)

        if element_orientations:
            np.savetxt(os.path.join(folder, 'element_orientations_%i.dat' % (ix)), element_orientations)

        if set_strings:
            np.savetxt(os.path.join(folder, 'node_and_element_sets.txt'), set_strings, fmt='%s', delimiter=',')


def save_pontoon_info(folder, node_labels, node_coordinates, pontoon_labels=None, pontoon_angles=None):
    if pontoon_labels==None:    # standard if no pontoon_labels are provided (integers!)
        pontoon_labels = np.linspace(1, len(node_labels), len(node_labels)).astype(int)

    if pontoon_angles==None:
        pontoon_angles = np.zeros(len(node_labels))    #if no angles are given, output zero for all pontoon angles

    pontooninfo = np.column_stack([pontoon_labels, node_coordinates,  node_labels, pontoon_angles])
    np.savetxt(os.path.join(folder, 'pontoon_info.dat'), pontooninfo)


def save_all_modal(folder, phi, suffix='', f=None, m=None, set_strings=None):

    if isinstance(phi, list):
        for ix, phi_i in enumerate(phi):
            np.savetxt(os.path.join(folder, 'phi_%s_%i.dat' % (suffix, ix)), phi_i)
        
        if set_strings:
            np.savetxt(os.path.join(folder, 'phi_%s_sets.txt' % (suffix)), set_strings, fmt='%s', delimiter=',')

    elif isinstance(phi, np.ndarray):
        np.savetxt(os.path.join(folder, 'phi_%s_%i.dat' % (suffix, 0)), phi)

    if f is not None:
        np.savetxt(os.path.join(folder, 'f.dat'), f)
    if m is not None:
        np.savetxt(os.path.join(folder, 'm.dat'), m)


#%% ONLY DEBUGGED IN BRIGADE
def mode2df(model, node_labels, phi, name, instance_name):
    nodes = tuple(np.repeat(node_labels,6).tolist())
    dofs = np.tile(np.arange(1,6+1), len(node_labels))

    dofs_and_mags = np.empty([np.shape(dofs)[0],2])
    dofs_and_mags[:, 0::2] = dofs[:, np.newaxis]
    dofs_and_mags[:, 1::2] = phi[:, np.newaxis]

    data = ((instance_name, 2, nodes, tuple(dofs_and_mags.flatten().tolist())),)
    df = model.DiscreteField(data=data, dataWidth=2, defaultValues=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), description='Mode displacement', fieldType=PRESCRIBEDCONDITION_DOF, location=NODES, name=name)

    return df


def apply_nodal_load(model, node_labels, step_name, loads, instance_name, prefix=''):
    instance = model.rootAssembly.instances[instance_name]
    all_node_labels = [node.label for node in instance.nodes]
    ndof = 6            # assumes 6 DOFs for all nodes - be aware!
    for node_ix, node_label in enumerate(node_labels):
        if all_node_labels.count(node_label) != None:   # if in node labels
            global_node_ix = all_node_labels.index(node_label)
            node_set = model.rootAssembly.Set(name='node_%i' % (node_label), nodes=instance.nodes[global_node_ix:global_node_ix+1])
            nodeloads = loads[node_ix*6:node_ix*6+6]

            if not np.all(nodeloads[0:3]==0):
                model.ConcentratedForce(cf1=nodeloads[0], cf2=nodeloads[1], cf3=nodeloads[2], createStepName=step_name, distributionType=UNIFORM, field='', localCsys=None, name='%sforces_node_%i' % (prefix, node_label), region=node_set)
            
            if not np.all(nodeloads[3:6]==0):
                model.Moment(cm1=nodeloads[3], cm2=nodeloads[4], cm3=nodeloads[5], createStepName=step_name, distributionType=UNIFORM, field='', localCsys=None, name='%smoments_node_%i' % (prefix, node_label), region=node_set)

        else:
            raise ValueError('Node %i does not exist in selected instance.' % (node_label))


def assign_modal_constraint_equation(model, instance_name, name, node_labels, displacement):
    ndof = 6            # assumes 6 DOFs for all nodes - be aware!
    instance = model.rootAssembly.instances[instance_name]
    all_node_labels = [node.label for node in instance.nodes]
    terms = []
    for node_ix, node_label in enumerate(node_labels):
        if all_node_labels.count(node_label) != None:   # if in node labels
            global_node_ix = all_node_labels.index(node_label)
            node_set_name = 'node_%i' % (node_label)
            node_set = model.rootAssembly.Set(name=node_set_name, nodes=instance.nodes[global_node_ix:global_node_ix+1])
            displacement_of_node = displacement[node_ix*ndof:node_ix*ndof+ndof]
            non_zero = np.where(displacement_of_node !=0 )[0]
            terms.append([(displacement_of_node[ldof], node_set_name, ldof+1) for ldof in non_zero])

    terms = tuple([term for sublist in terms for term in sublist])
    model.Equation(name=name, terms=terms)

