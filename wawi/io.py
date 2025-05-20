# -*- coding: utf-8 -*-
import numpy as np
import re
import json
import csv
import dill
from pathlib import Path

from wawi.model import ModalDry, PontoonType, Node, Model, Aero, AeroSection

def import_folder(model_folder, pontoon_stl='pontoon.stl', aero_sections='aero_sections.json',
                modal='modal.json', pontoon='pontoon.json', eldef='element.json', orientations='orientations.json',
                drag_elements='drag_elements.json', pontoon_types='pontoon_types.json', 
                sort_nodes_by_x=True, interpolation_kind='linear'):
    
    '''
    Import folder containing files defining a WAWI model.

    Parameters
    ---------------
    model_folder : str
        string defining path of root
    pontoon_stl : 'pontoon.stl', optional
        string to relative path of stl-file with pontoon (3d file)
    aero_sections : 'aero_sections.json', optional
        string to relative path of json file with aerodynamic sections
    modal : 'modal.json', optional
        string to relative path of dry modal definitions
    pontoon : 'pontoon.json', optional
        string to relative path of pontoon definitions
    eldef : 'element.json', optional
        string to relative path of element definitions (node and element matrices)
    orientations : 'orientations.json', optional
        string to relative path of orientations of aero elements
    drag_elements : 'drag_elements.json', optional
        string to relative path of file defining drag elements
    pontoon_types : 'pontoon_types.json', optional
        string to relative path of pontoon types
    sort_nodes_by_x : True, optional
        whether or not to sort the nodes of eldef by their x-coordinate
    interpolation_kind : {'quadratic', 'linear', ...}
        interpolation kind used for hydrodynamic transfer function and hydrodynamic system matrices

    Returns
    ---------------
    model : `wawi.model.Model`
        WAWI model object

    Notes
    ---------------
    Often, it is more convenient to define all (or many) of the model parameters in input files rather than in scripts. To simplify this process, a support function `wawi.io.import_folder` is provided in WAWI. An example of how to use the function is given in the example [Folder import of models.ipynb](./examples/0 Model generation and setup/JSON definition and folder import/Folder import of models.ipynb)

    This imports all files in the specified folder. The file names of the relevant files in the provided path are defined as inputs to the function (default values in parentheses): 

    - modal ('modal.json')
    - pontoon('pontoon.json')
    - pontoon_types ('pontoon_types.json')
    - pontoon_stl ('pontoon.stl')
    - eldef ('element.json')
    - aero_sections ('aero_sections.json')
    - orientations ('orientations.json')
    - drag_elements ('drag_elements.json')

    Modal definition (modal)
    ^^^^^^^^^^^^^^^^^^^^^^^^
    The first file, 'modal.json', defines the dry modes of the structure under consideration. It includes the following fields:

    ```json
    {
      "local": false,
      "phi": {
        "full": [[...]],
        "hydro": [[...]],
        "girder": [[...]],
        "girder_forces": [[...]]
      },
      "m": [...],
      "omega_n": [...]
    }
    ```

    The `local` parameter defines if the modal transformation matrices (mode shapes) are given in local or global coordinate system. Typically, a global system is more convenient, but both types should work fine in WAWI. The `phi` parameter gives a dictionary defining collections of mode shapes (or modal transformation matrices). The values should be list of lists (inner list is a row) describing the modal transformation matrix of a given key. The matrices have dimensions dofs-by-modes. The keys 'full' and 'hydro' are defaults in WAWI for the full model (used for plotting and convenient for aerodynamic analysis) and pontoon loading, respectively (they can be modified, but sticking with the defaults is a good choice). Furthermore, at least two of the following modal parameters must be given: `m` (modal mass) and `omega_n` (undamped natural frequencies in rad/s) or `m` and `k` (modal stiffness). All of these are 1d lists with length matching the number of columns in all entries in `phi`.

    Pontoon and pontoon types (pontoon, pontoon_types, pontoon_stl)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    All pontoon-related definitions are given by the files `pontoon.json`, `pontoon_types.json` and `pontoon.stl`. 

    Firstly, `pontoon.json` defines pontoon coordinates, orientations and pontoon type definition, as follows:

    ```json
    {
      "P1": {
        "coordinates": [0.0, 0.0, 0.0],
        "pontoon_type": "ptype_1",
        "rotation": 0,
        "node": 1
      },
      "P2": {
        "coordinates": [100.0, 0.0, 0.0],
        "pontoon_type": "ptype_1",
        "rotation": 12.5,
        "node": 2
      }
    }
    ```

    This defines two pontoons (named P1 and P2), places them at coordinates (0,0,0) and (100.0, 0, 0) with orientation (0 deg and 12.5 deg), assigns pontoon type from files "ptype_1.npz". Also, the `node` parameter connects the pontoons to nodes 1 and 2, which should be present in the node matrix of the full system (see `node_matrix` in `element.json`). When a pontoon is connected to a node, the pontoon is automatically plotted together with the elements when using the plotting methods of the model (both rotation and location of the node is used). Also, the coordinates of the pontoon is overwritten by the coordinates of the node. Only the x and y coordinates of the pontoon matters for the computation, as the resulting wave excitation from irregular wave fields is dependent on the horizontal coordinates. The pontoon type definition is an npz-file containing the following numpy nd arrays:

    * Added mass: 'M' (6 x 6 x len(omega))
    * Radiation/added damping: 'C' (6 x 6 x len(omega))
    * Linear hydrodynamic transfer function matrix: 'Q' (6 x len(theta_Q) x len(omega_Q))
    * omega (for mass and damping): 'omega' 
    * theta for transfer: 'theta_Q'
    * omega for transfer: 'omega_Q'

    Second, the file `pontoon_types.json` is usually not needed. It is merely added to enable adding additional properties to the pontoon types, not present in the npz-files. Currently, this is limited to drag coefficients and drag areas for inclusion of linearized drag effects (stochastic linearization). It could look something like this (drag terms ordered as relevant to x, y and z-directions, respectively):

    ```json
    {
        "ptype_1": {
            "Cd": [1.0, 1.0, 2.0],
            "area": [1.0, 2.0, 2.0]
        },

        "ptype_2": {
            "Cd": [0.5, 0.4, 0.8],
            "area": [1.0, 2.0, 2.0]
        }
    }
    ```

    If instead a wildcard name is used, these properties are used for all pontoons:

    ```json
    {
        "*": {
            "Cd": [1.0, 1.0, 2.0],
            "area": [1.0, 2.0, 2.0]
        }
    }
    ```

    Finally, the `pontoon.stl` file is simply a step file containing the mesh used to visualize pontoons together with the elements in the plotting methods.

    Element definition (eldef)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
    The file `element.json` defines the elements of the model (for aerodynamic analysis and plotting). It must contain the keys 'element_matrix' and 'node_matrix', and can also contain the fields 'sections' and 'section_assignments'. The two first are list of lists where the innermost lists are rows. These follow the typical convention in input files; rows of the node matrix consist of node label, x-coordinate, y-coordinate and z-coordinate, whereas the rows of the element matrix consist of the element label, label of the first node and label of the second node.

    The two latter properties of the files are not used inside WAWI directly (unless using drag elements, see below), but are used for the definition of the elements in the BEEF environment. It can therefore be useful for if the user for instance wants to calculate the stresses in an element, as all relevant properties are present inside the `eldef` attribute of the `model` object. The `sections` property is assumed self-explanatory from the json example provided. The `section_assignment` property is built up with a list of strings where each entry corresponds to the same-index row of the element matrix.

    ```json
    {
        "element_matrix": [[...]],
        "node_matrix":[[...]],

        "sections": 
        {
            "girder":
                {"E": 210000000000.0, 
                "Iz": 0.09285714285714286, 
                "Iy": 0.09285714285714286, 
                "A": 0.0052928571428571426, 
                "poisson": 0.3, 
                "J": 0.10714285714285714, 
                "m": 45.6272019496974},

            "column":
                {"E": 210000000000.0, 
                "Iz": 0.09285714285714286, 
                "Iy": 0.09285714285714286, 
                "A": 0.0052928571428571426, 
                "poisson": 0.3, 
                "J": 0.10714285714285714, 
                "m": 45.6272019496974}
        },

        "section_assignments": ["girder", "girder", "girder", "column", "column"]
    }
    ```

    Aerodynamic sections (aero_sections, orientations)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Aerodynamic sections relate aerodynamic properties to the elements defined in the section above. Two definitions are required, and are given in the files
    `aero_sections.json` and `orientations.json`.

    First, the aerodynamic sections are defined in the `aero_sections.json`-file, as follows:

    ```json
    {
        "girder":
            {
                "B":26.8,
                "D":3.715,

                "Cd":0.679,
                "dCd":-1.022,
                "Cl":-0.3896,
                "dCl":4.0116,
                "Cm":-0.0700,
                "dCm":1.0122,

                "elements": [1000, 1001, 1002, 1003, ...]
            },

        "column":
        {
            "B":3,
            "D":3,

            "Cd":0.679,
            "dCd":-1.022,
            "Cl":-0.3896,
            "dCl":4.0116,
            "Cm":-0.0700,
            "dCm":1.0122,

            "elements": [2001, 2002, 2003, 2004, 2005, ...]
        }
    }
    ```

    Here, two aerodynamic sections are defined. This includes definitions of width (B), height (D), static wind coefficients (Cd, dCd, Cl, dCl, Cm, and dCm - see `wawi.model.aero.AeroSection` for details on this) and a list of elements to apply the sections to.

    For aerodynamic analysis, the orientations of the elements are crucial. These are defined in the second file (`orientations.json`). The following structure is used for this:

    ```json
    {
        "girder": {
            "e3": [0,0,1],
            "elements": [1000, 1001, 1002, 1003, ...]
        },
        "column": {
            "e2": [0,1,0],
            "elements": [2001, 2002, 2003, 2004, 2005, ...]
        }
    }
    ```

    In the above snippet, two definitions of orientations are given. Note that the keys here are not used for anything (and are not related to the `aero_sections.json` keys); they are merely included to make it convenient for the user to maintain a structured model setup. The orientation is given by either `e2` or `e3`, indicating if the y/drag or z/lift of the element is defined. Note also that these definitions are assumed approximate; the procedure inside WAWI consists of taking a cross product between the supplied unit vector and the axial direction of the element, which gives a proper perpendicular vector. The last perpendicular vector is thereafter based on a secondary cross product between the calculated perpendicular unit vector and the longitudinal vector. Finally, a list of elements to assign the orientation to is provided. The resulting orientation is conveniently plotted in the model using `model.plot(tmat_on=['undeformed'], tmat_scaling=100)` (adjust `scaling` to make sense for your problem).

    Drag elements (drag_elements)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    To accommodate the possibility for floating bridges to have mooring lines with significant quadratic drag damping, an additional file for defining these properties is supported. Note that this, as for the drag of the pontoons, is only solved in a linearized sense using (approximate) stochastic linearization techniques. The `drag_elements.json` file is defined as follows:

    ```json
    {
        "mooring_wire":
            {   "sections": "MooringWire",
                "zrange": [-1e4, 0.0],
                "rho": 1025.0,
                "Cd": [0.1, 1.2, 1.2],
                "D": 0.146}
    }
    ```

    For the drag element group named "mooring_wire" above, all elements present in the elements assigned with the relevant section assignment(s) in `element.json` (either list of several strings or single string defined with property `sections`) and that have a z-coordinate inside the provided `zrange` will be given quadratic drag damping according to the specified density (`rho`), drag coefficients in local x, y and z-direction (`Cd`) and diameter (`D`).
    '''


    from beef.fe import Part, Section
    from beef.general import n2d_ix
    
    model_folder = Path(model_folder)
    
    # Load pontoon data
    pontoon_data = {}
    if pontoon is not None:
        try:
            with open(model_folder / pontoon, 'r') as f:
                pontoon_data = json.load(f)
        except:
            print('Valid pontoon data not found. No pontoons created.')
           

    # Load modal data
    with open(model_folder / modal, 'r') as f:
        modal = json.load(f)
    
    # Load element definitions
    if eldef is not None:
        try:
            with open(model_folder / eldef, 'r') as f:
                element_data = json.load(f)
        except:
            print('Valid element data not found. No eldef created.')
            element_data = {}
    else:
        element_data = {}
            
    if orientations is not None:
        try:
            with open(model_folder / orientations, 'r') as f:
                orientations = json.load(f)
        except:
            print('Specified orientations file found. No orientation definitions applied.')
            orientations = {}
    else:
        orientations = {}


    common_ptype_settings = None
    if pontoon_types is not None:
        try:
            with open(model_folder / pontoon_types, 'r') as f:
                pontoon_type_settings = json.load(f)
            
            if '*' in pontoon_type_settings:
                common_ptype_settings = pontoon_type_settings.pop('*')
            
        except:
            print('Valid pontoon type settings file not found. No definitions applied.')
            pontoon_type_settings = {}

    else:
        pontoon_type_settings = {}
    
    # Extract specific pontoon data
    pontoon_names = [key for key in pontoon_data]
    pontoon_nodes = [Node(pontoon_data[key]['node'], *pontoon_data[key]['coordinates']) for key in pontoon_data]
    pontoon_rotation = [pontoon_data[key]['rotation']*np.pi/180 for key in pontoon_data]
    pontoon_types = [pontoon_data[key]['pontoon_type'] for key in pontoon_data]
   
    # Pontoon types
    unique_pontoon_types = list(set(pontoon_types))
    files = [f'{model_folder}/{pt}.npz' for pt in unique_pontoon_types]
    ptypes = dict()    
    
    if len(pontoon_type_settings)==0:    
        pontoon_type_settings = {key: {} for key in unique_pontoon_types}
    
    if common_ptype_settings: #universal setting, overwriting others
        for name in pontoon_type_settings:
            for par in common_ptype_settings:
                pontoon_type_settings[name][par] = common_ptype_settings[par]
    
    for file in files:
        name = Path(file).stem
        P = np.load(file)
        
        if 'Cd' in P:
            Cd = P['Cd']
        else:
            Cd = np.array([0,0,0,0,0,0])
        
        if 'A' in P:
            A = P['A']
        else:
            A = np.array([0,0,0,0,0,0])
            
        ACd = {'area': A, 'Cd':Cd}
        ptypes[name] = PontoonType.from_numeric(interpolation_kind=interpolation_kind, A=P['M'], B=P['C'], 
                         omega=P['omega'], Q=P['Q'], theta=P['theta_Q'], omegaQ=P['omega_Q'], label=name, 
                         stl_path=f'{model_folder}/{pontoon_stl}', **{**ACd, **pontoon_type_settings[name]})    

    
    # Modal dry object 
    if 'xi0' in modal:
        xi0 = np.array(modal.pop('xi0'))
    else:
        xi0 = 0.0
        
    if 'm_min' in modal:
        m_min = modal.pop('m_min')
    else:
        m_min = 0.0
        
    if 'phi_x' in modal:
        phi_x = modal.pop('phi_x')
    else:
        phi_x = None
    
    if 'local' in modal:
        local_phi = modal.pop('local')
    else:
        local_phi = False    
    
    phi = modal.pop('phi')

    modal_dry = ModalDry(phi, xi0=xi0, local_phi=local_phi, phi_x=phi_x, m_min=m_min, **modal)
    
    # Element definition
    if element_data != {}:
        if 'sections' in element_data:
            sd = element_data['sections']
            section_dict = {key: Section(E=sd[key]['E'], poisson=sd[key]['poisson'], 
                                          A=sd[key]['A'], I_y=sd[key]['Iy'], I_z=sd[key]['Iz'], 
                                          J=sd[key]['J'], m=sd[key]['m'], name=key) for key in sd}
    
            sections = [section_dict[key] for key in element_data['section_assignment']]
        else:
            sections = None
            
        node_matrix = np.array(element_data['node_matrix'])
        node_matrix[:,0] = node_matrix[:,0].astype(int)
        element_matrix = np.array(element_data['element_matrix'])
        element_matrix = element_matrix.astype(int)

        # Remove elements without valid nodes
        remove_ix = []
        remove_els = []

        for ix,row in enumerate(element_matrix):
            el,node1,node2 = row
            # print(row)

            if (node1 not in node_matrix[:,0].astype(int)) or (node2 not in node_matrix[:,0].astype(int)):
               remove_ix.append(ix) 
               remove_els.append(el)

        if len(remove_els)>0:
            print(f'Elements {remove_els} do not have valid nodes - not included in model.')
            
        element_matrix = np.delete(element_matrix, remove_ix, axis=0)            

        eldef = Part(node_matrix, element_matrix, sections=sections,
                     assemble=False, forced_ndofs=6)
            
        if sort_nodes_by_x:
            ix = np.argsort([n.coordinates[0] for n in eldef.nodes])
            node_labels_sorted = eldef.get_node_labels()[ix]
            eldef.arrange_nodes(node_labels_sorted, arrange_dof_ixs=True)
              
            if 'full' in modal_dry.phi_full:  #adjust phi_full
                modal_dry.phi_full['full'] = modal_dry.phi_full['full'][n2d_ix(ix, n_dofs=6), :]

        # Assign orientation
        for key in orientations:
            if 'e2' in orientations[key]:
                e2 = orientations[key]['e2']
                e3 = None
            elif 'e3' in orientations[key]:
                e3 = orientations[key]['e3']
                e2 = None               
            else:
                raise ValueError('Orientations should contain either e2 or e3')

            elements = orientations[key]['elements']
            for el_label in elements:
                el = eldef.get_element(int(el_label))

                el.assign_e2(e2)
                el.assign_e3(e3)

                el.initiate_geometry()
    else:
        eldef = None

    # Create model object (only hydro part)
    model = Model.from_nodes_and_types(pontoon_nodes, [ptypes[pt] for pt in pontoon_types], modal_dry=modal_dry, 
                                       rotation=pontoon_rotation, eldef=eldef, labels=pontoon_names)
    
    # Aero mode
    if aero_sections is not None:
        # Load aero sections
        try:
            with open(model_folder / aero_sections, 'r') as f:
                data = json.load(f)
        
            sections = dict()
            element_assignments = dict()
            for key in data:
                element_assignments[key] = data[key].pop('elements')
                sections[key] = AeroSection(**data[key])
            
            model.aero = Aero(sections=sections, element_assignments=element_assignments)
        except:
            print('Specified aero_sections file found. No aerodynamics definitions applied.')

    # Drag elements model
    if drag_elements is not None:
        try:
            with open(model_folder / drag_elements, 'r') as f:
                data = json.load(f)
            
            model.assign_drag_elements(data)
        except:
            print('Specified drag_elements file found or invalid. No drag elements defined.')
    
    model.connect_eldef()
    model.assign_dry_modes()
    
    return model


def save_wwi(model, model_path):
    '''
    Save WAWI model object to specified model path.
    '''
    with open(model_path, 'wb') as f:
        dill.dump(model, f, -1)
    

def convert_to_wwi(model_path, output=None):
    '''
    Convert folder with WAWI object definition files to wwi (model) file
    '''
    model_path = Path(model_path)
    if output is None:
        output = model_path / 'model.wwi'
    
    model = load_model(model_path)

    if model_path.is_dir():
        with open(output, 'wb') as f:
            dill.dump(model, f, -1)


def load_model(model_path, save_if_nonexistent=False, sort_nodes_by_x=True):
    '''
    Load wwi-file with WAWI model object or folder with relevant definitions.

    Parameters
    -----------
    model_path : str
        path of wwi-file
    save_if_nonexistent : False, optional
        whether or not to save model as wwi if not existing
    sort_nodes_by_x : True, optional
        whether or not to sort the nodes of eldef by their x-coordinate

    Returns
    -----------
    model : `wawi.model.Model` 
      WAWI model
    '''

    model_path = Path(model_path)

    if model_path.is_dir() and not (model_path/'model.wwi').is_file():
        model = import_folder(model_path, sort_nodes_by_x=sort_nodes_by_x)
        if save_if_nonexistent:
            convert_to_wwi(model_path)   
    else:
        with open(model_path, 'rb') as f:
            model = dill.load(f)
    
    return model
            

def import_wadam_hydro_transfer(wadam_file):
    '''
    Import WADAM LIS output file.

    Parameters
    -----------
    wadam_file : str
        path to file to import
    
    Returns
    ----------
    omega : float
        numpy array describing numerical frequency axis
    theta : float
        numpy array describing numerical directional axis
    fhyd : float
        complex numpy 3d-array describing the transfer functions relating
        regular waves with unit height to the 6 relevant forces and moments
        6-by-len(theta)-by-len(omega)
    '''

    string = (r'.+W A V E  P E R I O D.+=\s+(?P<period>.+):.+\n.+'
              r'H E A D I N G   A N G L E.+=\s+(?P<theta>.+):(?:.*\n){5,10}'
              
              r'.+EXCITING FORCES AND MOMENTS FROM THE HASKIN RELATIONS(?:.*\n){4,6}\s+'
              
              r'-F1-\s+(?P<F1_real>[-?.E\d+]+)\s+(?P<F1_imag>[-?.E\d+]+)\s.+\n\s+'
              r'-F2-\s+(?P<F2_real>[-?.E\d+]+)\s+(?P<F2_imag>[-?.E\d+]+)\s.+\n\s+'
              r'-F3-\s+(?P<F3_real>[-?.E\d+]+)\s+(?P<F3_imag>[-?.E\d+]+)\s.+\n\s+'
              r'-F4-\s+(?P<F4_real>[-?.E\d+]+)\s+(?P<F4_imag>[-?.E\d+]+)\s.+\n\s+'
              r'-F5-\s+(?P<F5_real>[-?.E\d+]+)\s+(?P<F5_imag>[-?.E\d+]+)\s.+\n\s+'
              r'-F6-\s+(?P<F6_real>[-?.E\d+]+)\s+(?P<F6_imag>[-?.E\d+]+)\s.+\n')
    regex = re.compile(string) 
    
    nondim_string = (r'\s+NON-DIMENSIONALIZING FACTORS:\n(?:.*\n)+'
                     r'\s+RO\s+=\s+(?P<rho>[-?.E\d+]+)\n'
                     r'\s+G\s+=\s+(?P<g>[-?.E\d+]+)\n'
                     r'\s+VOL\s+=\s+(?P<vol>[-?.E\d+]+)\n'
                     r'\s+L\s+=\s+(?P<l>[-?.E\d+]+)\n'
                     r'\s+WA\s+=\s+(?P<wa>[-?.E\d+]+)')
    
    regex_nondim = re.compile(nondim_string) 
    
    with open(wadam_file, encoding='utf-8') as file:
        data = file.read()
    
    periods = []
    thetas = []
    Q = []
    
    for match in regex.finditer(data):
        data_dict = dict(zip(match.groupdict().keys(), [float(val) for val in match.groupdict().values()]))
        F1 = data_dict['F1_real'] + data_dict['F1_imag']*1j
        F2 = data_dict['F2_real'] + data_dict['F2_imag']*1j
        F3 = data_dict['F3_real'] + data_dict['F3_imag']*1j
        F4 = data_dict['F4_real'] + data_dict['F4_imag']*1j
        F5 = data_dict['F5_real'] + data_dict['F5_imag']*1j
        F6 = data_dict['F6_real'] + data_dict['F6_imag']*1j
        Q.append(np.array([F1,F2,F3,F4,F5,F6]))
        
        thetas.append(data_dict['theta'])
        periods.append(data_dict['period'])
    
    theta = np.unique(np.array(thetas))*np.pi/180.0
    period = np.unique(np.array(periods))
    omega = np.flip(2*np.pi/period, axis=0)
    Q = np.vstack(Q).T
    fhyd = np.zeros([6, len(theta), len(omega)]).astype('complex')
    for n in range(len(theta)):
        for k in range(len(omega)):
            fhyd[:,n,k] = Q[:, k*len(theta)+n] 
    
    # Re-dimensionalize and flip
    nondim_parameters = regex_nondim.search(data).groupdict()
    nd = dict(zip(nondim_parameters.keys(), [float(val) for val in nondim_parameters.values()]))
    
    rho, g, vol, wa, l = nd['rho'], nd['g'], nd['vol'], nd['wa'], nd['l']
    dim = 0*fhyd
    dim[0::6,:,:] = rho*vol*g*wa/l
    dim[1::6,:,:] = rho*vol*g*wa/l
    dim[2::6,:,:] = rho*vol*g*wa/l
    
    dim[3::6,:,:] = rho*vol*g*wa
    dim[4::6,:,:] = rho*vol*g*wa
    dim[5::6,:,:] = rho*vol*g*wa

    fhyd = np.flip(fhyd, axis=2) * dim

    return omega, theta, fhyd


def import_wadam_mat(wadam_file):
    '''
    Import system matrices given by input WADAM results (frequency dependent and constant).

    Parameters
    -------------
    wadam_file : str
        path to WADAM file to open
    
    Returns
    -------------
    mass : float
        6-by-6 added mass matrix of body
    damping : float
        6-by-6 radiation damping matrix of body
    stiffness : float
        6-by-6 restoring stiffness matrix of body
    static_mass : float
        6-by-6 static (constant) mass matrix of body
        this represents inertia of pontoon object itself
    omega : float
        numpy array describing numerical frequency axis (corresponding to mass and damping
        output)
    '''

    static_mass_target = 'MASS INERTIA COEFFICIENT MATRIX'
    stiffness_target = 'HYDROSTATIC RESTORING COEFFICIENT MATRIX'
    mass_target = 'ADDED MASS MATRIX                                   '
    damping_target = 'DAMPING MATRIX                                      '
    period_target = 'WAVE PERIOD  =   '
    non_dim_target = ' THE OUTPUT IS NON-DIMENSIONALIZED USING -'

    f = open(wadam_file)
    active_search = False
    current_elements = []
    current_range = [0]

    stiffness = np.empty([0,6,6])
    static_mass = np.empty([0,6,6])
    mass = np.empty([0,6,6])
    damping = np.empty([0,6,6])
    period = np.empty([0,1,1])
    non_dim = np.empty([0])

    temp_data=[]
    data_storage = {'static_mass': static_mass, 'stiffness': stiffness, 'period': period, 'mass':mass,'damping':damping,'non_dim':non_dim}
    append_axis = {'static_mass': 0, 'stiffness': 0, 'period': 0, 'mass':0,'damping':0,'non_dim':None}

    for lineno, line in enumerate(f):
        static_mass_switch = line.find(static_mass_target)!=-1
        stiffness_switch = line.find(stiffness_target)!=-1
        period_switch = line.find(period_target)!=-1
        mass_switch = line.find(mass_target)!=-1
        damping_switch = line.find(damping_target)!=-1
        non_dim_switch = line.find(non_dim_target)!=-1

        if static_mass_switch == True:
            active_search=True
            current_range = [el+lineno-1 for el in range(5,11)]
            current_elements = range(1,6+1)
            current_data_string='static_mass'
            current_data = data_storage[current_data_string]
        elif stiffness_switch:
            active_search=True
            current_range = [el+lineno-1 for el in range(5,11)]
            current_elements = range(1,6+1)
            current_data_string='stiffness'
            current_data = data_storage[current_data_string]
        elif period_switch:
             active_search=True
             current_range = [lineno]
             current_elements = [3]
             current_data_string='period'
             current_data = data_storage[current_data_string]
        elif mass_switch:
            active_search=True
            current_range = [el+lineno-1 for el in range(5,11)]
            current_elements = range(1,6+1)
            current_data_string='mass'
            current_data = data_storage[current_data_string]
        elif damping_switch:
            active_search=True
            current_range = [el+lineno-1 for el in range(5,11)]
            current_elements = range(1,6+1)
            current_data_string='damping'
            current_data = data_storage[current_data_string]
        elif non_dim_switch:
            active_search=True
            current_range = [el+lineno-1 for el in range(9,14)]
            current_elements = [2]
            current_data_string='non_dim'
            current_data = data_storage[current_data_string]

        if (active_search==True) and (lineno in current_range):
            data_line = [float(i) for i in [line.split()[el] for el in current_elements]]
            temp_data.append(data_line)
        elif (active_search==True) and lineno>=max(current_range):
             current_data=np.append(current_data,[temp_data],axis=append_axis[current_data_string])
             temp_data=[]
             current_range = [0]
             current_elements = []
             active_search=False
             data_storage[current_data_string] = current_data

    # Non-dimensionalizing factors
    non_dim = data_storage['non_dim']
    ro = non_dim[0]
    g = non_dim[1]
    vol = non_dim[2]
    l = non_dim[3]

    non_dim_defs = {'static_mass': np.array([[ro*vol, ro*vol*l],
                                             [ro*vol*l, ro*vol*l*l]]), 
                    'mass': np.array([[ro*vol, ro*vol*l],
                                      [ro*vol*l, ro*vol*l*l]]), 
                    'damping': np.array([[ro*vol*np.sqrt(g/l), ro*vol*np.sqrt(g*l)],
                                         [ro*vol*np.sqrt(g*l), ro*vol*l*np.sqrt(g*l)]]), 
                    'stiffness':np.array([[ro*vol*g/l, ro*vol*g],
                                          [ro*vol*g, ro*vol*g*l]])}

    # Re-dimensionalize data
    for current_data_string in non_dim_defs.keys():
        dim_matrix = np.repeat(np.repeat(non_dim_defs[current_data_string],3,axis=0),3,axis=1)
        data_storage[current_data_string] = np.multiply(data_storage[current_data_string], dim_matrix)

    static_mass = data_storage['static_mass'][0]
    stiffness = data_storage['stiffness'][0]
    mass = np.moveaxis(data_storage['mass'],0,2)
    damping = np.moveaxis(data_storage['damping'],0,2)
    omega = 2.0*np.pi/data_storage['period'].flatten()
    sortix = np.argsort(omega)
    omega = omega[sortix]
    
    mass = mass[:,:, sortix]
    damping = damping[:,:, sortix]
    stiffness = (stiffness + stiffness.T)/2

    return  mass, damping, stiffness, static_mass, omega



def import_wamit_mat(wamit_path, suffix=['hst', 'mmx', '1'], rho=1020, L=1.0, g=9.80665):
    '''
    Import system matrices from WAMIT results (frequency dependent and constant).

    Parameters
    -------------
    wamit_path : str
        path to WAMIT file to open
    suffix : ['hst', 'mmx', '1']
        all suffixes used - all default values required for full output
    rho : 1020, optional
        water mass density used to redimensionalize results
    L : 1.0, optional
        redimensionaling length - 1.0 is fine for most use cases
    g : 9.80665, optional
        gravitational constant for redimensionalization of data
    
    Returns
    -------------
    A : float
        6-by-6 added mass matrix of body
    B : float
        6-by-6 radiation damping matrix of body
    K0 : float
        6-by-6 restoring stiffness matrix of body
    M0 : float
        6-by-6 static (constant) mass matrix of body
        this represents inertia of pontoon object itself
    omega : float
        numpy array describing numerical frequency axis (corresponding to mass and damping
        output)
    '''
    if suffix.count('hst')>=1:
        # Hydrostatic stiffness
        k_list = np.loadtxt(wamit_path+'.hst')
        K0 = np.reshape(k_list[:,2], [6,6])   #assumes the order of elements in k_list
        dimK = 0*K0
        dimK[2, 2] = rho*g*L**2
        dimK[2, 3] = dimK[3, 2] = rho*g*L**3
        dimK[2, 4] = dimK[4, 2] = rho*g*L**3
        dimK[3::, :]  = rho*g*L**4
        dimK[:, 3::]  = rho*g*L**4
        
        K0 = K0*dimK
        
    else:
        K0 = 0

    if suffix.count('mmx')>=1:
        # Inertia matrix
        m_list = np.loadtxt(wamit_path+'.mmx', skiprows=12)
        M0 = np.reshape(m_list[:,2], [6,6]) 
    else: 
        M0 = 0

    if suffix.count('1')>=1:
        # Added mass and added damping
        mc_list = np.loadtxt(wamit_path+'.1')
        period = mc_list[0::10,0]
        Aper = np.zeros([6,6,len(period)])
        Bper = np.zeros([6,6,len(period)])

        for k in range(0,int(mc_list.shape[0]/10)):
            mc_this = mc_list[k*10:k*10+10,:]
            dof1, dof2 = mc_this[:,1].astype('int')-1, mc_this[:, 2].astype('int')-1

            for ix, i in enumerate(dof1):
                j = dof2[ix]
                Aper[i, j, k] = mc_this[ix, 3] 
                Bper[i, j, k] = mc_this[ix, 4] 

        # Re-dimensionalize and flip
        omega = np.flip(2*np.pi/period, axis=0)

        dimA = 0*Aper
        dimA[0:3, 0:3, :] = rho*L**3
        dimA[0:3, 3:-1, :] = rho*L**4
        dimA[3:-1, 0:3, :] = rho*L**4
        dimA[3:-1, 3:-1, :] = rho*L**5

        dimB = 0*Bper
        dimB[0:3, 0:3, :] = rho*L**3*omega
        dimB[0:3, 3:-1, :] = rho*L**4*omega
        dimB[3:-1, 0:3, :] = rho*L**4*omega
        dimB[3:-1, 3:-1, :] = rho*L**5*omega
        
        A = np.flip(Aper, axis=2)*dimA
        B = np.flip(Bper, axis=2)*dimB
    else:
        A = 0
        B = 0

    return A, B, K0, M0, omega


def import_wamit_force(wamit_path, A=1.0, rho=1020, L=1.0, g=9.80665):
    '''
    Import hydrodynamic transfer function from WAMIT '.2'-file.

    Parameters
    -----------
    wamit_path :
    A : 1.0, optional
        area used to redimensionalize results (1.0 is fine for most use cases)
    rho : 1020, optional
        water mass density used to redimensionalize results
    L : 1.0, optional
        redimensionaling length  (1.0 is fine for most use cases)
    g : 9.80665, optional
        gravitational constant for redimensionalization of data

    Returns
    ----------
    omega : float
        numpy array describing numerical frequency axis
    theta : float
        numpy array describing numerical directional axis
    fhyd : float
        complex numpy 3d-array describing the transfer functions relating
        regular waves with unit height to the 6 relevant forces and moments
        6-by-len(theta)-by-len(omega)
    '''

    # Transfer function
    q_list = np.loadtxt(wamit_path+'.2')    
    period = np.unique(q_list[:,0])
    n_omega = len(period)
    n_theta = int(len(q_list[:,0])/len(period)/6)
    theta = q_list[0:6*n_theta:6,1]
    f_hyd_per = np.zeros([6, n_theta, n_omega]).astype('complex')

    for k in range(0, n_omega):
        q_this = q_list[k*n_theta*6:k*n_theta*6+n_theta*6]
        for dof in range(0,6):
            f_hyd_per[dof, :, k] = q_this[dof::6, 5] + q_this[dof::6, 6]*1j
            
    omega = np.flip(2*np.pi/period, axis=0)
    
    # Re-dimensionalize and flip
    dim = 0*f_hyd_per
    dim[0::6,:,:] = rho*g*A*L**2
    dim[1::6,:,:] = rho*g*A*L**2
    dim[2::6,:,:] = rho*g*A*L**2
    dim[3::6,:,:] = rho*g*A*L**3
    dim[4::6,:,:] = rho*g*A*L**3
    dim[5::6,:,:] = rho*g*A*L**3

    f_hyd = np.flip(f_hyd_per, axis=2) * dim

    return omega, theta, f_hyd

