from sys import path
from collections import OrderedDict

path.append('C:/Users/knutankv/git-repos/wawi/')   # this is an easy quick fix to enable importing wawi package in Abaqus environment
savefolder = 'C:/Temp/'

import wawi.ext.abq
import json
import numpy as np

#%% Get (O) database object
db = wawi.ext.abq.get_db('odb')

#%% Definitions
frequency_step = 'Step-6'
part = db.rootAssembly.instances['BRIDGE-1']
step_obj = db.steps[frequency_step]

if 'ALL' not in part.elementSets:   #CREATE SET OF ALL ELEMENTS IN PART
    part.ElementSet('ALL', part.elements)

#%% Grab regions
region_full = part.elementSets['ALL']
region_hydro = part.nodeSets['SPRING']

#%% Get modal parameters
fn, m = wawi.ext.abq.get_modal_parameters(frequency_step)

#%% Get wind elements and mode shapes
node_matrix, element_matrix = wawi.ext.abq.get_element_matrices(region_full, obj=part)
node_labels = node_matrix[:,0]

# Export element definitions as json
el_data = dict(node_matrix=node_matrix.tolist(), element_matrix=element_matrix.tolist())

with open('element.json', 'w') as f:
    json.dump(el_data, f)

phi_full_disp = wawi.ext.abq.get_nodal_phi(step_obj, node_labels, flatten_components=True)

#%% Get pontoon data
node_matrix_pontoons = wawi.ext.abq.get_element_matrices(region_hydro, obj=part, sort_nodes_fun=None)
node_labels = node_matrix_pontoons[:,0]
phi_h = wawi.ext.abq.get_nodal_phi(step_obj, node_labels, flatten_components=True)

# Export pontoon.json
pontoon_types = ['ptype_1']*48
rotations = np.zeros(48)

pontoon_data = OrderedDict()

for ix, node in enumerate(node_labels):
    key = 'P'+str (ix+1)
    pontoon_data[key] = dict(coordinates=node_matrix_pontoons[ix, 1:].tolist(),
                            node=node,
                            rotation=rotations[ix],
                            pontoon_type=pontoon_types[ix])

with open('pontoon.json', 'w') as f:
    json.dump(pontoon_data, f)

## ------------------- EXPORT MODES ----------------
modal_data = dict(omega_n=(fn*2*np.pi).tolist(), m=m.tolist(), phi=dict(full=phi_full_disp.tolist(),
                                     hydro=phi_h.tolist()))

with open('modal.json', 'w') as f:
    json.dump(modal_data, f)