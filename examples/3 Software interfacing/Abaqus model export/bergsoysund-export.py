from sys import path
from collections import OrderedDict

path.append('C:/Users/knutankv/git-repos/wawi/')   # this is an easy quick fix to enable importing wawi package in Abaqus environment
savefolder = 'C:/Temp/'

# NOTE: Only linear beam elements are supported in relevant sets!!

import wawi.ext.abq
import json
import numpy as np

#%% Get (O) database object
db = wawi.ext.abq.get_db('odb')

#%% Definitions
frequency_step = 'Step-1'
part = db.rootAssembly.instances['TRUSSPART']
step_obj = db.steps[frequency_step]

if 'ALL' not in part.elementSets:   #CREATE SET OF ALL ELEMENTS IN PART
    part.ElementSet('ALL', part.elements)

#%% Grab regions
region_full = part.elementSets['ALL']
region_hydro = db.rootAssembly.nodeSets['PALL']

#%% Get modal parameters
fn, m = wawi.ext.abq.get_modal_parameters(frequency_step)

#%% Get wind elements and mode shapes
node_matrix, element_matrix = wawi.ext.abq.get_element_matrices(region_full, obj=part)
node_labels = node_matrix[:,0]
nodes = wawi.ext.abq.create_list_of_nodes(part, node_labels)

# Export element definitions as json
el_data = dict(node_matrix=node_matrix.tolist(), element_matrix=element_matrix.tolist())

with open('element.json', 'w') as f:
    json.dump(el_data, f)

phi_full_disp = wawi.ext.abq.get_nodal_phi(step_obj, nodes, flatten_components=True, field_outputs=['UT', 'UR'])

#%% Get pontoon data
pontoon_sets = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
pontoon_types = ['p17', 'p26', 'p345', 'p345', 'p345', 'p26', 'p17']
rotations = np.array([13.944407004001892, 9.2962713360012632, 4.6481356680006325, 0.0,
                     -4.6481356680006334, -9.2962713360012632, -13.944407004001883])

node_matrix_pontoons = np.vstack([wawi.ext.abq.get_element_matrices(db.rootAssembly.nodeSets[pset], obj=part) for pset in pontoon_sets])
node_labels_pontoons = node_matrix_pontoons[:,0]

# Export pontoon.json
pontoon_data = OrderedDict()

for ix, node in enumerate(node_labels_pontoons):
    key = pontoon_sets[ix]
    pontoon_data[key] = dict(coordinates=node_matrix_pontoons[ix, 1:].tolist(),
                            node=node,
                            rotation=rotations[ix],
                            pontoon_type=pontoon_types[ix])


nodes_pontoons = wawi.ext.abq.create_list_of_nodes(part, node_labels_pontoons)
phi_h = wawi.ext.abq.get_nodal_phi(step_obj, nodes_pontoons, flatten_components=True, field_outputs=['UT', 'UR'])


with open('pontoon.json', 'w') as f:
    json.dump(pontoon_data, f)

## ------------------- EXPORT MODES ----------------
modal_data = dict(omega_n=(fn*2*np.pi).tolist(), m=m.tolist(), phi=dict(full=phi_full_disp.tolist(),
                                     hydro=phi_h.tolist()))

with open('modal.json', 'w') as f:
    json.dump(modal_data, f)