from sys import path
from collections import OrderedDict

path.append('C:/Users/knutankv/git-repos/wawi/')   # this is an easy quick fix to enable importing wawi package in Abaqus environment
savefolder = 'C:/Temp/'

# NOTE: Only linear *beam* elements are supported in relevant sets!!

import wawi.ext.abq
import json
import numpy as np

#%% Get database object (ODB) 
db = wawi.ext.abq.get_db('odb')

#%% Definitions
frequency_step = 'Step-11'  # define set name to extract modal properties from
part = db.rootAssembly.instances['PART-1-1']    # define part 
step_obj = db.steps[frequency_step]             

part.ElementSet('ALL', part.elements)   # get all elements (for full element and node matrix)

#%% Grab regions
region_full = part.elementSets['ALL']

#%% Get modal parameters
fn, m = wawi.ext.abq.get_modal_parameters(frequency_step)

#%% Get wind elements and mode shapes
node_matrix, element_matrix = wawi.ext.abq.get_element_matrices(region_full, obj=part)
node_labels = node_matrix[:,0]
nodes = wawi.ext.abq.create_list_of_nodes(part, node_labels)

# Export element definitions as json
el_data = dict(node_matrix=node_matrix.tolist(), 
               element_matrix=element_matrix.tolist())

with open('element.json', 'w') as f:
    json.dump(el_data, f)

phi_full_disp = wawi.ext.abq.get_nodal_phi(step_obj, nodes, flatten_components=True, field_outputs=['U', 'UR'])

## ------------------- EXPORT MODES ----------------
modal_data = dict(omega_n=(fn*2*np.pi).tolist(), m=m.tolist(), phi=dict(full=phi_full_disp.tolist()))

with open('modal.json', 'w') as f:
    json.dump(modal_data, f)