import numpy as np

def terrain_roughness(z, z0=0.01, zmin=1.0):
    z0_ii = 0.01
    kr = 0.19 * (z0/z0_ii)**0.07
    
    if z<=zmin:
        return kr*np.log(zmin/z0)
    else:
        return kr*np.log(z/z0)


    
    
