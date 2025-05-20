import numpy as np

def terrain_roughness(z, z0=0.01, zmin=1.0):
    """
    Calculates the terrain roughness correction for wind speed at a given height.

    Parameters
    ----------
    z : float
        Height above ground level (in meters) at which to calculate the roughness correction.
    z0 : float, optional
        Surface roughness length (in meters). Default is 0.01.
    zmin : float, optional
        Minimum reference height (in meters) to use if `z` is less than or equal to `zmin`. Default is 1.0.

    Returns
    -------
    float
        The terrain roughness correction factor.

    Notes
    -----
    The function uses a logarithmic wind profile and a roughness correction factor `kr` based on the ratio of the provided roughness length `z0` to a reference value `z0_ii = 0.01`. If the height `z` is less than or equal to `zmin`, the calculation uses `zmin` instead of `z`.
    
    References
    ----------
    - EN 1991-1-4: Eurocode 1: Actions on structures – Part 1-4: General actions – Wind actions.
    """

    z0_ii = 0.01
    kr = 0.19 * (z0/z0_ii)**0.07
    
    if z<=zmin:
        return kr*np.log(zmin/z0)
    else:
        return kr*np.log(z/z0)


    
    
