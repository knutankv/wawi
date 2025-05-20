import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from wawi.wave import dispersion_relation_scalar as get_kappa
from scipy.ndimage import rotate, shift
from matplotlib import transforms
from scipy.interpolate import RectBivariateSpline, interp1d


def plot_ads(ad_dict, v, terms='stiffness', num=None, test_v=dict(), test_ad=dict(), zasso_type=False, ranges=None):
    """
    Plot aerodynamic derivative (AD) curves for multiple terms and test data.

    Parameters
    ----------
    ad_dict : dict
        Dictionary mapping term names (e.g., 'P4', 'H6', etc.) to functions that compute aerodynamic derivative values for a given `v`.
    v : array-like
        Array of reduced velocity (or reduced frequency) at which to evaluate the aerodynamic derivative functions.
    terms : str or list of list of str, optional
        Specifies which terms to plot. If 'stiffness' or 'damping', uses predefined groupings. Otherwise, expects a nested list of term names.
    num : int or None, optional
        Figure number for matplotlib. If None, a new figure is created.
    test_v : dict, optional
        Dictionary mapping term names to arrays of test reduced velocity values for overlaying test data.
    test_ad : dict, optional
        Dictionary mapping term names to arrays of test aerodynamic derivative data corresponding to `test_v`.
    zasso_type : bool, optional
        If True, applies special formatting and scaling for Zasso-type plots.
    ranges : dict or None, optional
        Dictionary mapping term names to (min, max) tuples, specifying valid ranges for plotting. Values outside the range are clipped.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plots.
    ax : numpy.ndarray of matplotlib.axes.Axes
        Array of Axes objects for each subplot.

    Notes
    -----
    - The function arranges subplots in a grid according to the structure of `terms`.
    - Each subplot shows the fitted aerodynamic derivative curve and, if provided, test data points.
    - Axis labels and scaling are adjusted depending on `zasso_type` and the term type.
    """

    if terms == 'stiffness':
        terms = [['P4', 'P6', 'P3'], ['H6', 'H4', 'H3'], ['A6', 'A4', 'A3']]
    elif terms == 'damping':
        terms = [['P1', 'P5', 'P2'], ['H5', 'H1', 'H2'], ['A5', 'A1', 'A2']]

    # Create exponent defs for K_normalized plotting
    K_exp = dict()
    stiffness_terms = ['P4', 'P6', 'P3', 'H6', 'H4', 'H3', 'A6', 'A4', 'A3']
    damping_terms = ['P1', 'P5', 'P2', 'H5', 'H1', 'H2', 'A5', 'A1', 'A2']
    K_exp.update(zip(stiffness_terms, [1]*len(stiffness_terms)))
    K_exp.update(zip(damping_terms, [2]*len(damping_terms)))
    
    K_label_add = dict()
    K_label_add.update(zip(stiffness_terms, ['$K$']*len(stiffness_terms)))
    K_label_add.update(zip(damping_terms, ['$K^2$']*len(damping_terms)))
        
    fig, ax = plt.subplots(nrows=len(terms[0]), ncols=len(terms), num=num, sharex=True)
    
    for row_ix, row in enumerate(terms):
        for col_ix, term in enumerate(row):
            axi = ax[row_ix, col_ix]
            
            if term in ad_dict:
                if ranges is not None and term in ranges:
                    ad_valid = ad_dict[term](v)*1
                    
                    v_min, v_max = ranges[term]
                    
                    ad_valid[v<v_min] = ad_dict[term](v_min)
                    ad_valid[v>v_max] = ad_dict[term](v_max)
                else:
                    ad_valid = ad_dict[term](v)
                
                axi.plot(v, ad_valid, label='Fit')
                
            label = zasso_type*K_label_add[term]+('$' + term[0] + '_' + term[1] + '^*' + '$')
            axi.set_ylabel(label)
            axi.grid('on')
            
            if term in test_v:
                if zasso_type:
                    vK = 1/test_v[term]
                    factor = vK**K_exp[term]
                else:
                    vK = test_v[term]
                    factor = 1.0
                    
                axi.plot(vK, test_ad[term]*factor, '.k', label='Test')
    
    for col_ix in range(len(terms)):
        if zasso_type:
            ax[-1, col_ix].set_xlabel(r'$K$')
        else:
            ax[-1, col_ix].set_xlabel(r'$V/(B\cdot \omega)$')
    
    fig.tight_layout()
    return fig, ax

def save_plot(pl, path, w=None, h=None):
    '''
    Saves pyvista plotter screenshot to file.

    Parameters
    ----------
    pl : pyvista.Plotter
        PyVista plotter object.
    path : str
        Path to save the screenshot.
    w : int, optional
        Width of the screenshot. If None, uses the width of the plotter window.
    h : int, optional
        Height of the screenshot. If None, uses the height of the plotter window.
    

    '''
    ws = pl.window_size
    if w is not None and h is None:
        w = int(np.round(w))
        h = int(np.round(ws[1] * w/ws[0]))
    elif h is not None and w is None:
        h = int(np.round(h))
        w = int(np.round(ws[0] * h/ws[1]))
    elif h is None and w is None:
        w,h = ws
    else:
        w = int(np.round(w))
        h = int(np.round(h))

    pl.screenshot(path, window_size=[w,h], return_img=False)

def plot_dir_and_crests(theta0, Tp, arrow_length=100, origin=np.array([0,0]), 
                        ax=None, n_repeats=2, crest_length=1000,
                        alpha_crests=0.2, arrow_options={}):
    
    """
    Plot wave direction arrow and wave crests on a matplotlib axis.

    Parameters
    ----------
    theta0 : float
        Wave direction in degrees (0 degrees is along the x-axis).
    Tp : float
        Peak wave period in seconds.
    arrow_length : float, optional
        Length of the direction arrow (default is 100).
    origin : np.ndarray, optional
        2D coordinates of the arrow origin (default is np.array([0, 0])).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, uses current axis (default is None).
    n_repeats : int, optional
        Number of wave crests to plot (default is 2).
    crest_length : float, optional
        Length of each wave crest line (default is 1000).
    alpha_crests : float, optional
        Transparency (alpha) for the crest lines (default is 0.2).
    arrow_options : dict, optional
        Additional keyword arguments for the arrow (default is {}).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plotted wave direction and crests.

    Notes
    -----
    - Requires `matplotlib.pyplot` as `plt` and `numpy` as `np` to be imported.
    - The function also requires a `get_kappa` function to compute the wavenumber.

    Docstring generated by GitHub Copilot.
    """
    arr_opts = {'head_width': 4, 'width':2, 'edgecolor':'none'}
    arr_opts.update(**arrow_options)

    if ax is None:
        ax = plt.gca()
        
    # Plot wave angle and crests
    v = np.array([np.cos(theta0*np.pi/180), np.sin(theta0*np.pi/180)])
    v_norm = np.array([-np.sin(theta0*np.pi/180), np.cos(theta0*np.pi/180)])
    wave_length = 2*np.pi/get_kappa(2*np.pi/Tp, U=0.0)

    plt.arrow(origin[0],origin[1], arrow_length*v[0], arrow_length*v[1], **arr_opts)
    plt.text(origin[0], origin[1], f'$\\theta_0$ = {theta0}$^o$\n $T_p$={Tp} s\n $\\lambda=${wave_length:.0f} m')

    dv = v*wave_length
    for n in range(n_repeats):
        p1 = origin-v_norm*crest_length/2
        p2 = origin+v_norm*crest_length/2
        pts = np.vstack([p1,p2])
        
        ax.plot(pts[:,0], pts[:,1], alpha=alpha_crests, color='black', zorder=0)
        origin = origin + dv

    return ax

def rotate_image_about_pivot(Z, x, y, angle, x0=0, y0=0):
    """
    Rotate an image array about a specified pivot point.
    This function rotates a 2D image array `Z` by a given angle (in degrees) about a pivot point (`x0`, `y0`), 
    where the image axes are defined by the coordinate arrays `x` and `y`. The rotation is performed in the 
    coordinate space defined by `x` and `y`, not necessarily pixel indices.

    Parameters
    ----------
    Z : ndarray
        2D array representing the image to be rotated.
    x : array_like
        1D array of x-coordinates corresponding to the columns of `Z`.
    y : array_like
        1D array of y-coordinates corresponding to the rows of `Z`.
    angle : float
        Rotation angle in degrees. Positive values correspond to counter-clockwise rotation.
    x0 : float, optional
        X-coordinate of the pivot point about which to rotate. Default is 0.
    y0 : float, optional
        Y-coordinate of the pivot point about which to rotate. Default is 0.

    Returns
    -------
    rotated_Z : ndarray
        The rotated image array, with the same shape as `Z`.

    Notes
    -----
    - The function uses interpolation to map between coordinate space and pixel indices.
    - The rotation is performed about the specified pivot point (`x0`, `y0`) in the coordinate system defined by `x` and `y`.
    - Requires `numpy`, `scipy.ndimage.shift`, `scipy.ndimage.rotate`, and `scipy.interpolate.interp1d`.

    Docstring generated by GitHub Copilot.
    """

    xc = np.mean(x)
    yc = np.mean(y)
    
    pixel_x = interp1d(x, np.arange(len(x)), fill_value='extrapolate') 
    pixel_y = interp1d(y, np.arange(len(y)), fill_value='extrapolate')   

    ds_x = pixel_x(x0) - pixel_x(xc) # sample shift x
    ds_y = pixel_y(y0) - pixel_y(yc) # sample shift y
    ds = np.array([ds_x, ds_y])*0
    
    T = np.array([[np.cos(angle*np.pi/180), np.sin(angle*np.pi/180)],
                  [-np.sin(angle*np.pi/180), np.cos(angle*np.pi/180)]])
                  
    ds_rot = T @ ds
    
    return shift(rotate(shift(Z, ds[::-1]), angle), -ds_rot[::-1])

def combine_eta(eta_fine, eta_course, x_fine, y_fine, x_course, y_course, x=None, y=None):
    """
    Combine two 2D fields (fine and coarse) into a single field, using the fine field where available.

    Parameters
    ----------
    eta_fine : ndarray
        2D array of the fine-resolution field values.
    eta_course : ndarray
        2D array of the coarse-resolution field values.
    x_fine : ndarray
        1D array of x-coordinates for the fine field.
    y_fine : ndarray
        1D array of y-coordinates for the fine field.
    x_course : ndarray
        1D array of x-coordinates for the coarse field.
    y_course : ndarray
        1D array of y-coordinates for the coarse field.
    x : ndarray, optional
        1D array of x-coordinates for the output grid. If None, generated from coarse grid.
    y : ndarray, optional
        1D array of y-coordinates for the output grid. If None, generated from coarse grid.

    Returns
    -------
    eta_combined : ndarray
        2D array of the combined field, using fine field values where available and coarse elsewhere.
    x : ndarray
        1D array of x-coordinates for the combined field.
    y : ndarray
        1D array of y-coordinates for the combined field.

    Notes
    -----
    The function interpolates both input fields onto a common grid. The fine field overwrites the coarse field
    in the region where it is defined.

    Docstring generated by GitHub Copilot.
    """

    dx_fine = x_fine[1]-x_fine[0]
    dy_fine = y_fine[1]-y_fine[0]
    
    if x is None:
        x = np.arange(np.min(x_course), np.max(x_course), dx_fine)
    
    if y is None:
        y = np.arange(np.min(y_course), np.max(y_course), dy_fine)
    
    eta_course_i = RectBivariateSpline(y_course,x_course,eta_course)
    eta_fine_i = RectBivariateSpline(y_fine,x_fine,eta_fine)
    
    eta_combined = eta_course_i(y,x)
    sel = np.ix_(
        (y >= np.min(y_fine)) & (y <= np.max(y_fine)),
        (x >= np.min(x_fine)) & (x <= np.max(x_fine)))
    
    eta_combined[sel] = eta_fine_i(y[(y >= np.min(y_fine)) & (y <= np.max(y_fine))],
                                   x[(x >= np.min(x_fine)) & (x <= np.max(x_fine))])
    
    return eta_combined, x, y


def animate_surface(eta, x, y, t, filename=None, fps=None, 
                    speed_ratio=1.0, figsize=None, writer='ffmpeg', 
                    ax=None, surface=None):
    """
    Animates a time-evolving eta (sea surface).

    Parameters
    ----------
    eta : ndarray
        3D array of surface values with shape (nx, ny, nt), where nt is the number of time steps.
    x : ndarray
        1D or 2D array representing the x-coordinates of the surface grid.
    y : ndarray
        1D or 2D array representing the y-coordinates of the surface grid.
    t : ndarray
        1D array of time points corresponding to the third dimension of `eta`.
    filename : str or None, optional
        If provided, the animation is saved to this file. If None, the animation is displayed interactively.
    fps : float or None, optional
        Frames per second for the animation. If None, it is computed from the time step and `speed_ratio`.
    speed_ratio : float, optional
        Ratio to speed up or slow down the animation relative to real time. Default is 1.0.
    figsize : tuple or None, optional
        Size of the figure in inches. Passed to `plt.subplots` if a new figure is created.
    writer : str, optional
        Writer to use for saving the animation (e.g., 'ffmpeg'). Default is 'ffmpeg'.
    ax : matplotlib.axes.Axes or None, optional
        Axes object to plot on. If None, a new figure and axes are created.
    surface : matplotlib.image.AxesImage or None, optional
        Existing surface image to update. If None, a new surface is created.

    Returns
    -------
    None
        The function either displays the animation or saves it to a file.

    Notes
    -----
    - Requires matplotlib and numpy.
    - The function uses `matplotlib.animation.FuncAnimation` for animation.
    - If `filename` is provided, the animation is saved and not displayed interactively.

    Docstring generated by GitHub Copilot.
    """
        
    if surface is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)   
        else:
            fig = fig.get_figure()
        surface, __ = plot_surface(eta[:,:,0], x, y, ax=ax, colorbar=False, labels=False)
    else:
        fig = surface.get_figure()
        ax = fig.axes[0]
    
    def animate(i): 
        surface.set_data(eta[:,:,i])
        return surface, 
    
    frames = np.arange(len(t))
    dt = t[1]-t[0]
    fs = 1/dt
    
    if filename is None:
        repeat = True
    else:
        repeat = False
    
    if fps is None:
        fps = fs*speed_ratio
        
    interval = 1/fps*1000
    
    anim = FuncAnimation(fig, animate, 
                         frames=frames, interval=interval, blit=True, repeat=repeat) 
    
    if filename is not None:
        anim.save(filename, writer=writer, fps=fps)
    else:
        plt.show()

def xy_to_latlon(latlon0, coors, rot=0):
    """
    Convert local Cartesian (x, y) coordinates to latitude and longitude using geodesic calculations.

    Parameters
    ----------
    latlon0 : array-like, shape (2,)
        The reference latitude and longitude (in degrees) as a 2-element array or list [lat, lon].
    coors : array-like, shape (N, 2)
        Array of local Cartesian coordinates (x, y) to be converted, where each row is a point.
    rot : float, optional
        Rotation angle in degrees to be subtracted from the computed azimuth, default is 0.

    Returns
    -------
    latlon : ndarray, shape (N, 2)
        Array of latitude and longitude pairs (in degrees) corresponding to the input coordinates.

    Notes
    -----
    Uses Cartopy's geodesic calculations to convert local (x, y) displacements from a reference point
    (latlon0) to geographic coordinates, accounting for Earth's curvature.

    Docstring generated by GitHub Copilot.
    """

    import cartopy.crs as ccrs, cartopy.geodesic as cgds
    dist = np.linalg.norm(coors, axis=1)
    azi = np.arctan2(coors[:,0], coors[:,1])*180/np.pi - rot # atan2(x,y) to get azimuth (relative to N-vector)
    geodesic = cgds.Geodesic()
    
    return np.vstack(geodesic.direct(np.tile(latlon0, [len(dist), 1]), azi, dist))[:,:2]


def plot_surface_in_map(eta, x, y, eta_geo0, extent, 
                        eta_scatter=None, 
                        wms_url='https://openwms.statkart.no/skwms1/wms.terrengmodell?request=GetCapabilities&service=WMS', 
                        wms_layers=['relieff'], ax=None,
                        cm='Blues_r', colorbar=True, figsize=None, eta_rot=0, labels=False):
    """
    Plot a 2D surface (e.g., elevation or other field) on a map with optional scatter points and WMS background.

    Parameters
    ----------
    eta : ndarray or None
        2D array representing the surface to plot (e.g., elevation values). If None, only scatter and WMS are shown.
    x : ndarray
        1D or 2D array of x-coordinates corresponding to `eta`.
    y : ndarray
        1D or 2D array of y-coordinates corresponding to `eta`.
    eta_geo0 : array-like
        Reference geographic coordinates (e.g., origin for coordinate transformation).
    extent : array-like
        Map extent in the format [xmin, xmax, ymin, ymax] (in longitude and latitude).
    eta_scatter : ndarray or None, optional
        Array of points to scatter on the map, shape (N, 2). Default is None.
    wms_url : str, optional
        URL to the WMS service for background map. Default is Statkart's terrain model.
    wms_layers : list of str, optional
        List of WMS layer names to display. Default is ['relieff'].
    ax : matplotlib.axes.Axes or None, optional
        Existing axes to plot on. If None, a new axes with Mercator projection is created.
    cm : str or Colormap, optional
        Colormap for the surface plot. Default is 'Blues_r'.
    colorbar : bool, optional
        If True, display a colorbar for the surface. Default is True.
    figsize : tuple or None, optional
        Figure size. Not used if `ax` is provided. Default is None.
    eta_rot : float, optional
        Rotation angle (degrees) to apply to `eta` and scatter points. Default is 0.
    labels : bool, optional
        If True, draw gridlines with labels. Default is False.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plotted map.
    scatter : matplotlib.collections.PathCollection or None
        The scatter plot object, or None if `eta_scatter` is None.
    surface : matplotlib.image.AxesImage or None
        The surface plot object, or None if `eta` is None.

    Notes
    -----
    - Requires cartopy and matplotlib.
    - Assumes existence of `xy_to_latlon` and `rotate` helper functions.

    Docstring generated by GitHub Copilot.
    """
    
    import cartopy.crs as ccrs, cartopy.geodesic as cgds
    
    proj = 'Mercator'
    proj = getattr(ccrs, proj)()
    

    if ax is None:
        ax = plt.axes(projection=proj)

    
    # Plot scatter
    if eta_scatter is None:
        scatter = None
    else:
        eta_scatter = xy_to_latlon(eta_geo0, eta_scatter, rot=eta_rot)
        scatter = ax.scatter(eta_scatter[:,0], eta_scatter[:,1], c='black', s=6, transform=ccrs.PlateCarree())
    
    # Plot eta
    if eta is not None:
        eta_max = np.max(np.abs(eta))
        corners = np.array([[np.min(x), np.min(y)],
                    [np.min(x), np.max(y)],
                    [np.max(x), np.max(y)],
                    [np.max(x), np.min(y)]])
        
        corners_latlon = xy_to_latlon(eta_geo0, corners, rot=eta_rot)

        extent_merc = np.vstack(proj.transform_points(ccrs.PlateCarree(), np.array([extent[0], extent[2]]), np.array([extent[1], extent[3]])))[:,:2]     
        extent_merc = [np.min(extent_merc[:,0]), np.max(extent_merc[:,0]), 
                       np.min(extent_merc[:,1]), np.max(extent_merc[:,1])]
        
        ax.imshow(np.zeros([2,2]), cmap=cm, origin='lower', interpolation='none', 
                    vmin=-eta_max, vmax=eta_max, extent=extent_merc)
        
        corners_latlon_new = np.vstack(proj.transform_points(ccrs.PlateCarree(), *corners_latlon.T))[:,:2]
        eta_extent_new = [np.min(corners_latlon_new[:,0]), np.max(corners_latlon_new[:,0]), 
                  np.min(corners_latlon_new[:,1]), np.max(corners_latlon_new[:,1])]
        
        eta_rotated = rotate(eta, -eta_rot)
        
        surface = ax.imshow(eta_rotated, cmap=cm, origin='lower', 
                            interpolation='none', extent=eta_extent_new, 
                            vmin=-eta_max, vmax=eta_max)   
        
        if colorbar:
            plt.colorbar(surface)        
            
    else:
        surface = None
    if labels:
        ax.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
        
    ax.add_wms(wms_url, layers=wms_layers, extent=extent)
    ax.set_extent(extent)

    return ax, scatter, surface


def plot_surface(eta, x, y, ax=None, 
                 cm='Blues_r', colorbar=True, 
                 labels=True, figsize=None, interpolation='none'): 
    """
    Plot a 2D surface using imshow.
    
    Parameters
    ----------
    eta : ndarray
        2D array representing the surface to plot (e.g., elevation values).
    x : ndarray
        1D or 2D array of x-coordinates corresponding to `eta`.
    y : ndarray
        1D or 2D array of y-coordinates corresponding to `eta`.
    ax : matplotlib.axes.Axes or None, optional
        Existing axes to plot on. If None, a new axes is created.
    cm : str or Colormap, optional
        Colormap for the surface plot. Default is 'Blues_r'.
    colorbar : bool, optional
        If True, display a colorbar for the surface. Default is True.
    labels : bool, optional
        If True, draw gridlines with labels. Default is True.
    figsize : tuple or None, optional
        Figure size. Not used if `ax` is provided. Default is None.
    interpolation : str, optional
        Interpolation method for the surface plot. Default is 'none'.

    Returns
    ------- 
    surface : matplotlib.image.AxesImage
        The surface plot object.
    ax : matplotlib.axes.Axes
        The axes with the plotted surface.

    Notes
    -----
    Docstring generated by GitHub Copilot.

"""

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    dx = (x[1]-x[0])/2.
    dy = (y[1]-y[0])/2.
    extent = [x[0]-dx, x[-1]+dx, y[0]-dy, y[-1]+dy]

    eta_max = np.max(np.abs(eta))   
    surface_plot = ax.imshow(eta, cmap=cm, extent=extent, origin='lower', vmin=-eta_max, vmax=eta_max, interpolation=interpolation)
    
    if labels:
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    if colorbar:
        cb = plt.colorbar(surface_plot)
    
    return surface_plot, ax


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def equal_3d(ax=None):
    if ax is None:
        ax = plt.gca()

    x_lims = np.array(ax.get_xlim())
    y_lims = np.array(ax.get_ylim())
    z_lims = np.array(ax.get_zlim())

    x_range = np.diff(x_lims)
    y_range = np.diff(y_lims)
    z_range = np.diff(z_lims)

    max_range = np.max([x_range,y_range,z_range])/2

    ax.set_xlim(np.mean(x_lims) - max_range, np.mean(x_lims) + max_range)
    ax.set_ylim(np.mean(y_lims) - max_range, np.mean(y_lims) + max_range)
    ax.set_zlim(np.mean(z_lims) - max_range, np.mean(z_lims) + max_range)
    # ax.set_aspect(1)
    
    return ax

def plot_transformation_mats(x,y,z,T,figno=None, ax=None, scaling='auto'):
    
    if ax==None:
        fig = plt.figure(figno)
        ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x,y,z,'.k')

    if scaling=='auto':
        xr = max(x)-min(x)
        yr = max(y)-min(y)
        zr = max(z)-min(z)
        r = np.sqrt(xr**2+yr**2+zr**2)
        scaling = 0.005*r

    compcolors = ['tab:red', 'tab:blue', 'tab:green']
    h = [None]*3
    for ix, Ti in enumerate(T):
        xi = x[ix]
        yi = y[ix]
        zi = z[ix]
        
        for comp in range(0,3):
            xunit = [xi, xi+Ti[comp,0]*scaling]
            yunit = [yi, yi+Ti[comp,1]*scaling]
            zunit = [zi, zi+Ti[comp,2]*scaling]

            h[comp] = plt.plot(xs=xunit,ys=yunit,zs=zunit, color=compcolors[comp])[0]

    plt.legend(h,['x', 'y', 'z'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    equal_3d(ax)
    return ax,h

def plot_2d(S2d, x1, x2, ax=None, levels=80, discrete=False, **kwargs):
    """
    Plot a 2D array as either a filled contour plot or a pseudocolor mesh.

    Parameters
    ----------
    S2d : array_like
        2D array of values to plot.
    x1 : array_like
        1D array representing the x-coordinates.
    x2 : array_like
        1D array representing the y-coordinates.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, uses the current axes.
    levels : int, optional
        Number of contour levels to use for filled contour plot. Default is 80.
    discrete : bool, optional
        If True, use `pcolormesh` for a discrete colormap. If False, use `contourf` for a filled contour plot.
    **kwargs
        Additional keyword arguments passed to the plotting function (`contourf` or `pcolormesh`).

    Returns
    -------
    contour : QuadMesh or QuadContourSet
        The resulting plot object from `pcolormesh` or `contourf`.

    Notes
    -----
    Docstring generated by GitHub Copilot.
    """
        

    if ax is None:
        ax = plt.gca()
        
    X, Y = np.meshgrid(x1, x2)
    if discrete:
        contour = ax.pcolormesh(x1,x2,S2d, **kwargs)
    else:        
        contour = ax.contourf(X, Y, S2d.T, levels=levels, **kwargs)
    return contour


def plot_S2d(S, omega, theta, D=None, omega_range=None, theta_range=None):
    """
    Plot a 2D spectral density (S2d) as a contour plot, with optional marginal plots (S and D).

    Parameters
    ----------
    S : array_like
        1D array of spectral density values as a function of omega.
    omega : array_like
        1D array of frequency values (rad/s).
    theta : array_like
        1D array of direction values (rad).
    D : array_like, optional
        1D array of directional spreading function values as a function of theta.
        If provided, marginal plots of S(omega) and D(theta) are shown.
    omega_range : list or tuple, optional
        Range [min, max] for the omega axis. If None, uses [0, max(omega)].
    theta_range : list or tuple, optional
        Range [min, max] for the theta axis. If None, uses [min(theta), max(theta)].

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    Notes
    -----
    - If `D` is provided, the function plots the 2D spectral density as a contour plot,
      with marginal line plots for S(omega) and D(theta).
    - If `D` is not provided, only the contour plot is shown.
    - The function handles NaN values in the spectral density by setting them to zero.

    Docstring generated by GitHub Copilot.
    """

    if theta_range is None:
        theta_range = [np.min(theta), np.max(theta)]
    
    if omega_range is None:
        omega_range = [0, np.max(omega)]

    X, Y = np.meshgrid(omega, theta)

    if D is None:
        SD = S*1
    else:
        SD = S[:,np.newaxis] @ D[np.newaxis,:]
    
    SD[np.isnan(SD)] = 0
    
    plt.figure(2).clf()
    fig = plt.figure(num=2, constrained_layout=True)
    
    if D is not None:
        widths = [2, 0.7]
        heights = [0.7, 2.0, 0.7]
        spec = fig.add_gridspec(ncols=2, nrows=3, width_ratios=widths,
                              height_ratios=heights, wspace=.1, hspace=.1)
    else:
        widths = [2]
        heights = [2.0, 0.7]
        spec = fig.add_gridspec(ncols=1, nrows=2, width_ratios=widths,
                              height_ratios=heights, wspace=.1, hspace=.1)        
    
    if D is not None:
        ax = [None]*3
        ax[0] = fig.add_subplot(spec[1,0])   # SD
        ax[1] = fig.add_subplot(spec[0,0])   # S
        ax[2] = fig.add_subplot(spec[1,1])   # D
        ax[1].set_yticklabels('')
        ax[2].set_xticklabels('')
        ax[2].set_yticklabels('')
        cbar_ax = fig.add_subplot(spec[2,0])
    else:
        ax = [fig.add_subplot(spec[0,0])]
        cbar_ax = fig.add_subplot(spec[1,0])

    cbar_ax.axis('off')

    # Contour plot
    contour = ax[0].contourf(X, Y, SD.T)
    ax[0].set_ylim(theta_range)
    ax[0].set_xlim(omega_range)
    ax[0].set_ylabel(r'$\theta$ [rad]')
    ax[0].set_xlabel(r'$\omega$ [rad/s]')
   
    if D is not None:
        # S line plot
        ax[1].plot(omega, S)
        ax[1].set_ylim(bottom=0)
        ax[1].set_xlim(omega_range)
        ax[1].set_xticklabels('')
        ax[1].set_yticks([])
        
        # D line plot

        ax[2].plot(D, theta)
        ax[2].set_ylim(theta_range)
        ax[2].set_xlim(left=0)
        ax[2].set_xticks([])
    
    # cbar_ax.axis('off')
    fig.colorbar(contour, ax=cbar_ax, orientation="horizontal", aspect=25, shrink=1.0)

    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    if D is not None:
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].set_ylabel(r'$S_\eta(\omega)$')
        
        ax[2].spines['bottom'].set_visible(False)
        ax[2].spines['right'].set_visible(False)
        ax[2].set_xlabel(r'$D(\theta)$',rotation=-90)
        ax[2].xaxis.set_label_position('top') 
    
        fig.subplots_adjust(top=0.97, bottom=0.08, left=0.16, right=0.97)
    
    return fig