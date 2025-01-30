import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from wawi.wave import dispersion_relation_scalar as get_kappa
from scipy.ndimage import rotate, shift
from matplotlib import transforms
from scipy.interpolate import RectBivariateSpline, interp1d


def plot_ads(ad_dict, v, terms='stiffness', num=None, test_v=dict(), test_ad=dict(), zasso_type=False, ranges=None):
    # v: v or K
    if terms is 'stiffness':
        terms = [['P4', 'P6', 'P3'], ['H6', 'H4', 'H3'], ['A6', 'A4', 'A3']]
    elif terms is 'damping':
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
            ax[-1, col_ix].set_xlabel('$K$')
        else:
            ax[-1, col_ix].set_xlabel('$V/(B\cdot \omega)$')
    
    fig.tight_layout()
    return fig, ax

def save_plot(pl, path, w=None, h=None):
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
    arr_opts = {'head_width': 4, 'width':2, 'edgecolor':'none'}
    arr_opts.update(**arrow_options)
    
    if ax is None:
        ax = plt.gca()
        
    # Plot wave angle and crests
    v = np.array([np.cos(theta0*np.pi/180), np.sin(theta0*np.pi/180)])
    v_norm = np.array([-np.sin(theta0*np.pi/180), np.cos(theta0*np.pi/180)])
    wave_length = 2*np.pi/get_kappa(2*np.pi/Tp, U=0.0)

    plt.arrow(origin[0],origin[1], arrow_length*v[0], arrow_length*v[1], **arr_opts)
    plt.text(origin[0], origin[1], f'$\\theta_0$ = {theta0}$^o$\n $T_p$={Tp} s\n $\lambda=${wave_length:.0f} m')

    dv = v*wave_length
    for n in range(n_repeats):
        p1 = origin-v_norm*crest_length/2
        p2 = origin+v_norm*crest_length/2
        pts = np.vstack([p1,p2])
        
        ax.plot(pts[:,0], pts[:,1], alpha=alpha_crests, color='black', zorder=0)
        origin = origin + dv

    return ax

def rotate_image_about_pivot(Z, x, y, angle, x0=0, y0=0):
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
    import matplotlib.pyplot as plt
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

def equal_3d(ax=plt.gca()):
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


def plot_elements(element_matrix, node_matrix, chosen_nodes_ix=[], disp=None, node_labels=False, element_labels=False, plot_nodes=True, plot_elements=True, ax=None, fig=None, element_settings={}, node_settings={}, node_label_settings={}, chosen_node_settings={}, disp_settings={}, element_label_settings={}, three_d=True):
    e_dict = {'color': 'LimeGreen', 'alpha': 1}
    e_dict.update(**element_settings)

    n_dict = {'color':'Black', 'linestyle':'', 'marker':'.', 'markersize':4, 'alpha':0.8}
    n_dict.update(**node_settings)
       
    n_chosen_dict = {'color':'GreenYellow', 'linestyle':'', 'marker':'o', 'markersize':8, 'alpha':1, 'markeredgecolor':'dimgray'}
    n_chosen_dict.update(**chosen_node_settings)
    
    disp_dict = {'color':'IndianRed', 'alpha':1}
    disp_dict.update(**disp_settings)

    l_nodes_dict = {'color':'Black', 'fontsize': 8, 'fontweight':'normal'}
    l_nodes_dict.update(**node_label_settings)
    
    l_elements_dict = {'color':'LimeGreen', 'fontsize': 8, 'fontweight':'bold', 'style':'italic'}
    l_elements_dict.update(**element_label_settings)
    
    if ax is None and fig is None:
        fig = plt.figure()
        
    if ax == None and three_d:
        ax = fig.gca(projection='3d')
    elif ax == None:
        ax = fig.gca()
    elif three_d:
        1
        # ax.set(projection='3d')  #mangler funksjonalitet her...
    
    element_handles = [None]*len(element_matrix[:,0])

    if plot_elements:
        for element_ix, __ in enumerate(element_matrix[:,0]):
            node1 = element_matrix[element_ix, 1]
            node2 = element_matrix[element_ix, 2]
            nodeix1 = np.where(node_matrix[:,0]==node1)[0]
            nodeix2 = np.where(node_matrix[:,0]==node2)[0]
            x01 = node_matrix[nodeix1,1:4]
            x02 = node_matrix[nodeix2,1:4]
            x0 = np.vstack([x01,x02])

            if three_d:
                element_handles[element_ix] = ax.plot(xs=x0[:,0], ys=x0[:,1], zs=x0[:,2], **e_dict)
            else:
                element_handles[element_ix] = ax.plot(x0[:,0], x0[:,1], **e_dict)
  
            if element_labels:
                xmean = np.mean(x0, axis=0)
                if three_d:
                    ax.text(xmean[0],xmean[1],xmean[2],'%i' % element_matrix[element_ix,0], **l_elements_dict)
                else:
                    ax.text(xmean[0],xmean[1],s='%i' % element_matrix[element_ix,0], **l_elements_dict)

            if disp is not None:
                disp_node1 = disp[nodeix1[0]*6:(nodeix1[0]*6+6)]
                disp_node2 = disp[nodeix2[0]*6:(nodeix2[0]*6+6)]
                x1 = x01+disp_node1[0:3]
                x2 = x02+disp_node2[0:3]
                x = np.vstack([x1,x2])
                
                if three_d:
                    ax.plot(xs=x[:,0], ys=x[:,1], zs=x[:,2], **disp_dict)
                else:
                    ax.plot(x[:,0], x[:,1], **disp_dict)

    if plot_nodes:
        if three_d:
            ax.plot(xs=node_matrix[:, 1], ys=node_matrix[:, 2], zs=node_matrix[:, 3], **n_dict)
        else:
           ax.plot(node_matrix[:, 1], node_matrix[:, 2], **n_dict)
        
        if chosen_nodes_ix != []:
            if three_d:
                ax.plot(xs=node_matrix[chosen_nodes_ix, 1], ys=node_matrix[chosen_nodes_ix, 2], zs=node_matrix[chosen_nodes_ix, 3], **n_chosen_dict)
            else:
               ax.plot(node_matrix[chosen_nodes_ix, 1], node_matrix[chosen_nodes_ix, 2], **n_chosen_dict)
        
    if node_labels:
        if three_d:
            for node_ix in range(0, np.shape(node_matrix)[0]):
                ax.text(node_matrix[node_ix, 1], node_matrix[node_ix, 2], node_matrix[node_ix, 3], '%i' % node_matrix[node_ix, 0], **l_nodes_dict)
        else:
            for node_ix in range(0, np.shape(node_matrix)[0]):
                ax.text(node_matrix[node_ix, 1], node_matrix[node_ix, 2], '%i' % node_matrix[node_ix, 0], **l_nodes_dict)
        
    if three_d:
        equal_3d(ax)
    else:
        ax.set_aspect('equal', adjustable='box')
    
    ax.grid('off')
    return ax, element_handles


def plot_2d(S2d, x1, x2, ax=None, levels=80, discrete=False, **kwargs):
    if ax is None:
        ax = plt.gca()
        
    X, Y = np.meshgrid(x1, x2)
    if discrete:
        contour = ax.pcolormesh(x1,x2,S2d, **kwargs)
    else:        
        contour = ax.contourf(X, Y, S2d.T, levels=levels, **kwargs)
    return contour


def plot_S2d(S, omega, theta, D=None, omega_range=None, theta_range=None):

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