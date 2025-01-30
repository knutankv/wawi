import numpy as np
from math import atan2
from scipy.interpolate import interp1d
from scipy.special import jv
from .general import wrap_to_pi, uniquetol, zero_pad_upsample, get_omega_upsampled
from .tools import print_progress as pp
from inspect import isfunction
from scipy.special import jv, gamma
from scipy.optimize import fsolve
from wawi.general import eval_fun_or_scalar

def linear_drag_damping(drag_coefficient, std_udot, area=1.0, rho=1020.0, as_matrix=True):
    damping = 0.5*rho*area*drag_coefficient*np.sqrt(8/np.pi)*std_udot

    if as_matrix == True and (len(damping)==3 or len(damping)==6):
        damping = np.diag(damping)

    return damping

def stochastic_linearize(C_quad, std_udot):
    # Input C_quad is assumed matrix form, std_udot is assumed matrix
    
    if np.ndim(std_udot)==1:
        std_udot = np.diag(std_udot)
        
    return C_quad*np.sqrt(8/np.pi)*std_udot

def harmonic_linearize(C_quad, udot):
    if np.ndim(udot)==2:
        udot = np.diag(np.diag(udot))
    else:
        udot = np.diag(udot)
        
    C_quad = np.diag(np.diag(C_quad))
    return 8/(3*np.pi)*C_quad*np.abs(udot)
    

def get_coh_fourier(omega, dx, dy, D, theta0, theta_shift=0.0, depth=np.inf, 
                    k_max=10, input_is_kappa=False):
    '''
    theta_shift is used to translate D, such
    that non-centered are allowed. Docs to come.
    '''

    L = np.sqrt(dx**2+dy**2)
    phi = np.arctan2(dy, dx)
    beta  = theta0 - phi

    if input_is_kappa:
        kappa = omega*1
    else:
        kappa = dispersion_relation(omega, h=depth)[:, np.newaxis]
    
    # Establish from Fourier coefficients
    k = np.arange(-k_max, k_max+1)[np.newaxis, :]
    theta = np.linspace(-np.pi, np.pi, k_max*2+1)     #ensures odd number of fft coeff.
    
    c = np.fft.ifft(D(theta + theta0-theta_shift))
    c = np.hstack([c[-k_max:], c[:k_max+1]])[np.newaxis, :]
    
    coh = 2*np.pi*np.sum(
        np.tile(c*1j**k*np.exp(-1j*k*beta), [len(kappa), 1]) 
        * jv(k, kappa*L), axis=1)
    
    return coh

def get_coh_cos2s(omega, dx, dy, s, theta0, k_max=10, depth=np.inf, 
                  input_is_kappa=False):
    if input_is_kappa:
        kappa = omega*1
    else:
        kappa = dispersion_relation(omega, h=depth)[:, np.newaxis]
        
    L = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx)
    beta  = theta0 - phi

    k = np.arange(-k_max, k_max+1)[np.newaxis, :]
    c = 1/(2*np.pi) * gamma(s+1)**2/(gamma(s-k+1)*gamma(s+k+1))
    coh = 2*np.pi * np.sum(np.tile(c*1j**k*np.exp(-1j*k*beta), 
                                   [len(kappa), 1]) * jv(k, kappa*L), axis=1)
    
    return coh

def get_coh(omega, dx, dy, D1, D2=None, depth=np.inf, n_theta=40, 
            theta_shift=0.0, input_is_kappa=False, twodimensional=False,
            include_D=True):
    
    if D2 is None:  #assumes the same as D1
        D2 = D1

    if input_is_kappa:
        kappa = omega*1
    else:
        kappa = dispersion_relation(omega, h=depth)
        
    theta = np.linspace(-np.pi, np.pi, n_theta)
    
    if include_D:
        D_eval = np.sqrt(D1(theta)*D2(theta))
    else:
        D_eval = 1.0
        
    coh2d = D_eval*np.exp(-1j*kappa[:, np.newaxis] @ ((np.cos(theta+theta_shift)*dx + np.sin(theta+theta_shift)*dy))[np.newaxis, :])

    if twodimensional:
        return coh2d, theta
    else:
        coh = np.trapz(coh2d, x=theta, axis=1)

    return coh

                                
def xsim(x, y, S, D, omega, fs=None, theta=None, n_theta=40, grid_mode=True, print_progress=True, 
         time_history=False, phase=None, return_phases=False, theta_shift=0.0):
    
    if fs is None:
        fs = np.max(omega)/2/np.pi

    if theta is None:
        theta = np.linspace(-np.pi, np.pi, n_theta)
    
    
    if not isfunction(S):
        Sfun = lambda x, y: S
    else:
        Sfun = S
    
    if not isfunction(D):
        Dfun = lambda x, y: D
    else:
        Dfun = D

    if grid_mode:
       xx,yy = np.meshgrid(x,y)
       xvec = x*1
       yvec = y*1
       x = xx.flatten()
       y = yy.flatten()
    
    domega = omega[1] - omega[0]
    
    if len(theta)>1:
        dtheta = theta[1] - theta[0]
    else:
        dtheta = 1.0
    
    omegai = get_omega_upsampled(omega, fs*2*np.pi)
    kappa = omega**2 / 9.81     #assume deep-water waves - can be generalized later (different depths at different positions possible also)
    
    # Create kappa grid
    # Attempt to fix function theta_shift (non centered dirdist definitions with theta0 as function)
    # kappax = lambda x,y: kappa[:, np.newaxis] @ np.cos(theta+eval_fun_or_scalar(theta_shift,x,y))[np.newaxis, :]
    # kappay = lambda x,y: kappa[:, np.newaxis] @ np.sin(theta+eval_fun_or_scalar(theta_shift,x,y))[np.newaxis, :]
    
    kappax = kappa[:, np.newaxis] @ np.cos(theta+theta_shift)[np.newaxis, :]
    kappay = kappa[:, np.newaxis] @ np.sin(theta+theta_shift)[np.newaxis, :]
   
    n_freqs = len(omega)
    n_freqs_i = len(omegai)
    n_angles = len(theta)
    
    if phase is None:
        phase = np.exp(1j*np.random.rand(n_freqs, n_angles)*2*np.pi)
    
    if time_history:
        eta = np.zeros([n_freqs_i, len(x)])
        selection = np.arange(n_freqs_i)
        n_t = n_freqs_i*1
    else:
        eta = np.zeros([1, len(x)])
        selection = np.array(0)
        n_t = 1
        
    for ix in range(len(x)):
        Sthis = Sfun(x[ix], y[ix])(omega)[:, np.newaxis]
        Dthis = Dfun(x[ix], y[ix])(theta)[np.newaxis, :]

        B0 = np.sqrt(2 * Sthis * Dthis * domega * dtheta)
        Bkr = B0*np.exp(-1j*(kappax*x[ix] + kappay*y[ix])) * phase          
        if Bkr.shape[1]>1:
            Bkr_sum = np.trapz(Bkr, axis=1)
        else:
            Bkr_sum = Bkr[:,0]
        
        Bkr_sum = zero_pad_upsample(Bkr_sum, omega, fs*2*np.pi)

        eta[:, ix] = np.fft.fftshift(len(omegai) * np.real(np.fft.ifft(Bkr_sum)))[selection]
        
        if print_progress:
            pp(ix+1, len(x), postfix=f'  |   x={x[ix]:.1f}m, y={y[ix]:.1f}m ')

    t = np.linspace(0, 2*np.pi/domega, n_freqs_i)[selection].T
    
    if grid_mode:
        if time_history:
            eta = np.swapaxes(eta, 0, 1)    # after swap: gridcombos x time
            eta = np.reshape(eta, [len(yvec), len(xvec), -1])
        else:
            eta = np.reshape(eta, [len(yvec), len(xvec)])

    # Return
    if return_phases:
        return eta, t, phase
    else:
        return eta, t  


def swh_from_gamma_alpha_Tp(gamma, alpha, Tp, g=9.81):
    wp = 2*np.pi/Tp
        
    Hs = (1.555 + 0.2596*gamma - 0.02231*gamma**2 + 0.01142*gamma**3)*g*np.sqrt(alpha)/wp**2
    return Hs

def sigma_from_sigma_range(sigma, wp):
    return lambda w: (sigma[0]+(sigma[1]-sigma[0])*(w>wp))

def peak_enhancement(gamma, Tp, sigma, normalize=True):
    wp = 2*np.pi/Tp
    sigma = sigma_from_sigma_range(sigma, wp)
    if normalize:
        A_gamma = (1 - 0.287*np.log(gamma))
        return lambda w: gamma**np.exp(-(w-wp)**2/(2*sigma(w)**2*wp**2)) * A_gamma
    else:
        return lambda w: gamma**np.exp(-(w-wp)**2/(2*sigma(w)**2*wp**2))  


def pm2(Hs, Tp, unit='Hz'):
    fp = 1/Tp
    A = 5*Hs**2*fp**4/(16)
    B = 5*fp**4/4
        
    if unit == 'Hz':
        return lambda f: A/f**5*np.exp(-B/f**4)
    elif unit == 'rad/s':
        return lambda w: A/(w/2/np.pi)**5*np.exp(-B/(w/2/np.pi)**4)/2/np.pi
    
    
def jonswap(Hs, Tp, gamma, g=9.81, sigma=[0.07, 0.09]):
    return lambda w: pm2(Hs, Tp, unit='rad/s')(w)*peak_enhancement(gamma, Tp, sigma, normalize=True)(w)
   
def jonswap_numerical(Hs, Tp, gamma, omega, g=9.81, sigma=[0.07, 0.09]):

    if omega[0] == 0:
        omega[0] = 1
        first_is_zero = True
    else:
        first_is_zero = False

    S = jonswap(Hs, Tp, gamma, g=g, sigma=sigma)(omega)
    
    if first_is_zero:
        S[0] = 0.0
        omega[0] = 0.0
    
    return S
   

def jonswap_dnv(Hs, Tp, gamma, sigma=[0.07, 0.09]):
    A = 1-0.287*np.log(gamma)
    wp = 2*np.pi/Tp

    sigma = sigma_from_sigma_range(sigma, wp)
    S = lambda omega: A*5.0/16.0*Hs**2*wp**4/(omega**5)*np.exp(-5/4*(omega/wp)**(-4))*gamma**(np.exp(-0.5*((omega-wp)/sigma(omega)/wp)**2))
        
    return S


def dirdist_decimal_inv(s, theta0=0, theta=None):
    if s>170:
        raise ValueError("Spreading exponent s cannot exceed 170. Please adjust!")
    C = gamma(s+1)/(2*np.sqrt(np.pi)*gamma(s+0.5))
    D = lambda theta: C*(np.abs(np.cos((theta+theta0)/2)))**(2*s)
    
    if theta!=None:
        D = D(theta)
    
    return D

def dirdist_decimal(s, theta0=0, theta=None):
    if s>170:
        raise ValueError("Spreading exponent s cannot exceed 170. Please adjust!")
    
    C = gamma(s+1)/(2*np.sqrt(np.pi)*gamma(s+0.5))
    D = lambda theta: C*(np.abs(np.cos((theta-theta0)/2)))**(2*s)
    
    if theta!=None:
        D = D(theta)
    
    return D

def dirdist(s, theta0=0, theta=None):
    if s>170:
        raise ValueError("Spreading exponent s cannot exceed 170. Please adjust!")
    C = gamma(s+1)/(2*np.sqrt(np.pi)*gamma(s+0.5))
    D = lambda theta: C*(np.cos((theta-theta0)/2))**(2*s)
    
    if theta!=None:
        D = D(theta)
    
    return D

def dirdist_robust(s, theta0=0, dtheta=1e-4, theta=None):
    theta_num = np.unique(np.hstack([np.arange(-np.pi, np.pi+dtheta, dtheta), wrap_to_pi(theta0)]))
    val = np.cos((theta_num-theta0)/2)**(2*s)
    scaling = 1/np.trapz(val, theta_num)

    def D(theta):
        return interp1d(theta_num, val*scaling)(wrap_to_pi(theta))    
        
    if theta!=None:
        D = D(theta)
    
    return D



def waveaction_fft(pontoons, omega, n_fourier=20, max_coherence_length=np.inf, print_progress=True):
    n_pontoons = len(pontoons)
    n_dofs = n_pontoons*6
    
    n_theta = n_fourier*2
    
    theta = np.linspace(-np.pi, np.pi-2*np.pi/n_theta, n_theta)
    S = np.zeros([n_dofs, n_dofs, len(omega)]).astype('complex')
    
    for i, pontoon_i in enumerate(pontoons):
        fi,__,__ = pontoon_i.evaluate_Q(omega, n_fourier*2)
        xi,yi = pontoon_i.node.coordinates[:2]
        
        for j, pontoon_j in enumerate(pontoons):
            xj,yj = pontoon_j.node.coordinates[:2]
            dx = xj-xi
            dy = yj-yi
            
            l = np.sqrt(dx**2+dy**2)
            
            if l<max_coherence_length:
                beta = atan2(dy,dx)
                fj,__,__ = pontoon_j.evaluate_Q(omega, n_fourier*2)
                
                depth = (pontoon_i.depth+pontoon_j.depth)/2
                kappa = np.array([dispersion_relation(omega_k, h=depth) for omega_k in omega])
                
                coh_2d = np.sqrt((pontoon_i.S(omega) * pontoon_j.S(omega))[:, np.newaxis] @ (pontoon_i.D(theta-pontoon_i.theta0) * pontoon_j.D(theta-pontoon_j.theta0))[np.newaxis, :])
       
                for dof_i in range(6):
                    for dof_j in range(6):
                        integrand = fi[dof_i,:] * fj[dof_j,:].conj() * coh_2d.T
                        c = np.fft.fft(integrand)
                        I = np.stack([np.exp(1j*n*beta)*1j**n*2*np.pi*jv(n, kappa*l) for n in range(-n_fourier, n_fourier)], axis=1)
                        
                        S[i*6+dof_i, j*6+dof_j, :] = np.sum(I*c)

            if print_progress:
                pp(i*n_pontoons+j, n_pontoons**2)
    
    return S


def waveaction(pontoon_group, omega, max_rel_error=1e-3, print_progress=True):
    n_pontoons = len(pontoon_group.pontoons)
    n_freqs = len(omega)
    n_dofs = n_pontoons*6
   
    if omega[0]==0:
        omega = omega[1::]
        first_is_zero = True
        n_freqs = n_freqs-1
    else:
        first_is_zero = False

    kappa = [None]*n_pontoons
    for pontoon_ix, pontoon in enumerate(pontoon_group.pontoons):
        kappa[pontoon_ix] = dispersion_relation(omega, pontoon.depth)

    Sqq = np.zeros([n_dofs, n_dofs, n_freqs]).astype('complex')
    
    for k, omega_k in enumerate(omega):
        if print_progress:
            pp(k+1, n_freqs)
    
        theta_int = pontoon_group.get_theta_int(omega_k)   
        dtheta = theta_int[2]-theta_int[1]
        
        n_theta = len(theta_int)
        Z = np.zeros([n_dofs, n_theta]).astype('complex')

        for pontoon_index, pontoon in enumerate(pontoon_group.pontoons):
            if pontoon.D.__code__.co_argcount==2:    # count number of inputs
                D = pontoon.D(theta_int, omega_k)
            else:
                D = pontoon.D(theta_int)

            # Shift current theta axis
            theta_pontoon = wrap_to_pi(pontoon.pontoon_type.theta + pontoon.rotation - pontoon.theta0)
            theta_pontoon, sort_ix = uniquetol(theta_pontoon, 1e-10)
            
            # Interpolate hydrodynamic transfer function
            Q_int = interp1d(theta_pontoon, pontoon.get_local_Q(omega_k)[:, sort_ix], fill_value=0, kind='quadratic', bounds_error=False)(theta_int)

            coh = np.exp(1j*kappa[pontoon_index][k] * (pontoon.node.x*np.cos(theta_int + pontoon.theta0) + pontoon.node.y*np.sin(theta_int + pontoon.theta0)))
            Z[pontoon_index*6:pontoon_index*6+6, :] = np.sqrt(pontoon.S(omega_k)) * Q_int * np.tile(np.sqrt(D), [6, 1]) * np.tile(coh, [6, 1])

        # first and last point in trapezoidal integration has 1/2 as factor, others have 1
        Z[:, 0] = np.sqrt(0.5)*Z[:, 0]
        Z[:, -1] = np.sqrt(0.5)*Z[:, -1]
          
        Sqq[:, :, k] = dtheta * pontoon_group.get_tmat().T @ (Z @ Z.conj().T) @ pontoon_group.get_tmat()            # verified to match for loop over angles and trapz integration.


    if first_is_zero==True:
        Sqq = np.insert(Sqq, 0, Sqq[:,:,0]*0, axis=2)
        
        
    return Sqq

    
def dispersion_relation_scalar(w, h=np.inf, g=9.81, U=0.0):
    if h==np.inf:
        f = lambda k: g*k - (w-k*U)**2
    else:
        f = lambda k: g*k*np.tanh(k*h) - (w-k*U)**2
        
    k0 = w**2/g     # deep-water, zero-current wave number
    
    k = fsolve(f, x0=k0)[0]

    return k

def dispersion_relation_scalar_legacy(w, h=np.inf, g=9.81):
    if h != np.inf:
        a = h*w**2/g
        
        # Initial guesses are provided by small value and large value approximations of x
        x = a*0
        x[a<=3/4] = np.sqrt((3-np.sqrt(9-12*a[a<=3/4]))/2)
        x[a>3/4] = a[a>3/4]
        
        for i in range(0,100):
            x = (a+(x**2)*(1-(np.tanh(x))**2))/(np.tanh(x)+x*(1-(np.tanh(x))**2))
            # The convergence criterion is chosen such that the wave numbers produce frequencies that don't deviate more than 1e-6*sqrt(g/h) from w.
            if np.max(abs(np.sqrt(x*np.tanh(x))-np.sqrt(a))) < 1e-6:
                break
        
        k = x/h
    else:
        return w**2/g

def dispersion_relation(w, h=np.inf, g=9.81):
    zero_ix = np.where(w==0)
    w = w[w!=0]

    if h != np.Inf:
        a = h*w**2/g

        # Initial guesses are provided by small value and large value approximations of x
        x = a*0
        x[a<=3/4] = np.sqrt((3-np.sqrt(9-12*a[a<=3/4]))/2)
        x[a>3/4] = a[a>3/4]
        
        for i in range(0,100):
            x = (a+(x**2)*(1-(np.tanh(x))**2))/(np.tanh(x)+x*(1-(np.tanh(x))**2))
            # The convergence criterion is chosen such that the wave numbers produce frequencies that don't deviate more than 1e-6*sqrt(g/h) from w.
            if np.max(abs(np.sqrt(x*np.tanh(x))-np.sqrt(a))) < 1e-6:
                break
        
        k = x/h
    else:
        k = w**2/g
    
    k = np.insert(k, zero_ix[0], 0)
    
    return k


def maxincrement(dl, kmax, a, b, max_relative_error):
    g = 9.81
    thetamax = np.pi/2
    K = abs(1j*kmax*(-(1/2)*np.pi)*dl*(-(1/2)*np.pi)*(np.cos(thetamax))*(-(1/2)*np.pi)*(np.exp(-1j*kmax*dl*np.cos(thetamax)))*(-(1/2)*np.pi)-kmax*(-(1/2)*np.pi)**2*dl*(-(1/2)*np.pi)**2*(np.sin(thetamax))*(-(1/2)*np.pi)**2*(np.exp(-1j*kmax*dl*np.cos(thetamax)))*(-(1/2)*np.pi))
    
    max_val = abs(np.exp(-1j*dl))
    max_error = max_val*max_relative_error
    N = np.sqrt((K*(b-a)**3)/(12*max_error))
    
    increment=(b-a)/N

    if dl==0:
        increment=b-a
        
    return increment
