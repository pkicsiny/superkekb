import numpy as np
import os
import xtrack as xt
import xfields as xf
import xpart as xp

import psutil
from pympler import muppy,summary
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from ipywidgets import interact, IntSlider
from scipy import constants as cst

"""
28/12/2021: add q0=+/-1*n_particles_per_macropart
"""

def stat_emittance_from_monitor(emits_dict, n_macroparticles, n_turns, alpha_x=0, alpha_y=0, beta_x=0, beta_y=0, normalize=False):
    """
    compute statistical emittances. First normalize coordinates by using (263) then (130) from
    https://arxiv.org/pdf/2107.02614.pdf
    """
        
    x     = np.reshape(emits_dict["x"],     (n_macroparticles, n_turns))
    px    = np.reshape(emits_dict["px"],    (n_macroparticles, n_turns))
    y     = np.reshape(emits_dict["y"],     (n_macroparticles, n_turns))
    py    = np.reshape(emits_dict["py"],    (n_macroparticles, n_turns))
    z     = np.reshape(emits_dict["zeta"],  (n_macroparticles, n_turns))
    delta = np.reshape(emits_dict["delta"], (n_macroparticles, n_turns))
        
    if normalize:
        x     = x / np.sqrt(beta_x)
        y     = y / np.sqrt(beta_y)
        px    = alpha_x / beta_x * x + beta_x * px
        py    = alpha_y / beta_y * y + beta_y * py   
    
    emit_x = np.sqrt(np.mean(( x -  np.mean(x, axis=0))**2, axis=0) *\
                     np.mean((px - np.mean(px, axis=0))**2, axis=0) -\
                     np.mean(( x -  np.mean(x, axis=0)) *\
                             (px - np.mean(px, axis=0)), axis=0)**2)
        
    emit_y = np.sqrt(np.mean(( y -  np.mean(y, axis=0))**2, axis=0) *\
                     np.mean((py - np.mean(py, axis=0))**2, axis=0) -\
                     np.mean(( y -  np.mean(y, axis=0)) *\
                             (py - np.mean(py, axis=0)), axis=0)**2)
        
    emit_s = np.sqrt(np.mean((    z - np.mean(    z, axis=0))**2, axis=0) *\
                     np.mean((delta - np.mean(delta, axis=0))**2, axis=0) -\
                     np.mean((    z - np.mean(    z, axis=0)) *\
                             (delta - np.mean(delta, axis=0)), axis=0)**2)
        
    return emit_x, emit_y, emit_s

#################
# Preprocessing #
#################

def get_effective_sigmas(beam_params, m_0=cst.m_e):
    """
    From the PhD thesis of D. Schulte
    https://inspirehep.net/files/9fd3c7e47d0ae0272382f7fd6159ab68
    
    Get the effective beam sizes and the disruption parameters.
    """

    def f(x):
        return -(1 + 2 * x**3) / (6 * x**3)
    
    piwi_x = beam_params["sigma_z"]/beam_params["sigma_x"]*np.tan(beam_params["phi"])
    piwi_factor = 1/np.sqrt(1 + piwi_x**2)
    r0 = -beam_params["q_b1"]*beam_params["q_b2"]*cst.e**2/(4*np.pi*cst.epsilon_0*m_0*cst.c**2)
    
    dx = 2*beam_params["bunch_intensity"]*r0*beam_params["sigma_z"] * piwi_factor / (beam_params["gamma"]*beam_params["sigma_x"]*(beam_params["sigma_x"] + beam_params["sigma_y"]))
    dy = 2*beam_params["bunch_intensity"]*r0*beam_params["sigma_z"] * piwi_factor / (beam_params["gamma"]*beam_params["sigma_y"]*(beam_params["sigma_x"] + beam_params["sigma_y"]))
    print("Dx: {}\nDy: {}".format(dx, dy))
    hx = 1 + dx**(.25) * dx**3 / (1 + dx**3) * (np.log(1 + np.sqrt(dx)) + np.log(beam_params["sigma_z"] / (.8 * beam_params["beta_x"])))
    hy = 1 + dy**(.25) * dy**3 / (1 + dy**3) * (np.log(1 + np.sqrt(dy)) + np.log(beam_params["sigma_z"] / (.8 * beam_params["beta_y"])))
    print("Hx: {}\nHy: {}\nf(σ_x/σ_y): {}".format(hx, hy, f(beam_params["sigma_x"] / beam_params["sigma_y"])))
    sigma_x_eff = beam_params["sigma_x"] * hx**(-.5)
    sigma_y_eff = beam_params["sigma_y"] * hy**(f(beam_params["sigma_x"] / beam_params["sigma_y"]))
    
    return sigma_x_eff, sigma_y_eff


###################
# Postprocessing #  
###################

def gauss(x,a,mu,sigma,offset):
    return offset + a*np.exp(-(x-mu)**2/(2*sigma**2))
    
    
def fit_gauss(x_data, y_data, peak, mean, gauss_window=6):

    x = x_data[peak-gauss_window:peak+gauss_window]
    xx = np.linspace( x[0], x[-1], 1000)
    y = y_data[peak-gauss_window:peak+gauss_window]

    # estimate Gaussian curve
    popt, pcov = curve_fit(gauss, x, y, p0=[1, mean, 1e-4, -5.5], maxfev=50000)
    return xx, popt


##########################
# Debugging & optimizing #
##########################

def add_new_value(elements_dict, key, value):
    try:
        elements_dict[key].append(value)
    except:
        elements_dict[key] = [value]

        
def displayMemoryUsage():
    ram_usage = int(psutil.virtual_memory().total - psutil.virtual_memory().available)/ 1024**3
    print(f'Memory usage {ram_usage}GB',flush=True)
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)
    
        
def print_info(turn, text, particles, n_macroparticles, print_to_file=False, overwrite_file=False):
    
    if not print_to_file:
        print(text)
        for ii in range(n_macroparticles):
            print("Macropart {}: x:     {:.10f}".format(    ii, particles.x[ii]))
            print("Macropart {}: px:    {:.10f}".format(   ii, particles.px[ii]))
            print("Macropart {}: y:     {:.10f}".format(    ii, particles.y[ii]))
            print("Macropart {}: py:    {:.10f}".format(   ii, particles.py[ii]))
            print("Macropart {}: z:     {:.10f}".format( ii, particles.zeta[ii]))
            print("Macropart {}: delta: {:.10f}".format(ii, particles.delta[ii]))
    else:  # write info to file
        fname = "outputs/coords_{}.txt".format(text.replace(" ", "_"))
        if overwrite_file:
            try:
                os.remove(fname)
            except OSError:
                pass
            f = open(fname, "w")
        else:
            f = open(fname, "a")
            
        f.write("Turn {}\n{}\n".format(turn, text))
        for ii in range(n_macroparticles):
            f.write("Macropart {}: x:     {:.10f}\n".format(    ii, particles.x[ii]))
            f.write("Macropart {}: px:    {:.10f}\n".format(   ii, particles.px[ii]))
            f.write("Macropart {}: y:     {:.10f}\n".format(    ii, particles.y[ii]))
            f.write("Macropart {}: py:    {:.10f}\n".format(   ii, particles.py[ii]))
            f.write("Macropart {}: z:     {:.10f}\n".format( ii, particles.zeta[ii]))
            f.write("Macropart {}: delta: {:.10f}\n".format(ii, particles.delta[ii]))
        f.write("\n")
        f.close()


def plot_single_part_trajectory(coords_dict, idx, beam=1):
  
    beam_key = "b{}".format(beam)
    
    fig, ax = plt.subplots(3,2, figsize=(24,24))
    
    ax[0,0].plot(    coords_dict[beam_key]["x"][idx])
    ax[0,1].plot(   coords_dict[beam_key]["px"][idx])
    ax[1,0].plot(    coords_dict[beam_key]["y"][idx])
    ax[1,1].plot(   coords_dict[beam_key]["py"][idx])
    ax[2,0].plot(    coords_dict[beam_key]["z"][idx])
    ax[2,1].plot(coords_dict[beam_key]["delta"][idx])
    
    ax[0,0].set_xlabel("Turn [1]")
    ax[0,1].set_xlabel("Turn [1]")
    ax[1,0].set_xlabel("Turn [1]")
    ax[1,1].set_xlabel("Turn [1]")
    ax[2,0].set_xlabel("Turn [1]")
    ax[2,1].set_xlabel("Turn [1]")
    
    ax[0,0].set_ylabel(    "x [m]")
    ax[0,1].set_ylabel(   "px [m]")
    ax[1,0].set_ylabel(    "y [m]")
    ax[1,1].set_ylabel(   "py [m]")
    ax[2,0].set_ylabel(    "z [m]")
    ax[2,1].set_ylabel("delta [1]")
    
    return fig, ax

        
# robustness checks
def test_inputs(params):
    betapereps_threshold = 6
    assert all(np.sqrt(params["beta_x"]/params["physemit_x"]) > betapereps_threshold), "β/ε too small! Decrease ε or increase β!"

##################
# Hirata's boost #
##################

"""
direct Lorentz boost for the weak beam
here no crossing plane angle assumed (α=0)
"""
def boost_hirata(x, y, z, px, py, e, sphi, cphi, tphi, n):


    for i in range(n):
    
        # h = total energy of particle i
        a = ( px[i]**2 + py[i]**2 ) / ( 1 + e[i] )**2
        sqr1a = np.sqrt(1-a)
        h = ( 1 + e[i] ) * a / ( 1 + sqr1a )
        
        # transform momenta
        px[i] = (px[i] - tphi*h) / cphi
        py[i] /= cphi
        e[i] -= sphi*px[i]
        
        # h1d = pz* in pdf
        a1 = ( px[i]**2 + py[i]**2 ) / ( 1 + e[i] )**2
        sqr1a = np.sqrt(1-a1)
        h1d = ( 1 + e[i] ) * sqr1a
        
        # derivatives of transformed Hamiltonian (h1z=-hσ* ??)
        h1x = px[i] / h1d
        h1y = py[i] / h1d
        h1z = a1 / (( 1 + sqr1a ) * sqr1a )
        
        # update coordinates
        x1 = tphi * z[i] + ( 1 + sphi * h1x ) * x[i]
        y[i] += sphi * h1y * x[i]
        z[i] /= cphi - sphi * h1z * x[i]
        x[i] = x1
        

def boost(x, y, z, px, py, e, sphi, cphi, tphi, verbose=False):

    for i in range(len(x)):
            
        # h = total energy of particle i
        h = e[i] + 1 - np.sqrt( (1 + e[i])**2 - px[i]**2 - py[i]**2 )

        # transform momenta
        px_ = (px[i] - h*tphi) / cphi
        py_ = py[i] / cphi
        e_  = e[i] - px[i]*tphi + h*tphi**2
        px[i] = px_
        py[i] = py_
        e[i]  = e_
        
        if verbose:
            print("h:", h, "px_:", px_,"py_:", py_,"e_:",e_)
        
        # pz*
        pz = np.sqrt( (1 + e[i])**2 - px[i]**2 - py[i]**2 )
        
        # derivatives of transformed Hamiltonian (hz=-hσ* ??)
        hx = px[i] / pz
        hy = py[i] / pz
        hs = 1 - ( e[i] + 1 ) / pz
        
        # update coordinates
        x_ = ( 1 + hx * sphi ) * x[i] + tphi * z[i]
        y_ = hy * sphi * x[i] + y[i]
        z_ = hs * sphi * x[i] + z[i] / cphi
        x[i] = x_
        y[i] = y_
        z[i] = z_
        
"""
inverse Lorentz boost
"""
def boost_inv_hirata(x, y, z, px, py, e, sphi, cphi, tphi, n):
    
    for i in range(n):
        
        # h = total energy of particle i
        a1 = ( px[i]**2 + py[i]**2 ) / ( 1 + e[i] )**2
        sqr1a = np.sqrt(1-a1)
        h1d = ( 1 + e[i] ) * sqr1a
        h1 = ( 1 + e[i] ) * a1 / ( 1 + sqr1a )
        h1x = px[i] / h1d
        h1y = py[i] / h1d
        h1z = a1 / (( 1 + sqr1a ) * sqr1a )
        
        # update coordinates
        det = 1 + sphi * ( h1x - sphi * h1z )
        x[i] = ( x[i] - sphi * z[i] ) / det
        z[i] = cphi * ( z[i] + sphi * h1z * x[i] )
        y[i] -= sphi * h1y * x[i]
        
        # transform momenta
        e[i] += sphi * px[i] 
        px[i] = ( px[i] + sphi * h1 ) * cphi
        py[i] *= cphi

        
def boost_inv(x, y, z, px, py, e, sphi, cphi, tphi, verbose=False):
    
    for i in range(len(x)):
        
        # h = total energy of particle i
        pz = np.sqrt( ( 1 + e[i] )**2 - px[i]**2 - py[i]**2 )
        hx = px[i] / pz
        hy = py[i] / pz
        hs = 1 - ( e[i] + 1 ) / pz

        # update coordinates
        det = 1/cphi + ( hx - hs*sphi ) * tphi
        x_ = x[i] / cphi - tphi * z[i]
        y_ = -tphi*hy * x[i] + (1/cphi + tphi*(hx - hs*sphi)) * y[i] + tphi*hy*sphi * z[i]
        z_ = -hs*sphi*x[i] + (1+hx*sphi)*z[i]
        x[i] = x_/det
        y[i] = y_/det
        z[i] = z_/det
        
        
        # transform momenta
        h = ( e[i] + 1 - pz ) * cphi**2
        px_ = px[i]*cphi + h*tphi
        py_ = py[i]*cphi
        e_  = e[i] + px[i]*cphi*tphi
        px[i] = px_
        py[i] = py_ 
        e[i]  = e_
        
        if verbose:
            print("h_:", h, "px__:", px_, "py_:", py, "e__:",e_)
            