import numpy as np
import os
import pandas as pd
import xpart as xp

def stat_emittance(beam, alpha_x=0, alpha_y=0, beta_x=0, beta_y=0):
    """
    05/09/2022: compute statistical emittances. First normalize coordinates by using (263) then (130) from
    https://arxiv.org/pdf/2107.02614.pdf
    """
        
    x_norm     = beam.x / np.sqrt(beta_x)
    y_norm     = beam.y / np.sqrt(beta_y)
    px_norm    = alpha_x / beta_x * beam.x + beta_x * beam.px
    py_norm    = alpha_y / beta_y * beam.y + beta_y * beam.py   
    
    emit_x = np.sqrt(np.mean(( x_norm -  np.mean(x_norm))**2) *\
                     np.mean((px_norm - np.mean(px_norm))**2) -\
                     np.mean(( x_norm -  np.mean(x_norm)) *\
                             (px_norm - np.mean(px_norm)))**2)
        
    emit_y = np.sqrt(np.mean(( y_norm -  np.mean(y_norm))**2) *\
                     np.mean((py_norm - np.mean(py_norm))**2) -\
                     np.mean(( y_norm -  np.mean(y_norm)) *\
                             (py_norm - np.mean(py_norm)))**2)
        
    emit_s = np.sqrt(np.mean(( beam.zeta -  np.mean(beam.zeta))**2) *\
                     np.mean((beam.delta - np.mean(beam.delta))**2) -\
                     np.mean(( beam.zeta -  np.mean(beam.zeta)) *\
                             (beam.delta - np.mean(beam.delta)))**2)
        
    return emit_x, emit_y, emit_s


def record_coordinates(coords_dict, particles, beam_params, turn_idx=-1):
    """
    dimensions of coords_dict expected: (n_turns X n_particles)
    29/10/21: add mean coordinate fields and only store particles equal to the length of the initialized dict.
    25/03/2022: simplify function. Add emittances.
    """
    particles_dict = particles.to_dict()
    
    for var in coords_dict.keys():
        if var in ["x", "px", "y", "py", "zeta", "delta"]:
            coords_dict[var][turn_idx] = particles_dict[var]
        if var == "energy":
            coords_dict[var][turn_idx] = (particles.energy0 + particles.ptau * particles.p0c)
        if var == "emit_x":        
            coords_dict[var][turn_idx] = 0.5*(beam_params["gamma_x"]*particles.x**2 + 2*beam_params["alpha_x"]*particles.x*particles.px + beam_params["beta_x"]*particles.px**2)
        if var == "emit_y":  
            coords_dict[var][turn_idx] = 0.5*(beam_params["gamma_y"]*particles.y**2 + 2*beam_params["alpha_y"]*particles.y*particles.py + beam_params["beta_y"]*particles.py**2)
        if var == "emit_s":  
            coords_dict[var][turn_idx] = 0.5*(particles.zeta**2/beam_params["beta_s"] + beam_params["beta_s"]*particles.delta**2)


def transpose_dict(coords_dict):
    """
    29/10/21: add mean coordinate fileds, just copied from old dict
    """

    coords_dict_transposed = {key_2: [] for key_2 in coords_dict}

    for var in coords_dict_transposed.keys():
        coords_dict_transposed[var] = np.transpose(coords_dict[var], (1,0))

    return coords_dict_transposed


def store_beam(particles, fname="../input_files/beamstrahlung/electron.ini", formatting="guineapig"):
    """
    Store beam coordinates in Guineapig format.
    GP prefers [um], [urad] and [GeV], xsuite prefers [m], [rad] and [eV].
    x,y,z - [m->um]; vx,vy - [rad->urad]; E_tot - [eV->GeV]
    
    Store beam coordinates in BBWS format.
    x [m], px [1], y [m], py [1], z [m], delta [1] 
    """
    
    assert formatting in ["guineapig", "bbws"], "formatting has to be 'guineapig' or 'bbws'!"
    
    # remove existing copy
    try:
        os.remove(fname)
    except OSError:
        pass
    
    f = open(fname, "w")
    
    if formatting == "guineapig":
        # x,y,z - [m->um]; px,py - [rad->urad]; E_tot - [eV->GeV]
        e_tot = (particles.energy0 + particles.ptau*particles.p0c)*1e-9
        x     = particles.x*1e6
        y     = particles.y*1e6
        z     = particles.zeta*1e6
        vx    = particles.px*1e6
        vy    = particles.py*1e6
        text  = ''.join(f'{e_tot_i:.16e} {x_i:.16e} {y_i:.16e} {z_i:.16e} {vx_i:.16e} {vy_i:.16e}\n' for e_tot_i, x_i, y_i, z_i, vx_i, vy_i in zip(e_tot, x, y, z, vx, vy))
    elif formatting == "bbws":
        x     = particles.x
        y     = particles.y
        z     = particles.zeta
        px    = particles.px
        py    = particles.py
        delta = particles.delta
        text  = ''.join(f'{x_i:.16e} {px_i:.16e} {y_i:.16e} {py_i:.16e} {z_i:.16e} {delta_i:.16e}\n' for x_i, px_i, y_i, py_i, z_i, delta_i in zip(x, py, y, py, z, delta))
        
    f.write(text)
    f.close()
    
def load_beam(beam_params, context, b, fname = "../input_files/beamstrahlung/electron.ini"):
    """
    Load beam coordinates stored in Guineapig format.
    GP prefers [um], [urad] and [GeV], xsuite prefers [m], [rad] and [eV].
    x,y,z - [um->m]; px,py - [urad->rad]; E_tot - [GeV->eV]
    """
    # TODO
    coords_df = pd.read_table(fname, delimiter=" ", header=None)
    e_tot = np.array(coords_df[0], dtype=float)*1e9
    x     = np.array(coords_df[1], dtype=float)*1e-6
    y     = np.array(coords_df[2], dtype=float)*1e-6
    z     = np.array(coords_df[3], dtype=float)*1e-6
    px    = np.array(coords_df[4], dtype=float)*1e-6
    py    = np.array(coords_df[5], dtype=float)*1e-6

    # back compute delta from total macropart energy
    pc = np.sqrt(e_tot**2 - beam_params["mass0"]**2)  # macropart kinetic E (pc)
    delta = pc/beam_params["p0c"] - 1
    
    xsuite_particles_object = xp.Particles(
            _context = context, 
            q0       = beam_params["q_b{}".format(b+1)],
            p0c      = beam_params["p0c"],
            mass0    = beam_params["mass0"],
            x        = x,
            px       = px,
            y        = y,
            py       = py,
            zeta     = z,
            delta    = delta,
            )
    xsuite_particles_object._init_random_number_generator()
    return xsuite_particles_object
    
