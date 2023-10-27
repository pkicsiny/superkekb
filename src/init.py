import numpy as np
import xtrack as xt
import xpart as xp
import xfields as xf
import os
import pandas as pd
import input_files.config as config

def load_beam_params(which):
    if which == "z_cdr":
        beam_params = config.fcc_z_cdr.copy()
    elif which == "ww_cdr":
        beam_params = config.fcc_ww_cdr.copy()
    elif which == "zh_cdr":
        beam_params = config.fcc_zh_cdr.copy()
    elif which == "ttbar1_cdr":
        beam_params = config.fcc_ttbar1_cdr.copy()
    elif which == "ttbar2_cdr":
        beam_params = config.fcc_ttbar2_cdr.copy()
    elif which == "z":
        beam_params = config.fcc_z.copy()
    elif which == "w":
        beam_params = config.fcc_w.copy()
    elif which == "h":
        beam_params = config.fcc_h.copy()
    elif which == "t":
        beam_params = config.fcc_t.copy()
    elif which == "skekb_ler":
        beam_params = config.skekb_ler.copy()
    elif which == "skekb_her":
        beam_params = config.skekb_her.copy()
        
    beam_params["sigma_x"]     = np.sqrt(beam_params["physemit_x"]*beam_params["beta_x"])
    beam_params["sigma_px"]    = np.sqrt(beam_params["physemit_x"]/beam_params["beta_x"])
    beam_params["sigma_y"]     = np.sqrt(beam_params["physemit_y"]*beam_params["beta_y"])
    beam_params["sigma_py"]    = np.sqrt(beam_params["physemit_y"]/beam_params["beta_y"])
    
    qx_superperiod = (beam_params["Qx_int"] + beam_params["Qx"]) / beam_params["n_ip"]
    qy_superperiod = (beam_params["Qy_int"] + beam_params["Qy"]) / beam_params["n_ip"]
    
    beam_params["Qx"] = np.round(np.round(qx_superperiod, 3) - int(qx_superperiod), 3)
    beam_params["Qy"] = np.round(np.round(qy_superperiod, 3) - int(qy_superperiod), 3)
    beam_params["Qs"] /= beam_params["n_ip"]
    
    return beam_params


def get_test_coord(x_vec, y_vec):
    for x in x_vec:
        for y in y_vec:
            yield x, y
            

def init_arcs(beam_params, sim_params, n_beams=2, n_arcs=1, precompile=True, damping=False):
    """
    initializes the arc elements
    """
    xtrack_arc = {}
    for b in range(n_beams):
        for s in range(n_arcs):
            
            if damping:
                arc = xt.LinearTransferMatrix(
                    _context=sim_params["context"],
                    beta_x_0             = beam_params["beta_x"],
                    beta_x_1             = beam_params["beta_x"],
                    beta_y_0             = beam_params["beta_y"],
                    beta_y_1             = beam_params["beta_y"],
                    alpha_x_0             = -10,
                    alpha_x_1             = -10,
                    alpha_y_0             = 1000,
                    alpha_y_1             = 1000,
                    Q_x                  = beam_params["Qx"]/n_arcs,
                    Q_y                  = beam_params["Qy"]/n_arcs,
                    Q_s                  = -beam_params["Qs"]/n_arcs,
                    beta_s               = beam_params["beta_s"],
                    damping_rate_x = beam_params["damping_rate_x"],
                    damping_rate_y = beam_params["damping_rate_y"],
                    damping_rate_s = beam_params["damping_rate_s"],
                    equ_emit_x = beam_params["physemit_x"],
                    equ_emit_y = beam_params["physemit_y"],
                    equ_emit_s = beam_params["physemit_s"],
                    )
            else:
                arc = xt.LinearTransferMatrix(
                    _context=sim_params["context"],
                    beta_x_0             = beam_params["beta_x"],
                    beta_x_1             = beam_params["beta_x"],
                    beta_y_0             = beam_params["beta_y"],
                    beta_y_1             = beam_params["beta_y"],
                    Q_x                  = beam_params["Qx"]/n_arcs,
                    Q_y                  = beam_params["Qy"]/n_arcs,
                    Q_s                  = -beam_params["Qs"]/n_arcs,
                    beta_s               = beam_params["beta_s"],
                    )
            xtrack_arc["section{}_b{}".format(s+1, b+1)] = arc
    
    if precompile:
        compile_tmp = xp.Particles(_context=sim_params["context"])
        compile_tmp._init_random_number_generator()
        xtrack_arc["section1_b1"].track(compile_tmp)
        
    return xtrack_arc


def init_particles(beam_params, sim_params, n_beams=2, draw_random=False, save_random=False, input_path="../input_files"):
    """
    initializes the particle ensembles
    """   
    random_numbers = {}
    for b in range(n_beams):
        if draw_random:    
            random_numbers["x_b{}".format(b+1)]     = np.random.randn(beam_params["n_macroparticles_b{}".format(b+1)])
            random_numbers["px_b{}".format(b+1)]    = np.random.randn(beam_params["n_macroparticles_b{}".format(b+1)])
            random_numbers["y_b{}".format(b+1)]     = np.random.randn(beam_params["n_macroparticles_b{}".format(b+1)])
            random_numbers["py_b{}".format(b+1)]    = np.random.randn(beam_params["n_macroparticles_b{}".format(b+1)])
            random_numbers["z_b{}".format(b+1)]     = np.random.randn(beam_params["n_macroparticles_b{}".format(b+1)])
            random_numbers["delta_b{}".format(b+1)] = np.random.randn(beam_params["n_macroparticles_b{}".format(b+1)])
            
            if save_random:
                print("Saving random numbers to: {}".format(input_path))
                np.savetxt(os.path.join(input_path, "random_x_b{}.txt".format(b+1)    ), random_numbers["x_b{}".format(b+1)]    )
                np.savetxt(os.path.join(input_path, "random_px_b{}.txt".format(b+1)   ), random_numbers["px_b{}".format(b+1)]   )
                np.savetxt(os.path.join(input_path, "random_y_b{}.txt".format(b+1)    ), random_numbers["y_b{}".format(b+1)]    )
                np.savetxt(os.path.join(input_path, "random_py_b{}.txt".format(b+1)   ), random_numbers["py_b{}".format(b+1)]   ) 
                np.savetxt(os.path.join(input_path, "random_z_b{}.txt".format(b+1)    ), random_numbers["z_b{}".format(b+1)]    )
                np.savetxt(os.path.join(input_path, "random_delta_b{}.txt".format(b+1)), random_numbers["delta_b{}".format(b+1)])
        else: 
            print("Loading random numbers from: {}".format(input_path))
            random_numbers["x_b{}".format(b+1)]     = np.loadtxt(os.path.join(input_path, "random_x_b{}.txt".format(b+1)    ))
            random_numbers["px_b{}".format(b+1)]    = np.loadtxt(os.path.join(input_path, "random_px_b{}.txt".format(b+1)   ))
            random_numbers["y_b{}".format(b+1)]     = np.loadtxt(os.path.join(input_path, "random_y_b{}.txt".format(b+1)    ))
            random_numbers["py_b{}".format(b+1)]    = np.loadtxt(os.path.join(input_path, "random_py_b{}.txt".format(b+1)   ))
            random_numbers["z_b{}".format(b+1)]     = np.loadtxt(os.path.join(input_path, "random_z_b{}.txt".format(b+1)    ))
            random_numbers["delta_b{}".format(b+1)] = np.loadtxt(os.path.join(input_path, "random_delta_b{}.txt".format(b+1)))

    
    xtrack_particles = {}
    for b in range(n_beams):
        xtrack_particles["b{}".format(b+1)] = xp.Particles(
            _context = sim_params["context"], 
            q0       = beam_params["q_b{}".format(b+1)],
            p0c      = beam_params["p0c"],
            mass0    = beam_params["mass0"],
            x        = beam_params["sigma_x"]    *random_numbers["x_b{}".format(b+1)] ,
            px       = beam_params["sigma_px"]   *random_numbers["px_b{}".format(b+1)],
            y        = beam_params["sigma_y"]    *random_numbers["y_b{}".format(b+1)] ,
            py       = beam_params["sigma_py"]   *random_numbers["py_b{}".format(b+1)],
            zeta     = beam_params["sigma_z"]    *random_numbers["z_b{}".format(b+1)],
            delta    = beam_params["sigma_delta"]*random_numbers["delta_b{}".format(b+1)],
            )
        xtrack_particles["b{}".format(b+1)]._init_random_number_generator()
    return xtrack_particles


def init_beambeam(beam_params, sim_params, n_beams=2, n_ips=1, precompile=True):
    """
    initializes the beambeam elements
    """
    xfields_beambeam = {}
    for b in range(n_beams):
        for i in range(n_ips):
            
            xfields_beambeam["boost_ip{}_b{}".format(i+1, b+1)] = xf.Boost3D(
                _context         = sim_params["context"],
                alpha            = beam_params["alpha"],
                phi              = beam_params["phi"],
                )

            xfields_beambeam["boostinv_ip{}_b{}".format(i+1, b+1)] = xf.BoostInv3D(
                _context         = sim_params["context"],
                alpha            = beam_params["alpha"],
                phi              = beam_params["phi"],
                )
            
            xfields_beambeam["sbc6d_full_ip{}_b{}".format(i+1, b+1)] = xf.Sbc6D_full(
                _context=sim_params["context"],
                n_bb     =np.zeros(sim_params["n_slices"]),  # need to init dynamic arrays to give them a size
                mean_x   =np.zeros(sim_params["n_slices"]),
                mean_xp  =np.zeros(sim_params["n_slices"]),
                mean_y   =np.zeros(sim_params["n_slices"]),
                mean_yp  =np.zeros(sim_params["n_slices"]),
                mean_z   =np.zeros(sim_params["n_slices"]),
                var_x    =np.zeros(sim_params["n_slices"]),
                cov_x_xp =np.zeros(sim_params["n_slices"]),
                cov_x_y  =np.zeros(sim_params["n_slices"]),
                cov_x_yp =np.zeros(sim_params["n_slices"]),
                var_xp   =np.zeros(sim_params["n_slices"]),
                cov_xp_y =np.zeros(sim_params["n_slices"]),
                cov_xp_yp=np.zeros(sim_params["n_slices"]),
                var_y    =np.zeros(sim_params["n_slices"]),
                cov_y_yp =np.zeros(sim_params["n_slices"]),
                var_yp   =np.zeros(sim_params["n_slices"]),
                var_z    =np.zeros(sim_params["n_slices"]),                
                min_sigma_diff     = sim_params["min_sigma_diff"],
                threshold_singular = 1e-28,  # must be larger than 0 but small
                dz=np.zeros(sim_params["n_slices"]),
                )
                
    # pre-compile elements
    if precompile:
        compile_tmp = xp.Particles(_context=sim_params["context"])
        compile_tmp._init_random_number_generator()
        
        xfields_beambeam["boost_ip1_b1"].track(compile_tmp)
        xfields_beambeam["boostinv_ip1_b1"].track(compile_tmp)
        xfields_beambeam["sbc6d_full_ip1_b1"].track(compile_tmp)
        
    return xfields_beambeam


def init_dynap_test_grid(beam_params, sim_params, q0=1, beam=1, bs=0, weight=1):
    beam_params[f"n_macroparticles_b{beam}"] = int(2)
    particles = xp.Particles(
                _context = sim_params["context"],
                q0        = q0,
                p0c       = beam_params[  "p0c"],
                mass0     = beam_params["mass0"],
                x         = [beam_params["sigma_x"]*.01, beam_params["sigma_x"]*.1],
                y         = [beam_params["sigma_y"]*.01, beam_params["sigma_y"]*.1],
                zeta      = [0,0],
                px        = [0,0],
                py        = [0,0],
                delta     = [0,0],
                )

    # grid settings in units of beam sigma
    j_vec     = np.linspace(-sim_params[    "j_max"], sim_params[    "j_max"], sim_params[    "n_j"])
    phi_vec   = np.linspace(                       0, sim_params[  "phi_max"], sim_params[  "n_phi"])[:-1]  # due to periodicity last element is the first (+2pi)
    delta_vec = np.linspace(-sim_params["delta_max"], sim_params["delta_max"], sim_params["n_delta"])

    # loop over deltas and add particles
    for delta_i in delta_vec:
        # add test particles for a given delta
        for j_i in j_vec:

            # round bc of numerical double precision 1e-15
            x_test  = np.round(j_i*np.cos(phi_vec), 14)
            px_test = np.round(j_i*np.sin(phi_vec), 14)

            empty_coord_vec = np.zeros_like(x_test)
            particles = add_test_particle(beam_params, sim_params['context'], particles,
                                                  x     = x_test,
                                                  y     = x_test,
                                                  px    = px_test,
                                                  py    = px_test,
                                                  z     = empty_coord_vec,
                                                  delta = empty_coord_vec+delta_i,
                                                  beam=1, bs=bs, weight=weight)  # this is in sigmas
    return particles


def add_test_particle(beam_params, context, particles, x=0, px=0, y=0, py=0, z=0, delta=0, beam=1, bs=0, weight=1):
    """
    Coordinates in units of rms beam size
    """
    x        = beam_params["sigma_x"]*x
    px       = beam_params["sigma_px"]*px
    y        = beam_params["sigma_y"]*y
    py       = beam_params["sigma_py"]*py
    
    if bs:
        zeta     = beam_params["sigma_z_tot"]*z
        delta    = beam_params["sigma_delta_tot"]*delta
    else:
        zeta     = beam_params["sigma_z"]*z
        delta    = beam_params["sigma_delta"]*delta
        
    # redefine the beam with the new particle added to it
    particles = xp.Particles(
                _context = context, 
                q0       = particles.q0,
                p0c      = particles.p0c[0],
                x        = np.hstack((particles.x,         x)),
                px       = np.hstack((particles.px,       px)),
                y        = np.hstack((particles.y,         y)),
                py       = np.hstack((particles.py,       py)),
                zeta     = np.hstack((particles.zeta,      z)),
                delta    = np.hstack((particles.delta, delta)),
                mass0    = particles.mass0,
                weight   = weight,
                )
    
    if type(x) == np.ndarray:
        beam_params["n_macroparticles_b{}".format(beam)] += np.shape(x)[0]
    else:
        beam_params["n_macroparticles_b{}".format(beam)] += 1

    return particles

    
def store_beam(particles, fname="../input_files/beamstrahlung/electron.ini"):
    """
    Store beam coordinates in Guineapig format.
    GP prefers [um], [urad] and [GeV], xsuite prefers [m], [rad] and [eV].
    x,y,z - [m->um]; px,py - [rad->urad]; E_tot - [eV->GeV]
    """
    # remove existing copy
    try:
        os.remove(fname)
    except OSError:
        pass
    
    # write coordinates
    f = open(fname, "w")
    for i in range(len(particles.x)):
        # x,y,z - [m->um]; px,py - [rad->urad]; E_tot - [eV->GeV]
        x = particles.x[i]*1e6
        y = particles.y[i]*1e6
        z = particles.z[i]*1e6
        px = particles.px[i]*1e6
        py = particles.py[i]*1e6
        e_tot = (particles.energy0 + particles.psigma*particles.p0c*particles.beta0)[i]*1e-9
        f.write("%.16e %.16e %.16e %.16e %.16e %.16e\n"%(e_tot, x, y, z, px, py))
    f.close()
    

def load_beam(beam_params, sim_params, b, fname = "../input_files/beamstrahlung/electron.ini"):
    """
    Load beam coordinates stored in Guineapig format.
    GP prefers [um], [urad] and [GeV], xsuite prefers [m], [rad] and [eV].
    x,y,z - [um->m]; px,py - [urad->rad]; E_tot - [GeV->eV]
    """

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
            _context = sim_params["context"], 
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
    
    return xsuite_particles_object
