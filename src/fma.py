import numpy as np
import os
from scipy import constants as cst
import harmonic_analysis as ha
from matplotlib import pyplot as plt
import pandas as pd

def plot_resonance_lines(axis, min_order=1, max_order=4, label=None, c=None, verbose=False, **kwargs):
    """
    07/03/2022: given a matplotlib.axes._subplots.AxesSubplot object, plots all resonance lines
    in the inclusive range [min_order, max_order] on the canvas.
    :param axis: matplotlib.axes._subplots.AxesSubplot, created e.g. by fig, axes = plt.subplots(...)
    :params min_order, max_order: int, to plot resonance lines from-to this order (inclusive on both ends)
    :param verbose: bool
    """
    
    if c is not None:
        set_colors = False
    else:
        set_colors = True
    
    color_vec = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for order in range(max_order, min_order-1, -1):  # resonance order up to and including N ([1,N])
        if set_colors:
            c = color_vec[(order-1)%len(color_vec)]
        if label == None:
            axis.plot(0,0, c=c, label=int(order))  # just for label
        
        resonance_lines = get_resonance_lines_order_n(order, verbose=verbose)
        for r in resonance_lines:
            axis.axline(r[0], r[1], c=c, **kwargs)
            

def get_resonance_lines_order_n(order_n=1, verbose=False):
    """
    07/03/2022: get a set of line segment endpoints which represent order N resonance lines inside the unit square.
    The resonance lines are defined via the diophantine equation:
    m*Qx + n*Qy = K, where the order of the resonance line is |m|+|n|. 
    m, n and K are all integers.
    For order N resonance, the allowed values for m and n are the integers within the interval [-N, N].
    In total, for order N, there are 2N+1 possible values for m and n.
    If |K| > N, the line will be out of the unit square, so K should be in [-N, N] too; 2N+1 possible values.
    :param order_n: int, order of resonance
    :param verbose: bool
    :return resonance_lines: list of 2 tuples of 2 tuples: ((xmin, ymin) (xmax, ymax))
    where all 4 values are on the boundary of the unit square
    """
    
    assert order_n >= 1, "Order has to be at least 1!"
    pairs_tot = (order_n+1)*4  # total number of (m,n) pairs if m,n are in [-N, N]
    K_vec = np.array(range(-order_n, order_n+1))  # possible values for K, len=2N+1
    
    # base pair: m,n >= 0; (+,+): (m, N-m)
    base_pairs = []  # list of tuples (m,n,K) for m*Qx+n*Qy=K for order N, len=(N+1)*(2N+1)
    for i,j in zip(range(order_n+1), reversed(range(order_n+1))):
        for k in K_vec:
            base_pairs.append((i,j,k))
        
    # make extended pairs (-,+), (+,-), (-,-)
    extended_pairs = base_pairs.copy()  # len=4*(N+1)*(2N+1)
    for p in base_pairs:
        extended_pairs.append((-p[0], p[1], p[2]))
        extended_pairs.append((p[0], -p[1], p[2]))
        extended_pairs.append((-p[0], -p[1], p[2]))
    extended_pairs = sorted(list(set(extended_pairs)))  # use set to remove 0 -0 redundancies (always 4 for each K, so 4*len(K_vec))

    # now len=4(N+1)(2N+1) - 4 = 4N(2N+1)
    if verbose:
        print("order N =", order_n)
        print("Total number of lines: 4(N+1)(2N+1)={}".format(pairs_tot*len(K_vec)))
        print("Trivial redundancies (4(2N+1)={}) removed: 4N(2N+1)={}".format(len(K_vec)*4, len(extended_pairs)))
    
    assert pairs_tot - 4 == len(extended_pairs)/len(K_vec), "Something's fishy"
    
    # loop over tuples (m,n,K), express y = ay*x+by, x = ax*y+bx
    resonance_lines = []
    for p in extended_pairs:
        ay = None
        by = None
        ax = None
        bx = None
        
        try:  # these are diagonal lines
            ay = -p[0]/p[1]  # ay = -m/n
            by =  p[2]/p[1]  # by =  K/n
            ax = -p[1]/p[0]  # ax = -n/m
            bx =  p[2]/p[0]  # bx =  K/m
            (x_min, y_min), (x_max, y_max) = get_segment(ay, by, ax, bx, verbose=verbose)
            
            # dont plot lines that touch corner from outside
            if np.abs(x_min - x_max) >1e-8 and np.abs(y_min - y_max) > 1e-8:
                resonance_lines.append(((x_min, y_min), (x_max, y_max)))  # 2 tuple of 2 tuples
                if verbose:
                    print("plotted (m,n,K):", p, "ay:", ay, "by:", by, "ax:", ax, "bx:", bx, (x_min, y_min), (x_max, y_max))
            elif verbose:
                print("not plotted (m,n,K):", p, "ay:", ay, "by:", by, "ax:", ax, "bx:", bx, (x_min, y_min), (x_max, y_max))

        except:  # these are vertical or horizontal lines
            if p[0] == 0:  # horizontal
                by = p[2]/p[1]
                if by <=1 and by >= 0:
                    resonance_lines.append(((0,by),(1,by)))
                    if verbose:
                        print("plotted (m,n,K):", p, "by:", by, "horizontal")
                elif verbose:
                    print("not plotted (m,n,K):", p, "by:", by, "horizontal")
            elif p[1] == 0:  # vertical
                bx = p[2]/p[0]
                if bx <=1 and bx >= 0:
                    resonance_lines.append(((bx,0),(bx,1)))
                    if verbose:
                        print("plotted (m,n,K):", p, "bx:", bx, "vertical")
                elif verbose:
                    print("not plotted (m,n,K):", p, "bx:", bx, "vertical")
                    
    
    if verbose:
        print("plotted {} lines".format(len(resonance_lines)))
    return resonance_lines


def get_segment(ay, by, ax, bx, verbose=False):
    """
    07/03/2022: get crossing points of unit square for resonance lines
    :params ay, by: float, y=f(x)=ay*x+by
    :params ax, bx: float, x=f-1(y)=ax*y+bx
    :param verbose: bool
    """
    ayby = ay + by
    axbx = ax + bx
    
    left   =      by <=1 and by >= 0  # f(0) in [0,1] ? (crosses left side?)
    right  =  ayby <=1 and ayby >= 0  # f(1) in [0,1] ? (crosses right side?)
    bottom =      bx <=1 and bx >= 0  # f-1(0) in [0,1] ? (crosses bottom side?)
    top    = axbx <= 1 and axbx >= 0  # just for corner cases; (crosses top side?)
    
    # get (x1, y1) (x2, y2)
    if left:  # crosses left
        x1_y1 = (0, by)
        if right:  # crosses left and right
            x2_y2 = (1, ayby)
        else:  # crosses left and not right
            if bottom:  # crosses left and bottom (can be corner)
                x2_y2 = (bx, 0)
                if top and x1_y1==x2_y2:  # left bottom corner in, need top out
                    x2_y2 = (axbx, 1) 
                    if verbose:
                        print("l r b t", left, right, bottom, top)
            else:  # crosses left and top and not bottom
                x2_y2 = (axbx, 1)
    else:
        if right:  # crosses right and not left
            x1_y1 = (1, ayby)
            if bottom:  # crosses right and bottom and not left (can be corner)
                x2_y2 = (bx, 0)
                if top and x1_y1==x2_y2:  # right bottom corner out, need top in
                    x2_y2 = (axbx, 1)
            else:  # crosses right and not left and not bottom
                x2_y2 = (axbx, 1)
        else:
            if bottom:  # crosses bottom and not left and not right
                x1_y1 = (bx, 0)
                x2_y2 = (axbx, 1) 
            else:  # outside unit square
                x1_y1 = (None, None)
                x2_y2 = (None, None)

    if verbose:
        print("l r b t", left, right, bottom, top)
    return x1_y1, x2_y2


def carthesian_to_action_angle(x, px, beta_x=0, alpha_x=0):
    """
    conversion from x,px to j,phi for one pair of coordinates or numpy array of coordinates
    :param x, px: float or numpy array containing particle coordinates
    """
    phi_x = np.arctan(-beta_x*px/x-alpha_x)
    j_x = 0.5*x**2/(beta_x*np.cos(phi_x)**2)
    return j_x, phi_x


def compute_dq_anal(beam_params, yokoya=1.3, m_0=cst.m_e, sigma_z_key="sigma_z_tot"):

    tunes = {}
    
    # particle radius
    r0 = -beam_params["q_b1"]*beam_params["q_b2"]*cst.e**2/(4*np.pi*cst.epsilon_0*m_0*cst.c**2) # - if pp
    
    # geometric reduction factor, piwinski angle
    phi_x = np.arctan(np.tan(beam_params["phi"])*np.cos(beam_params["alpha"]))
    phi_y = np.arctan(np.tan(beam_params["phi"])*np.sin(beam_params["alpha"]))
    
    piwi_x = beam_params[sigma_z_key]/beam_params["sigma_x"]*np.tan(phi_x)
    piwi_y = beam_params[sigma_z_key]/beam_params["sigma_y"]*np.tan(phi_y)
    
    geometric_factor_x = np.sqrt(1 + piwi_x**2)
    geometric_factor_y = np.sqrt(1 + piwi_y**2)
    
    # get exact ξ with formula, when far from res. it is the tune shift (to incoherent mode) for each parameter in parameter scan
    tunes["xi_x"] = beam_params["bunch_intensity"]*beam_params["beta_x"]*r0 / (2*np.pi*beam_params["gamma"]) / \
    (beam_params["sigma_x"]*geometric_factor_x* \
    (beam_params["sigma_x"]*geometric_factor_x + beam_params["sigma_y"]*geometric_factor_y))
    
    tunes["xi_y"] = beam_params["bunch_intensity"]*beam_params["beta_y"]*r0 / (2*np.pi*beam_params["gamma"]) / \
    (beam_params["sigma_y"]*geometric_factor_y* \
    (beam_params["sigma_x"]*geometric_factor_x + beam_params["sigma_y"]*geometric_factor_y))
    
    print("xi_x: {}\nxi_y: {}".format(tunes["xi_x"], tunes["xi_y"]))
    
    # get analytical incoherent tune, plug in exact ξ from previous
    if beam_params["Qx"]-int(beam_params["Qx"]) <.5:
        tunes["qx_i_anal"] = (np.arccos(np.cos(2*np.pi*beam_params["Qx"]) - 2*np.pi*tunes["xi_x"]*np.sin(2*np.pi*beam_params["Qx"])))/(2*np.pi)
    else:
        tunes["qx_i_anal"] = 1 - (np.arccos(np.cos(2*np.pi*beam_params["Qx"]) - 2*np.pi*tunes["xi_x"]*np.sin(2*np.pi*beam_params["Qx"])))/(2*np.pi)
        
    if beam_params["Qy"]-int(beam_params["Qy"]) <.5:
        tunes["qy_i_anal"] = (np.arccos(np.cos(2*np.pi*beam_params["Qy"]) - 2*np.pi*tunes["xi_y"]*np.sin(2*np.pi*beam_params["Qy"])))/(2*np.pi)
    else:
        tunes["qy_i_anal"] = 1 - (np.arccos(np.cos(2*np.pi*beam_params["Qy"]) - 2*np.pi*tunes["xi_y"]*np.sin(2*np.pi*beam_params["Qy"])))/(2*np.pi)
        
    # get analytical tune shift (equals ξ when far from resonance)
    tunes["dqx_anal"] = tunes["qx_i_anal"] - beam_params["Qx"]
    tunes["dqy_anal"] = tunes["qy_i_anal"] - beam_params["Qy"]
    
    # analytical pi mode corrected by yokoya
    tunes["qx_pi_anal"] = beam_params["Qx"]+yokoya*tunes["dqx_anal"]
    tunes["qy_pi_anal"] = beam_params["Qy"]+yokoya*tunes["dqy_anal"]

    #print("Q_x_i: {}\nQ_y_i: {}".format(tunes["qx_i_anal"], tunes["qy_i_anal"]))
    #print("dQ_x: {}\ndQ_y: {}".format(tunes["dqx_anal"], tunes["dqy_anal"]))
    #print("Q_x_pi: {}\nQ_y_pi: {}".format(tunes["qx_pi_anal"], tunes["qy_pi_anal"]))
    
    return tunes

def normalize_phase_space(coords_buffer, alpha_x=0, alpha_y=0, beta_x=0, beta_y=0):
    """
    Normalizes phase space coordinates according to (263) in https://arxiv.org/pdf/2107.02614.pdf
    :param coords_buffer: dict or pandas dataframe. If df each column is a series of length n_turns. 
    If dict each value is an np array of shape (n_macroparticles, n_turns)
    This works only without coupling. With coupling it shoud be 6D, including distpersion.
    In general its the solution of eigenproblem, eigenvectors define normal modes: build normalization matrix.
    paper hirata oide ohmi 1999 envelope: transport matrix from tracking. 1e-4 sigma init x,y track single part, very small amplitude, can assume linearity, can find linear matrix: track 1 turn with test part with all 0, 2nd turn 1e-3*sigma_x, rest 0 track a second turn, subtract result of turn 1, then 1e-4*sigma_px - turn 1: not a closed eq. orbit
    """
    
    key_x, key_y, key_px, key_py = infer_buffer_type(coords_buffer)
    
    coords_buffer["px_norm"] = alpha_x/np.sqrt(beta_x)*coords_buffer[key_x] + np.sqrt(beta_x)*coords_buffer[key_px]
    coords_buffer["py_norm"] = alpha_y/np.sqrt(beta_y)*coords_buffer[key_y] + np.sqrt(beta_y)*coords_buffer[key_py]
    coords_buffer["x_norm"]  = coords_buffer[key_x]/np.sqrt(beta_x)
    coords_buffer["y_norm"]  = coords_buffer[key_y]/np.sqrt(beta_y)
    return coords_buffer


def infer_buffer_type(coords_buffer):
    
    # pattern of buffer keys. Key of y, px, py are inferred from this by replacing 'x'.
    if isinstance(coords_buffer, dict):
        key_x = "x"
    elif isinstance(coords_buffer, pd.DataFrame):
        key_x = "x_mean"
    else:
        raise TypeError("coords_buffer must be a dict or pandas DataFrame.")
    key_y  = key_x.replace("x","y")
    key_px = key_x.replace("x","px")
    key_py = key_x.replace("x","py")  
    return key_x, key_y, key_px, key_py
        

def fma_dump(coords_buffer, qx, qy, n_macroparts, n_turns, tunes, fname_idx=0, window=3, laskar=True, laskar_n_peaks=2, out_path="../outputs/fma", out_name=None, alpha_x=0, alpha_y=0, beta_x=0, beta_y=0):
    """
    :param coords_buffer: dict or pandas dataframe. If df each column is a series of length n_turns. 
    If dict each value is an np array of shape (n_macroparticles, n_turns)
    """
        
    #normalize    
    if beta_x > 0:
        coords_buffer = normalize_phase_space(coords_buffer, alpha_x=alpha_x, alpha_y=alpha_y, beta_x=beta_x, beta_y=beta_y)
        key_x, key_y, key_px, key_py = "x_norm", "y_norm", "px_norm", "py_norm"
    else:
        key_x, key_y, key_px, key_py = infer_buffer_type(coords_buffer)      
        
    print("Computing tune spectra...")
    q_rel, fft_x_single_part, fft_y_single_part, qx_i_sim, qy_i_sim = fma(coords_buffer, n_macroparts, n_turns,
                                                                      qx, qy,
                                                                      tunes["qx_i_anal"],
                                                                      tunes["qy_i_anal"],
                                                                      window=window, 
                                                                      laskar=laskar,
                                                                      laskar_n_peaks=laskar_n_peaks,
                                                                      key_x=key_x, key_y=key_y)
    # write running tunes
    print(f"Saving incoherent tunes to {out_path}")
    if out_name is None:
        fname = os.path.join(out_path, "q_i_sim_{}.txt".format(np.char.zfill(str(fname_idx),3)))
    else:
        fname = os.path.join(out_path, out_name)
    np.savetxt(fname, np.c_[qx_i_sim, qy_i_sim], header="qx_i_sim qy_i_sim")
    

def fma_simple(coords_dict, n_macroparts, n_turns, key_x="x", key_y="y"):
    """
    :param coords_dict: dict, each value is an np array of shape (n_macroparticles, n_turns)
    """    

    coords_x = np.reshape(coords_dict[key_x], (n_macroparts, n_turns))
    coords_y = np.reshape(coords_dict[key_y], (n_macroparts, n_turns))
    
    length = n_macroparts  # number of macroparticles
    fft_resolution = n_turns  # fft resolution is equal to the number of time samples
    fft_x_single_part = np.zeros((length, fft_resolution))  # tune spectrum x
    fft_y_single_part = np.zeros((length, fft_resolution))  # tune spectrum y
    qxi_sim_laskar = np.zeros((length))  # incoherent tune peak x
    qyi_sim_laskar = np.zeros((length))  # incoherent tune peak y
    q_rel = np.fft.fftshift(np.fft.fftfreq(fft_resolution))  # fft x axis
    
    # get spectrum of all particle trajectories
    for part_i in range(length):
        fft_x_single_part[part_i]  = np.log10(np.abs(np.fft.fftshift(np.fft.fft(coords_x[part_i]))))
        fft_y_single_part[part_i]  = np.log10(np.abs(np.fft.fftshift(np.fft.fft(coords_y[part_i]))))
    
        # find shifted incoherent tune
        # better approximation with Laskar frequency analysis: https://link.springer.com/content/pdf/10.1007/BF00699731.pdf
        fft_harpy_x_single_part = ha.HarmonicAnalysis(coords_x[part_i])
        fft_harpy_y_single_part = ha.HarmonicAnalysis(coords_y[part_i])
        f_x_single_part, coeff_x_single_part = fft_harpy_x_single_part.laskar_method(1)
        f_y_single_part, coeff_y_single_part = fft_harpy_y_single_part.laskar_method(1)
        qx_i_sim[part_i] = f_x_single_part[0]
        qy_i_sim[part_i] = f_y_single_part[0]
        
    return q_rel, fft_x_single_part, fft_y_single_part, qx_i_sim, qy_i_sim  # these arrays of length n_macroparts


def fma(coords_dict, n_macroparts, n_turns, qx, qy, qx_i_anal, qy_i_anal, window=3, laskar=True, laskar_n_peaks=2, key_x="x", key_y="y"):
    """
    :param coords_dict: dict, each value is an np array of shape (n_macroparticles, n_turns)
    :param window: int, search for peak in this neighborhood of analytical pi mode
    :param shift: if true, the fft x axis will be shifted; use for tunes between 0-0.5. Use False for tunes between .5-1.
    """    
    
    coords_x = np.reshape(coords_dict[key_x], (n_macroparts, n_turns))
    coords_y = np.reshape(coords_dict[key_y], (n_macroparts, n_turns))
    
    length = n_macroparts  # number of macroparticles
    fft_resolution = n_turns  # fft resolution is equal to the number of time samples
    fft_x_single_part = np.zeros((length, fft_resolution))  # tune spectrum x
    fft_y_single_part = np.zeros((length, fft_resolution))  # tune spectrum y
    qx_i_sim = np.zeros((length))  # incoherent tune peak x
    qy_i_sim = np.zeros((length))  # incoherent tune peak y
    
    if qx_i_anal>.5:
        q_rel = np.fft.fftshift(np.fft.fftfreq(fft_resolution))  # fft x axis
    else:
        q_rel = np.fft.fftfreq(fft_resolution)  # fft x axis

    # get spectrum of all particle trajectories
    for part_i in range(length):
        
        if qx_i_anal>.5:
            fft_x_single_part[part_i]  = np.log10(np.abs(np.fft.fft(coords_x[part_i])))
        else:
            fft_x_single_part[part_i]  = np.log10(np.abs(np.fft.fftshift(np.fft.fft(coords_x[part_i]))))

        if qy_i_anal>.5:
            fft_y_single_part[part_i]  = np.log10(np.abs(np.fft.fft(coords_y[part_i])))
        else:
            fft_y_single_part[part_i]  = np.log10(np.abs(np.fft.fftshift(np.fft.fft(coords_y[part_i]))))
            
        # find shifted incoherent tune
        if laskar:
            
            # better approximation with Laskar frequency analysis: https://link.springer.com/content/pdf/10.1007/BF00699731.pdf
            fft_harpy_x_single_part = ha.HarmonicAnalysis(coords_x[part_i])
            fft_harpy_y_single_part = ha.HarmonicAnalysis(coords_y[part_i])
            f_x_single_part, coeff_x_single_part = fft_harpy_x_single_part.laskar_method(laskar_n_peaks)
            f_y_single_part, coeff_y_single_part = fft_harpy_y_single_part.laskar_method(laskar_n_peaks)

            coeff_x_single_part = np.array(coeff_x_single_part)[(np.array(f_x_single_part)>=qx*.95) & (np.array(f_x_single_part)<=qx_i_anal*1.05)]
            coeff_y_single_part = np.array(coeff_y_single_part)[(np.array(f_y_single_part)>=qy*.95) & (np.array(f_y_single_part)<=qy_i_anal*1.05)]
            f_x_single_part = np.array(f_x_single_part)[(np.array(f_x_single_part)>=qx*.95) & (np.array(f_x_single_part)<=qx_i_anal*1.05)]
            f_y_single_part = np.array(f_y_single_part)[(np.array(f_y_single_part)>=qy*.95) & (np.array(f_y_single_part)<=qy_i_anal*1.05)]
            
            #print(part_i, f_y_single_part, np.abs(coeff_y_single_part), "\n")

            # take peak that is closest to the analytical incoherent tune computed with formulas outside
            if len(f_x_single_part)>0:
                qx_i_sim[part_i] = f_x_single_part[np.argmax(np.abs(coeff_x_single_part))]
                #qx_i_sim[part_i] = f_x_single_part[np.argmin(np.abs(np.array(f_x_single_part)-(qx_i_anal)))]
            else: 
                qx_i_sim[part_i] = 0
            if len(f_y_single_part)>0:
                qy_i_sim[part_i] = f_y_single_part[np.argmax(np.abs(coeff_y_single_part))]
                #qy_i_sim[part_i] = f_y_single_part[np.argmin(np.abs(np.array(f_y_single_part)-(qy_i_anal)))]
            else:
                qy_i_sim[part_i] = 0 
                
        else:
            
            # conversion from tune value to fft channel idx
            qx_i_anal_idx_in_fft = (qx_i_anal*fft_resolution + fft_resolution/2)
            qy_i_anal_idx_in_fft = (qy_i_anal*fft_resolution + fft_resolution/2)
        
            # take peak that is closest to the analytical incoherent tune computed with formulas outside
            qx_i_sim_idx_in_fft = int(qx_i_anal_idx_in_fft)-window + np.argmax(fft_x_single_part[part_i][int(qx_i_anal_idx_in_fft)-window:int(qx_i_anal_idx_in_fft)+window])
            qy_i_sim_idx_in_fft = int(qy_i_anal_idx_in_fft)-window + np.argmax(fft_y_single_part[part_i][int(qy_i_anal_idx_in_fft)-window:int(qy_i_anal_idx_in_fft)+window])

            # simulated incoherent tune from vanilla fft
            qx_i_sim[part_i] = q_rel[qx_i_sim_idx_in_fft]
            qy_i_sim[part_i] = q_rel[qy_i_sim_idx_in_fft]

    return q_rel, fft_x_single_part, fft_y_single_part, qx_i_sim, qy_i_sim  # these arrays of length n_macroparts


def fma_coherent(coords_buffer, n_macroparts, n_turns, qx, qy, qx_pi_anal, qy_pi_anal, window=3, laskar=True, laskar_n_peaks=2, key_x="x", key_y="y"):
    """
    :param coords_buffer: dict or pandas dataframe. If df each column is a series of length n_turns. 
    If dict each value is an np array of shape (n_macroparticles, n_turns)
    :param window: int, search for peak in this neighborhood of analytical pi mode
    """

    if isinstance(coords_buffer, dict):
        coords_x = np.reshape(coords_buffer[key_x], (n_macroparts, n_turns))
        coords_y = np.reshape(coords_buffer[key_y], (n_macroparts, n_turns))
          
        # calculate beam mean coordinates over turns
        mean_coords_x = np.mean(coords_x, axis=0)
        mean_coords_y = np.mean(coords_y, axis=0)
    elif isinstance(coords_buffer, pd.DataFrame):
        mean_coords_x = np.array(coords_buffer[key_x])
        mean_coords_y = np.array(coords_buffer[key_y])
    else:
        raise TypeError("coords_buffer must be a dict or pandas DataFrame.")
        
    length = n_macroparts  # number of macroparticles
    fft_resolution = n_turns  # fft resolution is equal to the number of time samples

    if qx_pi_anal>.5:
        q_rel = np.fft.fftshift(np.fft.fftfreq(fft_resolution))  # fft x axis
    else:
        q_rel = np.fft.fftfreq(fft_resolution)  # fft x axis
        
    # get coherent spectrum
    if qx_pi_anal>.5:
        fft_x_mean  = np.log10(np.abs(np.fft.fft(mean_coords_x)))
    else:
        fft_x_mean  = np.log10(np.abs(np.fft.fftshift(np.fft.fft(mean_coords_x))))

    if qy_pi_anal>.5:
        fft_y_mean  = np.log10(np.abs(np.fft.fft(mean_coords_y)))   
    else:
        fft_y_mean  = np.log10(np.abs(np.fft.fftshift(np.fft.fft(mean_coords_y))))

    # find coherent pi mode
    if laskar:
        
        # better approximation with Laskar frequency analysis: https://link.springer.com/content/pdf/10.1007/BF00699731.pdf
        fft_harpy_x_mean = ha.HarmonicAnalysis(mean_coords_x)
        fft_harpy_y_mean = ha.HarmonicAnalysis(mean_coords_y)
        f_x_mean, coeff_x_mean = fft_harpy_x_mean.laskar_method(laskar_n_peaks)  # find laskar_n_peaks biggest peaks (sigma and pi, both sides)
        f_y_mean, coeff_y_mean = fft_harpy_y_mean.laskar_method(laskar_n_peaks)
        
        f_x_mean = np.array(f_x_mean)[(np.array(f_x_mean)>=qx*.95) & (np.array(f_x_mean)<=qx_pi_anal*1.05)]
        f_y_mean = np.array(f_y_mean)[(np.array(f_y_mean)>=qy*.95) & (np.array(f_y_mean)<=qy_pi_anal*1.05)]
        
        # take peak that is closest to the analytical pi mode computed with formulas outside
        qx_pi_sim = f_x_mean[np.argmin(np.abs(np.array(f_x_mean)-qx_pi_anal))]
        qy_pi_sim = f_y_mean[np.argmin(np.abs(np.array(f_y_mean)-qy_pi_anal))]
    else:
        
        # conversion from tune value to fft channel idx
        qx_pi_anal_idx_in_fft = (qx_pi_anal*fft_resolution + fft_resolution/2)
        qy_pi_anal_idx_in_fft = (qy_pi_anal*fft_resolution + fft_resolution/2)
        
        # take peak that is closest to the analytical pi mode computed with formulas outside
        qx_pi_sim_idx_in_fft = int(qx_pi_anal_idx_in_fft)-window + np.argmax(fft_x_mean[int(qx_pi_anal_idx_in_fft)-window:int(qx_pi_anal_idx_in_fft)+window])
        qy_pi_sim_idx_in_fft = int(qy_pi_anal_idx_in_fft)-window + np.argmax(fft_y_mean[int(qy_pi_anal_idx_in_fft)-window:int(qy_pi_anal_idx_in_fft)+window])
        
        # simulated pi mode from vanilla fft
        qx_pi_sim = q_rel[qx_pi_sim_idx_in_fft]
        qy_pi_sim = q_rel[qy_pi_sim_idx_in_fft]
        
    return q_rel, fft_x_mean, fft_y_mean, qx_pi_sim, qy_pi_sim


