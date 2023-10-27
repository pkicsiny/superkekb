import numpy as np
import xfields as xf

def _make_bb_lens(nb, phi, sigma_z, alpha, n_slices, other_beam_q0,
                  sigma_x, sigma_px, sigma_y, sigma_py, beamstrahlung_on=False, compt_x_min=1, binning_mode="unicharge"):

    slicer = xf.TempSlicer(n_slices=n_slices, sigma_z=sigma_z, mode=binning_mode)

    el_beambeam = xf.BeamBeamBiGaussian3D(
            #_context=context,
            config_for_update = None,
            other_beam_q0=other_beam_q0,
            phi=phi, # half-crossing angle in radians
            alpha=alpha, # crossing plane
            # decide between round or elliptical kick formula
            min_sigma_diff = 1e-28,
            # slice intensity [num. real particles] n_slices inferred from length of this
            slices_other_beam_num_particles = slicer.bin_weights * nb,
            # unboosted strong beam moments
            slices_other_beam_zeta_center = slicer.bin_centers,
            slices_other_beam_Sigma_11    = n_slices*[sigma_x**2], # Beam sizes for the other beam, assuming the same is approximation
            slices_other_beam_Sigma_22    = n_slices*[sigma_px**2],
            slices_other_beam_Sigma_33    = n_slices*[sigma_y**2],
            slices_other_beam_Sigma_44    = n_slices*[sigma_py**2],
            # only if BS on
            slices_other_beam_zeta_bin_width_star_beamstrahlung = None if not beamstrahlung_on else slicer.bin_widths_beamstrahlung / np.cos(phi),  # boosted dz
            # has to be set
            slices_other_beam_Sigma_12    = n_slices*[0],
            slices_other_beam_Sigma_34    = n_slices*[0],
            compt_x_min = compt_x_min,
        )
    el_beambeam.iscollective = False # Disable in twiss

    return el_beambeam


def _insert_beambeam_elements(line, bb_def_list, twiss_table, emit):

    print(f"Beam-beam definitions provided, installing beam-beam elements at: {', '.join([bbd['at_element'] for bbd in bb_def_list])}")

    for bb_def in bb_def_list:
        element_name = bb_def['at_element']
 
        # the beam-beam lenses are thin and have no effects on optics so no need to re-compute twiss
        element_twiss_index = list(twiss_table.name).index(element_name)

        # get the line index every time as it changes when elements are installed
        element_line_index = line.element_names.index(element_name)
        sigmas = twiss_table.get_betatron_sigmas(*emit if hasattr(emit, '__iter__') else (emit, emit))

        bb_elem = _make_bb_lens(nb=float(bb_def['bunch_intensity']),
                                phi=float(bb_def['crossing_angle']),
                                sigma_z=float(bb_def['sigma_z']),
                                n_slices=int(bb_def['n_slices']),
                                other_beam_q0=int(bb_def['other_beam_q0']),
                                alpha=bb_def['alpha'], # Put it to zero, it is okay for this use case
                                sigma_x =np.sqrt(sigmas['Sigma11'][element_twiss_index]),
                                sigma_px=np.sqrt(sigmas['Sigma22'][element_twiss_index]),
                                sigma_y =np.sqrt(sigmas['Sigma33'][element_twiss_index]),
                                sigma_py=np.sqrt(sigmas['Sigma44'][element_twiss_index]),
                                beamstrahlung_on=bb_def['beamstrahlung'], compt_x_min=bb_def["compt_x_min"], binning_mode=bb_def["binning_mode"])

        line.insert_element(index=element_line_index,
                            element=bb_elem,
                            name=f'beambeam_{element_name}')
        

def get_num_survived_turns_per_part(mon_atturn):
    """
    For each particle get the maximum number of turns survived i.e. the turn in which it got lost
    :param mon_atturn: numpy array of shape (n_macroparts, n_turns) extracted from the xtrack monitor:
    monitor.data.at_turn
    :return: numpy array of shape (n_macroparts,) containing the turn index in which each particle was lost
    """
    mon_lost_at = []
    for i in range(np.shape(mon_atturn)[0]):
        mon_lost_at.append(np.max(mon_atturn[i]))
    mon_lost_at = np.array(mon_lost_at)
    return mon_lost_at


def compute_contour(dynap_plot, n_j, n_delta, n_turns, verbose_pos=False, verbose_neg=False):
    """
    find dynamic aperture curve
    split array into 2 halves depending on J<0 and J>=0
    top half: start from J=0 (wont be exactly 0 but a small positive) line and start moving up
    and and see when the first particle gets lost earlier than the max. number of turns i.e. the first 0
    bottom half: start from bottom of array (J=-j_max and move up and see where the first 1 occurs
    :param dynap_plot: numpy array of shape (n_delta, n_j) containing the turn indices at which the particle was lost
    :param n_j: int, number of points on the J axis
    :param n_delta: int, number of points on the delta axis
    :param n_turns: int, number of turns simulated
    :return: 2 numpy arrays of shape (n_delta,)
    """

    dynap_binary = np.where(dynap_plot >= n_turns-1, 1, 0)

    # for a given delta find the first 0 in the array
    positive_j = dynap_binary[:,int(n_j/2):]
    negative_j = dynap_binary[:,:int(n_j/2)]
    
    # length of delta_vec
    dynap_curve_pos = []
    dynap_curve_neg = []
    
    for d in range(n_delta):
        
        # starting from j=0 find last occurence of 1 (first occurence of a 0 - 1 + n_delta/2 since this is the upper half)
        last_1_pos = np.where(positive_j[d] == 0)[0][0] - 1 + int(n_j / 2)
        if last_1_pos == int(n_j / 2)-1:
            last_1_pos = None
            
        if verbose_pos:
            print(positive_j[d],last_1_pos)
    
        # starting from -j_max find first occurence of 1 (last occurence of a 0 + 1)
        last_1_neg = np.where(negative_j[d] == 0)[0][-1] + 1
        if last_1_neg == int(n_j / 2):
                last_1_neg = None
                
        if verbose_neg:
            print(negative_j[d],last_1_neg)
    
        dynap_curve_pos.append(last_1_pos)
        dynap_curve_neg.append(last_1_neg)
    
    return dynap_curve_pos, dynap_curve_neg
