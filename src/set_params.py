import xobjects as xo
import numpy as np
import os
import xtrack as xt
import json

def set_binning_mode(sim_params, arg_binning_mode):
    print("[exec.py] setting 'binning_mode'...")
    if arg_binning_mode == 0:
        sim_params["binning_mode"] = "unibin"
    elif arg_binning_mode == 1:
        sim_params["binning_mode"] = "unicharge"
    elif arg_binning_mode == 2:
        sim_params["binning_mode"] = "shatilov"
    else:
        raise ValueError(f"Wrong value for 'arg_binning_mode': {arg_binning_mode}!")
    print(f"[exec.py] successfully set 'binning_mode' to: {sim_params['binning_mode']}")


def set_physemit_y(beam_params, arg_physemit_y, none_tag=-999):
    print("[exec.py] setting 'physemit_y'...")
    if arg_physemit_y != none_tag:
        beam_params["physemit_y"] = arg_physemit_y
    else:
        print("[exec.py] using default value...")
    print(f"[exec.py] successfully set 'physemit_y': {beam_params['physemit_y']:.4e} [m]")


def set_phi(beam_params, arg_phi, none_tag=-999):
    print("[exec.py] setting 'phi'...")
    if arg_phi != none_tag:
        beam_params["phi"] = arg_phi
    else:
        print("[exec.py] using default value...")
    print(f"[exec.py] successfully set 'phi': {beam_params['phi']:.4e} [rad]")


def set_bunch_intensity(beam_params, arg_bunch_intensity, none_tag=-999):
    print("[exec.py] setting 'bunch_intensity'...")
    if arg_bunch_intensity != none_tag:
        if arg_bunch_intensity >= 0:
            beam_params["bunch_intensity"] = int(arg_bunch_intensity)
        else:
            beam_params["bunch_intensity"] *= np.abs(arg_bunch_intensity)
    else:
        print("[exec.py] using default value...")
    print(f"[exec.py] successfully set 'bunch_intensity': {beam_params['bunch_intensity']:.4e} [e]")


def set_sigma_z_sim_sigma_delta_sim(beam_params, sim_params):
    """
    WS only. Requires: arg_bs, arg_sigma_z, arg_sigma_delta, arg_sigma_z_tot, arg_sigma_delta_tot
    """
    print("[exec.py] setting weak bunch longitudinal parameters based on beamstrahlung flag...")
    assert "flag_beamstrahlung" in  sim_params, "'flag_beamstrahlung' not in 'sim_params'!"

    if sim_params["flag_beamstrahlung"] == 0:
        assert     "sigma_z" in beam_params, "'sigma_z' not in 'beam_params'!"
        assert "sigma_delta" in beam_params, "'sigma_delta' not in 'beam_params'!"

        sim_params[    "sigma_z_sim"] = beam_params[    "sigma_z"]
        sim_params["sigma_delta_sim"] = beam_params["sigma_delta"]
    else:
        assert     "sigma_z_tot" in beam_params, "'sigma_z_tot' not in 'beam_params'!"
        assert "sigma_delta_tot" in beam_params, "'sigma_delta_tot' not in 'beam_params'!"

        sim_params[    "sigma_z_sim"] = beam_params[    "sigma_z_tot"]
        sim_params["sigma_delta_sim"] = beam_params["sigma_delta_tot"]
    print(f"[exec.py] successfully set 'sigma_z_sim': {sim_params['sigma_z_sim']} [m], 'sigma_delta_sim': {sim_params['sigma_delta_sim']} [1]")


def set_n_slices(beam_params, sim_params, arg_n_slices, none_tag=-999):
    """
    Requires: arg_phi, arg_sigma_z_tot, arg_beta_y, arg_sigma_x
    """
    print("[exec.py] setting 'n_slices'...")
    if arg_n_slices != none_tag:
        sim_params["n_slices"] = arg_n_slices
    else:
        print("[exec.py] calculating 'n_slices' from interaction length...")
        assert         "phi" in beam_params, "'phi' not in 'beam_params'!"
        assert "sigma_z_tot" in beam_params, "'sigma_z_tot' not in 'beam_params'!"
        assert      "beta_y" in beam_params, "'beta_y' not in 'beam_params'!"
        assert     "sigma_x" in beam_params, "'sigma_x' not in 'beam_params'!"

        interaction_length = beam_params["sigma_z_tot"] / np.sqrt(1 + (beam_params["sigma_z_tot"] / beam_params["sigma_x"] * np.tan(beam_params["phi"]))**2)
        sim_params['n_slices'] = int(max(100, 100*round(10*beam_params["sigma_z_tot"]/(min(interaction_length, beam_params["beta_y"])) / 100)))
    print(f"[exec.py] successfully set 'n_slices': {sim_params['n_slices']}")


def set_context(sim_params):
    """
    Requires: arg_n_slices, arg_n_threads
    """
    print("[exec.py] setting 'context'...")
    assert "n_threads" in sim_params, "'n_threads' not in 'sim_params'!"

    if sim_params["n_threads"] >= 0:
        sim_params["context"] = xo.ContextCpu(omp_num_threads=sim_params["n_threads"])
        print(f"[exec.py] successfully set 'context': CPU with {sim_params['n_threads']} threads")
    else:
        assert  "n_slices" in sim_params, "'n_slices' not in 'sim_params'!"

        sim_params["context"] = xo.ContextCupy()
        print(f"[exec.py] successfully set 'context': cupy")


def load_ebe_lattice(seq_name, path="/afs/cern.ch/work/p/pkicsiny/private/git/madx_lattice"):
    seq_path = os.path.join(path, seq_name)
    
    print(f"[exec.py] loading MAD-X lattice: {seq_path}")
    with open(seq_path, 'r') as f:
        line = xt.Line.from_dict(json.load(f))
    print("[exec.py] successfully loaded MAD-X lattice")
    return line

def set_dynap_delta(beam_params, arg_dynap_delta, none_tag=-999):
    print("[exec.py] setting 'dynap_delta'...")
    if arg_dynap_delta != none_tag:
        beam_params["dynap_delta"] = arg_dynap_delta
    else:
        print("[exec.py] using default value...")
    print(f"[exec.py] successfully set 'dynap_delta': {beam_params['dynap_delta']} [1]")


def set_tunes(beam_params, arg_qx, arg_qy, n_ip=2, none_tag=-999):
    """
    Requires: arg_qx, arg_qx_int, arg_qy, arg_qy_int, arg_qs
    """
    print("[exec.py] setting 'Qxys'...")
    if arg_qx != none_tag:
        beam_params["Qx"] = arg_qx
    else:
        print(f"[exec.py] using 'Qx' superperiod for {n_ip} IPs...")
        assert     "Qx" in beam_params, "'Qx' not in 'beam_params'!"
        assert "Qx_int" in beam_params, "'Qx_int' not in 'beam_params'!"

        qx_superperiod = (beam_params["Qx_int"] + beam_params["Qx"]) / n_ip
        beam_params["Qx"] = np.round(np.round(qx_superperiod, 3) - int(qx_superperiod), 3)

    if arg_qy != none_tag:
        beam_params["Qy"] = arg_qy
    else:
        print(f"[exec.py] using 'Qy' superperiod for {n_ip} IPs...")
        assert     "Qy" in beam_params, "'Qy' not in 'beam_params'!"
        assert "Qy_int" in beam_params, "'Qy_int' not in 'beam_params'!"

        qy_superperiod = (beam_params["Qy_int"] + beam_params["Qy"]) / n_ip
        beam_params["Qy"] = np.round(np.round(qy_superperiod, 3) - int(qy_superperiod), 3)

    print(f"[exec.py] using 'Qs' superperiod for {n_ip} IPs...")
    assert "Qs" in beam_params, "'Qs' not in 'beam_params'!"

    beam_params["Qs"] /= n_ip

    print(f"[exec.py] successfully set 'Qx': {beam_params['Qx']:.4e} [1], 'Qy': {beam_params['Qy']:.4e} [1], 'Qs': {beam_params['Qs']:.4e} [1]")


def set_damping_rates_lattice_emits_rf_energy(beam_params, sim_params, arg_rf_energy, n_ip=2, none_tag=-999):
    """
    Requires: arg_sr, arg_energy, arg_physemit_x, arg_physemit_y, arg_physemit_s, arg_u_sr, arg_u_bs
    """
    print("[exec.py] setting 'arc_damping_rate_xys', 'arc_physemit_xys' and 'arc_rf_energy'...")
    assert "flag_synrad" in  sim_params, "'flag_synrad' not in 'sim_params'!"

    if sim_params['flag_synrad'] > 0:
        assert       "U_SR" in beam_params, "'U_SR' not in 'beam_params'!"
        assert     "energy" in beam_params, "'energy' not in 'beam_params'!"
        assert "physemit_x" in beam_params, "'physemit_x' not in 'beam_params'!"
        assert "physemit_y" in beam_params, "'physemit_y' not in 'beam_params'!"
        assert "physemit_s" in beam_params, "'physemit_s' not in 'beam_params'!"

        # 1 1 2, damping equilibrium parameters, from CDR with SR wo BS
        # line segment map uses damping rate of emittance
        # damping rate of emittance is twice that of sigma
        # divide by number of ips
        sim_params['arc_damping_rate_s'] = 2 * beam_params["U_SR"]/beam_params["energy"] / n_ip # ~ 1e-3 [GeV]                                                 
        sim_params['arc_damping_rate_x'] = sim_params['arc_damping_rate_s'] / 2.0 # should be same as y, half it for half turn
        sim_params['arc_damping_rate_y'] = sim_params['arc_damping_rate_s'] / 2.0
        sim_params[    'arc_physemit_x'] = beam_params["physemit_x"]
        sim_params[    'arc_physemit_y'] = beam_params["physemit_y"]
        sim_params[    'arc_physemit_s'] = beam_params["physemit_s"] # only here i need sigma_z delta SR
        if arg_rf_energy != none_tag:
            sim_params['arc_rf_energy'] = arg_rf_energy
        else:
            assert "U_BS" in beam_params, "'U_BS' not in 'beam_params'!"

            sim_params['arc_rf_energy'] = beam_params["U_BS"]*1e9 / n_ip  # [eV]
    else:
        sim_params['arc_damping_rate_x'] = 0
        sim_params['arc_damping_rate_y'] = 0
        sim_params['arc_damping_rate_s'] = 0
        sim_params[    'arc_physemit_x'] = 0
        sim_params[    'arc_physemit_y'] = 0
        sim_params[    'arc_physemit_s'] = 0
        sim_params[     'arc_rf_energy'] = 0
    print(f"[exec.py] successfully set arc (emittance) damping rates: 'arc_damping_rate_x'={sim_params['arc_damping_rate_x']} [1/superperiod], 'arc_damping_raite_y'={sim_params['arc_damping_rate_y']} [1/superperiod], 'arc_damping_rate_s'={sim_params['arc_damping_rate_s']} [1/superperiod]")
    print(f"[exec.py] successfully set arc equilibrium emittances: 'arc_physemit_x'={sim_params['arc_physemit_x']} [m], 'arc_physemit_y'={sim_params['arc_physemit_y']} [m], 'arc_physemit_s'={sim_params['arc_physemit_s']} [m], 'arc_rf_energy'={sim_params['arc_rf_energy']} [eV]")
