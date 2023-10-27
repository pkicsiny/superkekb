# -*- coding: utf-8 -*-

import numpy as np

# from: https://arxiv.org/abs/2306.02681 and Demin's emails
skekb_ler = {
  "circumference":  3016.315,  # 3016.315 [m]
           "q_b1":         1,  # [e], positron beam
           "n_ip":         1,  # 1 [1], number of IPs
          "mass0":   0.511e6,  # 0.511e6 [eV], particle rest mass
            "phi":    0.0415,  # 0.0415 [rad]
         "energy":         4,  # 4 [GeV]
            "p0c":       4e9,  # 4e9 [eV], reference kin. energy
   "beam_current":   0.71e-3,  # 0.71 [mA]
"bunch_intensity":   6.25e10,  # 6.25e10 [e]
     "physemit_x":      4e-9,  # 4 [nm]
     "physemit_y":    20e-12,  # 20 [pm], lattice
     "physemit_s": 3.4625e-6,  # 3.4625 [um], lattice
         "beta_x":     80e-3,  # 80 [mm]
         "beta_y":      1e-3,  # 1 [mm]
        "sigma_z":    4.6e-3,  # 4.6 [mm], lattice
         "Qx_int":        44,  # 44 [1], integer part of tune
         "Qy_int":        46,  # 46 [1], integer part of tune
             "Qx":     0.524,  # 0.524 [1], fractional full turn tune
             "Qy":     0.589,  # 0.589 [1], fractional full turn tune
             "Qs":     0.023,  # 0.023 [1]
      "k2_factor":       0.8,  # 80 [%]
      "n_bunches":      1174,  # 1174 [1]
           "xi_x":    0.0036,  # 0.0036 [1]
           "xi_y":     0.052,  # 0.052 [1]
 "damping_rate_x":  2.203E-4,  # 2.203E-4 [1/turn], RMS beamsize damping rates
 "damping_rate_y":  2.203E-4,  # 2.203E-4 [1/turn]
 "damping_rate_s": 4.4033E-4,  # 4.4033E-4 [1/turn]
}
skekb_ler["gamma"]       = skekb_ler[    "energy"] /(skekb_ler["mass0"]*1e-9)  # [1]
skekb_ler["sigma_delta"] = skekb_ler["physemit_s"] / skekb_ler[    "sigma_z"]  # [1]
skekb_ler["beta_s"]      = skekb_ler[   "sigma_z"] / skekb_ler["sigma_delta"]  # [m]
skekb_ler["U_SR"]        = skekb_ler[ "n_ip"] * skekb_ler["energy"] * skekb_ler["damping_rate_s"]  # [GeV/turn]

skekb_her = {
  "circumference":  3016.315,  # 3016.315 [m]
           "q_b1":        -1,  # [e], electron beam
           "n_ip":         1,  # 1 [1], number of IPs
          "mass0":   0.511e6,  # 0.511e6 [eV], particle rest mass
            "phi":    0.0415,  # 0.0415 [rad]
         "energy":   7.00729,  # 7.00729 [GeV]
            "p0c": 7.00729e9,  # 7.00729e9 [eV], reference kin. energy
   "beam_current":   0.57e-3,  # 0.57 [mA]
"bunch_intensity":      5e10,  # 5e10 [e]
     "physemit_x":    4.6e-9,  # 4.6 [nm]
     "physemit_y":    35e-12,  # 35 [pm], lattice
     "physemit_s": 3.1824e-6,  # 3.1824 [um], lattice
         "beta_x":     60e-3,  # 60 [mm]
         "beta_y":      1e-3,  # 1 [mm]
        "sigma_z":    5.1e-3,  # 5.1 [mm], lattice
         "Qx_int":        45,  # 45 [1], integer part of tune
         "Qy_int":        43,  # 43 [1], integer part of tune
             "Qx":     0.532,  # 0.532 [1], fractional full turn tune
             "Qy":     0.572,  # 0.572 [1], fractional full turn tune
             "Qs":     0.027,  # 0.027 [1]
      "k2_factor":       0.4,  # 40 [%]
      "n_bunches":      1174,  # 1174 [1]
           "xi_x":    0.0024,  # 0.0024 [1]
           "xi_y":     0.044,  # 0.044 [1]
 "damping_rate_x":  1.736E-4,  # 1.736E-4 [1/turn], RMS beamsize damping rates
 "damping_rate_y":  1.736E-4,  # 1.736E-4 [1/turn]
 "damping_rate_s": 3.4706E-4,  # 3.4706E-4 [1/turn]
}
skekb_her["gamma"]       = skekb_her[    "energy"] /(skekb_her["mass0"]*1e-9)  # [1]
skekb_her["sigma_delta"] = skekb_her["physemit_s"] / skekb_her[    "sigma_z"]  # [1]
skekb_her["beta_s"]      = skekb_her[   "sigma_z"] / skekb_her["sigma_delta"]  # [m]
skekb_her["U_SR"]        = skekb_her[ "n_ip"] * skekb_her["energy"] * skekb_her["damping_rate_s"]  # [GeV/turn]

# new lattice 4IP: https://gitlab.cern.ch/acc-models/fcc/fcc-ee-lattice/-/blob/V23_dev/reference_parameters.json#L14
# https://indico.cern.ch/event/1202105/contributions/5408583/attachments/2659051/4608141/FCCWeek_Optics_Oide_230606.pdf
fcc_t = {
  "circumference":   90658.816,  # 90658.816 [m]
           "n_ip":           4,  # 4 [1], number of IPs
           "q_b1":          -1,  # -1 [e] 
           "q_b2":           1,  # 1 [e]
          "mass0":     0.511e6,  # 0.511e6 [eV], particle rest mass
          "alpha":           0,  # 0 [rad]
            "phi":       15e-3,  # 15e-3 [rad]
         "energy":       182.5,  # 182.5 [GeV]
            "p0c":     182.5e9,  # 182.5e9 [eV], reference kin. energy
   "beam_current":      4.9e-3,  # 4.9e-3 [A]
      "n_bunches":          60,  # 60 [1]
           "lumi": 1.25e34*1e4,  # 1.25e38 [m-2s-1], for the bunch train per IP
"bunch_intensity":     1.55e11,  # 1.55e11 [e]
     "physemit_x":     1.59e-9,  # 1.59e-9 [m]
     "physemit_y":     0.9e-12,  # 0.9e-12 [m], lattice emittance
  "physemit_y_bs":     1.6e-12,  # 1.6e-12 [m], with bb+bs from nonlinear lattice
        "alpha_c":      7.4e-6,  # 7.4e-6 [1], momentum compaction
         "beta_x":           1,  # 1 [m]
         "beta_y":      0.0016,  # 0.0016 [m]
    "sigma_delta":     0.16e-2,  # 0.16e-2 [1]
"sigma_delta_tot":    0.192e-2,  # 0.192e-2 [1]
        "sigma_z":     1.81e-3,  # 1.81e-3 [m]
    "sigma_z_tot":     2.17e-3,  # 2.17e-3 [m]
      "k2_factor":         0.4,  # 40 [%], crab waist ratio
           "U_SR":       10.42,  # 10.42 [GeV], eloss per turn due to sr
           "U_BS":  3.8268e-02,  # 3.8268e-02 [GeV], eloss per turn due to bs, using Khoi's model
         "Qx_int":         398,  # 398 [1], integer part of tune
         "Qy_int":         398,  # 398 [1], integer part of tune
             "Qx":       0.148,  # 0.148 [1]
             "Qy":       0.182,  # 0.182 [1]
             "Qs":       0.091,  # 0.091 [1]
   "tau_in_turns":        18.3,  # 18.3 [turns], longitudinal damping time due to sr
           "xi_x":       0.073,  # 0.073 [1]
           "xi_y":       0.134,  # 0.134 [1]
    "dynap_delta":      2.5e-2,  # 2.5 [%], energy acceptance
}
fcc_t["gamma"]          = fcc_t["energy"]      /(fcc_t["mass0"]*1e-9)  # [1]
fcc_t["beta_s"]         = fcc_t["sigma_z"]     / fcc_t["sigma_delta"]  # [m]
fcc_t["physemit_s"]     = fcc_t["sigma_z"]     * fcc_t["sigma_delta"]  # [m]
fcc_t["beta_s_tot"]     = fcc_t["sigma_z_tot"] / fcc_t["sigma_delta_tot"]  # [m]
fcc_t["physemit_s_tot"] = fcc_t["sigma_z_tot"] * fcc_t["sigma_delta_tot"]  # [m]

fcc_h = {
  "circumference":   90658.816,  # 90658.816 [m]
           "n_ip":           4,  # 4 [1], number of IPs
           "q_b1":          -1,  # -1 [e] 
           "q_b2":           1,  # 1 [e]
          "mass0":     0.511e6,  # 0.511e6 [eV], particle rest mass
          "alpha":           0,  # 0 [rad]
            "phi":       15e-3,  # 15e-3 [rad]
         "energy":         120,  # 120 [GeV]
            "p0c":       120e9,  # 120e9 [eV], reference kin. energy
   "beam_current":     26.7e-3,  # 26.7e-3 [A]
      "n_bunches":         440,  # 440 [1]
           "lumi":    5e34*1e4,  # 5e38 [m-2s-1], for the bunch train per IP
"bunch_intensity":     1.15e11,  # 1.15e11 [e]
     "physemit_x":     0.71e-9,  # 0.71e-9 [m]
     "physemit_y":    0.85e-12,  # 0.85e-12 [m], lattice emittance
  "physemit_y_bs":     1.4e-12,  # 1.4e-12 [m], with bb+bs from nonlinear lattice
        "alpha_c":      7.4e-6,  # 7.4e-6 [1], momentum compaction
         "beta_x":       0.240,  # 0.240 [m]
         "beta_y":       0.001,  # 0.001 [m]
    "sigma_delta":    0.104e-2,  # 0.104e-2 [1]
"sigma_delta_tot":    0.143e-2,  # 0.143e-2 [1]
        "sigma_z":      3.4e-3,  # 3.4e-3 [m]
    "sigma_z_tot":      4.7e-3,  # 4.7e-3 [m]
      "k2_factor":         0.5,  # 50 [%], crab waist ratio
           "U_SR":        1.89,  # 1.89 [GeV], eloss per turn due to sr
           "U_BS":  9.8440e-03,  # 9.8440e-03 [GeV], eloss per turn due to bs, using Khoi's model
         "Qx_int":         398,  # 398 [1], integer part of tune
         "Qy_int":         398,  # 398 [1], integer part of tune
             "Qx":       0.192,  # 0.192 [1]
             "Qy":       0.358,  # 0.358 [1]
             "Qs":       0.032,  # 0.032 [1]
   "tau_in_turns":          64,  # 64 [turns], longitudinal damping time due to sr
           "xi_x":        0.01,  # 0.01 [1]
           "xi_y":       0.088,  # 0.088 [1]
    "dynap_delta":      1.6e-2,  # 1.6 [%], energy acceptance
}
fcc_h["gamma"]          = fcc_h["energy"]      /(fcc_h["mass0"]*1e-9)  # [1]
fcc_h["beta_s"]         = fcc_h["sigma_z"]     / fcc_h["sigma_delta"]  # [m]
fcc_h["physemit_s"]     = fcc_h["sigma_z"]     * fcc_h["sigma_delta"]  # [m]
fcc_h["beta_s_tot"]     = fcc_h["sigma_z_tot"] / fcc_h["sigma_delta_tot"]  # [m]
fcc_h["physemit_s_tot"] = fcc_h["sigma_z_tot"] * fcc_h["sigma_delta_tot"]  # [m]

fcc_w = {
  "circumference":   90658.816,  # 90658.816 [m]
           "n_ip":           4,  # 4 [1], number of IPs
           "q_b1":          -1,  # -1 [e] 
           "q_b2":           1,  # 1 [e]
          "mass0":     0.511e6,  # 0.511e6 [eV], particle rest mass
          "alpha":           0,  # 0 [rad]
            "phi":       15e-3,  # 15e-3 [rad]
         "energy":          80,  # 80 [GeV]
            "p0c":        80e9,  # 80e9 [eV], reference kin. energy
   "beam_current":      137e-3,  # 137e-3 [A]
      "n_bunches":        1780,  # 1780 [1]
           "lumi":   20e34*1e4,  # 20e38 [m-2s-1], for the bunch train per IP
"bunch_intensity":     1.45e11,  # 1.45e11 [e]
     "physemit_x":     2.17e-9,  # 2.17e-9 [m]
     "physemit_y":    1.25e-12,  # 1.25e-12 [m], lattice emittance
  "physemit_y_bs":     2.2e-12,  # 2.2e-12[m], with bb+bs from nonlinear lattice
        "alpha_c":     28.6e-6,  # 28.6e-6 [1], momentum compaction
         "beta_x":        0.22,  # 0.22 [m]
         "beta_y":       0.001,  # 0.001 [m]
    "sigma_delta":     0.07e-2,  # 0.07e-2 [1]
"sigma_delta_tot":    0.109e-2,  # 0.109e-2 [1]
        "sigma_z":     3.47e-3,  # 3.47e-3 [m]
    "sigma_z_tot":     5.41e-3,  # 5.41e-3 [m]
      "k2_factor":        0.55,  # 55 [%], crab waist ratio
           "U_SR":       0.374,  # 0.374 [GeV], eloss per turn due to sr
           "U_BS":  3.0681e-03,  # 3.0681e-03 [GeV], eloss per turn due to bs, using Khoi's model
         "Qx_int":         218,  # 218 [1], integer part of tune
         "Qy_int":         222,  # 222 [1], integer part of tune
             "Qx":       0.186,  # 0.186 [1]
             "Qy":       0.220,  # 0.220 [1]
             "Qs":       0.081,  # 0.081 [1]
   "tau_in_turns":         219,  # 219 [turns], longitudinal damping time due to sr
           "xi_x":       0.013,  # 0.013 [1]
           "xi_y":       0.128,  # 0.128 [1]
    "dynap_delta":        1e-2,  # 1 [%], energy acceptance
}
fcc_w["gamma"]          = fcc_w["energy"]      /(fcc_w["mass0"]*1e-9)  # [1]
fcc_w["beta_s"]         = fcc_w["sigma_z"]     / fcc_w["sigma_delta"]  # [m]
fcc_w["physemit_s"]     = fcc_w["sigma_z"]     * fcc_w["sigma_delta"]  # [m]
fcc_w["beta_s_tot"]     = fcc_w["sigma_z_tot"] / fcc_w["sigma_delta_tot"]  # [m]
fcc_w["physemit_s_tot"] = fcc_w["sigma_z_tot"] * fcc_w["sigma_delta_tot"]  # [m]

fcc_z = {
  "circumference":   90658.816,  # 90658.816 [m]
           "n_ip":           4,  # 4 [1], number of IPs
           "q_b1":          -1,  # -1 [e] 
           "q_b2":           1,  # 1 [e]
          "mass0":     0.511e6,  # 0.511e6 [eV], particle rest mass
          "alpha":           0,  # 0 [rad]
            "phi":       15e-3,  # 15e-3 [rad]
         "energy":       45.6 ,  # 45.6 [GeV]
            "p0c":      45.6e9,  # 45.6e9 [eV], reference kin. energy
   "beam_current":     1270e-3,  # 1270e-3 [A]
      "n_bunches":       15880,  # 15880 [1]
           "lumi":  140e34*1e4,  # 140e38 [m-2s-1], for the bunch train per IP
"bunch_intensity":     1.51e11,  # 1.51e11 [e]
     "physemit_x":     0.71e-9,  # 0.71e-9 [m]
     "physemit_y":    0.75e-12,  # 1.75e-12 [m], lattice emittance
  "physemit_y_bs":     1.4e-12,  # 1.4e-12[m], with bb+bs from nonlinear lattice
        "alpha_c":     28.6e-6,  # 28.6e-6 [1], momentum compaction
         "beta_x":        0.11,  # 0.11 [m]
         "beta_y":      0.7e-3,  # 0.0007 [m]
    "sigma_delta":    0.039e-2,  # 0.039e-2 [1]
"sigma_delta_tot":    0.089e-2,  # 0.089e-2 [1]
        "sigma_z":      5.6e-3,  # 5.6e-3 [m]
    "sigma_z_tot":     12.7e-3,  # 12.7e-3 [m]
      "k2_factor":         0.7,  # 70 [%], crab waist ratio
           "U_SR":      0.0394,  # 0.0394 [GeV], eloss per turn due to sr
           "U_BS":  5.0541e-04,  # 5.0541e-04 [GeV], eloss per turn due to bs, using Khoi's model
         "Qx_int":         218,  # 218 [1], integer part of tune
         "Qy_int":         222,  # 222 [1], integer part of tune
             "Qx":       0.158,  # 0.158 [1]
             "Qy":        0.20,  # 0.20 [1]
             "Qs":      0.0288,  # 0.0288 [1]
   "tau_in_turns":        1158,  # 1158 [turns], longitudinal damping time due to sr
           "xi_x":      0.0023,  # 0.0023 [1]
           "xi_y":       0.096,  # 0.096 [1]
    "dynap_delta":        1e-2,  # 1 [%], energy acceptance
}
fcc_z["gamma"]          = fcc_z["energy"]      /(fcc_z["mass0"]*1e-9)  # [1]
fcc_z["beta_s"]         = fcc_z["sigma_z"]     / fcc_z["sigma_delta"]  # [m]
fcc_z["physemit_s"]     = fcc_z["sigma_z"]     * fcc_z["sigma_delta"]  # [m]
fcc_z["beta_s_tot"]     = fcc_z["sigma_z_tot"] / fcc_z["sigma_delta_tot"]  # [m]
fcc_z["physemit_s_tot"] = fcc_z["sigma_z_tot"] * fcc_z["sigma_delta_tot"]  # [m]


# https://arxiv.org/pdf/2208.08615.pdf
higgs_tanajisen = {
"q_b1":-1,  # def.: -1 [e] 
"q_b2":1,  # def.: 1 [e]
"circumference": 16e3,  # [m]
"energy": 120,  # [GeV]
"mass0": 0.511e6,  # particle rest mass, def.: 0.511e6 [eV]
"bunch_intensity": 8.34e11,  # [1]
"physemit_x": 21e-9,  # [m]
"physemit_y": 0.05e-9, # [m]
"beta_x": 0.2,  # [m]
"beta_y": 0.001,  # [m]
"sigma_z_tot": 2.9e-3,  # [m]
"phi": 0,  # [rad] half xing
}
higgs_tanajisen["gamma"] = higgs_tanajisen["energy"] / (higgs_tanajisen["mass0"]*1e-9)  # [1]

fcc_tanajisen = {
"q_b1":-1,  # def.: -1 [e] 
"q_b2":1,  # def.: 1 [e]
"circumference": 97.75e3,  # [m]
"energy": 120,  # [GeV]
"mass0": 0.511e6,  # particle rest mass, def.: 0.511e6 [eV]
"bunch_intensity": 1.8e11,  # [1]
"physemit_x": 0.63e-9,  # [m]
"physemit_y": 1.3e-12, # [m]
"beta_x": 0.3,  # [m]
"beta_y": 0.001,  # [m]
"sigma_z_tot": 5.3e-3,  # [m]
"phi": 15e-3,  # [rad] half xing
}
fcc_tanajisen["gamma"] = fcc_tanajisen["energy"] / (fcc_tanajisen["mass0"]*1e-9)  # [1]


# fcc-ee https://cds.cern.ch/record/2651299?ln=en
fcc_ttbar2_cdr = {
     "circumference":    97.756e3,  # 97.756e3 [m]
              "n_ip":           2,  # 2 [1], number of IPs
              "q_b1":          -1,  # def.: -1 [e] 
              "q_b2":           1,  # def.: 1 [e]
             "mass0":     0.511e6,  # particle rest mass, def.: 0.511e6 [eV]
             "alpha":           0,  # def.: 0 [rad]
               "phi":       15e-3,  # def.: 15e-3 [rad]
            "energy":      182.5 ,  # def.: 182.5 [GeV]
               "p0c":     182.5e9,  # reference kin. energy, def.: 182.5e9 [eV]
      "beam_current":      5.4e-3,  # def.: 5.4e-3 [A]
         "n_bunches":          48,  # def.: 48 [1]
              "lumi": 1.55e34*1e4,  # def.: 1.55e38 [m-2s-1] for the bunch train
   "bunch_intensity":      2.3e11,  # def.: 2.3e11 [1]
        "physemit_x":     1.46e-9,  # def.: 1.46e-9 [m]
        "physemit_y":     2.9e-12,  # def.: 2.9e-12 [m]
           "alpha_c":      7.3e-6,  # def.: 7.3e-6 [1]
            "beta_x":           1,  # def.: 1 [m]
            "beta_y":      1.6e-3,  # def.: 1.6e-3 [m]
       "sigma_delta":     0.15e-2,  # def.: 0.15e-2 [1]
           "sigma_z":     1.94e-3,  # def.: 1.94e-3 [m]
   "sigma_delta_tot":    0.192e-2,  # def.: 0.192e-2 [1]
       "sigma_z_tot":     2.54e-3,  # def.: 2.54e-3 [m]
"interaction_length":      1.8e-3,  # def.: 1.8e-3 [m]
         "k2_factor":         0.4,  # def.: 40 [%]
              "U_SR":         9.2,  # def.: 9.2 [GeV]
              "U_BS":  3.6252e-02,  # def.: 11.4e-3 [GeV]
            "Qx_int":         389,  # integer part of tune, def.: 389 [1]
            "Qy_int":         389,  # integer part of tune, def.: 389 [1]
                "Qx":      0.108 ,  # def.: 0.108 [1]
                "Qy":      0.175 ,  # def.: 0.175 [1]
                "Qs":      0.0872,  # def.: 0.0872 [1]
      "tau_in_turns":        20.4,  # def.: 20.4 [turns]
              "xi_x":      9.9e-2,  # def.: 9.9e-2 [1]
              "xi_y":     1.26e-1,  # def.: 1.26e-1 [1]
       "dynap_delta":      2.5e-2,  # def.: 2.5e-2 [1]
}
fcc_ttbar2_cdr["gamma"]          = fcc_ttbar2_cdr["energy"]     / (fcc_ttbar2_cdr["mass0"]*1e-9)  # [1]
fcc_ttbar2_cdr["beta_s"]         = fcc_ttbar2_cdr["sigma_z"]     / fcc_ttbar2_cdr["sigma_delta"]  # [m]
fcc_ttbar2_cdr["physemit_s"]     = fcc_ttbar2_cdr["sigma_z"]     * fcc_ttbar2_cdr["sigma_delta"]  # [m]
fcc_ttbar2_cdr["beta_s_tot"]     = fcc_ttbar2_cdr["sigma_z_tot"] / fcc_ttbar2_cdr["sigma_delta_tot"]  # [m]
fcc_ttbar2_cdr["physemit_s_tot"] = fcc_ttbar2_cdr["sigma_z_tot"] * fcc_ttbar2_cdr["sigma_delta_tot"]  # [m]

fcc_ttbar1_cdr = {
     "circumference":    97.756e3,  # 97.756e3 [m]
              "n_ip":           2,  # 2 [1], number of IPs
              "q_b1":          -1,  # def.: -1 [e] 
              "q_b2":           1,  # def.: 1 [e]
             "mass0":     0.511e6,  # particle rest mass, def.: 0.511e6 [eV]
             "alpha":           0,  # def.: 0 [rad]
               "phi":       15e-3,  # def.: 15e-3 [rad]
            "energy":         175,  # def.: 175 [GeV]
               "p0c":       175e9,  # reference kin. energy, def.: 175e9 [eV]
      "beam_current":      6.4e-3,  # def.: 6.4e-3 [A]
         "n_bunches":          59,  # def.: 59 [1]
              "lumi":  1.8e34*1e4,  # def.: 1.8e38 [m-2s-1] for the bunch train
   "bunch_intensity":      2.2e11,  # def.: 2.2e11 [1]
        "physemit_x":     1.34e-9,  # def.: 1.34e-9 [m]
        "physemit_y":     2.7e-12,  # def.: 2.7e-12 [m]
           "alpha_c":      7.3e-6,  # def.: 7.3e-6 [1]
            "beta_x":           1,  # def.: 1 [m]
            "beta_y":      1.6e-3,  # def.: 1.6e-3 [m]
       "sigma_delta":    0.144e-2,  # def.: 0.144e-2 [1]
   "sigma_delta_tot":    0.186e-2,  # def.: 0.186e-2 [1]  
           "sigma_z":     2.01e-3,  # def.: 2.01e-3 [m]
       "sigma_z_tot":     2.62e-3,  # def.: 2.62e-3 [m]
"interaction_length":      1.8e-3,  # def.: 1.8e-3 [m]
         "k2_factor":         0.4,  # def.: 40 [%]
              "U_SR":         7.8,  # def.: 7.8 [GeV]
              "U_BS":  3.6252e-02,  # def.: 9.84e-3 [GeV]
            "Qx_int":         389,  # integer part of tune, def.: 389 [1]
            "Qy_int":         389,  # integer part of tune, def.: 389 [1]
                "Qx":       0.108,  # def.: 0.108 [1]
                "Qy":       0.175,  # def.: 0.175 [1]
                "Qs":      0.0818,  # def.: 0.0818 [1]
      "tau_in_turns":        23.1,  # def.: 23.1 [turns]
              "xi_x":      9.7e-2,  # def.: 9.7e-2 [1]
              "xi_y":     1.28e-1,  # def.: 1.28e-1 [1]
       "dynap_delta":      2.5e-2,  # def.: 2.5e-2 [1]
}
fcc_ttbar1_cdr["gamma"]          = fcc_ttbar1_cdr["energy"]      /(fcc_ttbar1_cdr["mass0"]*1e-9)  # [1]
fcc_ttbar1_cdr["beta_s"]         = fcc_ttbar1_cdr["sigma_z"]     / fcc_ttbar1_cdr["sigma_delta"]  # [m]
fcc_ttbar1_cdr["physemit_s"]     = fcc_ttbar1_cdr["sigma_z"]     * fcc_ttbar1_cdr["sigma_delta"]  # [m]
fcc_ttbar1_cdr["beta_s_tot"]     = fcc_ttbar1_cdr["sigma_z_tot"] / fcc_ttbar1_cdr["sigma_delta_tot"]  # [m]
fcc_ttbar1_cdr["physemit_s_tot"] = fcc_ttbar1_cdr["sigma_z_tot"] * fcc_ttbar1_cdr["sigma_delta_tot"]  # [m]

fcc_zh_cdr = {
     "circumference":    97.756e3,  # 97.756e3 [m]
              "n_ip":           2,  # 2 [1], number of IPs
              "q_b1":          -1,  # def.: -1 [e] 
              "q_b2":           1,  # def.: 1 [e]
             "mass0":     0.511e6,  # particle rest mass, def.: 0.511e6 [eV]
             "alpha":           0,  # def.: 0 [rad]
               "phi":       15e-3,  # def.: 15e-3 [rad]
            "energy":         120,  # def.: 120 [GeV]
               "p0c":       120e9,  # reference kin. energy, def.: 120e9 [eV]
      "beam_current":       29e-3,  # def.: 29e-3 [A]
         "n_bunches":         328,  # def.: 328 [1]
              "lumi":  8.5e34*1e4,  # def.: 8.5e38 [m-2s-1] for the bunch train
   "bunch_intensity":      1.8e11,  # def.: 1.8e11 [1]
        "physemit_x":     0.63e-9,  # def.: 0.63e-9 [m]
        "physemit_y":     1.3e-12,  # def.: 1.3e-12 [m]
           "alpha_c":      7.3e-6,  # def.: 7.3e-6 [1]
            "beta_x":         0.3,  # def.: 0.3 [m]
            "beta_y":        1e-3,  # def.: 1e-3 [m]
       "sigma_delta":    0.099e-2,  # def.: 0.099e-2 [1]
   "sigma_delta_tot":    0.165e-2,  # def.: 0.165e-2 [1]
           "sigma_z":     3.15e-3,  # def.: 3.15e-3 [m]
       "sigma_z_tot":      5.3e-3, # def.: 5.3e-3 [m]
"interaction_length":      0.9e-3,  # def.: 0.9e-3 [m]
         "k2_factor":         0.8,  # def.: 80 [%]
              "U_SR":        1.72,  # def.: 1.72 [GeV]
              "U_BS":  9.0274e-03,  # def.: 1.97e-3 [GeV]
            "Qx_int":         389,  # integer part of tune, def.: 389 [1]
            "Qy_int":         389,  # integer part of tune, def.: 389 [1]
                "Qx":       0.129,  # def.: 0.129 [1]
                "Qy":       0.199,  # def.: 0.199 [1]
                "Qs":      0.0358,  # def.: 0.0358 [1]
      "tau_in_turns":        70.3,  # def.: 70.3 [turns]
              "xi_x":      1.6e-2,  # def.: 1.6e-2 [1]
              "xi_y":     1.18e-1,  # def.: 1.18e-1 [1]
       "dynap_delta":      1.7e-2,  # def.: 1.7e-2 [1]
}
fcc_zh_cdr["gamma"]          = fcc_zh_cdr["energy"]      /(fcc_zh_cdr["mass0"]*1e-9)  # [1]
fcc_zh_cdr["beta_s"]         = fcc_zh_cdr["sigma_z"]     / fcc_zh_cdr["sigma_delta"]  # [m]
fcc_zh_cdr["physemit_s"]     = fcc_zh_cdr["sigma_z"]     * fcc_zh_cdr["sigma_delta"]  # [m]
fcc_zh_cdr["beta_s_tot"]     = fcc_zh_cdr["sigma_z_tot"] / fcc_zh_cdr["sigma_delta_tot"]  # [m]
fcc_zh_cdr["physemit_s_tot"] = fcc_zh_cdr["sigma_z_tot"] * fcc_zh_cdr["sigma_delta_tot"]  # [m]

fcc_ww_cdr = {
     "circumference":    97.756e3,  # 97.756e3 [m]
              "n_ip":           2,  # 2 [1], number of IPs
              "q_b1":          -1,  # def.: -1 [e] 
              "q_b2":           1,  # def.: 1 [e]
             "mass0":     0.511e6,  # particle rest mass, def.: 0.511e6 [eV]
             "alpha":           0,  # def.: 0 [rad]
               "phi":       15e-3,  # def.: 15e-3 [rad]
            "energy":          80,  # def.: 80 [GeV]
               "p0c":        80e9,  # reference kin. energy, def.: 80e9 [eV]
      "beam_current":      147e-3,  # def.: 147e-3 [A]
         "n_bunches":        2000,  # def.: 2000 [1]
              "lumi":   28e34*1e4,  # def.: 28e38 [m-2s-1] for the bunch train
   "bunch_intensity":      1.5e11,  # def.: 1.5e11 [1]
        "physemit_x":     0.84e-9,  # def.: 0.84e-9 [m]
        "physemit_y":     1.7e-12,  # def.: 1.7e-12 [m]
           "alpha_c":     14.8e-6,  # def.: 14.8e-6 [1]
            "beta_x":         0.2,  # def.: 0.2 [m]
            "beta_y":        1e-3,  # def.: 1e-3 [m]
       "sigma_delta":    0.066e-2,  # def.: 0.066e-2 [1]
   "sigma_delta_tot":    0.131e-2,  # def.: 0.131e-2 [1]
           "sigma_z":        3e-3,  # def.: 3e-3 [m]
       "sigma_z_tot":        6e-3, # def.: 6e-3 [m]
"interaction_length":     0.85e-3,  # def.: 0.85e-3 [m]
         "k2_factor":        0.87,  # def.: 87 [%]
              "U_SR":        0.34,  # def.: 0.34 [GeV]
              "U_BS":  2.3185e-03,  # def.: 0.445e-3 [GeV]
            "Qx_int":         269,  # integer part of tune, def.: 269 [1]
            "Qy_int":         269,  # integer part of tune, def.: 269 [1]
                "Qx":       0.124,  # def.: 0.124 [1]
                "Qy":       0.199,  # def.: 0.199 [1]
                "Qs":      0.0506,  # def.: 0.0506 [1]
      "tau_in_turns":         236,  # def.: 236 [turns]
              "xi_x":        1e-2,  # def.: 1e-2 [1]
              "xi_y":     1.13e-1,  # def.: 1.13e-1 [1]
       "dynap_delta":      1.3e-2,  # def.: 1.3e-2 [1]
}
fcc_ww_cdr["gamma"]          = fcc_ww_cdr["energy"]      /(fcc_ww_cdr["mass0"]*1e-9)  # [1]
fcc_ww_cdr["beta_s"]         = fcc_ww_cdr["sigma_z"]     / fcc_ww_cdr["sigma_delta"]  # [m]
fcc_ww_cdr["physemit_s"]     = fcc_ww_cdr["sigma_z"]     * fcc_ww_cdr["sigma_delta"]  # [m]
fcc_ww_cdr["beta_s_tot"]     = fcc_ww_cdr["sigma_z_tot"] / fcc_ww_cdr["sigma_delta_tot"]  # [m]
fcc_ww_cdr["physemit_s_tot"] = fcc_ww_cdr["sigma_z_tot"] * fcc_ww_cdr["sigma_delta_tot"]  # [m]

fcc_z_cdr = {
     "circumference":    97.756e3,  # 97.756e3 [m]
              "n_ip":           2,  # 2 [1], number of IPs
              "q_b1":          -1,  # def.: -1 [e] 
              "q_b2":           1,  # def.: 1 [e]
             "mass0":     0.511e6,  # particle rest mass, def.: 0.511e6 [eV]
             "alpha":           0,  # def.: 0 [rad]
               "phi":       15e-3,  # def.: 15e-3 [rad]
            "energy":        45.6,  # def.: 45.6 [GeV]
               "p0c":      45.6e9,  # reference kin. energy, def.: 45.6e9 [eV]
      "beam_current":     1390e-3,  # def.: 1390e-3 [A]
         "n_bunches":       16640,  # def.: 16640 [1]
              "lumi":  230e34*1e4,  # def.: 230e38 [m-2s-1] for the bunch train
   "bunch_intensity":      1.7e11,  # def.: 1.7e11 [1]
        "physemit_x":     0.27e-9,  # def.: 0.27e-9 [m]
        "physemit_y":       1e-12,  # def.: 1e-12 [m]
           "alpha_c":     14.8e-6,  # def.: 14.8e-6 [1]
            "beta_x":        0.15,  # def.: 0.15 [m]
            "beta_y":      0.8e-3,  # def.: 0.8e-3 [m]
       "sigma_delta":    0.038e-2,  # def.: 0.038e-2 [1]
   "sigma_delta_tot":    0.132e-2,  # def.: 0.132e-2 [1]
           "sigma_z":      3.5e-3,  # def.: 3.5e-3 [m]
       "sigma_z_tot":     12.1e-3,  # def.: 12.1e-3 [m]
"interaction_length":     0.42e-3,  # def.: 0.42e-3 [m]
         "k2_factor":        0.97,  # def.: 97 [%]
              "U_SR":       0.036,  # def.: 0.036 [GeV]
              "U_BS":  4.9028e-04,  # def.: 0.21e-3 [GeV]
            "Qx_int":         269,  # integer part of tune, def.: 269 [1]
            "Qy_int":         269,  # integer part of tune, def.: 269 [1]
                "Qx":       0.139,  # def.: 0.139 [1]
                "Qy":       0.219,  # def.: 0.219 [1]
                "Qs":       0.025,  # def.: 0.025 [1]
      "tau_in_turns":        1273,  # def.: 1273 [turns]
              "xi_x":        4e-3,  # def.: 4e-3 [1]
              "xi_y":     1.33e-1,  # def.: 1.33e-1 [1]
       "dynap_delta":      1.3e-2,  # def.: 1.3e-2 [1]
}
fcc_z_cdr["gamma"]          = fcc_z_cdr["energy"]      /(fcc_z_cdr["mass0"]*1e-9)  # [1]
fcc_z_cdr["beta_s"]         = fcc_z_cdr["sigma_z"]     / fcc_z_cdr["sigma_delta"]  # [m]
fcc_z_cdr["physemit_s"]     = fcc_z_cdr["sigma_z"]     * fcc_z_cdr["sigma_delta"]  # [m]
fcc_z_cdr["beta_s_tot"]     = fcc_z_cdr["sigma_z_tot"] / fcc_z_cdr["sigma_delta_tot"]  # [m]
fcc_z_cdr["physemit_s_tot"] = fcc_z_cdr["sigma_z_tot"] * fcc_z_cdr["sigma_delta_tot"]  # [m]

fcc_z_halfbeta = {
"n_macroparticles_b1": int(1e3), 
"n_macroparticles_b2": int(1e3), # 1e4-5, large intensity makes fft more ragged
"q_b1":-1,  # def.: -1 [e] 
"q_b2":1,  # def.: 1 [e]
"mass0": 0.511e6,  # particle rest mass, def.: 0.511e6 [eV]
"alpha": 0,  # def.: 0 [rad]
"phi": 15e-3,  # def.: 15e-3 [rad]
"energy": 45.6 ,  # def.: 45.6 [GeV]
"p0c": 45.6e9,  # reference kin. energy, def.: 45.6e9 [eV]
"beam_current": 1390e-3,  # def.: 1390e-3 [A]
"n_bunches": 16640,  # def.: 16640 [1]
"bunch_intensity": 1.7e11,  # def.: 1.7e11 [1]
"physemit_x": 0.27e-9,  # def.: 0.27e-9 [m]
"physemit_y": 1e-12,  # def.: 1e-12 [m]
"alpha_c": 14.8e-6,  # def.: 14.8e-6 [1]
"beta_x": 0.075,  # def.: 0.15 [m]
"beta_y": 0.8e-3,  # def.: 0.8e-3 [m]
"sigma_delta": 0.038e-2,  # def.: 0.038e-2 [1]
"sigma_delta_tot": 0.132e-2,  # def.: 0.132e-2 [1]
"sigma_z": 3.5e-3,  # def.: 3.5e-3 [m]
"sigma_z_tot": 12.1e-3,  # def.: 12.1e-3 [m]
"interaction_length": 0.42e-3,  # def.: 0.42e-3 [m]
"k2_factor": 0.97,  # def.: 97 [%]
"U_SR": 0.036,  # def.: 0.036 [GeV]
"U_BS": 0.21e-3,  # def.: 0.21e-3 [GeV]
"Q": 269,  # integer part of tune, def.: 269 [1]
"Qx": 0.139 ,  # def.: 0.139 [1]
"Qy": 0.219 ,  # def.: 0.219 [1]
"Qs": 0.025,  # def.: 0.025 [1]
"tau_in_turns": 1273,  # def.: 1273 [turns]
"xi_x": 4e-3,  # def.: 4e-3 [1]
"xi_y": 1.33e-1,  # def.: 1.33e-1 [1]
}
fcc_z_halfbeta["gamma"]  = fcc_z_halfbeta["energy"]/(fcc_z_halfbeta["mass0"]*1e-9)  # [1]
fcc_z_halfbeta["beta_s"]         = fcc_z_halfbeta["sigma_z"] / fcc_z_halfbeta["sigma_delta"]  # [m]
fcc_z_halfbeta["physemit_s"]     = fcc_z_halfbeta["sigma_z"] * fcc_z_halfbeta["sigma_delta"]  # [m]
fcc_z_halfbeta["beta_s_tot"]     = fcc_z_halfbeta["sigma_z_tot"] / fcc_z_halfbeta["sigma_delta_tot"]  # [m]
fcc_z_halfbeta["physemit_s_tot"] = fcc_z_halfbeta["sigma_z_tot"] * fcc_z_halfbeta["sigma_delta_tot"]  # [m]

example_005 = {
"q_b1":-1,
"q_b2":1,
"mass0": 0.511e6,
"energy": 45.6,
"p0c": 45.6e9,
"physemit_x": 0.3e-9,
"physemit_y": 1e-12,
"beta_x": 1.0,
"beta_y": 10.0,
"sigma_delta": 3.8E-4,
"sigma_z": 3.5E-3,
"Qx": .18,
"Qy": .22,
"Qs": 0.025,
"alpha_x": -10.0,
"alpha_y": 1000.0,
"damping_rate_x": 5E-4,
"damping_rate_y": 1E-3,
"damping_rate_s": 2E-3,
}
example_005["gamma_x"]    = (1.0+example_005["alpha_x"]**2)/example_005["beta_x"]
example_005["gamma_y"]    = (1.0+example_005["alpha_y"]**2)/example_005["beta_y"]
example_005["beta_s"]     = example_005["sigma_z"] / example_005["sigma_delta"]  # [m]
example_005["physemit_s"] = example_005["sigma_z"] * example_005["sigma_delta"]  # [m]

# https://cds.cern.ch/record/2159684?ln=en
fcc_cw = {
"n_macroparticles_b1": int(1e3), 
"n_macroparticles_b2": int(1e3), # 1e4-5, large intensity makes fft more ragged
"q_b1":-1,  # def.: -1 [e] 
"q_b2":1,  # def.: 1 [e]
"mass0": 0.511e6,  # particle rest mass, def.: 0.511e6 [eV]
"alpha": 0,  # def.: 0 [rad]
"phi": 15e-3,  # def.: 15e-3 [rad]
"energy": 45.6 ,  # def.: 45.6 [GeV]
"p0c": 45.6e9,  # reference kin. energy, def.: 45.6e9 [eV]
"beam_current": 1.4503e3,  # [A]
"n_bunches": 91500,
"bunch_intensity": 3.3e10,  # def.: 3.3e10 [1]
"physemit_x": 0.09e-9,  # def.: 0.09e-9 [m]
"physemit_y": 1e-12,  # def.: 1e-12 [m]
"alpha_c": 7e-6,  # def.: 7e-6 [1]
"beta_x": 1,  # def.: 1 [m]
"beta_y": 2e-3,  # def.: 2e-3 [m]
"sigma_delta": 0.04e-2,  # def.: 0.04e-2 [1]
"sigma_delta_tot": 0.09e-2, # def.: 0.09e-2 [1]
"sigma_z": 1.6e-3,  # def.: 1.6e-3 [m]
"sigma_z_tot": 3.8e-3,  # def.: 3.8e-3 [m]
"U_SR": 0.03,  # def.: 0.03 [GeV]
"U_BS": 0.5e-3,  # def.: 0.5e-3 [GeV]
"Qx": 0.139 ,  # def.: 0.139 [1]
"Qy": 0.219 ,  # def.: 0.219 [1]
"Qs": 0.025,  # def.: 0.025 [1]
"tau_in_turns": 1320,  # def.: 1320 [turns]
"xi_x": 5e-2,  # def.: 5e-2 [1]
"xi_y": 13e-2,  # def.: 13e-2 [1]
"ups_max": 1.7e-4,  # def.: 1.7e-4 [1]
"ups_avg": 0.7e-4,  # def.: 0.7e-4 [1]
}
fcc_cw["gamma"]  = fcc_cw["energy"]/(fcc_cw["mass0"]*1e-9)  # [1]
fcc_cw["beta_s"]         = fcc_cw["sigma_z"] / fcc_cw["sigma_delta"]  # [m]
fcc_cw["physemit_s"]     = fcc_cw["sigma_z"] * fcc_cw["sigma_delta"]  # [m]
fcc_cw["beta_s_tot"]     = fcc_cw["sigma_z_tot"] / fcc_cw["sigma_delta_tot"]  # [m]
fcc_cw["physemit_s_tot"] = fcc_cw["sigma_z_tot"] * fcc_cw["sigma_delta_tot"]  # [m]

fcc_ho = {
"n_macroparticles_b1": int(1e3), 
"n_macroparticles_b2": int(1e3), # 1e4-5, large intensity makes fft more ragged
"q_b1":-1,  # def.: -1 [e] 
"q_b2":1,  # def.: 1 [e]
"mass0": 0.511e6,  # particle rest mass, def.: 0.511e6 [eV]
"alpha": 0,  # def.: 0 [rad]
"phi": 0,  # def.: 0 [rad]
"energy": 62.5 ,  # def.: 62.5 [GeV]
"p0c": 62.5e9,  # reference kin. energy, def.: 62.5e9 [eV]
"beam_current": 0.4083e3,  # [A]
"n_bunches": 80960,
"bunch_intensity": 0.7e10,  # def.: 0.7e10 [1]
"physemit_x": 0.17e-9,  # def.: 0.17e-9 [m]
"physemit_y": 1e-12,  # def.: 1e-12 [m]
"alpha_c": 7e-6,  # def.: 7e-6 [1]
"beta_x": 1,  # def.: 1 [m]
"beta_y": 2e-3,  # def.: 2e-3 [m]   
"sigma_delta": 0.06e-2,  # def.: 0.06e-2 [1]
"sigma_delta_tot": 0.06e-2,  # def.: 0.06e-2 [1]
"sigma_z": 1.8e-3,  # def.: 1.8e-3 [m]
"sigma_z_tot": 1.8e-3,  # def.: 1.8e-3 [m]
"U_SR": 0.12,  # def.: 0.12 [GeV]
"U_BS": 0.05e-3,  # def.: 0.05e-3 [GeV]
"Qx": 0.139 ,  # def.: 0.139 [1]
"Qy": 0.219 ,  # def.: 0.219 [1]
"Qs": 0.03,  # def.: 0.03 [1]
"tau_in_turns": 509,  # def.: 509 [turns]
"xi_x": 12e-2,  # def.: 12e-2 [1]
"xi_y": 15e-2,  # def.: 15e-2 [1]
"ups_max": 0.8e-4,  # def.: 0.8e-4 [1]
"ups_avg": 0.3e-4,  # def.: 0.3e-4 [1]
}
fcc_ho["gamma"]      =  fcc_ho["energy"] / (fcc_ho["mass0"]*1e-9)  # [1]
fcc_ho["beta_s"]     = fcc_ho["sigma_z"] / fcc_ho["sigma_delta"]  # [m]
fcc_ho["physemit_s"] = fcc_ho["sigma_z"] * fcc_ho["sigma_delta"]  # [m]

# hl-lhc
hl_lhc_params = {
"n_macroparticles_b1": int(1e3),
"n_macroparticles_b2": int(1e3), # 1e4-5, large intensity makes fft more ragged
"q_b1":1,  # def.: 1 [e]
"q_b2":1,  # def.: 1 [e]
"mass0": 938.27208816e6,  # particle rest mass, def.: 938.27208816e6 [eV]
"alpha": 0,  # def.: 0 [rad]
"phi": 0,  # def.: 0 [rad]
"energy": 7e3 ,  # def.: 7e3 [GeV]
"p0c": 7000e9, # reference kin. energy, def.: 7000e9 [eV]
"bunch_intensity": 2.2e11,  # def.: 2.2e11 [1]
"beta_x": 1,  # def.:1 [m]
"beta_y": 1,  # def.: 1 [m]
"sigma_delta": 1e-4,  # def.: 1e-4 [1]
"sigma_delta_tot": 1e-4,  # def.: 1e-4 [1]
"sigma_z": 0.08,  # def.: 0.08 [m]
"sigma_z_tot": 0.08,  # def.: 0.08 [m]
"Qx": 0.31 ,  # def.: 0.31 [1]
"Qy": 0.32 ,  # def.: 0.32 [1]
"Qs": 1e-3,  # def.: 1e-3 [1]
}
hl_lhc_params["gamma"] = hl_lhc_params["energy"]/(hl_lhc_params["mass0"]*1e-9)  # [1]
hl_lhc_params["physemit_x"] = 2E-6/hl_lhc_params["gamma"]  # [m]
hl_lhc_params["physemit_y"] = 2E-6/hl_lhc_params["gamma"]  # [m]
hl_lhc_params["beta_s"]         = hl_lhc_params["sigma_z"] / hl_lhc_params["sigma_delta"]  # [m]
hl_lhc_params["physemit_s"]     = hl_lhc_params["sigma_z"] * hl_lhc_params["sigma_delta"]  # [m]
hl_lhc_params["beta_s_tot"]     = hl_lhc_params["sigma_z_tot"] / hl_lhc_params["sigma_delta_tot"]  # [m]
hl_lhc_params["physemit_s_tot"] = hl_lhc_params["sigma_z_tot"] * hl_lhc_params["sigma_delta_tot"]  # [m]

# default lhc
lhc = {
"n_macroparticles_b1": int(1e3),
"n_macroparticles_b2": int(1e3), # 1e4-5, large intensity makes fft more ragged
"q_b1":1,  # def.: 1 [e]
"q_b2":1,  # def.: 1 [e]
"mass0": 938.27208816e6,  # particle rest mass, def.: 938.27208816e6 [eV]
"alpha": 0,  # def.: 0 [rad]
"phi": 0,  # def.: 0 [rad]
"energy": 7e3 ,  # def.: 7e3 [GeV]
"p0c": 7000e9, # reference energy 7000e9 [eV]
"bunch_intensity": 1.05e11,  # def.: 1.05e11 [1]
"beta_x": 0.55,  # def.: 0.55 [m]
"beta_y": 0.55,  # def.: 0.55 [m]
"sigma_delta": 1.13e-4,  # def.: 1.13e-4 [1]
"sigma_delta_tot": 1.13e-4,  # def.: 1.13e-4 [1]
"sigma_z": 0.0755,  # def.: 0.0755 [m]
"sigma_z_tot": 0.0755,  # def.: 0.0755 [m]
"Qx": 0.28 ,  # def.: 0.28 [1]
"Qy": 0.31 ,  # def.: 0.31 [1]
"Qs": 2e-3,  # def.: 2e-3 [1]
}
lhc["gamma"] = lhc["energy"]/(lhc["mass0"]*1e-9)  # [1]
lhc["physemit_x"] = 3.75E-6/lhc["gamma"]  # [m]
lhc["physemit_y"] = 3.75E-6/lhc["gamma"]  # [m]
lhc["beta_s"]         = lhc["sigma_z"] / lhc["sigma_delta"]  # [m]
lhc["physemit_s"]     = lhc["sigma_z"] * lhc["sigma_delta"]  # [m]
lhc["beta_s_tot"]     = lhc["sigma_z_tot"] / lhc["sigma_delta_tot"]  # [m]
lhc["physemit_s_tot"] = lhc["sigma_z_tot"] * lhc["sigma_delta_tot"]  # [m]

# Dimitry study small crossing angles and large piwi
# https://www.lnf.infn.it/acceleratori/dafne/NOTEDAFNE/G/G-59.pdf
dimitry_small = {
"n_macroparticles_b1": int(1e3),
"n_macroparticles_b2": int(1e3), # 1e4-5, large intensity makes fft more ragged
"q_b1":-1,  # def.: -1 [e]
"q_b2":1,  # def.: 1 [e]
"mass0": 0.511e6,  # particle rest mass, def.: 0.511e6 [eV]
"alpha": 0,  # def.: 0 [rad]
"phi": 0,  # def.: 0 [rad]
"energy": 0.51,  # def.: 0.51 [GeV]
"p0c": 0.51e9, # reference kin. energy, def.: 7000e9 [eV]  
"bunch_intensity": 8.873e9,  # def.: 8.873e9 [1]
"physemit_x" : 5e-7,  # def.: 5e-7 [m]
"physemit_y" : 1e-9,  # def.: 1e-9 [m]   
"beta_x": 1.5,  # def.: 1.5 [m]
"beta_y": 0.2,  # def.: 0.2 [m]
"sigma_delta": 1e-4,  # def.: 1e-4 [1]
"sigma_z": 0.03,  # def.: 0.03 [m]
"Qx": 0.057 ,  # def.: 0.057 [1]
"Qy": 0.097 ,  # def.: 0.097 [1]
"Qs": 0.011,  # def.: 0.011 [1]
}
dimitry_small["gamma"]      =  dimitry_small["energy"] / (dimitry_small["mass0"]*1e-9)  # [1]
dimitry_small["beta_s"]     = dimitry_small["sigma_z"] / dimitry_small["sigma_delta"]  # [m]
dimitry_small["physemit_s"] = dimitry_small["sigma_z"] * dimitry_small["sigma_delta"]  # [m]

# large crossing angles
dimitry_large = {
"n_macroparticles_b1": int(1e3),
"n_macroparticles_b2": int(1e3), # 1e4-5, large intensity makes fft more ragged
"q_b1":-1,  # def.: -1 [e]
"q_b2":1,  # def.: 1 [e]
"mass0": 0.511e6,  # particle rest mass, def.: 0.511e6 [eV]    
"alpha": 0,  # def.: 0 [rad]
"phi": 0,  # def.: 0 [rad]
"energy": 0.51,  # def.: 0.51 [GeV]
"p0c": 0.51e9, # reference energy, def.: 7000e9 [eV]
"bunch_intensity": 2.0e+11, # def.: 2.0e+11 [1]
"physemit_x" : 2.25e-6,  # def.: 2.25e-6 [m]
"physemit_y" : 1e-6,  # def.: 1e-6 [m] 
"beta_x": 1,  # def.: 1 [m]
"beta_y": 1,  # def.: 1 [m]
"sigma_delta": 1e-4,  # def.: 1e-4 [1]
"sigma_z": 0.003,  # def.: 0.003 [m]
"Qx": 0.057 ,  # def.: 0.057 [1]
"Qy": 0.097 ,  # def.: 0.097 [1]
"Qs": 0.011,  # def.: 0.011 [1]
}
dimitry_large["gamma"]      =  dimitry_large["energy"] / (dimitry_large["mass0"]*1e-9)  # [1]
dimitry_large["beta_s"]     = dimitry_large["sigma_z"] / dimitry_large["sigma_delta"]  # [m]
dimitry_large["physemit_s"] = dimitry_large["sigma_z"] * dimitry_large["sigma_delta"]  # [m]

# accelerator dictionary
acc_dict = {
"skekb_ler": skekb_ler,
"skekb_her": skekb_her,
"higgs_tanajisen": higgs_tanajisen,
"fcc_tanajisen": fcc_tanajisen,
"fcc_z_halfbeta": fcc_z_halfbeta,
"fcc_z": fcc_z,
"fcc_w": fcc_w,
"fcc_h": fcc_h,
"fcc_t": fcc_t,
"fcc_z_cdr": fcc_z_cdr,
"fcc_ww_cdr": fcc_ww_cdr,
"fcc_zh_cdr": fcc_zh_cdr,
"fcc_ttbar1_cdr": fcc_ttbar1_cdr,
"fcc_ttbar2_cdr": fcc_ttbar2_cdr,
"fcc_cw": fcc_cw,
"fcc_ho": fcc_ho,
"hl_lhc": hl_lhc_params,
"lhc": lhc,
"dimitry_small": dimitry_small,
"dimitry_large": dimitry_large,
"example_005": example_005,
}

param_vec = {
"phi": np.linspace(0,.5,6)*1e-3,  # [rad]
#"phi": np.array([0, .1, .2, .5, 1, 2, 5, 10, 20, 50, 1e2, 2e2])*1e-3, #, 5e2, 1e3])*1e-3,
"alpha": np.array([0, np.pi/2]),  # [rad]
"beta_x": np.array([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]),  # [1]
"phi_y" : np.array([0, 1, 2, 3, 4, 5])*1e-3,  # 100, 200, 500, 1000])*1e-3,  # [rad]
"phi_x" : np.array([0, 20, 40, 60, 80, 100])*1e-3,  #, 200, 300, 400, 500, 600, 700, 800, 900, 1000])*1e-3,  #15 [rad]
"n_slices": np.array([1, 2, 3, 5]),  # [1]
}
      