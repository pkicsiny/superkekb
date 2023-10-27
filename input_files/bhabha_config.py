import sys
sys.path.append('/Users/pkicsiny/phd/cern/xsuite')
import input_files.config as config
import pandas as pd

# sources: 
# CDR: https://cds.cern.ch/record/2651299/files/CERN-ACC-2018-0057.pdf
# Oide's FCC week 2023 slides: https://indico.cern.ch/event/1202105/contributions/5408583/attachments/2659051/4608141/FCCWeek_Optics_Oide_230606.pdf

#############################
# revolution frequency [Hz] #
#############################

f_rev_dict = {"cdr": 3e8 / 97.756e3, "4ip": 3e8 / 90658.816}

#################
# number of IPs #
#################

n_ip_dict = {"cdr": 2, "4ip": 4}

###################################
# bunch intensity [e] from config #
###################################

bunchint_cdr = {
    "cdr":{
     "z_cdr":config.fcc_z_cdr["bunch_intensity"],
    "ww_cdr":config.fcc_ww_cdr["bunch_intensity"],
    "zh_cdr":config.fcc_zh_cdr["bunch_intensity"],
"ttbar1_cdr":config.fcc_ttbar1_cdr["bunch_intensity"],
"ttbar2_cdr":config.fcc_ttbar2_cdr["bunch_intensity"], 
    }, "4ip":{
         "z":config.fcc_z["bunch_intensity"],
         "w":config.fcc_w["bunch_intensity"],
         "h":config.fcc_h["bunch_intensity"],
         "t":config.fcc_t["bunch_intensity"],
    }
}


#####################
# lumunosity [m^-2] #
#####################

# lumi fine from guineapig C and bhabha logbook, cdr: [cutxyz=60x200x60 nxy=500x500x100] 4ip: [cutxyz=CUTXx4x4 nxy=64x64x100]
lumi_guinea = {
    "cdr":{
      "z_cdr":  4.32 * 1e32,
     "ww_cdr":  4.58 * 1e32,
     "zh_cdr":  8.52 * 1e32,
 "ttbar1_cdr": 10.31 * 1e32,
 "ttbar2_cdr": 10.69 * 1e32,
    }, "4ip":{
          "z":  3.45 * 1e32,
          "w":  5.28 * 1e32,
          "h":  5.19 * 1e32,
          "t": 10.22 * 1e32,
    }
}

# from n113b_lxplus_bhabha_lumi_count_convergence, guinea[6:].mean with cutx, 4, 4, nxy scanned
lumi_guinea_new ={
    "cdr":{
      "z_cdr": 4.674272*1e32,
     "ww_cdr": 4.907765*1e32,
     "zh_cdr": 9.120521*1e32,
 "ttbar1_cdr": 10.98323*1e32,
 "ttbar2_cdr": 11.33219*1e32,
    }, "4ip":{
          "z": 4.045142*1e32,
          "w": 5.325122*1e32,
          "h": 5.256648*1e32,
          "t": 10.18669*1e32,
    }
}

# ws from n89_luminosity_github_test and bhabha notebook, 1e4 mp
lumi_xsuite = {
    "cdr":{
      "z_cdr": 4.48*1e32,
     "ww_cdr": 4.51*1e32,
     "zh_cdr": 8.30*1e32,
 "ttbar1_cdr": 9.86*1e32,
 "ttbar2_cdr": 10.3*1e32,
    }, "4ip":{
          "z": 3.95*1e32,
          "w": 4.79*1e32,
          "h": 4.73*1e32,
          "t": 9.06*1e32,
    }
}

# ss from n89_luminosity_github_test and bhabha notebook, beam 1, 1e4 mp
lumi_xsuite_ss = {
    "cdr":{
      "z_cdr":  4.85*1e32,
     "ww_cdr":  4.82*1e32,
     "zh_cdr":  8.87*1e32,
 "ttbar1_cdr": 10.63*1e32,
 "ttbar2_cdr": 11.10*1e32,
    }, "4ip":{
          "z":  4.23*1e32,
          "w":  5.01*1e32,
          "h":  5.04*1e32,
          "t":  9.95*1e32,
    }
}

# 1IP 1 bunch lumi from cdr or oides fccweek slides, from bhabha logbook
lumi_cdr = { 
    "cdr":{
      "z_cdr":  4.60 * 1e32,
     "ww_cdr":  4.66 * 1e32,
     "zh_cdr":  8.63 * 1e32,
 "ttbar1_cdr": 10.16 * 1e32,
 "ttbar2_cdr": 10.76 * 1e32, 
    }, "4ip":{  # this is with sigma_y_bs, ~1.5 less than simulated
          "z":  2.66 * 1e32,
          "w":  3.40 * 1e32,
          "h":  3.43 * 1e32,
          "t":  6.30 * 1e32,
    }
}
# formula from n89_luminosity_github_test, from bhabha logbook
lumi_formula = {
    "cdr":{
      "z_cdr":  4.47 * 1e32,
     "ww_cdr":  4.77 * 1e32,
     "zh_cdr":  8.86 * 1e32,
 "ttbar1_cdr": 10.91 * 1e32,
 "ttbar2_cdr": 11.45 * 1e32,
    }, "4ip":{  # this is with sigma_y_bs, ~1.5 less than simulated
          "z":  3.04 * 1e32, 
          "w":  4.24 * 1e32,
          "h":  3.92 * 1e32,
          "t":  7.34 * 1e32,
    }
}

# formula from n89_luminosity_github_test, from bhabha logbook, using sigma_y_lattice
lumi_formula_lattice = {
"4ip":{
    "z":  4.15 * 1e32,  
    "w":  5.63 * 1e32,
    "h":  5.03 * 1e32,
    "t":  9.79 * 1e32,
    }
}

#####################################
# cross sections from BBBREM [m^-2] #
#####################################

# bbbrem mean from 100 runs without and with beamsize effect, n113_lxplus_bhabha_tracking
wo_beamsize_bbbrem={  # k=0.01
      "z_cdr":.319595*1e-28, 
     "ww_cdr":.333325*1e-28, 
     "zh_cdr":.343229*1e-28, 
 "ttbar1_cdr":.349812*1e-28, 
 "ttbar2_cdr":.353252*1e-28,
}
wi_beamsize_bbbrem={  # k=0.01
     "z_cdr": .151306*1e-28,
    "ww_cdr": .156034*1e-28,
    "zh_cdr": .154402*1e-28,
"ttbar1_cdr": .161813*1e-28,
"ttbar2_cdr": .162229*1e-28,
}
wo_beamsize_dynap_cdr_bbbrem={  # k=CDR
    "cdr":{
     "z_cdr": .296196*1e-28,
    "ww_cdr": .309133*1e-28,
    "zh_cdr": .294169*1e-28,
"ttbar1_cdr": .267066*1e-28,
"ttbar2_cdr": .267870*1e-28,
    }, "4ip": {
         "z": .318713*1e-28,
         "w": .332649*1e-28,
         "h": .299678*1e-28,
         "t": .267870*1e-28,   
    }
}
wi_beamsize_dynap_cdr_bbbrem={  # k=CDR
    "cdr":{
     "z_cdr": .141507*1e-28,
    "ww_cdr": .145832*1e-28,
    "zh_cdr": .134095*1e-28,
"ttbar1_cdr": .125361*1e-28,
"ttbar2_cdr": .125690*1e-28,
    }, "4ip": {
         "z": .148719*1e-28,
         "w": .154139*1e-28,
         "h": .134116*1e-28,
         "t": .120096*1e-28,
    }
}
wo_beamsize_dynap_2em2_bbbrem={  # k=0.02
     "z_cdr": .259847*1e-28,
    "ww_cdr": .271332*1e-28,
    "zh_cdr": .279612*1e-28,
"ttbar1_cdr": .287323*1e-28,
"ttbar2_cdr": .288182*1e-28,   
}

wi_beamsize_dynap_2em2_bbbrem={  # k=0.02
     "z_cdr" : .125476*1e-28,
    "ww_cdr" : .129323*1e-28,
    "zh_cdr" : .127945*1e-28,
"ttbar1_cdr" : .134100*1e-28,
"ttbar2_cdr" : .134447*1e-28,    
}

##########################
# bhabha lifetimes tau [min] = N_b/(σ*L_int*f_rev*N_ip*60) #
##########################

# k = 0.01
wo_beamsize_bbbrem_tau={
      "z_cdr": bunchint_cdr["cdr"][      "z_cdr"] / (wo_beamsize_bbbrem[     "z_cdr"] * lumi_cdr["cdr"][      "z_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "ww_cdr": bunchint_cdr["cdr"][     "ww_cdr"] / (wo_beamsize_bbbrem[    "ww_cdr"] * lumi_cdr["cdr"][     "ww_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "zh_cdr": bunchint_cdr["cdr"][     "zh_cdr"] / (wo_beamsize_bbbrem[    "zh_cdr"] * lumi_cdr["cdr"][     "zh_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar1_cdr": bunchint_cdr["cdr"][ "ttbar1_cdr"] / (wo_beamsize_bbbrem["ttbar1_cdr"] * lumi_cdr["cdr"][ "ttbar1_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar2_cdr": bunchint_cdr["cdr"][ "ttbar2_cdr"] / (wo_beamsize_bbbrem["ttbar2_cdr"] * lumi_cdr["cdr"][ "ttbar2_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60),}
wo_beamsize_bbbrem_tau_df = pd.Series(wo_beamsize_bbbrem_tau)

wi_beamsize_bbbrem_tau={
      "z_cdr": bunchint_cdr["cdr"][      "z_cdr"] / (wi_beamsize_bbbrem[     "z_cdr"] * lumi_cdr["cdr"][      "z_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "ww_cdr": bunchint_cdr["cdr"][     "ww_cdr"] / (wi_beamsize_bbbrem[    "ww_cdr"] * lumi_cdr["cdr"][     "ww_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "zh_cdr": bunchint_cdr["cdr"][     "zh_cdr"] / (wi_beamsize_bbbrem[    "zh_cdr"] * lumi_cdr["cdr"][     "zh_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar1_cdr": bunchint_cdr["cdr"][ "ttbar1_cdr"] / (wi_beamsize_bbbrem["ttbar1_cdr"] * lumi_cdr["cdr"][ "ttbar1_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar2_cdr": bunchint_cdr["cdr"][ "ttbar2_cdr"] / (wi_beamsize_bbbrem["ttbar2_cdr"] * lumi_cdr["cdr"][ "ttbar2_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60),}
wi_beamsize_bbbrem_tau_df = pd.Series(wi_beamsize_bbbrem_tau)

# k = cdr
wo_beamsize_dynap_cdr_bbbrem_tau={
    "cdr":{
      "z_cdr": bunchint_cdr["cdr"][      "z_cdr"] / (wo_beamsize_dynap_cdr_bbbrem["cdr"][     "z_cdr"] * lumi_cdr["cdr"][      "z_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "ww_cdr": bunchint_cdr["cdr"][     "ww_cdr"] / (wo_beamsize_dynap_cdr_bbbrem["cdr"][    "ww_cdr"] * lumi_cdr["cdr"][     "ww_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "zh_cdr": bunchint_cdr["cdr"][     "zh_cdr"] / (wo_beamsize_dynap_cdr_bbbrem["cdr"][    "zh_cdr"] * lumi_cdr["cdr"][     "zh_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar1_cdr": bunchint_cdr["cdr"][ "ttbar1_cdr"] / (wo_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar1_cdr"] * lumi_cdr["cdr"][ "ttbar1_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar2_cdr": bunchint_cdr["cdr"][ "ttbar2_cdr"] / (wo_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar2_cdr"] * lumi_cdr["cdr"][ "ttbar2_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60),
    }, 
    "4ip":{
          "z": bunchint_cdr["4ip"]["z"] / (wo_beamsize_dynap_cdr_bbbrem["4ip"]["z"] * lumi_cdr["4ip"]["z"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "w": bunchint_cdr["4ip"]["w"] / (wo_beamsize_dynap_cdr_bbbrem["4ip"]["w"] * lumi_cdr["4ip"]["w"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "h": bunchint_cdr["4ip"]["h"] / (wo_beamsize_dynap_cdr_bbbrem["4ip"]["h"] * lumi_cdr["4ip"]["h"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "t": bunchint_cdr["4ip"]["t"] / (wo_beamsize_dynap_cdr_bbbrem["4ip"]["t"] * lumi_cdr["4ip"]["t"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
    }
}
wo_beamsize_dynap_cdr_bbbrem_tau_df = pd.DataFrame(wo_beamsize_dynap_cdr_bbbrem_tau)

wi_beamsize_dynap_cdr_bbbrem_tau={
    "cdr":{
      "z_cdr": bunchint_cdr["cdr"][      "z_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][     "z_cdr"] * lumi_cdr["cdr"][      "z_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "ww_cdr": bunchint_cdr["cdr"][     "ww_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][    "ww_cdr"] * lumi_cdr["cdr"][     "ww_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "zh_cdr": bunchint_cdr["cdr"][     "zh_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][    "zh_cdr"] * lumi_cdr["cdr"][     "zh_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar1_cdr": bunchint_cdr["cdr"][ "ttbar1_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar1_cdr"] * lumi_cdr["cdr"][ "ttbar1_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar2_cdr": bunchint_cdr["cdr"][ "ttbar2_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar2_cdr"] * lumi_cdr["cdr"][ "ttbar2_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60),
    }, 
}
wi_beamsize_dynap_cdr_bbbrem_tau_df = pd.DataFrame(wi_beamsize_dynap_cdr_bbbrem_tau)

wi_beamsize_dynap_cdr_bbbrem_lumi_guinea_tau={
    "cdr":{
      "z_cdr": bunchint_cdr["cdr"][      "z_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][     "z_cdr"] * lumi_guinea["cdr"][      "z_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "ww_cdr": bunchint_cdr["cdr"][     "ww_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][    "ww_cdr"] * lumi_guinea["cdr"][     "ww_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "zh_cdr": bunchint_cdr["cdr"][     "zh_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][    "zh_cdr"] * lumi_guinea["cdr"][     "zh_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar1_cdr": bunchint_cdr["cdr"][ "ttbar1_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar1_cdr"] * lumi_guinea["cdr"][ "ttbar1_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar2_cdr": bunchint_cdr["cdr"][ "ttbar2_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar2_cdr"] * lumi_guinea["cdr"][ "ttbar2_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60),
    }, 
    "4ip":{ 
          "z": bunchint_cdr["4ip"]["z"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["z"] * lumi_guinea["4ip"]["z"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "w": bunchint_cdr["4ip"]["w"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["w"] * lumi_guinea["4ip"]["w"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "h": bunchint_cdr["4ip"]["h"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["h"] * lumi_guinea["4ip"]["h"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "t": bunchint_cdr["4ip"]["t"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["t"] * lumi_guinea["4ip"]["t"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60),
         }
}
wi_beamsize_dynap_cdr_bbbrem_lumi_guinea_tau_df = pd.DataFrame(wi_beamsize_dynap_cdr_bbbrem_lumi_guinea_tau)

wi_beamsize_dynap_cdr_bbbrem_lumi_guinea_new_tau={
    "cdr":{
      "z_cdr": bunchint_cdr["cdr"][      "z_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][     "z_cdr"] * lumi_guinea_new["cdr"][      "z_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "ww_cdr": bunchint_cdr["cdr"][     "ww_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][    "ww_cdr"] * lumi_guinea_new["cdr"][     "ww_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "zh_cdr": bunchint_cdr["cdr"][     "zh_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][    "zh_cdr"] * lumi_guinea_new["cdr"][     "zh_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar1_cdr": bunchint_cdr["cdr"][ "ttbar1_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar1_cdr"] * lumi_guinea_new["cdr"][ "ttbar1_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar2_cdr": bunchint_cdr["cdr"][ "ttbar2_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar2_cdr"] * lumi_guinea_new["cdr"][ "ttbar2_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60),
    }, 
    "4ip":{ 
          "z": bunchint_cdr["4ip"]["z"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["z"] * lumi_guinea_new["4ip"]["z"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "w": bunchint_cdr["4ip"]["w"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["w"] * lumi_guinea_new["4ip"]["w"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "h": bunchint_cdr["4ip"]["h"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["h"] * lumi_guinea_new["4ip"]["h"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "t": bunchint_cdr["4ip"]["t"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["t"] * lumi_guinea_new["4ip"]["t"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60),
         }
}
wi_beamsize_dynap_cdr_bbbrem_lumi_guinea_new_tau_df = pd.DataFrame(wi_beamsize_dynap_cdr_bbbrem_lumi_guinea_new_tau)

wi_beamsize_dynap_cdr_bbbrem_lumi_xsuite_ws_tau={
    "cdr":{
      "z_cdr": bunchint_cdr["cdr"][      "z_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][     "z_cdr"] * lumi_xsuite["cdr"][      "z_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "ww_cdr": bunchint_cdr["cdr"][     "ww_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][    "ww_cdr"] * lumi_xsuite["cdr"][     "ww_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "zh_cdr": bunchint_cdr["cdr"][     "zh_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][    "zh_cdr"] * lumi_xsuite["cdr"][     "zh_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar1_cdr": bunchint_cdr["cdr"][ "ttbar1_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar1_cdr"] * lumi_xsuite["cdr"][ "ttbar1_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar2_cdr": bunchint_cdr["cdr"][ "ttbar2_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar2_cdr"] * lumi_xsuite["cdr"][ "ttbar2_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60),
    }, 
    "4ip":{ 
          "z": bunchint_cdr["4ip"]["z"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["z"] * lumi_xsuite["4ip"]["z"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "w": bunchint_cdr["4ip"]["w"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["w"] * lumi_xsuite["4ip"]["w"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "h": bunchint_cdr["4ip"]["h"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["h"] * lumi_xsuite["4ip"]["h"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "t": bunchint_cdr["4ip"]["t"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["t"] * lumi_xsuite["4ip"]["t"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60),
         }
}
wi_beamsize_dynap_cdr_bbbrem_lumi_xsuite_ws_tau_df = pd.DataFrame(wi_beamsize_dynap_cdr_bbbrem_lumi_xsuite_ws_tau)

wi_beamsize_dynap_cdr_bbbrem_lumi_xsuite_ss_tau={
    "cdr":{
      "z_cdr": bunchint_cdr["cdr"][      "z_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][     "z_cdr"] * lumi_xsuite_ss["cdr"][      "z_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "ww_cdr": bunchint_cdr["cdr"][     "ww_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][    "ww_cdr"] * lumi_xsuite_ss["cdr"][     "ww_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "zh_cdr": bunchint_cdr["cdr"][     "zh_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][    "zh_cdr"] * lumi_xsuite_ss["cdr"][     "zh_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar1_cdr": bunchint_cdr["cdr"][ "ttbar1_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar1_cdr"] * lumi_xsuite_ss["cdr"][ "ttbar1_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar2_cdr": bunchint_cdr["cdr"][ "ttbar2_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar2_cdr"] * lumi_xsuite_ss["cdr"][ "ttbar2_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60),
    }, 
    "4ip":{ 
          "z": bunchint_cdr["4ip"]["z"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["z"] * lumi_xsuite_ss["4ip"]["z"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "w": bunchint_cdr["4ip"]["w"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["w"] * lumi_xsuite_ss["4ip"]["w"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "h": bunchint_cdr["4ip"]["h"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["h"] * lumi_xsuite_ss["4ip"]["h"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "t": bunchint_cdr["4ip"]["t"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["t"] * lumi_xsuite_ss["4ip"]["t"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60),
         }
}
wi_beamsize_dynap_cdr_bbbrem_lumi_xsuite_ss_tau_df = pd.DataFrame(wi_beamsize_dynap_cdr_bbbrem_lumi_xsuite_ss_tau)

wi_beamsize_dynap_cdr_bbbrem_lumi_formula_tau={
    "cdr":{
      "z_cdr": bunchint_cdr["cdr"][      "z_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][     "z_cdr"] * lumi_formula["cdr"][      "z_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "ww_cdr": bunchint_cdr["cdr"][     "ww_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][    "ww_cdr"] * lumi_formula["cdr"][     "ww_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "zh_cdr": bunchint_cdr["cdr"][     "zh_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"][    "zh_cdr"] * lumi_formula["cdr"][     "zh_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar1_cdr": bunchint_cdr["cdr"][ "ttbar1_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar1_cdr"] * lumi_formula["cdr"][ "ttbar1_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar2_cdr": bunchint_cdr["cdr"][ "ttbar2_cdr"] / (wi_beamsize_dynap_cdr_bbbrem["cdr"]["ttbar2_cdr"] * lumi_formula["cdr"][ "ttbar2_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60),
    }, 
    "4ip":{ 
          "z": bunchint_cdr["4ip"]["z"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["z"] * lumi_formula_lattice["4ip"]["z"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "w": bunchint_cdr["4ip"]["w"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["w"] * lumi_formula_lattice["4ip"]["w"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "h": bunchint_cdr["4ip"]["h"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["h"] * lumi_formula_lattice["4ip"]["h"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60), 
          "t": bunchint_cdr["4ip"]["t"] / (wi_beamsize_dynap_cdr_bbbrem["4ip"]["t"] * lumi_formula_lattice["4ip"]["t"] * n_ip_dict["4ip"] * f_rev_dict["4ip"] * 60),
         }
}
wi_beamsize_dynap_cdr_bbbrem_lumi_formula_tau_df = pd.DataFrame(wi_beamsize_dynap_cdr_bbbrem_lumi_formula_tau)

# k = 0.02
wo_beamsize_dynap_2em2_bbbrem_tau={
      "z_cdr": bunchint_cdr["cdr"][      "z_cdr"] / (wo_beamsize_dynap_2em2_bbbrem[     "z_cdr"] * lumi_cdr["cdr"][      "z_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "ww_cdr": bunchint_cdr["cdr"][     "ww_cdr"] / (wo_beamsize_dynap_2em2_bbbrem[    "ww_cdr"] * lumi_cdr["cdr"][     "ww_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "zh_cdr": bunchint_cdr["cdr"][     "zh_cdr"] / (wo_beamsize_dynap_2em2_bbbrem[    "zh_cdr"] * lumi_cdr["cdr"][     "zh_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar1_cdr": bunchint_cdr["cdr"][ "ttbar1_cdr"] / (wo_beamsize_dynap_2em2_bbbrem["ttbar1_cdr"] * lumi_cdr["cdr"][ "ttbar1_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar2_cdr": bunchint_cdr["cdr"][ "ttbar2_cdr"] / (wo_beamsize_dynap_2em2_bbbrem["ttbar2_cdr"] * lumi_cdr["cdr"][ "ttbar2_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60),}
wo_beamsize_dynap_2em2_bbbrem_tau_df = pd.Series(wo_beamsize_dynap_2em2_bbbrem_tau)

wi_beamsize_dynap_2em2_bbbrem_tau={
      "z_cdr": bunchint_cdr["cdr"][      "z_cdr"] / (wi_beamsize_dynap_2em2_bbbrem[     "z_cdr"] * lumi_cdr["cdr"][      "z_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "ww_cdr": bunchint_cdr["cdr"][     "ww_cdr"] / (wi_beamsize_dynap_2em2_bbbrem[    "ww_cdr"] * lumi_cdr["cdr"][     "ww_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
     "zh_cdr": bunchint_cdr["cdr"][     "zh_cdr"] / (wi_beamsize_dynap_2em2_bbbrem[    "zh_cdr"] * lumi_cdr["cdr"][     "zh_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar1_cdr": bunchint_cdr["cdr"][ "ttbar1_cdr"] / (wi_beamsize_dynap_2em2_bbbrem["ttbar1_cdr"] * lumi_cdr["cdr"][ "ttbar1_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60), 
 "ttbar2_cdr": bunchint_cdr["cdr"][ "ttbar2_cdr"] / (wi_beamsize_dynap_2em2_bbbrem["ttbar2_cdr"] * lumi_cdr["cdr"][ "ttbar2_cdr"] * n_ip_dict["cdr"] * f_rev_dict["cdr"] * 60),}
wi_beamsize_dynap_2em2_bbbrem_tau_df = pd.Series(wi_beamsize_dynap_2em2_bbbrem_tau)
