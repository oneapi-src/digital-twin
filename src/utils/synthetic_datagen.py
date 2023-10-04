# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Functions for synthetic data generation
"""
from itertools import product
import pandas as pd
import numpy as np
import os



def sub_vth(w_l, vgs, vth, temperature=300):
    """
    Helper function to calculate sub-vth current analytically.
    Uses randomized parameters to mimic
    measurement noise and manufacturing/material variability
    """
    q_charge = 1.60218e-19  # Electron charge
    k = 1.3806e-23  # Boltzman constant
    eta_cap = 1.2+0.01*np.random.normal()  # Capacitance factor randomized
    # Mobility factor/coefficient randomized
    w_l = w_l*(1+0.01*np.random.normal())
    temperature = temperature*(1+0.1*np.random.normal())
    v_thresh = w_l*np.exp(q_charge*(vgs-vth)/(eta_cap*k*temperature))
    return eta_cap, temperature, v_thresh


def sub_vth_aug(w_l, vgs, vth, temperature=300):
    """
    Helper function to calculate sub-vth current analytically.
    Uses randomized parameters to mimic
    measurement noise and manufacturing/material variability
    """
    eta_cap = 1.2+0.01*np.random.normal()  # Capacitance factor randomized
    # Mobility factor/coefficient randomized
    w_l = w_l*(1+0.01*np.random.normal())
    temperature = temperature*(1+0.1*np.random.normal())
    return eta_cap, temperature


def data_gen(linspace_factor):
    w_l_list = [1e-3*i for i in np.linspace(1, 30.5, 30*linspace_factor)]
    vgs_list = [0.01*i for i in np.linspace(1, 100.5, 100*linspace_factor)]
    vth_list = [0.05*i for i in np.linspace(21, 40.5, 40*linspace_factor)]
    comb = list(product(w_l_list, vgs_list, vth_list))

    w_l_bins = [x*0.001 for x in [0, 6, 11, 16, 21, 26, 31]]
    w_l_labels = [1, 2, 3, 4, 5, 6]
    vgs_bins = [x*0.01 for x in [0, 21, 41, 66, 81, 101]]
    vgs_labels = [1, 2, 3, 4, 5]
    vth_bins = [x*0.05 for x in [20, 26, 31, 36, 41]]
    vth_labels = [1, 2, 3, 4]

    data_dict = {'w_l': [], 'vgs': [], 'vth': [],
                 'eta': [], 'temp': [], 'sub-vth': []}
    for c in comb:
        data_dict['w_l'].append(c[0])
        data_dict['vgs'].append(c[1])
        data_dict['vth'].append(c[2])
        eta, temp, v_th = sub_vth(c[0], c[1], c[2])
        data_dict['eta'].append(eta)
        data_dict['temp'].append(temp)
        data_dict['sub-vth'].append(v_th)

    df = pd.DataFrame(data=data_dict, columns=[
                      'w_l', 'vgs', 'vth', 'eta', 'temp', 'sub-vth'])
    df['w_l_bins'] = pd.cut(df['w_l'], w_l_bins, labels=w_l_labels)
    df['vgs_bins'] = pd.cut(df['vgs'], vgs_bins, labels=vgs_labels)
    df['vth_bins'] = pd.cut(df['vth'], vth_bins, labels=vth_labels)
    df['log-leakage'] = -np.log10(df['sub-vth'])
    print("--------- Synthetic Dataset Overview ---------")
    print(df.head(5))
    print("Done ✓")
    data_dir=os.environ.get('DATA_DIR')
    saving_path=os.path.join(data_dir,"synthetic_data.csv")
    df.to_csv(os.path.join(data_dir,"synthetic_data.csv"))
    print("Data saved in: ", saving_path)
    #return df


def data_gen_aug(linspace_factor):
    w_l_list = [1e-3*i for i in np.linspace(1, 30.5, 30*linspace_factor)]
    vgs_list = [0.01*i for i in np.linspace(1, 100.5, 100*linspace_factor)]
    vth_list = [0.05*i for i in np.linspace(21, 40.5, 40*linspace_factor)]
    comb = list(product(w_l_list, vgs_list, vth_list))

    w_l_bins = [x*0.001 for x in [0, 6, 11, 16, 21, 26, 31]]
    w_l_labels = [1, 2, 3, 4, 5, 6]
    vgs_bins = [x*0.01 for x in [0, 21, 41, 66, 81, 101]]
    vgs_labels = [1, 2, 3, 4, 5]
    vth_bins = [x*0.05 for x in [20, 26, 31, 36, 41]]
    vth_labels = [1, 2, 3, 4]

    data_dict = {'w_l': [], 'vgs': [], 'vth': [],
                 'eta': [], 'temp': []}
    for c in comb:
        data_dict['w_l'].append(c[0])
        data_dict['vgs'].append(c[1])
        data_dict['vth'].append(c[2])
        eta, temp = sub_vth_aug(c[0], c[1], c[2])
        data_dict['eta'].append(eta)
        data_dict['temp'].append(temp)

    df = pd.DataFrame(data=data_dict, columns=[
                      'w_l', 'vgs', 'vth', 'eta', 'temp'])
    df['w_l_bins'] = pd.cut(df['w_l'], w_l_bins, labels=w_l_labels)
    df['vgs_bins'] = pd.cut(df['vgs'], vgs_bins, labels=vgs_labels)
    df['vth_bins'] = pd.cut(df['vth'], vth_bins, labels=vth_labels)
    print("--------- Synthetic Dataset Overview ---------")
    print(df.head(5))
    print("Done ✓")
    data_dir=os.environ.get('DATA_DIR')
    saving_path=os.path.join(data_dir,"synthetic_data_aug.csv")
    df.to_csv(os.path.join(data_dir,"synthetic_data_aug.csv"))
    print("Data saved in: ", saving_path)
    return df
