"""
run_topology.py

Main script to run CMB covariance matrix computations for non-trivial topologies.
Configures and executes simulations for specified topologies and parameters.
"""

import sys
from src.E1 import E1
from src.E2 import E2
from src.E3 import E3
from src.E4 import E4
from src.E5 import E5
from src.E6 import E6
import numpy as np
import time
import parameter_files.default_E1 as parameter_file
import matplotlib.pyplot as plt

def run_topology(topology, Lx, Ly, Lz, l_max, l_min=2, beta=90, alpha=90, gamma=0,
                 x0=np.array([0.0, 0.0, 0.0]), l_range=np.array([[2, 2]]),
                 lp_range=np.array([[2, 2]])):
    """Run covariance matrix computation for a specified topology.

    Args:
        topology (str): Topology type ('E1', 'E2', 'E3').
        Lx (float): Length scale in x-direction (in units of L_LSS).
        Ly (float): Length scale in y-direction (in units of L_LSS).
        Lz (float): Length scale in z-direction (in units of L_LSS).
        l_max (int): Maximum multipole.
        l_min (int, optional): Minimum multipole. Defaults to 2.
        beta (float, optional): Angle beta in degrees. Defaults to 90.
        alpha (float, optional): Angle alpha in degrees. Defaults to 90.
        gamma (float, optional): Angle gamma in degrees. Defaults to 0.
        x0 (np.ndarray, optional): Observer position. Defaults to [0, 0, 0].
        l_range (np.ndarray, optional): Multipole range [l_min, l_max]. Defaults to [[2, 2]].
        lp_range (np.ndarray, optional): Multipole range [lp_min, lp_max]. Defaults to [[2, 2]].
    """
    param = parameter_file.parameter
    param.update({
        'topology': topology,
        'c_l_accuracy': 0.99,
        'Lx': Lx,
        'Ly': Ly,
        'Lz': Lz,
        'l_max': l_max,
        'l_min': l_min,
        'beta': beta,
        'alpha': alpha,
        'gamma': gamma,
        'x0': x0,
        'do_polarization': False
    })

    if topology == 'E1':
        topo = E1(param=param, make_run_folder=True)
    elif topology == 'E2':
        topo = E2(param=param, make_run_folder=True)
    elif topology == 'E3':
        topo = E3(param=param, make_run_folder=True)
    elif topology == 'E4':
        topo = E4(param=param, make_run_folder=True)
    elif topology == 'E5':
        topo = E5(param=param, make_run_folder=True)
    elif topology == 'E6':
        topo = E6(param=param, make_run_folder=True)
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    topo.calculate_c_lmlpmp(
        normalize=False,
        plot_param={'l_ranges': l_range, 'lp_ranges': lp_range}
    )
    #to generate plots
    #C_l_type (int, optional): Correlation type (0: TT, 1: EE, 2: TE, 3: All). Defaults to 3.
    topo.plot_cov_matrix(normalize=True, C_l_type=0) 

if __name__ == '__main__':
    l_max = 30
    L_list = np.linspace(1, 1.5, 100)
    for L in L_list:
        run_topology(
            topology='E1',
            Lx=L,
            Ly=L,
            Lz=L,
            l_max=l_max,
            beta=90,
            alpha=90,
            gamma=0,
            x0=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            l_range=np.array([[2, l_max]]),
            lp_range=np.array([[2, l_max]])
        )