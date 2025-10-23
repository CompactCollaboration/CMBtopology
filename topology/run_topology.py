"""
run_topology.py

Main interface for the TopologyPy package to run CMB covariance matrix computations
for non-trivial topologies. Supports both programmatic and command-line usage.
"""

import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
from .src.E1 import E1
from .src.E2 import E2
from .src.E3 import E3
from .src.E4 import E4
from .src.E5 import E5
from .src.E6 import E6
from .src.E7 import E7
from .src.E8 import E8
from .src.E9 import E9
from .src.E10 import E10
from .src.topology import Topology  # Updated import

def run_topology(
    topology,
    l_max,
    l_min=2,
    do_polarization=False,
    number_of_a_lm_realizations=1,
    l_range=None,
    lp_range=None,
    **topology_params
):
    """Run covariance matrix computation for a specified topology.

    This function is the primary interface for the TopologyPy package, allowing users to compute
    CMB covariance matrices for various topologies. It loads default parameters from the topology's
    parameter file and updates them with provided topology-specific and general parameters.
    Invalid parameters (not present in the default parameter file) raise an error.

    Args:
        topology (str): Topology type (e.g., 'E1', 'E2', ..., 'E8').
        l_max (int): Maximum multipole.
        l_min (int, optional): Minimum multipole. Defaults to 2.
        do_polarization (bool, optional): Whether to include polarization in calculations. Defaults to False.
        number_of_a_lm_realizations (int, optional): Number of a_lm realizations. Defaults to 1.
        l_range (np.ndarray, optional): Multipole range [l_min, l_max]. If None, defaults to [[l_min, l_max]].
        lp_range (np.ndarray, optional): Multipole range [lp_min, lp_max]. If None, defaults to [[l_min, l_max]].
        **topology_params: Topology-specific parameters (e.g., Lx, Ly, Lz for E1; LAx, LAy for E7).
                          Must match keys in the default parameter file for the topology.

    Raises:
        ValueError: If an invalid parameter is provided or the topology is unsupported.
        ImportError: If the parameter file for the topology is not found.

    Example:
        >>> from topology.run_topology import run_topology
        >>> run_topology('E1', l_max=30, Lx=1.0, Ly=1.0, Lz=1.0, beta=90, alpha=90, do_polarization=True)
        >>> run_topology('E6', l_max=20, Lx=1.0, Ly=1.0, Lz=1.0, r_x=0.5, r_y=0.5, r_z=0.5)
        >>> run_topology('E7', l_max=20, LAx=1.0, LAy=1.0, L1y=1.0, L2x=1.0, L2z=1.0)
    """
    if l_range is None:
        l_range = np.array([[l_min, l_max]])
    if lp_range is None:
        lp_range = np.array([[l_min, l_max]])

    # Dynamically import the parameter file
    try:
        parameter_module = importlib.import_module(f".parameter_files.default_{topology}", package="topology")
        param = parameter_module.parameter
    except ImportError:
        raise ValueError(f"Parameter file for topology {topology} not found. Expected: topology/parameter_files/default_{topology}.py")

    # Validate topology-specific parameters
    for key in topology_params:
        if key not in param:
            raise ValueError(f"Invalid parameter '{key}' for topology '{topology}'. Valid parameters are: {list(param.keys())}")

    # Update parameters
    param.update(topology_params)
    param.update({
        'topology': topology,
        'c_l_accuracy': 0.99,
        'l_max': l_max,
        'l_min': l_min,
        'do_polarization': do_polarization,
        'number_of_a_lm_realizations': number_of_a_lm_realizations
    })

    # Initialize topology class
    topology_classes = {
        'E1': E1,
        'E2': E2,
        'E3': E3,
        'E4': E4,
        'E5': E5,
        'E6': E6,
        'E7': E7,
        'E8': E8
    }

    if topology not in topology_classes:
        raise ValueError(f"Unsupported topology: {topology}. Supported topologies are: {list(topology_classes.keys())}")

    topo = topology_classes[topology](param=param, make_run_folder=True)

    topo.calculate_c_lmlpmp(
        normalize=False,
        plot_param={'l_ranges': l_range, 'lp_ranges': lp_range}
    )
    topo.plot_cov_matrix(normalize=True, C_l_type=0)

def main():
    """Command-line interface for TopologyPy."""
    parser = argparse.ArgumentParser(
        description="TopologyPy: Compute CMB covariance matrices for non-trivial topologies."
    )
    parser.add_argument(
        "--topology",
        type=str,
        required=True,
        help="Topology type (e.g., E1, E6, E7, E8)"
    )
    parser.add_argument(
        "--l_max",
        type=int,
        required=True,
        help="Maximum multipole"
    )
    parser.add_argument(
        "--l_min",
        type=int,
        default=2,
        help="Minimum multipole (default: 2)"
    )
    parser.add_argument(
        "--do_polarization",
        action="store_true",
        help="Include polarization in calculations"
    )
    parser.add_argument(
        "--number_of_a_lm_realizations",
        type=int,
        default=1,
        help="Number of a_lm realizations (default: 1)"
    )
    # Allow topology-specific parameters as optional arguments
    parser.add_argument("--Lx", type=float, help="Length scale in x-direction (for E1-E6)")
    parser.add_argument("--Ly", type=float, help="Length scale in y-direction (for E1-E6)")
    parser.add_argument("--Lz", type=float, help="Length scale in z-direction (for E1-E6)")
    parser.add_argument("--beta", type=float, help="Angle beta in degrees (for E1-E6)")
    parser.add_argument("--alpha", type=float, help="Angle alpha in degrees (for E1-E6)")
    parser.add_argument("--gamma", type=float, default=0, help="Angle gamma in degrees (for E1-E6)")
    parser.add_argument("--r_x", type=float, help="r_x parameter (for E6)")
    parser.add_argument("--r_y", type=float, help="r_y parameter (for E6)")
    parser.add_argument("--r_z", type=float, help="r_z parameter (for E6)")
    parser.add_argument("--LAx", type=float, help="LAx parameter (for E7-E8)")
    parser.add_argument("--LAy", type=float, help="LAy parameter (for E7-E8)")
    parser.add_argument("--L1y", type=float, help="L1y parameter (for E7)")
    parser.add_argument("--L2x", type=float, help="L2x parameter (for E7)")
    parser.add_argument("--L2z", type=float, help="L2z parameter (for E7)")
    parser.add_argument("--LBx", type=float, help="LBx parameter (for E8)")
    parser.add_argument("--LBz", type=float, help="LBz parameter (for E8)")
    parser.add_argument("--LCy", type=float, help="LCy parameter (for E8)")
    parser.add_argument(
        "--x0",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Observer position as three floats (default: [0.0, 0.0, 0.0])"
    )

    args = parser.parse_args()

    # Collect topology-specific parameters
    topology_params = {}
    for param in [
        'Lx', 'Ly', 'Lz', 'beta', 'alpha', 'gamma',
        'r_x', 'r_y', 'r_z',
        'LAx', 'LAy', 'L1y', 'L2x', 'L2z', 'LBx', 'LBz', 'LCy', 'x0'
    ]:
        if hasattr(args, param) and getattr(args, param) is not None:
            topology_params[param] = getattr(args, param)

    # Convert x0 to numpy array
    if 'x0' in topology_params:
        topology_params['x0'] = np.array(topology_params['x0'], dtype=np.float64)

    # Run the topology simulation
    run_topology(
        topology=args.topology,
        l_max=args.l_max,
        l_min=args.l_min,
        do_polarization=args.do_polarization,
        number_of_a_lm_realizations=args.number_of_a_lm_realizations,
        **topology_params
    )

if __name__ == '__main__':
    main()