"""
example_usage.py

A Python script demonstrating usage of the run_topology interface from TopologyPy.
This script runs examples for topologies E1, E6, E7, and E10, showcasing different parameters.
The normalize option is explicitly set to True or False for each example, as required for
calculations and plotting of covariance matrices.

To run: python examples/example_usage.py

These examples are inspired by the COMPACT Collaboration papers:
- "Cosmic topology. Part IIa. Eigenmodes, correlation matrices, and detectability of orientable Euclidean manifolds" (arXiv:2306.17112)
- "Cosmic topology. Part IIb. Eigenmodes, correlation matrices, and detectability of non-orientable Euclidean manifolds" (arXiv:2510.05030)

Copyright (c) 2025 Amirhossein Samandar
Licensed under the MIT License. See LICENSE for details.
"""

import numpy as np
from topology.run_topology import run_topology

def run_examples():
    # Example 1: E1 topology (3-torus, orientable from Part IIa) with normalize=True
    print("Running Example 1: E1 Topology (Orientable Euclidean Manifold) with normalize=True")
    run_topology(
        topology='E1',
        l_max=5,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        beta=90,
        alpha=90,
        gamma=0,
        do_polarization=False,
        normalize=True,
        l_range=np.array([[2, 5]]),
        lp_range=np.array([[2, 5]])
    )
    print("Example 1 completed.\n")

    # Example 2: E6 topology with custom parameters (orientable from Part IIa) with normalize=False
    print("Running Example 2: E6 Topology (Orientable Euclidean Manifold) with normalize=False")
    run_topology(
        topology='E6',
        l_max=5,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        r_x=0.5,
        r_y=0.5,
        r_z=0.5,
        x0 = np.array([0.1, 0.2, 0.3]),
        do_polarization=False,
        normalize=False,
        l_range=np.array([[2, 5]]),
        lp_range=np.array([[2, 5]])
    )
    print("Example 2 completed.\n")

    # Example 3: E7 topology (non-orientable from Part IIb) with normalize=True
    print("Running Example 3: E7 Topology (Non-orientable Euclidean Manifold) with normalize=True")
    run_topology(
        topology='E7',
        l_max=5,
        LAx=1.0,
        LAy=1.0,
        L1y=1.0,
        L2x=1.0,
        L2z=1.0,
        x0 = np.array([0.1, 0.2, 0.3]),
        do_polarization=False,
        normalize=True,
        l_range=np.array([[2, 5]]),
        lp_range=np.array([[2, 5]])
    )
    print("Example 3 completed.\n")

    # Example 4: E10 topology with default parameters (non-orientable from Part IIb) with normalize=False
    print("Running Example 4: E10 Topology with normalize=False")
    run_topology(
        topology='E10',
        l_max=5,
        LAx=1.0,
        LAy=1.0,
        LBx=1.0,
        LBz=1.0,
        LCy=1.0,
        do_polarization=False,
        normalize=False,
        l_range=np.array([[2, 5]]),
        lp_range=np.array([[2, 5]])
    )
    print("Example 4 completed.\n")

if __name__ == '__main__':
    run_examples()