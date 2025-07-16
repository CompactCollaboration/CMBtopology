"""
test_topology.py

Unit tests for the cmb_topology library.
"""

import pytest
import numpy as np
from cmb_topology import E1, run_topology

# Placeholder for tests (requires tools.py and default_E1.py for full functionality)
def test_e1_initialization():
    """Test initialization of E1 topology."""
    param = {
        'topology': 'E1',
        'Lx': 1.2,
        'Ly': 1.2,
        'Lz': 1.2,
        'l_max': 10,
        'l_min': 2,
        'beta': 90,
        'alpha': 90,
        'gamma': 0,
        'x0': np.array([0.0, 0.0, 0.0]),
        'c_l_accuracy': 0.99,
        'do_polarization': False
    }
    topo = E1(param=param, make_run_folder=False)
    assert topo.topology == 'E1'
    assert topo.l_max == 10
    assert topo.c_l_accuracy == 0.99

# Add more tests for covariance computation, plotting, etc., once tools.py is available

if __name__ == "__main__":
    pytest.main()