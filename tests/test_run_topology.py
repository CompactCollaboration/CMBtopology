"""
test_run_topology.py

Unit tests for the run_topology interface in TopologyPy.
These tests verify parameter validation, error handling, and basic execution for selected topologies.

Run from the project root: python -m unittest discover tests

Copyright (c) 2025 Amirhossein Samandar
Licensed under the MIT License. See LICENSE for details.
"""

import unittest
import numpy as np
from topology.run_topology import run_topology

class TestRunTopology(unittest.TestCase):
    def test_e1_valid_parameters(self):
        """Test E1 topology with valid parameters; should not raise errors."""
        try:
            run_topology(
                topology='E1',
                l_max=20,
                Lx=1.0,
                Ly=1.0,
                Lz=1.0,
                beta=90,
                alpha=90,
                gamma=0,
                do_polarization=False,
                l_range=np.array([[2, 20]]),
                lp_range=np.array([[2, 20]])
            )
        except Exception as e:
            self.fail(f"Unexpected error for valid E1 parameters: {e}")

    def test_e6_valid_parameters(self):
        """Test E6 topology with valid parameters; should not raise errors."""
        try:
            run_topology(
                topology='E6',
                l_max=15,
                Lx=1.0,
                Ly=1.0,
                Lz=1.0,
                r_x=0.5,
                r_y=0.5,
                r_z=0.5,
                do_polarization=True,
                l_range=np.array([[2, 15]]),
                lp_range=np.array([[2, 15]])
            )
        except Exception as e:
            self.fail(f"Unexpected error for valid E6 parameters: {e}")

    def test_invalid_topology(self):
        """Test unsupported topology; should raise ValueError."""
        with self.assertRaises(ValueError):
            run_topology(
                topology='InvalidTopology',
                l_max=20
            )

    def test_invalid_parameter_for_e1(self):
        """Test invalid parameter for E1 (e.g., r_x for E1); should raise ValueError."""
        with self.assertRaises(ValueError):
            run_topology(
                topology='E1',
                l_max=20,
                r_x=0.5  # Invalid for E1
            )

    def test_missing_required_parameter(self):
        """Test missing required parameter (e.g., l_max); should raise TypeError or ValueError."""
        with self.assertRaises(TypeError):
            run_topology(topology='E1')  # Missing l_max

if __name__ == '__main__':
    unittest.main()