# TopologyPy

A Python package for computing CMB covariance matrices for non-trivial topologies (E1–E8), developed by Amirhossein Samandar under the auspices of the COMPACT Collaboration.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/TopologyPy.svg)](https://pypi.org/project/TopologyPy/)
[![Documentation](https://readthedocs.org/projects/topologypy/badge/?version=latest)](https://topologypy.readthedocs.io/)

## Overview
TopologyPy enables efficient computation of temperature and polarization covariance matrices using CAMB transfer functions, Numba-accelerated integrals, and multiprocessing. It supports topologies like the 3-torus (E1) and advanced cases (E6–E8) with parameters such as `Lx`, `r_x`, etc.

This package aligns with COMPACT's research on cosmic topology signatures in CMB data (e.g., eigenmodes and detectability in JCAP papers).

## Installation
```bash
pip install TopologyPy
