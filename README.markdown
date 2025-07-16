# CMB Topology Covariance Matrix Library

The `cmb_topology` library provides tools for computing the covariance matrices of the Cosmic Microwave Background (CMB) temperature and polarization for universes with non-trivial topologies, as described in the associated paper (see Section: Numerical Analysis). It calculates the 2-point correlation matrix of spherical-harmonic coefficients, $C_{\ell m \ell' m'}^{E_{i};XY}$, for compact, orientable Euclidean topologies (currently E1-E6). The library is designed for researchers studying topological signatures in the CMB, with a focus on modularity and performance.

## Features

- Computes CMB covariance matrices for temperature (TT), polarization (EE, TE), and cross-correlations.
- Supports the E1-E6 orientable Euclidean topologies with configurable length scales ($L_x$, $L_y$, $L_z$) and angles ($\alpha$, $\beta$, $\gamma$).
- Integrates with CAMB for transfer functions and primordial power spectra using Planck 2018 $\Lambda$CDM parameters.
- Uses a multipole-dependent wavevector cutoff ($|\vec{k}_{\text{max}}(\ell)|$) for 99% accuracy relative to $\Lambda$CDM.
- Employs Numba and multiprocessing for performance optimization.
- Generates rescaled covariance matrix plots ($\Xi^{E_{i}}_{\ell m \ell' m'}$).

## Installation

### Prerequisites
- Python 3.12+
- Dependencies: `camb`, `numpy`, `scipy`, `matplotlib`, `numba`, `tqdm`, `spherical`, `quaternionic`

### Install from PyPI (Future)
Once published to PyPI:
```bash
pip install cmb-topology
```

### Install from Source
1. Clone the repository:
   ```bash
   git clone https://github.com/CompactCollaboration/cmb-topology-covariance.git
   cd cmb-topology-covariance
   ```
2. Install the library:
   ```bash
   pip install .
   ```
   Or for development:
   ```bash
   pip install -e .
   ```

3. Verify installation:
   ```python
   import cmb_topology
   print(cmb_topology.__version__)
   ```

## Usage

### Basic Example
Compute the covariance matrix for an E1 topology:
```python
from cmb_topology import E1, run_topology
import numpy as np

param = {
    'topology': 'E1',
    'Lx': 1.2,
    'Ly': 1.2,
    'Lz': 1.2,
    'l_max': 30,
    'l_min': 2,
    'beta': 90,
    'alpha': 90,
    'gamma': 0,
    'x0': np.array([0.0, 0.0, 0.0]),
    'c_l_accuracy': 0.99,
    'do_polarization': False
}

# Initialize topology
topo = E1(param=param, make_run_folder=True)

# Compute covariance matrix
cov_matrix = topo.calculate_c_lmlpmp(
    normalize=False,
    plot_param={'l_ranges': np.array([[2, 30]]), 'lp_ranges': np.array([[2, 30]])}
)

# Optionally plot (uncomment in code)
# topo.plot_cov_matrix(normalize=True, C_l_type=0)
```

### Running Example Script
The `examples/run_topology.py` script demonstrates batch processing:
```bash
python examples/run_topology.py
```

### Parameters
- `topology`: 'E1' (others require implementation).
- `Lx`, `Ly`, `Lz`: Length scales in units of $L_{\text{LSS}}$ (13824.9 Mpc).
- `alpha`, `beta`, `gamma`: Topology angles in degrees.
- `l_max`, `l_min`: Maximum and minimum multipoles.
- `c_l_accuracy`: Wavevector cutoff accuracy (default: 0.99).
- `do_polarization`: `True` for TT/EE/TE, `False` for TT only.
- `l_ranges`, `lp_ranges`: Multipole ranges for covariance computation.

### Outputs
- **Covariance Matrices**: Saved as `.npy` files in `runs/[topology]_Lx_[...]/`.
- **k_max_list**: Wavevector cutoffs saved as `k_max_list.npy`.
- **Plots**: (Optional) Rescaled covariance matrix plots.

## Package Structure

- **`cmb_topology/__init__.py`**: Exposes `Topology`, `E1`, `run_topology`.
- **`cmb_topology/topology.py`**: Base `Topology` class for CMB covariance computation.
- **`cmb_topology/E1.py`**: E1 topology implementation.
- **`cmb_topology/E2.py`**: E2 topology implementation.
- **`cmb_topology/E3.py`**: E3 topology implementation.
- **`cmb_topology/E4.py`**: E4 topology implementation.
- **`cmb_topology/E5.py`**: E5 topology implementation.
- **`cmb_topology/E6.py`**: E6 topology implementation.
- **`cmb_topology/tools.py`**: Utility functions.
- **`examples/run_topology.py`**: Example script for running simulations.
- **`parameter_files/default_E1.py`**: Default parameters.
- **`parameter_files/default_E2.py`**: Default parameters.
- **`parameter_files/default_E3.py`**: Default parameters.
- **`parameter_files/default_E4.py`**: Default parameters.
- **`parameter_files/default_E5.py`**: Default parameters.
- **`parameter_files/default_E6.py`**: Default parameters.
- **`tests/test_topology.py`**: (Placeholder) Unit tests.

## Numerical Analysis

The library implements the numerical methods from the paper’s "Numerical Analysis" section:

**Section: Numerical Analysis**

As described above, to the extent that the CMB is Gaussian, all the information about the temperature anisotropies is given by the 2-point correlation matrix of the spherical-harmonic coefficients, namely the covariance matrix $C_{\ell m\ell'm'}^{E_{i};XY} = \langle a_{\ell m}^{E_{i};X}  a_{\ell' m'}^{E_{i};Y*} \rangle$ \cite{Hu2002}.
In the familiar case of the isotropic covering space, we have the usual formula  $\langle  a_{\ell m}^{E_{18};X}  a_{\ell' m'}^{E_{18};Y*} \rangle = C_\ell^{E_{18};XY} \delta^{K}_{\ell\ell'}\delta^{K}_{mm'}$.
But a non-trivial topology breaks isotropy (see, e.g., \rcite{Riazuelo2004:prd}), inducing non-zero off-diagonal components.
[...]
We solve these equations using optimized Python code, leveraging \texttt{CAMB} with Planck 2018 $\Lambda$CDM parameters \cite{Planck:2018vyg}.
The rescaled covariance matrix is defined as:
$$
\Xi^{E_{i}}_{\ell m\ell'm'} \equiv \frac{C^{E_{i}}_{\ell m\ell'm'}} {\sqrt{C^{\Lambda \mathrm{CDM}}_{\ell}C^{\Lambda \mathrm{CDM}}_{\ell'}}},
$$
used for visualizing correlations and computing KL divergence.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

- **Tests**: Add unit tests to `tests/test_topology.py`.
- **Issues**: Report bugs or suggest improvements via GitHub Issues.

## Citation

Please cite the associated paper if using this library:


## Contact

For questions, contact Amirhossein Samandar at amirsamandar@gmail.com or open a GitHub issue.
