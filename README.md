# CMBtopology

A Python package for computing CMB covariance matrices for non-trivial topologies (E1–E10), developed by Amirhossein Samandar and Johannes R. Eskilt under the auspices of the COMPACT Collaboration.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/CMBtopology.svg)](https://pypi.org/project/CMBtopology/)
[![Documentation](https://readthedocs.org/projects/CMBtopology/badge/?version=latest)](https://CMBtopology.readthedocs.io/)
[![arXiv Part IIa](https://img.shields.io/badge/arXiv-2306.17112-b31b1b.svg)](https://arxiv.org/abs/2306.17112)
[![arXiv Part IIb](https://img.shields.io/badge/arXiv-2510.05030-b31b1b.svg)](https://arxiv.org/abs/2510.05030)

## Overview

CMBtopology is a Python package designed to compute Cosmic Microwave Background (CMB) covariance matrices for non-trivial topologies, including orientable and non-orientable Euclidean manifolds. The package enables efficient calculation of temperature and polarization covariance matrices using CAMB transfer functions, Numba-accelerated integrals, and multiprocessing. It supports all the fully compact Euclidean topologies E1--E10.

This package is based on the research presented in two key papers from the COMPACT Collaboration:

### Cosmic Topology. Part IIa: Eigenmodes, Correlation Matrices, and Detectability of Orientable Euclidean Manifolds
- **Authors**: (COMPACT Collaboration) Johannes R. Eskilt, Yashar Akrami, Stefano Anselmi, Craig J. Copi, Andrew H. Jaffe, Arthur Kosowsky, Deyan P. Mihaylov, Glenn D. Starkman, Andrius Tamosiunas, James B. Mertens, Pip Petersen, Samanta Saha, Quinn Taylor, Özenç Güngör
- **Abstract**: If the Universe has non-trivial spatial topology, observables depend on both the parameters of the spatial manifold and the position and orientation of the observer. In infinite Euclidean space, most cosmological observables arise from the amplitudes of Fourier modes of primordial scalar curvature perturbations. Topological boundary conditions replace the full set of Fourier modes with specific linear combinations of selected Fourier modes as the eigenmodes of the scalar Laplacian. We present formulas for eigenmodes in orientable Euclidean manifolds with the topologies E1–E6, E11, E12, E16, and E18 that encompass the full range of manifold parameters and observer positions, generalizing previous treatments. Under the assumption that the amplitudes of primordial scalar curvature eigenmodes are independent random variables, for each topology we obtain the correlation matrices of Fourier-mode amplitudes (of scalar fields linearly related to the scalar curvature) and the correlation matrices of spherical-harmonic coefficients of such fields sampled on a sphere, such as the temperature of the cosmic microwave background (CMB). We evaluate the detectability of these correlations given the cosmic variance of the observed CMB sky. We find that topologies where the distance to our nearest clone is less than about 1.2 times the diameter of the last scattering surface of the CMB give a correlation signal that is larger than cosmic variance noise in the CMB. This implies that if cosmic topology is the explanation of large-angle anomalies in the CMB, then the distance to our nearest clone is not much larger than the diameter of the last scattering surface. We argue that the topological information is likely to be better preserved in three-dimensional data, such as will eventually be available from large-scale structure surveys.
- **Link**: [arXiv:2306.17112](https://arxiv.org/abs/2306.17112)

### Cosmic Topology. Part IIb: Eigenmodes, Correlation Matrices, and Detectability of Non-Orientable Euclidean Manifolds
- **Authors**: (COMPACT Collaboration) Craig J. Copi, Amirhossein Samandar, Glenn D. Starkman, Javier Carrón Duque, Yashar Akrami, Stefano Anselmi, Andrew H. Jaffe, Arthur Kosowsky, Fernando Cornet-Gomez, Johannes R. Eskilt, Mikel Martin Barandiaran, Deyan P. Mihaylov, Anna Negro, Joline Noltmann, Thiago S. Pereira, Andrius Tamosiunas
- **Abstract**: If the Universe has non-trivial spatial topology, observables depend on both the parameters of the spatial manifold and the position and orientation of the observer. In infinite Euclidean space, most cosmological observables arise from the amplitudes of Fourier modes of primordial scalar curvature perturbations. Topological boundary conditions replace the full set of Fourier modes with specific linear combinations of selected Fourier modes as the eigenmodes of the scalar Laplacian. In this paper we consider the non-orientable Euclidean topologies E7–E10, E13–E15, and E17, encompassing the full range of manifold parameters and observer positions, generalizing previous treatments. Under the assumption that the amplitudes of primordial scalar curvature eigenmodes are independent random variables, for each topology we obtain the correlation matrices of Fourier-mode amplitudes (of scalar fields linearly related to the scalar curvature) and the correlation matrices of spherical-harmonic coefficients of such fields sampled on a sphere, such as the temperature of the cosmic microwave background (CMB). We evaluate the detectability of these correlations given the cosmic variance of the CMB sky. We find that in manifolds where the distance to our nearest clone is less than about 1.2 times the diameter of the last scattering surface of the CMB, we expect a correlation signal that is larger than cosmic variance noise in the CMB. Our limited selection of manifold parameters are exemplary of interesting behaviors, but not necessarily representative. Future searches for topology will require a thorough exploration of the parameter space to determine what values of the parameters predict statistical correlations that are convincingly attributable to topology.[Abridged]
- **Link**: [arXiv:2510.05030](https://arxiv.org/abs/2510.05030)

## Installation

To install CMBtopology, use pip:(The package will be released soon!)

```bash
pip install CMBtopology
```

Alternatively, install from the GitHub repository:

```bash
pip install git+https://github.com/CompactCollaboration/CMBtopology.git
```

Requirements
------------

CMBtopology requires the following Python packages:

- numpy>=1.20
- matplotlib>=3.5
- scipy>=1.8
- healpy>=1.18.0
- camb>=1.3
- tqdm>=4.60
- numba>=0.60
- quaternionic>=1.0.13
- spherical>=1.0.14

## Usage

CMBtopology provides a flexible interface for computing CMB covariance matrices. You can use it programmatically or via the command-line interface (CLI).

Programmatic Usage
-----------------

To compute the covariance matrix for the E1 topology (3-torus):

```python
from topology.run_topology import run_topology

run_topology(
    topology='E1',
    l_max=30,
    Lx=1.0, Ly=1.0, Lz=1.0,
    beta=90, alpha=90,
    x0 = np.array([0.1, 0.2, 0.3]),
    do_polarization=False,
    normalize=True,
)
```

For the E6 topology with specific parameters:

```python
run_topology(
    topology='E6',
    l_max=20,
    Lx=1.0, Ly=1.0, Lz=1.0,
    r_x=0.5, r_y=0.5, r_z=0.5,
    do_polarization=False,
)
```

## Testing
Run tests: `python -m unittest discover tests`

## Examples
Run examples: `python examples/example_usage.py`

Command-Line Interface
---------------------

Run CMBtopology from the command line:

```bash
CMBtopology --topology E6 --l_max 20 --Lx 1.0 --Ly 1.0 --Lz 1.0 --r_x 0.5 --r_y 0.5 --r_z 0.5 --do_polarization
```

For more details, see the :ref:`api-reference` section in the [documentation](https://CMBtopology.readthedocs.io/).

## Documentation

Full documentation is available at [https://CMBtopology.readthedocs.io/](https://CMBtopology.readthedocs.io/), including API reference, installation guide, and usage examples.

## Citation

If you use CMBtopology in your research, please cite the following papers:

- Eskilt, J. R., et al. (COMPACT Collaboration). "Cosmic topology. Part IIa. Eigenmodes, correlation matrices, and detectability of orientable Euclidean manifolds." arXiv:2306.17112 (2023).

  BibTeX:
  ```
  @article{eskilt2024cosmic,
  title={Cosmic topology. Part IIa. Eigenmodes, correlation matrices, and detectability of orientable Euclidean manifolds},
  author={Eskilt, Johannes R and others},
  journal={Journal of Cosmology and Astroparticle Physics},
  volume={2024},
  number={03},
  pages={036},
  year={2024},
  publisher={IOP Publishing}
    }
  ```

- Samandar, A., et al. (COMPACT Collaboration). "Cosmic topology. Part IIb. Eigenmodes, correlation matrices, and detectability of non-orientable Euclidean manifolds." arXiv:2510.05030 (2025).

  BibTeX:
  ```
  @article{copi2025cosmic,
  title={Cosmic topology. Part IIb. Eigenmodes, correlation matrices, and detectability of non-orientable Euclidean manifolds},
  author={Samandar, Amirhossein and others},
  journal={arXiv preprint arXiv:2510.05030},
  year={2025}
    }
  ```

You can also cite the package directly:
```
@software{CMBtopology,
  author = {Samandar, Amirhossein and Eskilt, Johannes R. and Mihaylov, Deyan P. and Duque, Javier Carrón},
  title = {CMBtopology: A Python package for computing CMB covariance matrices for non-trivial topologies},
  version = {0.1.0},
  year = {2025},
  url = {https://github.com/CompactCollaboration/CMBtopology}
}
```

## Contributors

- **Amirhossein Samandar** (Major Contributor): Lead developer, responsible for package architecture, implementation of eigenmodes, correlation matrices, and core functionalities.
- **Johannes R. Eskilt** (Major Contributor): Key contributor to the code, particularly in eigenmode calculations, CMB integration, and optimization for orientable topologies.
- **Deyan P. Mihaylov** (Minor Contributor): Contributed to utility functions, testing, and speeding up key functions (e.g., covariance matrix computations and eigenmode evaluations).
- **Javier Carrón Duque** (Minor Contributor): Provided input on correlation matrix computations, debugging code, and suggestions for error raising and validation.
- **Andrius Tamošiūnas** (Minor Contributor): Contributed to testing early versions of the code used in the ML paper [arXiv:2404.01236](https://arxiv.org/abs/2404.01236), helped identify issues with the implementation of the reality conditions in the $a_{\ell m}$ generation.

Contributions from the COMPACT Collaboration are welcome! See the [Contributing](#contributing) section.

## Contributing

Contributions to CMBtopology are welcome! Please submit issues or pull requests on the [GitHub repository](https://github.com/CompactCollaboration/CMBtopology).

To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Add tests in the `tests/` directory.
4. Update documentation in the `docs/` directory.
5. Submit a pull request.

By contributing, you agree to the Contributor License Agreement (CLA) outlined in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License – Copyright (c) 2025 Amirhossein Samandar and Johannes R. Eskilt. See [LICENSE](LICENSE) for details.

Affiliated with the [COMPACT Collaboration](https://github.com/CompactCollaboration).
```
