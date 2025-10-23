Usage
=====

TopologyPy provides a flexible interface for computing CMB covariance matrices. You can use it programmatically or via the command-line interface (CLI).

Programmatic Usage
==================

To compute the covariance matrix for the E1 topology (3-torus):

.. code-block:: python

   from topology.run_topology import run_topology

   run_topology(
       topology='E1',
       l_max=30,
       Lx=1.0, Ly=1.0, Lz=1.0,
       beta=90, alpha=90,
       do_polarization=True,
   )

For the E6 topology with specific parameters:

.. code-block:: python

   run_topology(
       topology='E6',
       l_max=20,
       Lx=1.0, Ly=1.0, Lz=1.0,
       r_x=0.5, r_y=0.5, r_z=0.5,
       do_polarization=False,
   )

Command-Line Interface
======================

Run TopologyPy from the command line:

.. code-block:: bash

   topologypy --topology E6 --l_max 20 --Lx 1.0 --Ly 1.0 --Lz 1.0 --r_x 0.5 --r_y 0.5 --r_z 0.5 --do_polarization

For more details, see the :ref:`api-reference` section.
