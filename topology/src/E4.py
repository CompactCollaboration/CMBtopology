"""
E4.py

Implementation of the E4 topology (third-turn) for CMB covariance matrix computation.
Extends the base Topology class with E2-specific wavevector generation and covariance
calculations.
"""
from .topology import Topology
from .tools import *
from .tools_E2_E3_E4_E5 import *
import numpy as np
from numpy import pi, sin, cos, exp, sqrt, tan
from numba import njit, prange

class E4(Topology):
  """E4 topology (quater-turn) for CMB covariance matrix computation.

    Attributes:
        L12 (float): Length scale in square base (Mpc).
        Lz (float): Length scale in z-direction (Mpc).
        alpha (float): Angle alpha in radians.
        beta (float): Angle beta in radians.
        gamma (float): Angle gamma in radians.
        V (float): Volume of the fundamental domain.
        cubic (bool): True if LA = Lz and alpha = beta = 90 degrees.
  """
  def __init__(self, param, debug=True, make_run_folder = False):
    """Initialize the E4 topology.

        Args:
            param (dict): Parameters including L12, Lz, alpha, beta, l_max.
            debug (bool, optional): Enable debug output. Defaults to True.
            make_run_folder (bool, optional): Create output directories. Defaults to False.
    """
    L_LSS = 13824.9 * 2
    self.L12 = param['Lx'] * L_LSS
    self.Lz = param['Lz'] * L_LSS
    
    self.x0 = param['x0'] * L_LSS
    if np.linalg.norm(param['x0']) < 1e-6 and np.abs(param['beta'] - 90) < 1e-6  and np.abs(param['alpha'] - 90) < 1e-6:
      self.no_shift = True
    else:
      self.no_shift = False
    print('No shift:', self.no_shift)
    self.beta = param['beta'] * np.pi / 180
    self.alpha = param['alpha'] * np.pi / 180
    self.V = sqrt(3)/2 * self.L12**2 * self.Lz * sin(self.beta)
    self.l_max = param['l_max']
    self.param = param

    self.root = 'runs/{}_L12_{}_Lz_{}_beta_{}_alpha_{}_x_{}_y_{}_z_{}_l_max_{}_accuracy_{}_percent/'.format(
        param['topology'],
        "{:.2f}".format(param['Lx']),
        "{:.2f}".format(param['Lz']),
        int(param['beta']),
        int(param['alpha']),
        "{:.2f}".format(param['x0'][0]),
        "{:.2f}".format(param['x0'][1]),
        "{:.2f}".format(param['x0'][2]),
        self.l_max,
        int(param['c_l_accuracy']*100)
    )

    Topology.__init__(self, param, debug, make_run_folder)

  def get_c_lmlpmp_per_process_multi(
    self,
    process_i,
    return_dict,
    min_index,
    max_index,
    V,
    k_amp, 
    phi, 
    theta_unique_index,
    k_amp_unique_index,
    k_max_list, 
    l_max,
    lm_2_index,
    wigner_d_l_m_2,
    integrand,
    ell_range,
    ell_p_range,
  ):
    """Compute covariance matrix for a process in multiprocessing.
      This function seems unnecessary, but Numba does not allow return_dict
      which is of type multiprocessing.Manager

        Args:
            process_i (int): Process index.
            return_dict (multiprocessing.Manager.dict): Shared dictionary for results.
            min_ell (int): Minimum multipole for this process.
            max_ell (int): Maximum multipole for this process.
            V (float): Volume of the fundamental domain.
            k_amp (np.ndarray): Wavevector magnitudes.
            phi (np.ndarray): Wavevector azimuthal angles.
            theta_unique_index (np.ndarray): Indices for unique theta.
            k_amp_unique_index (np.ndarray): Indices for unique k_amp.
            k_max_list (np.ndarray): Wavevector cutoffs.
            l_max (int): Maximum multipole.
            lm_index (np.ndarray): Spherical harmonic indices.
            sph_harm_no_phase (np.ndarray): Spherical harmonics without phase.
            integrand (np.ndarray): Preprocessed integrand.
            ell_range (np.ndarray): Multipole range [l_min, l_max].
            ell_p_range (np.ndarray): Multipole range [lp_min, lp_max].
        """

    if self.do_polarization:
      raise NotImplementedError(
          "E1 topology with polarization covariance matrices is not implemented yet "
          "and will be released in the next version."
      )
    else:
      return_dict[process_i] = E2_E3_E4_E5_get_c_TT_lmlpmp(
      min_index,
      max_index,
      V,
      k_amp, 
      phi, 
      theta_unique_index,
      k_amp_unique_index,
      k_max_list, 
      l_max,
      lm_2_index,
      wigner_d_l_m_2,
      integrand,
      ell_range,
      ell_p_range,
      tilde_xi = self.tilde_xi,
      tilde_xi_delta_m = 3, 
      )

  def get_list_of_k_phi_theta(self):
    """Generate wavevector magnitudes and angles for E4 topology.

        Returns:
            tuple: (k_amp, phi, theta) arrays of wavevector magnitudes and angles.
    """
    M_B_1 = np.array([
        [-1/2,       sqrt(3)/2, 0],
        [-sqrt(3)/2, -1/2,      0],
        [0,          0,         1]
      ], dtype=np.float64)
    M_B_j = np.zeros((3, 3, 3), dtype=np.float64)
    for j in range(3):
      M_B_j[j, :, :] = np.linalg.matrix_power(M_B_1, j)
    
    M_B_j_minus_identity = M_B_j - np.identity(3)

    M_0j = np.zeros((3, 3, 3), dtype=np.float64)
    M_0j[0, :, :] = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
    ]
    M_0j[1, :, :] = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
    ]
    M_0j[2, :, :]= [
      [1/2,        sqrt(3)/2, 0],
      [-sqrt(3)/2, 1,         0],
      [0,          0,         2]
    ]

    k_amp, phi, theta, tilde_xi = get_list_of_k_phi_theta(
      max(self.k_max_list),
      self.L12,
      self.Lz,
      self.x0,
      self.beta,
      M_B_j_minus_identity,
      M_0j
    )
    print('Size of tilde xi: {} MB.'.format(round(tilde_xi.size * tilde_xi.itemsize / 1024 / 1024, 2)))
    print('Shape tilde xi:', tilde_xi.shape, '\n')
    self.tilde_xi = tilde_xi
    return k_amp, phi, theta
  

@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, L_12, L_z, x0, beta, M_B_j_minus_identity, M_0j):
    """Compute wavevector magnitudes and angles for E4 topology.

    Args:
        k_max (float): Maximum wavevector magnitude.
        k_min (float): Minimum wavevector magnitude.
        L_12 (float): Length scale in x&y-direction.
        L_z (float): Length scale in z-direction.
        beta (float): Angle beta in radians.

    Returns:
        tuple: (k_amp, phi, theta) arrays of wavevector magnitudes and angles.
    """
    sin_b_inv = 1/sin(beta)

    n_x_max = int(np.ceil(k_max * L_12 / (2*pi)))
    n_y_max = 2*int(np.ceil(k_max * L_12 / (2*pi)))
    n_z_max = int(np.ceil(k_max * L_z * 3 / (2*pi))) # Because of eigenmode 1
 
    k_amp = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    phi = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    theta = np.zeros(n_x_max * n_y_max * n_z_max * 8)

    tilde_xi = np.zeros((n_x_max * n_y_max * n_z_max * 8, 3), dtype=np.complex128)

    T_B = L_z * np.array([cos(beta), 0, sin(beta)])
    k_max_squared = k_max**2

    cur_index = 0

    # Eigenmode 1
    k_x = 0
    k_y = 0
    # Only n_z = 0 mod 3
    for n_z in range(-n_z_max, n_z_max+1):
      if n_z % 3 != 0 or n_z == 0:
        continue
      k_z = 2*pi * n_z * sin_b_inv / (3 * L_z)
      k_xyz = sqrt(k_z**2)
      if k_xyz > k_max:
        continue
      
      k_amp[cur_index] = k_xyz
      cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
      phi[cur_index] = cur_phi
      theta[cur_index] = cur_theta
      tilde_xi[cur_index, :] = exp(- 1j * k_z * x0[2])
      cur_index += 1
    print(cur_index)

    # Eigenmode 2
    for n_z in range(-n_z_max, n_z_max+1):
      k_z = 2*pi * n_z * sin_b_inv / (3 * L_z)
      k_z_squared = k_z**2
      if k_z_squared > k_max_squared:
        continue

      for n_x in range(-n_x_max, n_x_max+1):
        k_x = 2*pi * n_x / L_12

        k_xz_squared = k_x**2 + k_z_squared
        if k_xz_squared > k_max**2:
          continue

        if n_x >= 0:
          n_y_start = 0
          n_y_end = n_y_max
        else:
          n_y_start = -n_y_max
          n_y_end = -1

        for n_y in range(n_y_start, n_y_end+1):
          if n_y == 0 and n_x == 0:
            continue

          k_y = 4*pi * n_y / (sqrt(3)*L_12) + k_x/np.sqrt(3)

          k_xyz = sqrt(k_xz_squared + k_y**2)

          if k_xyz > k_max or k_xyz < 1e-6:
            continue

          k_vec = np.array([k_x, k_y, k_z])

          k_amp[cur_index] = k_xyz
          cur_phi, cur_theta = cart2spherical(k_vec/k_xyz)
          phi[cur_index] = cur_phi
          theta[cur_index] = cur_theta
         
          j_contribution = np.zeros(3, dtype=np.complex128)
          for k in range(3):
            j_contribution[k] = exp(-1j*np.dot(k_vec, np.dot(M_B_j_minus_identity[k], x0))) * exp(1j*np.dot(k_vec, np.dot(M_0j[k], T_B)))
          for m_mod_3 in range(3):
            tilde_xi[cur_index, m_mod_3] = np.sum(exp(1j * m_mod_3 * np.arange(3) * 2*np.pi/3) * j_contribution)
          tilde_xi[cur_index, :] *= exp(- 1j * np.dot(k_vec, x0))/sqrt(3) 
          
          cur_index += 1
    k_amp = k_amp[:cur_index]
    phi = phi[:cur_index]   
    theta = theta[:cur_index]
    tilde_xi = tilde_xi[:cur_index, :]

    print('Final num of elements:', k_amp.size, 'Minimum k_amp', np.amin(k_amp), 'n_x_max', n_x_max, 'n_z_max', n_z_max)
    return k_amp, phi, theta, tilde_xi