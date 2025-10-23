"""
E1.py

Implementation of the E1 topology (3-torus) for CMB covariance matrix computation.
Extends the base Topology class with E1-specific wavevector generation and covariance
calculations.
"""


from .tools import *
from .topology import Topology
import numpy as np
from numpy import pi, sin, tan, sqrt, cos, conjugate
from numba import njit, prange, jit
from numba_progress import ProgressBar

class E1(Topology):
  """E1 topology (3-torus) for CMB covariance matrix computation.

    Attributes:
        Lx (float): Length scale in x-direction (Mpc).
        Ly (float): Length scale in y-direction (Mpc).
        Lz (float): Length scale in z-direction (Mpc).
        alpha (float): Angle alpha in radians.
        beta (float): Angle beta in radians.
        gamma (float): Angle gamma in radians.
        V (float): Volume of the fundamental domain.
        cubic (bool): True if Lx = Ly = Lz and alpha = beta = 90 degrees.
    """
  
  def __init__(self, param, debug=True, make_run_folder = False):
    """Initialize the E1 topology.

        Args:
            param (dict): Parameters including Lx, Ly, Lz, alpha, beta, gamma, l_max.
            debug (bool, optional): Enable debug output. Defaults to True.
            make_run_folder (bool, optional): Create output directories. Defaults to False.
    """
    L_LSS = 13824.9 * 2  # Last scattering surface diameter (Mpc)
    self.Lx = param['Lx'] * L_LSS
    self.Ly = param['Ly'] * L_LSS
    self.Lz = param['Lz'] * L_LSS
    self.alpha = param['alpha'] * np.pi / 180
    self.beta = param['beta'] * np.pi / 180
    self.gamma = param['gamma'] * np.pi / 180
    self.V = self.Lx * self.Ly * self.Lz * sin(self.beta) * sin(self.alpha)
    self.l_max = param['l_max']
    self.param = param

    if param['Lx'] == param['Ly'] and param['Ly'] == param['Lz'] and param['beta'] == 90 and param['alpha'] == 90:
      print('Cubic Torus')
      self.cubic = True
    else:
      self.cubic = False

    print('Running - E1 l_max={}, Lx={}'.format(self.l_max, int(self.Lx)))
    self.root = 'runs/{}_Lx_{}_Ly_{}_Lz_{}_beta_{}_alpha_{}_l_max_{}_accuracy_{}_percent/'.format(
        param['topology'],
        "{:.2f}".format(param['Lx']),
        "{:.2f}".format(param['Ly']),
        "{:.2f}".format(param['Lz']),
        int(param['beta']),
        int(param['alpha']),
        self.l_max,
        int(param['c_l_accuracy']*100)
    )

    Topology.__init__(self, param, debug, make_run_folder)


  def get_c_lmlpmp_per_process_multi(
    self,
    process_i,
    return_dict,
    min_ell,
    max_ell,
    V,
    k_amp, 
    phi, 
    theta_unique_index,
    k_amp_unique_index,
    k_max_list,
    l_max,
    lm_index,
    sph_harm_no_phase,
    integrand,
    ell_range,
    ell_p_range
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
      return_dict[process_i] = get_c_lmlpmp(
      min_ell,
      max_ell,
      V,
      k_amp, 
      phi, 
      theta_unique_index,
      k_amp_unique_index,
      k_max_list,
      l_max,
      lm_index,
      sph_harm_no_phase,
      integrand,
      ell_range,
      ell_p_range
      )
    else:
      return_dict[process_i] = get_c_TT_lmlpmp(
      min_ell,
      max_ell,
      V,
      k_amp, 
      phi, 
      theta_unique_index,
      k_amp_unique_index,
      k_max_list,
      l_max,
      lm_index,
      sph_harm_no_phase,
      integrand,
      ell_range,
      ell_p_range
      )

  def get_list_of_k_phi_theta(self):
    """Generate wavevector magnitudes and angles for E1 topology.

        Returns:
            tuple: (k_amp, phi, theta) arrays of wavevector magnitudes and angles.
    """
    k_amp, phi, theta = get_list_of_k_phi_theta(max(self.k_max_list), self.k_list[0], 
                                                self.Lx, self.Ly, self.Lz, self.beta, 
                                                self.alpha, self.gamma)

    return k_amp, phi, theta
  

@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, k_min, L_x, L_y, L_z, beta, alpha, gamma):
    """Compute wavevector magnitudes and angles for E1 topology.

    Args:
        k_max (float): Maximum wavevector magnitude.
        k_min (float): Minimum wavevector magnitude.
        L_x (float): Length scale in x-direction.
        L_y (float): Length scale in y-direction.
        L_z (float): Length scale in z-direction.
        beta (float): Angle beta in radians.
        alpha (float): Angle alpha in radians.
        gamma (float): Angle gamma in radians.

    Returns:
        tuple: (k_amp, phi, theta) arrays of wavevector magnitudes and angles.
    """
    tan_b_inv = 1/tan(beta)
    sin_b_inv = 1/sin(beta)
    tan_a_inv = 1/tan(alpha)
    sin_a_inv = 1/sin(alpha)
    print('inv sin and tan', sin_b_inv, tan_b_inv)

    n_x_max = int(np.ceil(k_max * L_x / (2*pi)))
    n_y_max = int(np.ceil(k_max * L_y / (2*pi)))  
    n_z_max = int(np.ceil(k_max * L_z / (2*pi)))
    
    k_amp = np.zeros(n_x_max * n_y_max * n_z_max * 8) # *2(n_x> or n_x<0) *2(n_y> or n_y<0)*2(n_z> or n_z<0)
    phi = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    theta = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    
    cur_index = 0

    for n_x in prange(-n_x_max, n_x_max + 1):
      k_x = 2*pi * n_x / L_x

      for n_y in range(-n_y_max, n_y_max+1):
        k_y = 2*pi * n_y * sin_a_inv / L_y - k_x * tan_a_inv

        k_xy_square = k_x * k_x + k_y*k_y
        if k_xy_square > k_max*k_max:
          continue

        for n_z in range(-n_z_max, n_z_max+1):        
          k_z = (
              2*pi * n_z * sin_b_inv / L_z - 2*pi * n_y * tan_b_inv * sin(gamma) / sin(alpha) / L_y
              - k_x * tan_b_inv * (cos(gamma) - tan_a_inv * sin(gamma))
          )
          k_xyz = sqrt(k_xy_square + k_z*k_z)

          if k_xyz > k_max or k_xyz < k_min:
            continue

          k_amp[cur_index] = k_xyz
          cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
          phi[cur_index] = cur_phi
          theta[cur_index] = cur_theta
          cur_index += 1
    
    k_amp = k_amp[:cur_index]
    phi = phi[:cur_index]   
    theta = theta[:cur_index]
    print("done")
    print('Final num of elements:', k_amp.size, 'Minimum k_amp', np.amin(k_amp), 'n_x_max', n_x_max, 'n_z_max', n_z_max)
    return k_amp, phi, theta



@njit(nogil=True, parallel = False)
def get_c_lmlpmp(
    min_ell,
    max_ell,
    V,
    k_amp, 
    phi, 
    theta_unique_index,
    k_amp_unique_index,
    k_max_list,
    l_max,
    lm_index,
    sph_harm_no_phase,
    integrand,
    ell_range,
    ell_p_range
    ):

    """Compute covariance matrix for polarization (TT, EE, TE).

    Args:
        min_ell (int): Minimum multipole.
        max_ell (int): Maximum multipole.
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

    Returns:
        np.ndarray: Covariance matrix (TT, EE, TE).
    """

    num_l_m = ell_range[1] * (ell_range[1] + 1) + ell_range[1] + 1 - ell_range[0]**2     # = Sum[2 l+1, {l , l_min, l_max}]
    num_l_m_p = ell_p_range[1] * (ell_p_range[1] + 1) + ell_p_range[1] + 1 - ell_p_range[0]**2 
    C_lmlpmp = np.zeros((3, num_l_m, num_l_m_p), dtype=np.complex128)    
    m_list = np.arange(0, l_max+1)
    ipow = np.array([1, 1j, -1, -1j])
    shortle = np.array([1, -1])
    min_k_amp = np.min(k_amp)
    eig_num = k_amp.size
    min_ell_p_square = ell_p_range[0]**2
    min_ell_square = ell_range[0]**2
    for i in range(eig_num):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      sph_harm_index = theta_unique_index[i] # wigner_d for especific theta[i]

      phase_list_plus = np.exp(1j * phi[i] * m_list)
      phase_list_minus = np.exp(-1j * phi[i] * m_list)

      for l in range(min_ell, max_ell + 1):
        ell_times_ell_plus_one_minus_min_ell = l * (l + 1) - min_ell_square
        coeff_E_ell = sqrt((l + 2) * (l + 1) * l * (l - 1)) 

        if ell_p_range[0] > l:
          l_start = ell_p_range[0]
        else:
          l_start = l

        for l_p in range(l_start, ell_p_range[1]+1):
          if k_amp_cur > np.sqrt(k_max_list[l]*k_max_list[l_p]):
            continue
          ell_p_times_ell_p_plus_one_minus_min_ell_p = l_p * (l_p + 1) - min_ell_p_square
          coeff_E_l_p = sqrt((l_p + 2) * (l_p + 1) * l_p * (l_p - 1)) 

          ell_ell_p_integrand_TT = integrand[0, k_unique_index_cur, l, l_p] * ipow[(l-l_p)%4]
          ell_ell_p_integrand_EE = ( 
                                  coeff_E_ell * coeff_E_l_p * 
                                  integrand[1, k_unique_index_cur, l, l_p] 
                                  * ipow[(l-l_p)%4]
                                )
          ell_ell_p_integrand_TE = (
                                  coeff_E_l_p* integrand[2, k_unique_index_cur, l, l_p] 
                                  * ipow[(l-l_p)%4]
                                )
          ell_p_ell_integrand_TE = (
                                  coeff_E_ell* integrand[2, k_unique_index_cur, l_p, l] 
                                  * ipow[(l_p - l)%4]
                                )

          for m in range(-l, l + 1):
            lm_index_cur = ell_times_ell_plus_one_minus_min_ell + m 
            abs_m = np.abs(m)
            sph_cur_index = lm_index[l, abs_m]

            sph_harm_l_m = sph_harm_no_phase[sph_harm_index, sph_cur_index] * phase_list_plus[abs_m]
            if m<0:
              sph_harm_l_m = shortle[abs_m%2] * conjugate(sph_harm_l_m)

            # Only do m-mp = 0 mod 2  
            for m_p in range(0, l_p + 1):
              if (m_p-m)%2 ==1:
                continue
              lm_p_index_cur = ell_p_times_ell_p_plus_one_minus_min_ell_p + m_p  
              sph_p_cur_index = lm_index[l_p, m_p]

              sph_harm_l_m_p = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * phase_list_plus[m_p]

              Xi = conjugate(sph_harm_l_m) * sph_harm_l_m_p

              C_lmlpmp[0, lm_index_cur, lm_p_index_cur] += ell_ell_p_integrand_TT * Xi
              C_lmlpmp[1, lm_index_cur, lm_p_index_cur] += ell_ell_p_integrand_EE * Xi
              C_lmlpmp[2, lm_index_cur, lm_p_index_cur] += ell_ell_p_integrand_TE * Xi
              if l != l_p:
                C_lmlpmp[2, lm_p_index_cur, lm_index_cur] += ell_p_ell_integrand_TE* conjugate(Xi)

    for l in prange(min_ell, max_ell + 1):
      for l_p in range(l, ell_p_range[1]+1):
        for m in range(-l, l + 1):
          lm_index_new = l * (l+1) + m - ell_range[0]*ell_range[0] # l**2 + l +m - l_min**2
          lm_index_cal = l * (l+1) - m - ell_range[0]*ell_range[0]
          for m_p in range(-l_p, 0):
            lm_p_index_new = l_p * (l_p+1) + m_p  - ell_range[0]*ell_range[0]
            lm_p_index_cal = l_p * (l_p+1) - m_p  - ell_range[0]*ell_range[0]
            C_lmlpmp[:, lm_index_new, lm_p_index_new] = shortle[(m+m_p)%2] * conjugate(C_lmlpmp[:, lm_index_cal, lm_p_index_cal])
            if l != l_p:
              C_lmlpmp[2, lm_p_index_new, lm_index_new] = shortle[(m+m_p)%2] * conjugate(C_lmlpmp[2, lm_p_index_cal, lm_index_cal])

    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    
    return C_lmlpmp



@njit(nogil=True, parallel = False)
def get_c_TT_lmlpmp(
    min_ell,
    max_ell,
    V,
    k_amp, 
    phi, 
    theta_unique_index,
    k_amp_unique_index,
    k_max_list,
    l_max,
    lm_index,
    sph_harm_no_phase,
    integrand,
    ell_range,
    ell_p_range
    ):

    """Compute covariance matrix for temperature (TT) only.

    Args:
        min_ell (int): Minimum multipole.
        max_ell (int): Maximum multipole.
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

    Returns:
        np.ndarray: TT covariance matrix.
    """
    num_l_m = ell_range[1] * (ell_range[1] + 1) + ell_range[1] + 1 - ell_range[0]**2     # = Sum[2 l+1, {l , l_min, l_max}]
    num_l_m_p = ell_p_range[1] * (ell_p_range[1] + 1) + ell_p_range[1] + 1 - ell_p_range[0]**2 
    C_lmlpmp = np.zeros((num_l_m, num_l_m_p), dtype=np.complex128)    
    m_list = np.arange(0, l_max+1)
    ipow = np.array([1, 1j, -1, -1j])
    shortle = np.array([1, -1])
    eig_num = k_amp.size
    min_ell_p_square = ell_p_range[0]**2
    min_ell_square = ell_range[0]**2
    for i in range(eig_num):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      sph_harm_index = theta_unique_index[i] # wigner_d for especific theta[i]

      phase_list_plus = np.exp(1j * phi[i] * m_list)

      for l in range(min_ell, max_ell + 1):
        ell_times_ell_plus_one_minus_min_ell = l * (l + 1) - min_ell_square

        if ell_p_range[0] > l:
          l_start = ell_p_range[0]
        else:
          l_start = l

        for l_p in range(l_start, ell_p_range[1]+1):
          if k_amp_cur > np.sqrt(k_max_list[l]*k_max_list[l_p]):
            continue
          ell_p_times_ell_p_plus_one_minus_min_ell_p = l_p * (l_p + 1) - min_ell_p_square

          ell_ell_p_integrand_TT = integrand[k_unique_index_cur, l, l_p] * ipow[(l-l_p)%4]

          for m in range(-l, l + 1):
            lm_index_cur = ell_times_ell_plus_one_minus_min_ell + m 
            abs_m = np.abs(m)
            sph_cur_index = lm_index[l, abs_m]

            sph_harm_l_m = sph_harm_no_phase[sph_harm_index, sph_cur_index] * phase_list_plus[abs_m]
            if m<0:
              sph_harm_l_m = shortle[abs_m%2] * conjugate(sph_harm_l_m)

            # Only do m-mp = 0 mod 2  
            for m_p in range(0, l_p + 1):
              if (m_p-m)%2 ==1:
                continue
              lm_p_index_cur = ell_p_times_ell_p_plus_one_minus_min_ell_p + m_p  
              sph_p_cur_index = lm_index[l_p, m_p]

              sph_harm_l_m_p = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * phase_list_plus[m_p]

              Xi = conjugate(sph_harm_l_m) * sph_harm_l_m_p

              C_lmlpmp[lm_index_cur, lm_p_index_cur] += ell_ell_p_integrand_TT * Xi

    for l in prange(min_ell, max_ell + 1):
      ell_times_ell_plus_one_minus_min_ell = l * (l + 1) - min_ell_square
      for l_p in range(l, ell_p_range[1]+1):
        ell_p_times_ell_p_plus_one_minus_min_ell_p = l_p * (l_p + 1) - min_ell_p_square
        for m in range(-l, l + 1):
          lm_index_new = ell_times_ell_plus_one_minus_min_ell + m  # l**2 + l +m - l_min**2
          lm_index_cal = ell_times_ell_plus_one_minus_min_ell - m 
          for m_p in range(-l_p, 0):
            lm_p_index_new = ell_p_times_ell_p_plus_one_minus_min_ell_p + m_p 
            lm_p_index_cal = ell_p_times_ell_p_plus_one_minus_min_ell_p - m_p 
            C_lmlpmp[lm_index_new, lm_p_index_new] = shortle[(m+m_p)%2] * conjugate(C_lmlpmp[lm_index_cal, lm_p_index_cal])
  
    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    
    return C_lmlpmp