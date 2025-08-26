"""
E6.py

Implementation of the E6 topology (Hantzsche-Wendt space) for CMB covariance matrix computation.
Extends the base Topology class with E6-specific wavevector generation and covariance
calculations.
"""
from .topology import Topology
from .tools import *
import numpy as np
import healpy as hp
from numpy import pi, sin, cos, exp, sqrt, conjugate
from numba import njit, prange
from numba_progress import ProgressBar
import pyshtools as pysh

class E6(Topology):
  def __init__(self, param, debug=True, make_run_folder = False):
    L_LSS = 13824.9 * 2
    self.LAx = param['Lx'] * L_LSS
    self.LBy = param['Ly'] * L_LSS
    self.LCz = param['Lz'] * L_LSS
    self.alpha_x = param['alpha_x']
    self.alpha_y = param['alpha_y']
    self.alpha_z = param['alpha_z']

    self.V = self.LAx * self.LBy * self.LCz * 2

    l_max = param['l_max']
    lm_index = np.zeros((l_max+1, l_max+1), dtype=int)
    for l in range(l_max+1):
        for m in range(l+1):
            cur_index = hp.Alm.getidx(l_max, l, m)
            lm_index[l, m] = cur_index
    
    '''
    num_l_m = int((l_max + 1)*(l_max + 2)/2)
    sph_harm_no_phase_theta_0_pi_over_2 = np.zeros((2, num_l_m), dtype=np.float64)

    all_sph_harm_theta_0_no_phase = np.real(pysh.expand.spharm(l_max, 0, 0, kind = 'complex', degrees = False, normalization = 'ortho', csphase=-1))
    all_sph_harm_theta_pi_over_2_no_phase = np.real(pysh.expand.spharm(l_max, pi/2, 0, kind = 'complex', degrees = False, normalization = 'ortho', csphase=-1))
    for l in range(l_max+1):
        for m in range(l+1):
            cur_index = lm_index[l, m]
            sph_harm_no_phase_theta_0_pi_over_2[0, cur_index] = all_sph_harm_theta_0_no_phase[0, l, m]
            sph_harm_no_phase_theta_0_pi_over_2[1, cur_index] = all_sph_harm_theta_pi_over_2_no_phase[0, l, m]  
    self.sph_harm_no_phase_theta_0_pi_over_2 = sph_harm_no_phase_theta_0_pi_over_2'''

    self.x0 = param['x0'] * L_LSS
    if np.linalg.norm(param['x0']) < 1e-6 and np.abs(param['alpha_x']-0.5) < 1e-6  and np.abs(param['alpha_y']-0.5) < 1e-6 and np.abs(param['alpha_z']-0.5) < 1e-6:
      self.no_shift = True
    else:
      self.no_shift = False
    print('No shift:', self.no_shift)
    self.l_max = param['l_max']
    self.param = param

    self.root = 'runs/{}_LAx_{}_LBy_{}_LCz_{}_alphax_{}_alphay_{}_alphaz_{}_x_{}_y_{}_z_{}_l_max_{}_accuracy_{}_percent/'.format(
        param['topology'],
        "{:.2f}".format(param['Lx']),
        "{:.2f}".format(param['Ly']),
        "{:.2f}".format(param['Lz']),
        "{:.2f}".format(param['alpha_x']),
        "{:.2f}".format(param['alpha_y']),
        "{:.2f}".format(param['alpha_z']),
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
          min_index,
          max_index,
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
          ell_p_range,
          tilde_xi = self.tilde_xi,
          eigenmode_index_split = self.eigenmode_index_split,
      )
    else: 
      return_dict[process_i] = get_c_lmlpmp(
          min_index,
          max_index,
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
          ell_p_range,
          tilde_xi = self.tilde_xi,
          eigenmode_index_split = self.eigenmode_index_split,
      )

  def get_list_of_k_phi_theta(self):
    k_amp, phi, theta, tilde_xi, eigenmode_index_split = get_list_of_k_phi_theta(
      max(self.k_max_list),
      self.LAx,
      self.LBy,
      self.LCz,
      self.alpha_x,
      self.alpha_y,
      self.alpha_z,
      self.x0,
    )
    print('Size of tilde xi: {} MB.'.format(round(tilde_xi.size * tilde_xi.itemsize / 1024 / 1024, 2)))
    print('Shape tilde xi:', tilde_xi.shape, '\n')
    self.tilde_xi = tilde_xi
    self.eigenmode_index_split = eigenmode_index_split
    return k_amp, phi, theta
  

@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, L_Ax, L_By, L_Cz, alpha_x, alpha_y, alpha_z, x0):
    # Returns list of k, phi, and theta for this topology

    n_x_max = int(np.ceil(k_max * L_Ax / (pi)))
    n_y_max = int(np.ceil(k_max * L_By / (pi)))
    n_z_max = int(np.ceil(k_max * L_Cz / (pi)))

    k_amp = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    phi = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    theta = np.zeros(n_x_max * n_y_max * n_z_max * 8)

    # First index is i, second is l%2, third is m%2. The fourth index splits the fourth eigenmode into two pieces
    tilde_xi = np.zeros((n_x_max * n_y_max * n_z_max * 8, 2, 2, 2), dtype=np.complex128)

    T_A = np.array([L_Ax,                   (alpha_y + 1/2)*L_By,   (alpha_z - 1/2) * L_Cz])
    T_B = np.array([(alpha_x - 1/2) * L_Ax, L_By,                   (alpha_z + 1/2)*L_Cz])
    T_C = np.array([(alpha_x + 1/2) * L_Ax, (alpha_y - 1/2) * L_By, L_Cz])

    M_A_minus_id = np.array([
      [1, 0, 0],
      [0, -1, 0],
      [0, 0, -1]
    ], dtype=np.float64) - np.identity(3)

    M_B_minus_id = np.array([
      [-1, 0, 0],
      [0, 1, 0],
      [0, 0, -1]
    ], dtype=np.float64) - np.identity(3)

    M_C_minus_id = np.array([
      [-1, 0, 0],
      [0, -1, 0],
      [0, 0, 1]
    ], dtype=np.float64) - np.identity(3)

    cur_index = 0

    # Eigenmode 1
    k_y = 0
    k_z = 0
    for n_x in range(2, n_x_max+1, 2):
      k_x = pi * n_x / L_Ax
      k_xyz = sqrt(k_x**2)
      if k_xyz > k_max:
        continue
      
      k_amp[cur_index] = k_xyz
      cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
      phi[cur_index] = cur_phi
      theta[cur_index] = cur_theta
      for l_mod_2 in range(2):
        tilde_xi[cur_index, l_mod_2, :, :] = 1/sqrt(2) * (exp(- 1j * k_x * x0[0]) + (-1)**l_mod_2 * exp(1j * k_x * x0[0]))
      cur_index += 1

    # Eigenmode 2
    k_x = 0
    k_z = 0
    for n_y in range(2, n_y_max+1, 2):
      k_y = pi * n_y / L_By
      k_xyz = sqrt(k_y**2)
      if k_xyz > k_max:
        continue
      
      k_amp[cur_index] = k_xyz
      cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
      phi[cur_index] = cur_phi
      theta[cur_index] = cur_theta
      for l_mod_2 in range(2):
        tilde_xi[cur_index, l_mod_2, :, :] = 1/sqrt(2) * (exp(- 1j * k_y * x0[1]) + (-1)**l_mod_2 * exp(1j * k_y * x0[1]))
      cur_index += 1

    # Eigenmode 3
    k_x = 0
    k_y = 0
    for n_z in range(2, n_z_max+1, 2):
      k_z = pi * n_z / L_Cz
      k_xyz = sqrt(k_z**2)
      if k_xyz > k_max:
        continue
      
      k_amp[cur_index] = k_xyz
      cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
      phi[cur_index] = cur_phi
      theta[cur_index] = cur_theta
      for l_mod_2 in range(2):
        tilde_xi[cur_index, l_mod_2, :, :] = 1/sqrt(2) * (exp(- 1j * k_z * x0[2]) + (-1)**l_mod_2 * exp(1j * k_z * x0[2]))
      cur_index += 1
    eigenmode_index_split = cur_index

    # Eigenmode 4
    for n_z in range(-n_z_max, n_z_max+1):
      k_z = pi * n_z / L_Cz

      if n_z <= 0:
        n_x_start = 1
        n_x_end = n_x_max
        
        n_y_start = 1
        n_y_end = n_y_max
      else:
        n_x_start = 0
        n_x_end = n_x_max
        
        n_y_start = 0
        n_y_end = n_y_max

      for n_x in range(n_x_start, n_x_end+1):
        k_x = pi * n_x / L_Ax
        
        k_zx_squared = k_z**2 + k_x**2
        if k_zx_squared > k_max**2:
          continue

        for n_y in range(n_y_start, n_y_end+1):
          if n_x == 0 and n_y == 0:
            continue

          k_y = pi * n_y / L_By
          
          k_xyz = sqrt(k_zx_squared + k_y**2)

          if k_xyz > k_max:
            continue

          k_vec = np.array([k_x, k_y, k_z])

          k_amp[cur_index] = k_xyz
          cur_phi, cur_theta = cart2spherical(k_vec/k_xyz)
          phi[cur_index] = cur_phi
          theta[cur_index] = cur_theta
         
          for l_mod_2 in range(2):
            for m_mod_2 in range(2):
              tilde_xi[cur_index, l_mod_2, m_mod_2, 0] = 1 + (-1)**m_mod_2 * exp(-1j * np.dot(k_vec, np.dot(M_C_minus_id, x0))) * exp(1j * np.dot(k_vec, T_C))
              tilde_xi[cur_index, l_mod_2, m_mod_2, 1] = (-1)**(l_mod_2+m_mod_2) * exp(-1j * np.dot(k_vec, np.dot(M_A_minus_id, x0))) * exp(1j * np.dot(k_vec, T_A)) + (-1)**l_mod_2 * exp(-1j * np.dot(k_vec, np.dot(M_B_minus_id, x0))) * exp(1j * np.dot(k_vec, T_B))
          tilde_xi[cur_index, :, :, :] *= exp(- 1j * np.dot(k_vec, x0))/2
          cur_index += 1
    k_amp = k_amp[:cur_index]
    phi = phi[:cur_index]   
    theta = theta[:cur_index]
    tilde_xi = tilde_xi[:cur_index, :, :, :]

    print(cur_index, eigenmode_index_split, 'split')
    print('Final num of elements:', k_amp.size, 'Minimum k_amp', np.amin(k_amp), 'n_x_max', n_x_max, 'n_z_max', n_z_max)
    return k_amp, phi, theta, tilde_xi, eigenmode_index_split

@njit(nogil=True, parallel=False)
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
    ell_p_range,
    tilde_xi,
    eigenmode_index_split,
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
    C_lmlpmp = np.zeros((num_l_m, num_l_m_p), dtype=np.complex128)    
    m_list = np.arange(0, l_max+1)
    ipow = np.array([1, 1j, -1, -1j])
    shortle = np.array([1, -1])
    # min_k_amp = np.min(k_amp)
    eig_num = k_amp.size
    min_ell_p_square = ell_p_range[0]**2
    min_ell_square = ell_range[0]**2

    for i in range(eig_num):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      sph_harm_index = theta_unique_index[i]

      m_list = np.arange(0, l_max+1)
      phase_list = np.exp(1j * phi[i] * m_list)

      cur_tilde_xi = tilde_xi[i, :, :, :]      

      for l in range(min_ell, max_ell + 1):
        ell_times_ell_plus_one_minus_min_ell = l * (l + 1) - min_ell_square
        # coeff_E_ell = sqrt((l + 2) * (l + 1) * l * (l - 1)) 

        if ell_p_range[0] > l:
          l_start = ell_p_range[0]
        else:
          l_start = l

        for l_p in range(l_start, ell_p_range[1]+1):
          if k_amp_cur > np.sqrt(k_max_list[l]*k_max_list[l_p]):
            continue
          ell_p_times_ell_p_plus_one_minus_min_ell_p = l_p * (l_p + 1) - min_ell_p_square
          # coeff_E_l_p = sqrt((l_p + 2) * (l_p + 1) * l_p * (l_p - 1)) 

          ell_ell_p_integrand_TT = integrand[k_unique_index_cur, l, l_p] * ipow[(l-l_p)%4]    # removed index from integrand because we are doing TT only
          # ell_ell_p_integrand_EE = ( 
          #                         coeff_E_ell * coeff_E_l_p * 
          #                         integrand[1, k_unique_index_cur, l, l_p] 
          #                         * ipow[(l-l_p)%4]
          #                       )
          # ell_ell_p_integrand_TE = (
          #                         coeff_E_l_p* integrand[2, k_unique_index_cur, l, l_p] 
          #                         * ipow[(l-l_p)%4]
          #                       )
          # ell_p_ell_integrand_TE = (
          #                         coeff_E_ell* integrand[2, k_unique_index_cur, l_p, l] 
          #                         * ipow[(l_p - l)%4]
          #                       )
          for m in range(-l, l + 1):
            lm_index_cur = ell_times_ell_plus_one_minus_min_ell + m 
            abs_m = np.abs(m)
            sph_cur_index = lm_index[l, abs_m]

            # This is really Y_{l|m|}
            Y_lm = sph_harm_no_phase[sph_harm_index, sph_cur_index] * phase_list[abs_m]

            if i <= eigenmode_index_split:              
              if m < 0:
                xi_lm = shortle[m%2] * Y_lm * cur_tilde_xi[l%2, m%2, 0]
              else:
                xi_lm = np.conjugate(Y_lm) * cur_tilde_xi[l%2, m%2, 0]
            else:
              if m < 0:
                xi_lm = Y_lm * cur_tilde_xi[l%2, m%2, 0] + np.conjugate(Y_lm) * cur_tilde_xi[l%2, m%2, 1]
                xi_lm *= shortle[m%2]
              else:
                xi_lm = np.conjugate(Y_lm) * cur_tilde_xi[l%2, m%2, 0] + Y_lm * cur_tilde_xi[l%2, m%2, 1]
            
            for m_p in range(0, l_p + 1):   # this indentation was unclear, is it correct like this?
              lm_p_index_cur = ell_p_times_ell_p_plus_one_minus_min_ell_p + m_p  
              sph_p_cur_index = lm_index[l_p, m_p]
              Y_lm_p = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * phase_list[m_p]

              if i <= eigenmode_index_split:
                # xi depends on Y^*
                # xi^* depends on Y
                # We use Y_lm^*
                
                if m_p < 0:
                  xi_lm_p = shortle[m_p%2] * Y_lm_p * cur_tilde_xi[l_p%2, m_p%2, 0]
                else:
                  xi_lm_p = np.conjugate(Y_lm_p) * cur_tilde_xi[l_p%2, m_p%2, 0]
              else:
                if m_p < 0:
                  xi_lm_p = Y_lm_p * cur_tilde_xi[l_p%2, m_p%2, 0] + np.conjugate(Y_lm_p) * cur_tilde_xi[l_p%2, m_p%2, 1]
                  xi_lm_p *= shortle[m_p%2]
                else:
                  xi_lm_p = np.conjugate(Y_lm_p) * cur_tilde_xi[l_p%2, m_p%2, 0] + Y_lm_p * cur_tilde_xi[l_p%2, m_p%2, 1]

                Xi = xi_lm * np.conjugate(xi_lm_p)
                C_lmlpmp[lm_index_cur, lm_p_index_cur] += ell_ell_p_integrand_TT * Xi
                # C_lmlpmp[1, lm_index_cur, lm_p_index_cur] += ell_ell_p_integrand_EE * Xi
                # C_lmlpmp[2, lm_index_cur, lm_p_index_cur] += ell_ell_p_integrand_TE * Xi
                # if l != l_p:
                #   C_lmlpmp[2, lm_p_index_cur, lm_index_cur] += ell_p_ell_integrand_TE* conjugate(Xi)

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
            # if l != l_p:
            #   C_lmlpmp[2, lm_p_index_new, lm_index_new] = shortle[(m+m_p)%2] * conjugate(C_lmlpmp[2, lm_p_index_cal, lm_index_cal])

    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    return C_lmlpmp



@njit(nogil=True, parallel=False)
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
    ell_p_range,
    tilde_xi,
    eigenmode_index_split,
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
    C_lmlpmp = np.zeros((num_l_m, num_l_m_p), dtype=np.complex128)    
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
      sph_harm_index = theta_unique_index[i]

      m_list = np.arange(0, l_max+1)
      phase_list = np.exp(1j * phi[i] * m_list)

      cur_tilde_xi = tilde_xi[i, :, :, :]      

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
          ell_ell_p_integrand_TT = integrand[0, k_unique_index_cur, l, l_p] * ipow[(l-l_p)%4]
        for m in range(-l, l + 1):
            lm_index_cur = ell_times_ell_plus_one_minus_min_ell + m 
            abs_m = np.abs(m)
            sph_cur_index = lm_index[l, abs_m]

            # This is really Y_{l|m|}
            Y_lm = sph_harm_no_phase[sph_harm_index, sph_cur_index] * phase_list[abs_m]

            if i <= eigenmode_index_split:              
              if m < 0:
                xi_lm = shortle[m%2] * Y_lm * cur_tilde_xi[l%2, m%2, 0]
              else:
                xi_lm = np.conjugate(Y_lm) * cur_tilde_xi[l%2, m%2, 0]
            else:
              if m < 0:
                xi_lm = Y_lm * cur_tilde_xi[l%2, m%2, 0] + np.conjugate(Y_lm) * cur_tilde_xi[l%2, m%2, 1]
                xi_lm *= shortle[m%2]
              else:
                xi_lm = np.conjugate(Y_lm) * cur_tilde_xi[l%2, m%2, 0] + Y_lm * cur_tilde_xi[l%2, m%2, 1]
            
              for m_p in range(0, l_p + 1):
                  lm_p_index_cur = ell_p_times_ell_p_plus_one_minus_min_ell_p + m_p  
                  sph_p_cur_index = lm_index[l_p, m_p]
                  Y_lm_p = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * phase_list[m_p]

                  if i <= eigenmode_index_split:
                    # xi depends on Y^*
                    # xi^* depends on Y
                    # We use Y_lm^*
                    
                    if m_p < 0:
                      xi_lm_p = shortle[m_p%2] * Y_lm_p * cur_tilde_xi[l_p%2, m_p%2, 0]
                    else:
                      xi_lm_p = np.conjugate(Y_lm_p) * cur_tilde_xi[l_p%2, m_p%2, 0]
                  else:
                    if m_p < 0:
                      xi_lm_p = Y_lm_p * cur_tilde_xi[l_p%2, m_p%2, 0] + np.conjugate(Y_lm_p) * cur_tilde_xi[l_p%2, m_p%2, 1]
                      xi_lm_p *= shortle[m_p%2]
                    else:
                      xi_lm_p = np.conjugate(Y_lm_p) * cur_tilde_xi[l_p%2, m_p%2, 0] + Y_lm_p * cur_tilde_xi[l_p%2, m_p%2, 1]

                  Xi = xi_lm * np.conjugate(xi_lm_p)
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

