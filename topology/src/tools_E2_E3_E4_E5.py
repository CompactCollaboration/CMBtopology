from .topology import Topology
from .tools import *
import numpy as np
from numpy import pi, sin, cos, exp, sqrt, tan, conjugate
from numba import njit, prange



# computing the TT correlations
@njit(nogil=True, parallel=False)
def E2_E3_E4_E5_get_c_TT_lmlpmp(
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
    tilde_xi_delta_m,
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
    num_l_m = ell_range[1] * (ell_range[1] + 1) + ell_range[1] + 1 - ell_range[0]**2     
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

      cur_tilde_xi = tilde_xi[i, :]

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
          ell_ell_p_integ_TT = integrand[k_unique_index_cur, l, l_p] * ipow[(l - l_p)%4]

          for m in range(-l, l + 1):
            lm_index_cur = ell_times_ell_plus_one_minus_min_ell + m 
            abs_m = np.abs(m)
            sph_cur_index = lm_index[l, abs_m]

            sph_harm_l_m = sph_harm_no_phase[sph_harm_index, sph_cur_index] * phase_list_plus[abs_m]
            if m<0:
              sph_harm_l_m = shortle[abs_m%2] * conjugate(sph_harm_l_m)

            # Then tilde xi
            xi_m = cur_tilde_xi[m % tilde_xi_delta_m]
          
            for m_p in range(0, l_p + 1):
              lm_p_index_cur = ell_p_times_ell_p_plus_one_minus_min_ell_p + m_p  
              sph_p_cur_index = lm_index[l_p, m_p]

              sph_harm_l_m_p = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * phase_list_plus[m_p] 

              xi_m_p_conj = np.conjugate(cur_tilde_xi[m_p % tilde_xi_delta_m])

              Xi = np.conjugate(sph_harm_l_m)* sph_harm_l_m_p * xi_m * xi_m_p_conj

              C_lmlpmp[lm_index_cur, lm_p_index_cur] += ell_ell_p_integ_TT * Xi

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