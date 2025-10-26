from .tools import *
from .topology import Topology
from .config import L_LSS
import numpy as np
from numpy import pi, sqrt, conjugate, exp, dot
from numba import njit, prange

class E7(Topology):
  def __init__(self, param, debug=True, make_run_folder = False):
    self.LAx = param['LAx'] * L_LSS
    self.LAy = param['LAy'] * L_LSS
    self.L1y = param['L1y'] * L_LSS
    self.L2x = param['L2x'] * L_LSS
    self.L2z = param['L2z'] * L_LSS
    self.x0 = param['x0'] * L_LSS
 
    self.V = self.LAx * self.L1y * self.L2z 
    self.l_max = param['l_max']
    self.l_min = param['l_min']

    self.param = param
    
    print('Running - E7 l_max={}_LAx_{}_LAy_{}_L1y_{}_L2x_{}_L2z_{}_x_{}_y_{}_z_{}'.format(self.l_max, 
    int(self.LAx), int(self.LAy), 
    int(self.L1y), int(self.L2x), int(self.L2z), 
    "{:.2f}".format(param['x0'][0]),
    "{:.2f}".format(param['x0'][1]),
    "{:.2f}".format(param['x0'][2])))
    self.root = 'runs/{}_LAx_{}_LAy_{}_L1y_{}_L2x_{}_L2z_{}_x_{}_y_{}_z_{}_l_max_{}/'.format(
        param['topology'],
        "{:.2f}".format(param['LAx']),
        "{:.2f}".format(param['LAy']),
        "{:.2f}".format(param['L1y']),
        "{:.2f}".format(param['L2x']),
        "{:.2f}".format(param['L2z']),
        "{:.2f}".format(param['x0'][0]),
        "{:.2f}".format(param['x0'][1]),
        "{:.2f}".format(param['x0'][2]),
        self.l_max
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
    # This function seems unnecessary, but Numba does not allow return_dict
    # which is of type multiprocessing.Manager
    if self.do_polarization:
      raise NotImplementedError(
          "E1 topology with polarization covariance matrices is not implemented yet "
          "and will be released in the next version."
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
        ell_p_range,
        self.tilde_xi,
        self.split_index,
      )



  def get_list_of_k_phi_theta(self):

    M_A = np.diag([1.0, -1.0, 1.0])
    k_amp, phi, theta, tilde_xi, split_index = get_list_of_k_phi_theta(max(self.k_max_list), self.k_list[0], self.LAx, 
                                                                       self.LAy, self.L1y, self.L2x, self.L2z, M_A, self.x0)
    self.tilde_xi = tilde_xi
    self.split_index = split_index
    return k_amp, phi, theta 
  
 
@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, k_min, LAx, LAy, L1y, L2x, L2z, M_A, x0):
    # Returns list of k, phi, and theta for this topology

    n_x_max = int(np.ceil(k_max * LAx / pi))
    n_y_max = int(np.ceil(k_max * L1y / (2*pi)))
    # abs(L2x) < LAx so the extreme case for n_z will hapen for L2x = LAx
    n_z_max = int(np.ceil(k_max * L2z / (2*pi) + n_x_max / 2)) 
 
    list_length = n_x_max * n_y_max * n_z_max * 8
    k_amp = np.zeros(list_length)
    phi = np.zeros(list_length)
    theta = np.zeros(list_length)

    tilde_xi = np.zeros((list_length, 2), dtype=np.complex128)

    T_A = np.array([LAx, LAy, 0])

    cur_index = 0
    
    # Eigenmode 1
    k_y = 0
    # Only n_z = 0 mod 2
    for n_x in range(-n_x_max, n_x_max + 1):
      if n_x % 2 == 1 or n_x == 0:
        continue
      k_x = pi * n_x / LAx
      for n_z in range(-n_z_max, n_z_max + 1):
        if n_z == 0:
          continue
        k_z = 2*pi / L2z * (n_z - L2x * n_x / (2 * LAx))
        k_xyz = sqrt(k_x * k_x + k_z * k_z)
        if k_xyz > k_max or k_xyz < k_min:
          continue
        
        k_amp[cur_index] = k_xyz
        cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
        phi[cur_index] = cur_phi
        theta[cur_index] = cur_theta
        tilde_xi[cur_index, :] = exp(- 1j * 
                                     (k_x * x0[0] + k_z * x0[2])
                                    )
        cur_index += 1
    print('Eigenmode 1:', cur_index)
    split_index = cur_index
    # Eigenmode 2
    for n_x in range(-n_x_max, n_x_max+1):
      k_x = pi * n_x / LAx
      for n_y in range(1, n_y_max+1):
        k_y = 2*pi * n_y / L1y 

        k_xy_squared = k_x*k_x + k_y*k_y
        if k_xy_squared > k_max*k_max:
          continue
        
        for n_z in range(-n_z_max, n_z_max+1):
          k_z = 2*pi / L2z * (n_z - L2x * n_x / 2 / LAx)
          
          k_xyz = sqrt(k_xy_squared + k_z*k_z)

          if k_xyz > k_max or k_xyz < k_min:
            continue

          k_vec = np.array([k_x, k_y, k_z])

          k_amp[cur_index] = k_xyz
          cur_phi, cur_theta = cart2spherical(k_vec/k_xyz)
          phi[cur_index] = cur_phi
          theta[cur_index] = cur_theta
          tilde_xi[cur_index, 0] = exp(- 1j * np.dot(k_vec, x0))/sqrt(2) 
          tilde_xi[cur_index, 1] = exp(- 1j * dot(k_vec, M_A @ x0 - T_A))/sqrt(2) 

          cur_index += 1


    print('Eigenmode 2:', cur_index)
    k_amp = k_amp[:cur_index]
    phi = phi[:cur_index]   
    theta = theta[:cur_index]
    tilde_xi = tilde_xi[:cur_index, :]

    print('E7 Final num of elements:', k_amp.size, 'Minimum k_amp', np.amin(k_amp), 'n_x_max', n_x_max, 'n_z_max', n_z_max)
    return k_amp, phi, theta, tilde_xi, split_index


 

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
    ell_p_range,
    tilde_xi,
    split_index,
    ):

    num_l_m = ell_range[1] * (ell_range[1] + 1) + ell_range[1] + 1 - ell_range[0]**2     # = Sum[2 l+1, {l , l_min, l_max}]
    num_l_m_p = ell_p_range[1] * (ell_p_range[1] + 1) + ell_p_range[1] + 1 - ell_p_range[0]**2 
    C_lmlpmp = np.zeros((num_l_m, num_l_m_p), dtype=np.complex128)    
    m_list = np.arange(0, l_max+1)
    shortle = np.array([1, -1])
    ipow = np.array([1, 1j, -1, -1j])
    min_k_amp = np.min(k_amp)
    eig_num = k_amp.size
    min_ell_p_square = ell_p_range[0]**2
    min_ell_square = ell_range[0]**2
    for i in range(eig_num):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      sph_harm_index = theta_unique_index[i] # wigner_d for especific theta[i]
      cur_tilde_xi = tilde_xi[i, :]

      phase_list_plus = np.exp(1j * phi[i] * m_list)

      for l in range(min_ell, max_ell + 1):
        ell_times_ell_plus_one_minus_min_ell = l * (l + 1) - min_ell_square

        if ell_p_range[0] > l:
          l_start = ell_p_range[0]
        else:
          l_start = l

        for l_p in range(l_start, ell_p_range[1]+1):
          if k_amp_cur > np.sqrt(k_max_list[l]*k_max_list[l_p]) or k_amp_cur < min_k_amp:
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

            if i < split_index:
              xi_lm = np.conjugate(sph_harm_l_m) * cur_tilde_xi[0]
            else:
              xi_lm = np.conjugate(sph_harm_l_m) * cur_tilde_xi[0] + sph_harm_l_m * cur_tilde_xi[1]

            for m_p in range(0, l_p + 1):
              lm_p_index_cur = ell_p_times_ell_p_plus_one_minus_min_ell_p + m_p  

              sph_p_cur_index = lm_index[l_p, m_p]
              sph_harm_l_m_p = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * phase_list_plus[m_p]
              
              if i < split_index:
                xi_lm_p = np.conjugate(sph_harm_l_m_p) * cur_tilde_xi[0]
              else:
                xi_lm_p = np.conjugate(sph_harm_l_m_p) * cur_tilde_xi[0] + sph_harm_l_m_p * cur_tilde_xi[1]

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

    C_lmlpmp *= 32 * pi * pi * pi * pi / V
    
    return C_lmlpmp