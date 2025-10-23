
from numba import njit, prange
import numba
import numpy as np
from numpy import pi, sqrt, exp
import quaternionic
import spherical

@njit
def cart2phi(x, y):
    return np.arctan2(y, x)

@njit
def cart2theta(xy_squared, z):
    return np.arctan2(np.sqrt(xy_squared), z)

@njit
def cart2spherical(xyz):
    # Calculates spherical coordinates from cartesian coordinates
    xy = xyz[0]**2 + xyz[1]**2

    # phi
    phi = np.arctan2(xyz[1], xyz[0])
    # theta
    theta = np.arctan2(np.sqrt(xy), xyz[2])
    
    return phi, theta


def get_D_l(c_l):
    return np.array([c_l[l] * l * (l+1) / (2*np.pi) for l in range(c_l.size)])

@njit
def isclose(param_list, param):
  # Check if a parameter is already in a list of that parameter
  # We use this for example to not recalculate spherical harmonics twice for same theta
  param_abs = np.abs(param)
  for i in range(param_list.size):
    if param_abs < 1e-8:
      if np.abs(param_list[i] - param) < 1e-8:
        return i
    elif np.abs((param_list[i] - param) / param)  < 1e-8:
      return i
  return -1


def eigen(m):
    if m.shape[0] != m.shape[1]:
        raise TypeError("The matrix should be square!")
    mat_size = np.int32(m.shape[0])
    blcok_size = np.int32(mat_size/2)
    block_mat = np.zeros((3, blcok_size), dtype=np.complex128)
    eigen_vec = np.zeros((mat_size, mat_size), dtype=np.complex128)
    for i in range(blcok_size):
        block_mat[0, i] = m[i, i]
        if m[i, blcok_size + i] != m[i + blcok_size, i]:
            raise TypeError("The matrix should be symmetric!")
        block_mat[1, i] = m[i, blcok_size + i]
        block_mat[2, i] = m[i + blcok_size, i + blcok_size]

    eig_values_1 = 0.5 * (block_mat[0]+ block_mat[2] 
                - sqrt(block_mat[0] * block_mat[0]
                       + 4 * block_mat[1] * block_mat[1]
                       - 2 * block_mat[0] * block_mat[2]
                       + block_mat[2] * block_mat[2])
                )
    eig_values_2 = 0.5 * (block_mat[0]+ block_mat[2] 
                + sqrt(block_mat[0] * block_mat[0]
                       + 4 * block_mat[1] * block_mat[1]
                       - 2 * block_mat[0] * block_mat[2]
                       + block_mat[2] * block_mat[2])
                )
    
    for i in range(blcok_size):
        first_eig = - 0.5 * (- block_mat[0]+ block_mat[2] 
                + sqrt(block_mat[0] * block_mat[0]
                       + 4 * block_mat[1] * block_mat[1]
                       - 2 * block_mat[0] * block_mat[2]
                       + block_mat[2] * block_mat[2])
                ) / block_mat[1]
        eigen_vec[i , i] = first_eig[i]
        eigen_vec[i , i + blcok_size] = 1
        second_eig = - 0.5 * (- block_mat[0]+ block_mat[2] 
                - sqrt(block_mat[0] * block_mat[0]
                       + 4 * block_mat[1] * block_mat[1]
                       - 2 * block_mat[0] * block_mat[2]
                       + block_mat[2] * block_mat[2])
                ) / block_mat[1]
        eigen_vec[blcok_size + i, i] = second_eig[i]
        eigen_vec[blcok_size + i , i + blcok_size] = 1
    # eigen_vec = eigen_vec / np.linalg.norm(eigen_vec, axis=1)[:, np.newaxis]
    norms = sqrt(np.sum(eigen_vec ** 2, axis=1))
    eigen_vec = eigen_vec / norms[:, np.newaxis]

    return np.flip(np.hstack((eig_values_1, eig_values_2))), np.flipud(eigen_vec).T
    

@njit
def get_idx(l_max, l, m):
    # From the Healpy library. But we copy it here so that Numba can use it
    return m * (2 * l_max + 1 - m) // 2 + l

@njit(parallel=True)
def get_unique_array_indices(full_param_array, param_repeat):
    # The point of this function is to save all unique parameters of a list
    # into a new list. For example:
    # There are many numbers in the theta list that are the same
    # but we do not want to calculate the spherical harmonics for same theta twice or more
    # So we find only the spherical harmonics for each unique theta
    # We also save a lot of memory by saving sperical harmonics for unique theta

    assert(full_param_array.size == param_repeat.size)
    num_indices = full_param_array.size
    unique_param_length = np.count_nonzero(param_repeat==-1)
    param_unique, param_unique_index = np.zeros(unique_param_length), np.zeros(num_indices, dtype=numba.int32)

    max_index = 0
    for i in prange(num_indices):
      if param_repeat[i] != -1:
        # This theta is not unique
        index = isclose(param_unique[:max_index], full_param_array[i])
        assert (index != -1)
        param_unique_index[i] = index
      else:
        # We found a new unique parameter in the list
        # Save it
        param_unique_index[i] = max_index
        param_unique[max_index] = full_param_array[i]
        max_index += 1
    return param_unique, param_unique_index



@njit
def get_c_l_from_c_lmlpmp(c_lmlpmp, l_max):
    c_l = np.zeros(l_max+1, dtype=np.complex128)

    one_dim = c_lmlpmp.ndim == 1
    for l in range(2, l_max+1):
        for m in range(-l, l+1):
            if one_dim:
                lm_id = l * (l+1) + m
                c_l[l] += c_lmlpmp[lm_id]
            else:
                lm_id = l * (l+1) + m - 2**2
                c_l[l] += c_lmlpmp[lm_id, lm_id]
        c_l[l] /= 2*l + 1
    return np.real(c_l)


@njit
def get_lm_idx(l_min, l, m):
    """
    Returns the index corresponding to (l, m) in an array describing alm up.

    Parameters:
    - l_min (int): The minimum l, defines the alm layout.
    - l (int): The l for which to get the index.
    - m (int): The m for which to get the index.

    Returns:
    - idx (int): The index corresponding to (l, m).
    """
    if l < l_min or m < 0 or m > l:
        raise ValueError("Invalid values for l and m")
    
    return int((l_min + l +1) * (l - l_min) / 2 + m)




def get_sph_harm_parallel(i, l_max, theta, lm_index, num_l_m):
  # Get the spherical harmonics with no phase (phi=0) for a given index i
  # Get Wigner_d matrix for a given index i
  sph_harm_no_phase_i = np.zeros(num_l_m, dtype=np.float64)
  theta_cur = theta[i]
  wigner = spherical.Wigner(l_max)
  R = quaternionic.array.from_spherical_coordinates(theta_cur, 0)
  Y = wigner.sYlm(0, R)
  for l in range(l_max+1):
      for m in range(l+1):
          sph_harm_no_phase_i[lm_index[l, m]] = np.real(Y[wigner.Yindex(l, m)])    
          
  return sph_harm_no_phase_i


## KL Divergence
def calculate_kl_divergence(matrix):
  w, _ = np.linalg.eig(matrix)
  kl_P_assuming_Q = 0
  kl_Q_assuming_P = 0

  for eig in w:
    kl_P_assuming_Q += (np.log(np.abs(eig)) + 1/eig - 1)/2
    kl_Q_assuming_P += (-np.log(np.abs(eig)) + eig - 1)/2

  return np.real(kl_P_assuming_Q), np.real(kl_Q_assuming_P)




def get_wigner_d(i, l_max, theta, lm_2_index, num_l_m_2):
  
  # Get Wigner_d matrix for a given index i
  wigner_d_i = np.zeros(2* num_l_m_2, dtype=np.float64)
  theta_cur = theta[i]
  wigner = spherical.Wigner(l_max)
  R = quaternionic.array.from_spherical_coordinates(theta_cur, 0)
  D = wigner.D(R)
  for l in range(2, l_max+1):
      for m in range(l+1):
          wigner_d_i[lm_2_index[l, m]] = np.real(D[wigner.Dindex(l, m, 2)])
          wigner_d_i[num_l_m_2+ lm_2_index[l, m]] = np.real(D[wigner.Dindex(l, m, -2)])
          
  return wigner_d_i


def transfer_parallel(i, l_max, k_amp, transfer_interpolate_k_l_list):
  # Get the transfer function for a given index i
  cur_transfer_i = np.zeros(l_max+1)
  for l in range(2, l_max+1):
    cur_transfer_i[l] = transfer_interpolate_k_l_list[l](k_amp[i])

  return cur_transfer_i






def get_k_theta_index_repeat(k_amp, theta):
    # k and theta often repeats themselves in the full list of allowed wavenumber list
    # So we want to know when they have repeated values so that we dont have to
    # recalculate spherical harmonics for example.

    # The unique lists are the lists of only unique elements
    # The unique_index lists are the indices going form the full parameter list to the
    # unique list.
    # For example:
    # k = [1, 2, 3, 1, 2, 5]
    # k_unique = [1, 2, 3, 5]
    # k_unique_index = [0, 1, 2, 0, 1, 3]
    
    length = theta.size

    print('Getting repeated theta and k elements')
    k_amp_unique, k_amp_unique_index = np.unique(np.round(k_amp, decimals=7), return_inverse=True)
    theta_unique, theta_unique_index = np.unique(np.round(theta, decimals=7), return_inverse=True)
    # k_amp_unique, k_amp_unique_index = np.unique(k_amp, return_inverse=True)
    # theta_unique, theta_unique_index = np.unique(theta, return_inverse=True)
    print('Ratio of unique theta:', theta_unique.size / length)
    print('Ratio of unique |k|:', k_amp_unique.size / length)

    return k_amp_unique, k_amp_unique_index, theta_unique, theta_unique_index


@njit
def do_integrand_pre_processing(unique_k_amp, tensor_pk_k3, transfer_delta_kl_X, transfer_delta_kl_Y, l_max):
    # Calculating P(k) / k^3 * Delta_ell_X(k) * Delta_ell'_Y(k) 
    num_k_amp = unique_k_amp.size
    integrand = np.zeros((num_k_amp, l_max+1, l_max+1))
    for i in range(num_k_amp):
        tensor_pk_k3_cur = tensor_pk_k3[i]
        for l in range(2, l_max+1): 
            for lp in range(2, l_max+1):
                integrand[i, l, lp] = tensor_pk_k3_cur * transfer_delta_kl_X[i, l] * transfer_delta_kl_Y[i, lp]

    return integrand

@njit
def normalize_c_lmlpmp(c_lmlpmp, camb_c_l_X, camb_c_l_Y, l_min, l_max, lp_min, lp_max, cl_accuracy=1):
    # Normalize the covariance matrix by dividing sqrt(c_l * c_l'). This is used in the KL divergence.
    # I also divide by cl_accuracy as an approximation of the lost power along the diagonal.
    # Wihtout it we get that the diagonal in the L->infinity limit becomes cl_accuracy and not 1. This
    # adds a contribution to the KL divergence which should not be there.
    normalized_c_lmlpmp = np.zeros(c_lmlpmp.shape, dtype=np.complex128)
    for l in range(l_min, l_max+1):
        for m in range(-l, l+1):
            index = l * (l+1) + m - l_min**2
            for lp in range(lp_min, lp_max+1):
                for mp in range(-lp, lp+1):
                    index_p = lp * (lp+1) + mp - lp_min**2                    
                    normalized_c_lmlpmp[index, index_p] = c_lmlpmp[index, index_p] / (np.sqrt(camb_c_l_X[l]*camb_c_l_Y[lp]) * cl_accuracy)
    return normalized_c_lmlpmp



@njit
def normalize_kl_c_lmlpmp(c_lmlpmp, camb_c_l_X, l_min, l_max, lp_min, lp_max):
    # Normalize the covariance matrix by dividing sqrt(c_l * c_l'). This is used in the KL divergence.
    # I also divide by cl_accuracy as an approximation of the lost power along the diagonal.
    # Wihtout it we get that the diagonal in the L->infinity limit becomes cl_accuracy and not 1. This
    # adds a contribution to the KL divergence which should not be there.
    normalized_c_lmlpmp = np.zeros(c_lmlpmp.shape, dtype=np.complex128)
    for l in range(l_min, l_max+1):
        for m in range(-l, l+1):
            index = l * (l+1) + m - l_min**2
            for lp in range(lp_min, lp_max+1):
                for mp in range(-lp, lp+1):
                    index_p = lp * (lp+1) + mp - lp_min**2                    
                    normalized_c_lmlpmp[index, index_p] = c_lmlpmp[index, index_p] / camb_c_l_X[l]
    return normalized_c_lmlpmp