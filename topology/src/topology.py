"""
topology.py

Base class for computing CMB covariance matrices for non-trivial topologies.
Implements preprocessing, CAMB integration, transfer function interpolation,
spherical harmonics computation, and covariance matrix calculations.
"""

from locale import normalize
import camb
import os
import numpy as np
from numpy import pi, sqrt
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import multiprocessing
from itertools import repeat
from tqdm import tqdm
import time
from .tools import *
from sys import getsizeof

class Topology:
    """Base class for CMB covariance matrix computation in non-trivial topologies.

    Attributes:
        param (dict): Configuration parameters (e.g., topology, l_max, c_l_accuracy).
        topology (str): Topology type (e.g., 'E1').
        l_max (int): Maximum multipole.
        l_min (int): Minimum multipole.
        c_l_accuracy (float): Accuracy for wavevector cutoff (e.g., 0.99).
        C_l_type_array (np.ndarray): Array of correlation types ['TT', 'EE', 'TE'].
        do_polarization (bool): Whether to compute polarization (EE, TE).
        fig_name (str): Base name for output figures.
        debug (bool): Enable debug output.
        make_run_folder (bool): Create output directories.
        root (str): Output directory path.
        powers (np.ndarray): CMB power spectra from CAMB.
        k_list (np.ndarray): Wavevector magnitudes from CAMB.
        ell_list (np.ndarray): Multipoles from CAMB.
        transfer_T_interpolate_k_l_list (dict): Interpolated T transfer functions.
        transfer_E_interpolate_k_l_list (dict): Interpolated E transfer functions.
        k_max_list (np.ndarray): Wavevector cutoffs per multipole.
        k_amp (np.ndarray): Wavevector magnitudes.
        phi (np.ndarray): Wavevector azimuthal angles.
        theta (np.ndarray): Wavevector polar angles.
        k_amp_unique (np.ndarray): Unique wavevector magnitudes.
        k_amp_unique_index (np.ndarray): Indices for unique k_amp.
        theta_unique (np.ndarray): Unique polar angles.
        theta_unique_index (np.ndarray): Indices for unique theta.
        scalar_pk_k3 (np.ndarray): Primordial power spectrum divided by k^3.
        transfer_T_delta_kl (np.ndarray): T transfer functions for unique k.
        transfer_E_delta_kl (np.ndarray): E transfer functions for unique k.
        lm_index (np.ndarray): Indices for spherical harmonic coefficients.
        integrand (np.ndarray): Preprocessed integrand for covariance computation.
        C_lmlpmp (np.ndarray): Covariance matrix.
        normalized_C_lmlpmp (np.ndarray): Rescaled covariance matrix.
    """
    def __init__(self, param, debug=True, make_run_folder = True):
        """Initialize the Topology class with simulation parameters.

        Args:
            param (dict): Parameters including topology, l_max, l_min, c_l_accuracy,
                          do_polarization, etc.
            debug (bool, optional): Enable debug output. Defaults to True.
            make_run_folder (bool, optional): Create output directories. Defaults to True.
        """
        print('Running', param)
        self.param = param
        self.topology = param['topology']
        self.l_max = param['l_max']
        self.l_min = param['l_min']
        self.c_l_accuracy = param['c_l_accuracy']
        C_l_type_array= np.array(['TT','EE','TE'])
        self.C_l_type_array = C_l_type_array
        self.do_polarization = param['do_polarization']

        self.fig_name = 'l_max_{}'.format(self.l_max)
        self.debug = debug
        self.make_run_folder = make_run_folder

        # Create output directories
        if make_run_folder and os.path.exists(self.root) == False:
            print('Making run folder:', self.root)
            os.makedirs(self.root)
            os.makedirs(self.root+'figs/')
            os.makedirs(self.root+'realizations/') 

        # Perform preprocessing
        start_time = time.time()
        self.do_pre_processing()
        print(f'Time to pre-process with l_max={self.l_max} and accuracy={self.c_l_accuracy}: '
              f'{time.time() - start_time:.2f} seconds')

    def do_pre_processing(self):
        """Perform preprocessing for covariance matrix computation.

        Sets up CAMB parameters, computes transfer functions, primordial power spectrum,
        and spherical harmonics, optimizing for memory efficiency.
        """
        # Configure CAMB parameters
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        pars.set_for_lmax(self.l_max)

        # More accurate transfer functions. Takes longer time to run
        pars.set_accuracy(AccuracyBoost = 2, lAccuracyBoost = 2, lSampleBoost = 50)
        pars.Accuracy.IntkAccuracyBoost = 2
        pars.Accuracy.SourcekAccuracyBoost = 2

        # Compute transfer functions and power spectra
        data = camb.get_transfer_functions(pars)
        results = camb.get_results(pars)
        self.powers = results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK')['unlensed_scalar']
        transfer_function = data.get_cmb_transfer_data(tp='scalar')


        # To get C_\ell in units of umK, we multiply by 1e6 (K to micro K) and the temperature of the CMB in K
        transfer_data = np.array(transfer_function.delta_p_l_k) * 1e6 * 2.7255  # Convert to muK
        print(f'Shape of transfer function from CAMB: {transfer_data.shape}')

        # CAMB gives the transfer data for a set of k and ell. Store these values
        # and later we will use interpolation for other k/ell values.
        #k is in Mpc^{-1}
        self.k_list = np.array(transfer_function.q)
        self.ell_list = np.array(transfer_function.L)

        # Calculates the transfer functions, primordial power spectrum,
        # and spherical harmonics and stores them with as little memory as possible

        l_max = self.l_max
        l_min = self.l_min

        # Get the transfer functions and put them into lists of interpolate objects.
        # We can therefore evaluate the transfer function for all possibel |k| later
        assert(self.ell_list[0] == 2 and self.ell_list[l_max-2] == l_max)
        self.transfer_T_interpolate_k_l_list ={}
        if self.do_polarization:
            self.transfer_E_interpolate_k_l_list = {}
        for l in range(2, l_max+1):
            self.transfer_T_interpolate_k_l_list[l] = scipy.interpolate.interp1d(self.k_list, transfer_data[0, l-2, :], kind='cubic') 
            if self.do_polarization:
                self.transfer_E_interpolate_k_l_list[l] = scipy.interpolate.interp1d(self.k_list, transfer_data[1, l-2, :], kind='cubic') 
            
        # Compute wavevector cutoffs
        self.get_kmax_as_function_of_ell(pars.scalar_power)
        
        # We find all allowed |k|, phi, theta and put them in big lists
        # The function get_list_of_k_phi_theta is specific to each topology
        start_time = time.time()
        self.k_amp, self.phi, self.theta = self.get_list_of_k_phi_theta()
        print('Time to get list of k, phi, theta:', time.time()-start_time, 'seconds')

        # |k|, phi and theta often repeats themselves. We do not want to recalculate spherical harmonics
        # twice or more so we store a list of all unique thetas. Same for |k| to quickly find transfer functions later
        # Optimize memory by storing unique k and theta
        start_time = time.time()
        self.k_amp_unique, self.k_amp_unique_index, self.theta_unique, self.theta_unique_index = get_k_theta_index_repeat(self.k_amp, self.theta)
        print('Time to get unique k and theta:', time.time()-start_time, 'seconds')


        # Get P(k) / k^3 for all unique |k| values
        self.scalar_pk_k3 = pars.scalar_power(self.k_amp_unique) / self.k_amp_unique**3

        # Get the transfer function for all unique |k| values
        self.transfer_T_delta_kl = self.get_transfer_functions_multi(self.transfer_T_interpolate_k_l_list)
        if self.do_polarization:
            self.transfer_E_delta_kl = self.get_transfer_functions_multi(self.transfer_E_interpolate_k_l_list)
        
        # Store spherical harmonic indices
        self.lm_index = np.zeros((l_max+1, l_max+1), dtype=np.int32)
        for l in range(l_min, l_max+1):
            for m in range(l+1):
                self.lm_index[l, m] = get_lm_idx(l_min, l, m)

        # Compute spherical harmonics 
        self.get_sph_harm()

        print('\n**************\nDone with all preprocessing\n**************')

        # Get the spherical harmonics without the phase (without exp(i*m*phi))
        # We store these in an array of size (all lm, all n_z, all n*x^2+n_y^2)
        # This is because theta can be found from nz and nx^2+ny^2, and we do not
        # care about phi since we can add the phase in the sum
        #start_time = time.time()

        

   
    def calculate_c_lmlpmp(self, normalize = False, plot_param={}):
        """Compute the CMB covariance matrix C_lmlpmp.

        Args:
            normalize (bool, optional): Normalize the covariance matrix. Defaults to False.
            plot_param (dict, optional): Parameters for plotting (e.g., l_ranges, lp_ranges).

        Returns:
            np.ndarray: Covariance matrix C_lmlpmp.
        """
        print('\nCalculating covariance matrix') 
        
        if self.do_polarization:
            integrand_TT = do_integrand_pre_processing(
                self.k_amp_unique, self.scalar_pk_k3, self.transfer_T_delta_kl,
                self.transfer_T_delta_kl, self.l_max)
            integrand_EE = do_integrand_pre_processing(
                self.k_amp_unique, self.scalar_pk_k3, self.transfer_E_delta_kl,
                self.transfer_E_delta_kl, self.l_max)
            integrand_TE = do_integrand_pre_processing(
                self.k_amp_unique, self.scalar_pk_k3, self.transfer_T_delta_kl,
                self.transfer_E_delta_kl, self.l_max)
            self.integrand = np.stack((integrand_TT, integrand_EE, integrand_TE), dtype=np.float32)
        else:
            self.integrand = do_integrand_pre_processing(
                self.k_amp_unique, self.scalar_pk_k3, self.transfer_T_delta_kl,
                self.transfer_T_delta_kl, self.l_max)

       
        print(f'Size of integrand: {round(self.integrand.size * self.integrand.itemsize / 1024 / 1024, 2)} MB')

        l_min = plot_param['l_ranges'][0, 0]
        l_max = plot_param['l_ranges'][0, 1]
        lp_min = plot_param['lp_ranges'][0, 0]
        lp_max = plot_param['lp_ranges'][0, 1]
        # Make sure the l_ranges do not overlap!
        ell_range = np.array(plot_param['l_ranges'][0, :])
        ell_p_range = np.array(plot_param['lp_ranges'][0, :])
 
        start_time = time.time()
        C_lmlpmp = self.get_c_lmlpmp_multiprocessing(
            ell_range = ell_range,
            ell_p_range = ell_p_range,)
        total_time_seconds = time.time() - start_time
        hours, remainder = divmod(total_time_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f'Time to get Correlation functions: {int(hours)}:{int(minutes)}:{int(seconds)}')
        self.C_lmlpmp = C_lmlpmp

        # Save covariance matrix
        filename = f'{"full" if self.do_polarization else "TT"}_corr_matrix_l_{l_min}_{l_max}_lp_{lp_min}_{lp_max}.npy'
        np.save(os.path.join(self.root, filename), C_lmlpmp)
            
        # Normalize if requested
        if normalize:
            normalized_C_lmlpmp = np.zeros_like(C_lmlpmp)
            if self.do_polarization:
                for i, cl_type in enumerate([0, 1, 3]):  # TT, EE, TE
                    normalized_C_lmlpmp[i] = normalize_c_lmlpmp(
                        C_lmlpmp[i], self.powers[:, cl_type], self.powers[:, cl_type],
                        l_min=l_min, l_max=l_max, lp_min=lp_min, lp_max=lp_max,
                        cl_accuracy=self.c_l_accuracy)
            else:
                normalized_C_lmlpmp = normalize_c_lmlpmp(
                    C_lmlpmp, self.powers[:, 0], self.powers[:, 0],
                    l_min=l_min, l_max=l_max, lp_min=lp_min, lp_max=lp_max,
                    cl_accuracy=self.c_l_accuracy)
            np.save(os.path.join(self.root, f'norm_{filename}'), normalized_C_lmlpmp)
            self.normalized_C_lmlpmp = normalized_C_lmlpmp
        
        return C_lmlpmp
    
    def plot_cov_matrix(self, normalize = False, C_l_type = 3):
        """Plot the covariance matrix.

        Args:
            normalize (bool, optional): Plot normalized matrix. Defaults to False.
            C_l_type (int, optional): Correlation type (0: TT, 1: EE, 2: TE, 3: All). Defaults to 3.
        """
        l_max = self.l_max
        ell_range = np.array([2, l_max])
        ell_p_range = np.array([2,l_max])

        if C_l_type == 3 and self.do_polarization:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.2, 6), dpi=500, sharex='all', sharey='all')
            fig.tight_layout()
            data = self.normalized_C_lmlpmp if normalize else self.C_lmlpmp
            self.do_cov_sub_plot(ax1, normalize, 1, data[2], ell_range, ell_p_range)
            ax1.set_title('$TE$', weight='bold', fontsize=11)
            self.do_cov_sub_plot(ax2, normalize, 2, data[0], ell_range, ell_p_range)
            ax2.set_title('$TT$', weight='bold', fontsize=11)
            self.do_cov_sub_plot(ax3, normalize, 3, data[1], ell_range, ell_p_range)
            ax3.set_title('$EE$', weight='bold', fontsize=11)
            im = self.do_cov_sub_plot(ax4, normalize, 4, data[2].conj().T, ell_range, ell_p_range)
            ax4.set_title('$ET$', weight='bold', fontsize=11)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.8, 0.07, 0.04, 0.9])
            fig.colorbar(im, cax=cbar_ax)
        else:
            fig, ax = plt.subplots(dpi=500)
            fig.tight_layout()
            data = self.normalized_C_lmlpmp if normalize else self.C_lmlpmp
            data = data[C_l_type] if self.do_polarization else data
            title = {0: '$TT$', 1: '$EE$', 2: '$TE$'}[C_l_type]
            im = self.do_cov_sub_plot(ax, normalize, 0, data, ell_range, ell_p_range)
            ax.set_title(title, weight='bold', fontsize=11)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.8, 0.07, 0.04, 0.9])
            fig.colorbar(im, cax=cbar_ax)
        # Save the plot in the fig folder of the respective run folder
        plot_filename = f"cov_matrix_C_l_{self.C_l_type_array[C_l_type]}_normalize_{normalize}.pdf"
        fig.savefig(os.path.join(self.root, "figs", plot_filename), bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory



        
    
    def do_cov_sub_plot(self, ax, normalize, ax_index, C_order, ell_range, ell_p_range):
        """Create a subplot for the covariance matrix.

        Args:
            ax (matplotlib.axes.Axes): Subplot axes.
            normalize (bool): Use normalized data.
            ax_index (int): Subplot index.
            C_order (np.ndarray): Covariance matrix data.
            ell_range (np.ndarray): Multipole range [l_min, l_max].
            ell_p_range (np.ndarray): Multipole range [lp_min, lp_max].

        Returns:
            matplotlib.image.AxesImage: Image object for colorbar.
        """
        
        l_min = ell_range[0]
        l_max = ell_range[1]
        lp_min = ell_p_range[0]
        lp_max = ell_p_range[1]
        C_order = np.where(np.abs(C_order) < 1e-12, 1e-12, np.abs(C_order))

        ell_to_s_map = np.array([l * (l+1) - l - l_min**2  for l in range(l_min, l_max+1)])
        ellp_to_s_map = np.array([l * (l+1) - l - lp_min**2  for l in range(lp_min, lp_max+1)])

        axim = ax.imshow(C_order.T, cmap='inferno', norm=LogNorm(), origin='lower', interpolation='nearest')

        if l_max-l_min > 20:
            jump = np.array([5, 10, 15, 20])-2
            ax.set_xticks(ell_to_s_map[jump]- 0.5)
            ax.set_xticklabels(np.arange(l_min, l_max+1)[jump])
        else:
            ax.set_xticks(ell_to_s_map-0.5)
            ax.set_xticklabels(np.arange(l_min, l_max+1))

        if lp_max-lp_min > 20:
            jump = np.array([5, 10, 15, 20])-2
            ax.set_yticks(ellp_to_s_map[jump]- 0.5)
            ax.set_yticklabels(np.arange(lp_min, lp_max+1)[jump])
        else:
            ax.set_yticks(ellp_to_s_map-0.5)
            ax.set_yticklabels(np.arange(lp_min, lp_max+1))
        
        if lp_max > 50 or l_max > 50:
            ax.set_title(str(ax_index+5), weight='bold', fontsize='20')
        else:
            ax.set_title(str(ax_index+1), weight='bold', fontsize='20')
        
        if ax_index == 3 or ax_index == 2:
            ax.set_xlabel(r"$\ell $")
        if ax_index == 0 or ax_index == 2: 
            ax.set_xlabel(r"$\ell $") 
            ax.set_ylabel(r"$\ell'$")
        if normalize:
            axim.set_clim(1e-6, 1e0)
        else:
            axim.set_clim(1e-10, 1e2)
        
        return axim

     
    def get_kmax_as_function_of_ell(self, scalar_power):
        """Compute wavevector cutoffs k_max as a function of multipole ell.

        Args:
            scalar_power (callable): Primordial power spectrum function from CAMB.
        """

        l_max = self.l_max
    
        # Do the integration up to k=0.08. This should be fine for ell=<250 and accuracy<=0.99
        print('\nFinding k_max as a function of ell')
        self.k_max_list = np.zeros(self.l_max + 1)
        k_list = np.linspace(self.k_list[0], self.k_list[-1], 200000)

        for l in tqdm(range(2, self.l_max + 1)):
            integrand_TT = 4 * pi * scalar_power(k_list) * self.transfer_T_interpolate_k_l_list[l](k_list)**2 / k_list
            if self.do_polarization:
                integrand_EE = (4 * pi * (l + 2) * (l + 1) * l * (l - 1) * scalar_power(k_list) *
                                self.transfer_E_interpolate_k_l_list[l](k_list)**2 / k_list)
                integrand_TE = (4 * pi * sqrt((l + 2) * (l + 1) * l * (l - 1)) * scalar_power(k_list) *
                                self.transfer_T_interpolate_k_l_list[l](k_list) *
                                self.transfer_E_interpolate_k_l_list[l](k_list) / k_list)
                cumulative_c_l_TT_ratio = scipy.integrate.cumulative_trapezoid(y=integrand_TT, x=k_list) / self.powers[l, 0]
                cumulative_c_l_EE_ratio = scipy.integrate.cumulative_trapezoid(y=integrand_EE, x=k_list) / self.powers[l, 1]
                cumulative_c_l_TE_ratio = scipy.integrate.cumulative_trapezoid(y=integrand_TE, x=k_list) / self.powers[l, 3]
                self.k_max_list[l] = max(
                    k_list[(np.abs(cumulative_c_l_TT_ratio - self.c_l_accuracy)).argmin()],
                    k_list[(np.abs(cumulative_c_l_EE_ratio - self.c_l_accuracy)).argmin()],
                    k_list[(np.abs(cumulative_c_l_TE_ratio - self.c_l_accuracy)).argmin()])
            else:
                cumulative_c_l_TT_ratio = scipy.integrate.cumulative_trapezoid(y=integrand_TT, x=k_list) / self.powers[l, 0]
                self.k_max_list[l] = k_list[(np.abs(cumulative_c_l_TT_ratio - self.c_l_accuracy)).argmin()]

        if self.make_run_folder:
            np.save(os.path.join(self.root, 'k_max_list.npy'), self.k_max_list)
        print(f'Done. k_max for ell_max = {self.k_max_list[self.l_max]}')
    
    
    def get_transfer_functions_multi(self, transfer_interpolate_k_l_list):
        """Compute transfer functions for unique wavevector magnitudes using multiprocessing.

        Args:
            transfer_interpolate_k_l_list (dict): Interpolated transfer functions.

        Returns:
            np.ndarray: Transfer functions for unique k.
        """
        
        num_k_amp_unique = self.k_amp_unique.size
        transfer_delta_kl = np.zeros((num_k_amp_unique, self.l_max+1))

        ncpus = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = '1'
        pool = multiprocessing.Pool(processes=ncpus)
        print('\nGetting transfer functions')
        
        args = zip(np.arange(num_k_amp_unique), repeat(self.l_max), repeat(self.k_amp_unique), repeat(transfer_interpolate_k_l_list))
        
        with multiprocessing.Pool(processes=ncpus) as pool:
            transfer_delta_kl = np.array(pool.starmap(transfer_parallel, tqdm(args, total=num_k_amp_unique)))
            print('Size of transfer function: {} MB.'.format(round(getsizeof(transfer_delta_kl) / 1024 / 1024, 2)), '\n')
        
        pool.close() 

        return transfer_delta_kl 
    
    
    
    def get_sph_harm(self):
        """Compute spherical harmonics without phase(phi=0) using multiprocessing.

        Stores results in self.sph_harm_no_phase.
        """
        num_l_m = int((self.l_max + 1)*(self.l_max + 2)/2)

        # We only find Y_lm for unique theta elements. We don't want to recalculate Y_lm unnecessarily
        unique_theta_length = self.theta_unique.size
        ncpus = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = '1'
        pool = multiprocessing.Pool(processes=ncpus)
        print('\nGetting spherical harmonics')

        args = zip(np.arange(unique_theta_length), repeat(self.l_max), repeat(self.theta_unique), repeat(self.lm_index), repeat(num_l_m))
        with multiprocessing.Pool(processes=ncpus) as pool:
            self.sph_harm_no_phase = np.array(pool.starmap(get_sph_harm_parallel, tqdm(args, total=unique_theta_length)), dtype=np.float32)
            print('The spherical harmonics array is', round(getsizeof(self.sph_harm_no_phase) / 1024 / 1024,2), 'MB \n')
            pool.close()



    def get_c_lmlpmp_multiprocessing(self, ell_range, ell_p_range):
        """Compute covariance matrix using multiprocessing.

        Args:
            ell_range (np.ndarray): Multipole range [l_min, l_max].
            ell_p_range (np.ndarray): Multipole range [lp_min, lp_max].

        Returns:
            np.ndarray: Covariance matrix C_lmlpmp.
        """
        l_max = self.l_max
        l_min = self.l_min
        num_l_m = ell_range[1] * (ell_range[1] + 1) + ell_range[1] + 1 - ell_range[0] * ell_range[0] # = Sum[2 l+1, {l , l_min, l_max}]
        if self.do_polarization:
            c_lmlpmp = np.zeros((3, num_l_m, num_l_m), dtype=np.complex128) 
        else:
            c_lmlpmp = np.zeros((num_l_m, num_l_m), dtype=np.complex128)
        
        ncpus = multiprocessing.cpu_count()
        semi_jumps = int(np.ceil(np.sum(2 * np.arange(l_min, l_max + 1) + 1)**2/2 / ncpus))
        index_thread_split = np.array([l_min], dtype= np.int16)

        i = 0
        for ell in range(index_thread_split[i] + 1, l_max + 1):
            if np.sum(2 * np.arange(index_thread_split[i], ell) + 1)* np.sum(2 * np.arange(index_thread_split[i], l_max+1) + 1)>= semi_jumps:
                index_thread_split = np.append(index_thread_split, ell)
                i += 1
                continue
        size = index_thread_split.size
        index_thread_split = np.append(index_thread_split, l_max + 1)


        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for i in range(size):
            # Spawn a process for each cpu that goes through parts of the summation each
            min_ell = index_thread_split[i] # minimum index in the k_amp list not unique 
            max_ell = index_thread_split[i+1] - 1
            
            args = (
                i,
                return_dict,
                min_ell,
                max_ell,
                self.V,
                self.k_amp, 
                self.phi, 
                self.theta_unique_index,
                self.k_amp_unique_index,
                self.k_max_list,
                l_max,
                self.lm_index,
                self.sph_harm_no_phase,
                self.integrand,
                ell_range,
                ell_p_range
            )

            p = multiprocessing.Process(target=self.get_c_lmlpmp_per_process_multi, args=args)
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join() #Waits for each process to complete before moving on.
            proc.close()
            

        # The final c_lmlpmp that is the sum of contribution from each process  
        c_lmlpmp = sum(return_dict.values())

        # We only compute the upper triangle of the covariance matrix, so we need to
        # conjugate the lower triangle to get the full covariance matrix.
        if self.do_polarization:
            for ell in range(2, l_max+1):
                for ell_p in range(ell + 1, l_max+1):
                    for m in range(-ell, ell+1):
                        for m_p in range(-ell_p, ell_p + 1):
                            lm_p_index = ell_p * (ell_p+1) + m_p - l_min * l_min
                            lm_index = ell * (ell+1) + m - l_min * l_min
                            c_lmlpmp[:2, lm_p_index, lm_index] = np.conjugate(c_lmlpmp[:2, lm_index, lm_p_index])
        else:
            for ell in range(2, l_max+1):
                for ell_p in range(ell + 1, l_max+1):
                    for m in range(-ell, ell+1):
                        for m_p in range(-ell_p, ell_p + 1):
                            lm_p_index = ell_p * (ell_p+1) + m_p - l_min * l_min
                            lm_index = ell * (ell+1) + m - l_min * l_min
                            c_lmlpmp[lm_p_index, lm_index] = np.conjugate(c_lmlpmp[lm_index, lm_p_index])

        return c_lmlpmp