import numpy as np
from twentyone.units import *
from twentyone.physics_standalone import alpha_recomb, peebles_C
from scipy.optimize import root
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, RegularGridInterpolator
from classy import Class

from twentyone.lya import LymanAlpha


class TwentyOne(LymanAlpha):
    def __init__(self, cosmo=None, z_reio=7.82, z_min=10.0, z_max=150.0, f_star_L=0.5, f_star_X=0.004, T_vir_cut=1e4 * Kelv, use_hirata_fits=True, sed_X="PL", sed_X_kwargs={}, hmf_kwargs={"mdef": "vir", "model": "despali16"}):
        """ Class to calculate Lyman-alpha heating

            :param cosmo: The cosmology, specified as an astropy.cosmology.FlatLambdaCDM instance
            :param z_reio: Redshift at reionization, passed to CLASS
            :param z_min: Minimum redshift (for interpolation tables)
            :param z_max: Maximum redshift (for interpolation tables)
            :param f_star_L: Efficiency of star formation, passed to Ly-A/X-ray classes
            :param f_star_X: Efficiency of star formation, X-ray, passed to Ly-A/X-ray classes
            :param T_vir_cut: Minimum virial temperature of star-forming halos, passed to Ly-A/X-ray classes
            :param use_hirata_fits: Whether to use fitting functions from Hirata for S_alpha and T_c
            :param sed_X: X-ray luminosity, "PL" or "Hirata"
            :param sed_X_kwargs: Parameters for X-ray luminosity
        """

        LymanAlpha.__init__(self, cosmo=cosmo, z_min=z_min, z_max=z_max, f_star_L=f_star_L, f_star_X=f_star_X, T_vir_cut=T_vir_cut, sed_X=sed_X, sed_X_kwargs=sed_X_kwargs, hmf_kwargs=hmf_kwargs)

        self.z_reio = z_reio  # Redshift at reionization
        self.use_hirata_fits = use_hirata_fits  # Whether to use fitting functions from Hirata for S_alpha and T_c
        self.load_constants()  # Set class-specific constants
        self.load_interpolations()  # Load interpolation tables
        self.initialize_class_inst()  # Initialize CLASS instance

    def load_constants(self):
        """ Load class-specific constants
        """
        self.nu_Lya = 2466 * 1e12 / Sec  # Ly-A absorption frequency, originally in THz
        self.gamma = 50 * 1e6 / Sec  # HWHM  of the 21-cm resonance, after Eq. (11) of astro-ph/0507102, originally in MHz
        self.T_21 = 68.2e-3  # 21-cm temperature, in K
        self.nu_21 = 1420405751.7667 / Sec  # Frequency of 21-cm transition
        self.A_21 = 2.86e-15 / Sec  # Spontaneous emission (Einstein-A) coefficient of 21-cm transition, after Eq. (3) of 0802.2102
        self.lambda_Lya = 121.567e-9 * Meter  # Ly-A absorption wavelength
        self.sigma_T = 0.66524587158 * barn  # Thomson scattering cross-section
        self.EHth = 13.6 * eV  # Hydrogen ground state energy

    def initialize_class_inst(self):
        """ Get electron ionization fraction from CLASS
        """
        class_parameters = {"H0": self.cosmo.H0.value, "Omega_b": self.cosmo.Ob0, "N_ur": self.cosmo.Neff, "Omega_cdm": self.cosmo.Odm0, "YHe": self.Y_p, "z_reio": self.z_reio}

        self.CLASS_inst = Class()
        self.CLASS_inst.set(class_parameters)
        self.CLASS_inst.compute()

    def x_e(self, z):
        """ Electron ionization fraction, from CLASS instance
        """
        return self.CLASS_inst.ionization_fraction(z)

    def T_b(self, z):
        """ Baryon temperature at given redshift, from CLASS instance
        """
        return self.CLASS_inst.baryon_temperature(z)

    def load_interpolations(self):
        """ Load interpolation tables
        """

        ## Load table from https://github.com/ntveem/lyaheating
        heffs = np.load("../data/heffs.npy")
        # heffs[:, :, :, 1, 0][np.where(heffs[:, :, :, 1, 0] < 0)] = 1e-15

        # Argument arrays for T_k, T_s,...
        l10_t_ary = np.linspace(np.log10(0.1), np.log10(100.0), num=175)
        # ... and tau_GP (Gunn-Peterson optical depth)
        l10_tau_gp_ary = np.linspace(4.0, 7.0)

        # Net energy loss efficiency, defined in Eq. (37) of 1804.02406
        self.E_c_interp = RegularGridInterpolator(points=[l10_t_ary, l10_t_ary, l10_tau_gp_ary], values=((heffs[:, :, :, 0, 0])), bounds_error=False, fill_value=None)
        self.E_i_interp = RegularGridInterpolator(points=[l10_t_ary, l10_t_ary, l10_tau_gp_ary], values=((heffs[:, :, :, 1, 0])), bounds_error=False, fill_value=None)

        # Energy loss to spins, defined in Eq. (32) of astro-ph/0507102
        self.S_alpha_c_interp = RegularGridInterpolator(points=[l10_t_ary, l10_t_ary, l10_tau_gp_ary], values=(np.log10(heffs[:, :, :, 0, 2])), bounds_error=False, fill_value=None)
        self.S_alpha_i_interp = RegularGridInterpolator(points=[l10_t_ary, l10_t_ary, l10_tau_gp_ary], values=(np.log10(heffs[:, :, :, 1, 2])), bounds_error=False, fill_value=None)

        # Effective colour temperature, defined in Eq. (32) of astro-ph/0507102
        self.T_c_c_interp = RegularGridInterpolator(points=[l10_t_ary, l10_t_ary, l10_tau_gp_ary], values=(np.log10(heffs[:, :, :, 0, 3])), bounds_error=False, fill_value=None)
        self.T_c_i_interp = RegularGridInterpolator(points=[l10_t_ary, l10_t_ary, l10_tau_gp_ary], values=(np.log10(heffs[:, :, :, 1, 3])), bounds_error=False, fill_value=None)

        ## Rate coefficients

        # From astro-ph/0608067, Table 1
        kappa_10_eH_ary = np.loadtxt("../data/kappa_10_eH_tab.dat")
        # From Zygelman (2005), http://adsabs.harvard.edu/abs/2005ApJ...622.1356Z, Table 2
        kappa_10_HH_ary = np.loadtxt("../data/kappa_10_HH_tab.dat")

        self.l10_kappa_10_eH_interp = interp1d(np.log10(kappa_10_eH_ary)[:, 0], np.log10(kappa_10_eH_ary * Centimeter ** 3 / Sec)[:, 1], bounds_error=False, fill_value="extrapolate")
        self.l10_kappa_10_HH_interp = interp1d(np.log10(kappa_10_HH_ary)[:, 0], np.log10(kappa_10_HH_ary * Centimeter ** 3 / Sec)[:, 1], bounds_error=False, fill_value="extrapolate")

    def S_alpha_Hirata(self, Tk, Ts, tauGP):
        """ Hirata fitting functional form for energy loss to spins
        """
        xi = (1e-7 * tauGP) ** (1.0 / 3.0) * Tk ** (-2.0 / 3.0)
        a = 1.0 - 0.0631789 / Tk + 0.115995 / Tk ** 2 - 0.401403 / Ts / Tk + 0.336463 / Ts / Tk ** 2
        b = 1.0 + 2.98394 * xi + 1.53583 * xi ** 2 + 3.85289 * xi ** 3

        return a / b

    def T_c_Hirata(self, Tk, Ts):
        """ Hirata fitting functional form for effective colour temperature
        """
        T_c_inv = Tk ** (-1.0) + 0.405535 * Tk ** (-1.0) * (Ts ** (-1.0) - Tk ** (-1.0))
        return 1 / T_c_inv

    def T_c_c(self, T_k, T_s, x_e, z):
        """ Effective color temperature from interpolation, continuum
        """
        if T_k <= 100.0:
            return 10 ** self.T_c_c_interp([np.log10(T_k), np.log10(T_s), np.log10(self.tau_GP(x_e, z))])
        else:
            return T_k / 100 * 10 ** self.T_c_c_interp([np.log10(100.0), np.log10(T_s), np.log10(self.tau_GP(x_e, z))])  # self.T_b(z)

    def T_c_i(self, T_k, T_s, x_e, z):
        """ Effective color temperature from interpolation, injected
        """
        if T_k <= 100.0:
            return 10 ** self.T_c_i_interp([np.log10(T_k), np.log10(T_s), np.log10(self.tau_GP(x_e, z))])
        else:
            return T_k / 100 * 10 ** self.T_c_i_interp([np.log10(100.0), np.log10(T_s), np.log10(self.tau_GP(x_e, z))])  # self.T_b(z)

    def x_HI(self, x_e):
        """ IGM neutral fraction, from electron ionization fraction
        """
        # return np.max(1 - x_e, 0)
        return np.where(x_e <= 1 + self.Y_p / 4 * (1 - self.Y_p), 1 - x_e * (1 - self.Y_p / (4 - 3 * self.Y_p)), 0)

    def T_CMB(self, z):
        """ CMB temperature, in K
        """
        return self.T_CMB_0 * (1 + z)

    def tau_GP(self, x_e, z):
        """ Gunn-Peterson optical depth, Eq. (35) of astro-ph/0507102 
        """
        H = self.cosmo.H(z).value * Kmps / Mpc
        return (3 * self.n_H(z) * self.x_HI(x_e) * self.gamma) / (2 * H * self.nu_Lya ** 3)

    def tau_21(self, T_s, x_e, z):
        """ 21-cm optical depth, Eq. (2) of 1804.02406
        """
        H = self.cosmo.H(z).value * Kmps / Mpc
        return 3 / (32 * np.pi) * (self.n_H(z) * self.x_HI(x_e) * self.A_21) / (self.nu_21 ** 3 * H) * self.T_21 / T_s

    def x_CMB(self, T_s, x_e, z):
        """ Spin-flip rate due to CMB heating, Eq. (14) of 1804.02406
        """
        return 1 / self.tau_21(T_s, x_e, z) * (1 - np.exp(-self.tau_21(T_s, x_e, z)))

    def x_c(self, T_k, T_gamma, x_e, z):
        """ Collisional coupling spin-flip rate coefficient, Eq (3) of 0802.2102
        """
        return 4 * self.T_21 / (3 * self.A_21 * T_gamma) * self.n_H(z) * (10 ** self.l10_kappa_10_eH_interp(np.log10(T_k)) * x_e + 10 ** self.l10_kappa_10_HH_interp(np.log10(T_k)))

    def x_alpha_c(self, T_k, T_s, T_gamma, x_e, J_c_o_J_0, z):
        """ Wouthuysen-Field spin-flip rate coefficient, continuum, Eq. (11) of astro-ph/0507102 
        """
        if self.use_hirata_fits:
            S_alpha_c = self.S_alpha_Hirata(T_k, T_s, self.tau_GP(x_e, z))
        else:
            S_alpha_c = 10 ** self.S_alpha_c_interp(np.log10([T_k, T_s, self.tau_GP(x_e, z)]))
        return 8 * np.pi * self.lambda_Lya ** 2 * self.gamma * self.T_21 / (9 * self.A_21 * T_gamma) * S_alpha_c * J_c_o_J_0 * self.J_0(z)

    def x_alpha_i(self, T_k, T_s, T_gamma, x_e, J_i_o_J_0, z):
        """ Wouthuysen-Field spin-flip rate coefficient, injected, Eq. (11) of astro-ph/0507102 
        """
        if self.use_hirata_fits:
            S_alpha_i = self.S_alpha_Hirata(T_k, T_s, self.tau_GP(x_e, z))
        else:
            S_alpha_i = 10 ** self.S_alpha_i_interp(np.log10([T_k, T_s, self.tau_GP(x_e, z)]))
        return 8 * np.pi * self.lambda_Lya ** 2 * self.gamma * self.T_21 / (9 * self.A_21 * T_gamma) * S_alpha_i * J_i_o_J_0 * self.J_0(z)

    def T_s_inv(self, T_s, T_k, T_gamma, x_e, J, z):
        """ Spin temperature (inverse), e.g. Eq (13) of 1804.02406
        """

        x_CMB = self.x_CMB(T_s, x_e, z)
        x_alpha_c = self.x_alpha_c(T_k, T_s, T_gamma, x_e, J[0], z)
        x_alpha_i = self.x_alpha_i(T_k, T_s, T_gamma, x_e, J[1], z)
        x_c = self.x_c(T_k, T_gamma, x_e, z)

        if self.use_hirata_fits:
            T_c_c = T_c_i = self.T_c_Hirata(T_k, T_s)
        else:
            T_c_c = self.T_c_c(T_k, T_s, x_e, z)
            T_c_i = self.T_c_i(T_k, T_s, x_e, z)

        return (x_CMB * T_gamma ** -1 + x_alpha_c * T_c_c ** -1 + x_alpha_i * T_c_i ** -1 + x_c * T_k ** -1) / (x_CMB + x_alpha_c + x_alpha_i + x_c)

    def T_s_solve(self, T_k, T_gamma, x_e, J, z):
        """ Solve for spin temperature
        """
        T_s = (root(lambda T_s: self.T_s_inv(T_s[0], T_k, T_gamma, x_e, J, z) - T_s ** -1, np.min([T_k, T_gamma])).x)[0]
        return T_s

    def E_CMB(self, T_s, T_gamma, T_k, x_e, z):
        """ Heating efficiency due to CMB, from Eq. (17) of 1804.02406
        """
        H = self.cosmo.H(z).value * Kmps / Mpc
        return self.x_HI(x_e) * self.A_21 / (2 * H) * self.x_CMB(T_s, x_e, z) * (T_gamma / T_s - 1) * self.T_21 / T_k

    def E_Compton(self, T_k, x_e, z):
        """ Compton heating efficiency, from Eq. (22) of 1312.4948 (TODO: but is it)
        """
        H = self.cosmo.H(z).value * Kmps / Mpc
        a_r = np.pi ** 2 / 15.0
        return 8 * self.sigma_T * a_r * (self.T_CMB(z) * Kelv) ** 4 * x_e / (3 * m_e * H) * (self.T_CMB(z) / T_k - 1)

    def dT_k_dz(self, T_s, T_k, T_gamma, x_e, J, z):
        """ Kinetic temperature evolution, from Eq (18) of 1804.02406
        """

        E_c = self.E_c_interp(np.log10([T_k, T_s, self.tau_GP(x_e, z)]))
        E_i = self.E_i_interp(np.log10([T_k, T_s, self.tau_GP(x_e, z)]))

        dT_k_dz = 1 / (1 + z) * (2 * T_k - 1 / (1 + self.f_He + x_e) * (E_c * J[0] + E_i * J[1] + self.E_CMB(T_s, T_gamma, T_k, x_e, z) + self.E_Compton(T_k, x_e, z)) * T_k)

        return dT_k_dz - 1 / (1 + z) * self.heat_coeff(z)

    def alpha_A(self, T):
        """ Case-A recombination coefficient, from Pequignot et al (1991), Eq. (1) and Table 1
        """

        zi = 1.0
        a = 5.596
        b = -0.6038
        c = 0.3436
        d = 0.4479
        t = 1e-4 * T / zi ** 2

        return zi * (a * t ** b) / (1 + c * t ** d) * 1e-13 * Centimeter ** 3 / Sec

    def alpha_B(self, T):
        """ Case-B recombination coefficient, from `DarkHistory`
        """
        return alpha_recomb((k_B * T * Kelv) / eV, species="HI") * Centimeter ** 3 / Sec

    def rec_rate(self, z, x_e, T_k, case="B"):
        """ Recombination rate (Eq. 29 of 1312.4948)
        """
        Gamma_rec = -peebles_C(1 - self.x_HI(x_e), 1 + z) * self.alpha_B(T_k) * x_e ** 2 * self.n_H(z)
        return self.dz_dt(z) ** -1 * Gamma_rec

    def reio_rate(self, z, x_e):
        """ Reionization rate (TODO: where from)
        """
        f_ion = 10 ** self.l10_f_ion_int(np.log10(x_e))
        f_heat = 10 ** self.l10_f_heat_int(np.log10(x_e))
        Gamma_ion = f_ion / (self.EHth * f_heat) * self.epsilon_X(z)
        return self.dz_dt(z) ** -1 * Gamma_ion

    def delta_T_b(self, T_s, T_gamma, x_e, z):
        """ 21-cm brightness temperature contrast from CMB, from Eq. (15) of 1804.02406
        """
        return self.x_CMB(T_s, x_e, z) * self.tau_21(T_s, x_e, z) / (1 + z) * (T_s - T_gamma)

    def mod_f_star(self, f_star_X=None, f_star_L=None, T_vir_cut=None, hmf_kwargs=None):

        if f_star_X is None:
            f_star_X = self.f_star_X

        if f_star_L is None:
            f_star_L = self.f_star_L

        if hmf_kwargs is not None:
            self.hmf_kwargs = hmf_kwargs

        if T_vir_cut is not None:
            self.T_vir_cut = T_vir_cut

        if (T_vir_cut is not None) or (hmf_kwargs is not None):
            self.make_D_int_halo_z_interp()

        self.D_int_halo_z_interp = interp1d(self.z_interp_ary, f_star_X / self.f_star_X * np.array(self.D_int_halo_z_ary), bounds_error=False, kind="linear", fill_value=0.0)  # Dodgy interpolation

        self.J_c_per_J_0_interp = interp1d(self.z_J_interp_ary, f_star_L / self.f_star_L * self.J_c_per_J_0_ary)
        self.J_i_per_J_0_interp = interp1d(self.z_J_interp_ary, f_star_L / self.f_star_L * self.J_i_per_J_0_ary)

        self.f_star_L = f_star_L
        self.f_star_X = f_star_X


class TwentyOneSolver:
    def __init__(self, to, z_ary, T_gamma=None, perfect_WF=False):
        """ Class for solving 21-cm evolution equations 

            :param to: Instance of class TwentyOne containing all the knibs and knobs
            :param z_ary: Redshift array at which to calculate various quantities, should be strictly descending
            :param T_gamma: Function returning photon temperature for a given redshift argument, defaults to CMB temperature
        """

        if T_gamma is None:
            self.T_gamma = to.T_CMB
        else:
            self.T_gamma = T_gamma
        self.to = to

        self.perfect_WF = perfect_WF

        # Check for strictly descending redshift array
        assert np.all(~(np.diff(z_ary) > 0)), "Redshift array z_ary must be strictly descending!"

        self.z_ary = z_ary
        self.z_min, self.z_max = z_ary[-1], z_ary[0]

    def vectorfield_ivp(self, z, y):
        """ System of ODEs for T_k(z) and x_e(z)
            :param y: [T_k(z), x_e(z)]
            :return: [T_k'(z), x_e'(z)]
        """

        T_k, x_e = y

        T_gamma = self.T_gamma(z)

        J_c = self.to.J_c_per_J_0_interp(z) + self.to.J_X_per_J_0(z, x_e)
        J_i = self.to.J_i_per_J_0_interp(z)
        T_s = self.to.T_s_solve(T_k, T_gamma, x_e, [J_c, J_i], z)

        # Functions for d(T_k)/dz and d(x_e)/dz
        f = [self.to.dT_k_dz(T_s, T_k, T_gamma, x_e, [J_c, J_i], z), self.to.rec_rate(z, x_e, T_k) + self.to.reio_rate(z, x_e)]

        return f

    def T_s(self, z, T_k, x_e):
        """ Solve for spin temperature
        """
        T_gamma = self.T_gamma(z)
        J_c = self.to.J_c_per_J_0_interp(z) + self.to.J_X_per_J_0(z, x_e)
        J_i = self.to.J_i_per_J_0_interp(z)
        if self.perfect_WF:
            T_s = T_k
        else:
            T_s = self.to.T_s_solve(T_k, T_gamma, x_e, [J_c, J_i], z)
        return T_s

    def solve(self):
        """ ODE solver for temperature evolution
        """
        # Solve evolution ODE
        # Initial conditions set for T_k to match gas temperature and x_e match class output at highest redshift
        with np.errstate(invalid="ignore"):
            solution_ivp = solve_ivp(
                fun=self.vectorfield_ivp,
                t_span=[self.z_max, self.z_min],
                y0=[self.to.T_b(self.z_max), self.to.x_e(self.z_max)],
                t_eval=self.z_ary,
                # method='LSODA',
                rtol=1e-6,
                atol=1e-12,
            )

        # Assign and/or calculate quantities
        self.T_k_ary = solution_ivp.y[0]
        self.x_e_ary = solution_ivp.y[1]
        self.T_s_ary = [self.T_s(z, T_k, x_e) for z, T_k, x_e in zip(solution_ivp.t, solution_ivp.y[0], solution_ivp.y[1])]
        self.T_gamma_ary = [self.T_gamma(z) for z in solution_ivp.t]
        self.delta_T_b_ary = [self.to.delta_T_b(T_s, T_gamma, x_e, z) for T_s, T_gamma, x_e, z in zip(self.T_s_ary, self.T_gamma_ary, self.x_e_ary, self.z_ary)]

        return solution_ivp
