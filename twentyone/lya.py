from pathlib import Path

import numpy as np
from scipy.optimize import root, minimize, brentq
from scipy.integrate import quad
from scipy.interpolate import interp1d
from tqdm.notebook import tqdm

from twentyone.units import *
from twentyone.xray import XRay


class LymanAlpha(XRay):
    def __init__(self, z_min, z_max, cosmo=None, f_star_L=0.5, f_star_X=0.004, T_vir_cut=1e5 * Kelv, sed_X="PL", sed_X_kwargs={}, hmf_kwargs={"mdef": "vir", "model": "despali16"}):
        """ Class to calculate Lyman-alpha heating

            :param z_min: Minimum redshift (for interpolation tables)
            :param z_max: Maximum redshift (for interpolation tables)
            :param cosmo: The cosmology, specified as an astropy.cosmology.FlatLambdaCDM instance
            :param f_star_L: Efficiency of star formation, Lyman-A
            :param f_star_X: Efficiency of star formation, X-ray, passed to X-ray class
            :param T_vir_cut: Minimum virial temperature of star-forming halos, passed to X-ray class
            :param sed_X: X-ray luminosity, "PL" or "Hirata"
            :param sed_X_kwargs: Parameters for X-ray luminosity
        """

        XRay.__init__(self, cosmo=cosmo, z_min=z_min, z_max=2 * z_max, f_star_X=f_star_X, T_vir_cut=T_vir_cut, sed=sed_X, sed_kwargs=sed_X_kwargs, hmf_kwargs=hmf_kwargs)

        self.xi_3 = 1.20205  # Apéry's constant, from Wikipedia
        self.Y_p = 0.247  # Helium-to-hydrogen mass fraction, from 1912.01132
        self.f_He = (1 / 4.0) * self.Y_p / (1 - self.Y_p)  # Helium-to-hydrogen number ratio, after Eq. (18) of 1804.02406
        self.nu_Lya = 2466 * 1e12 / Sec  # THz
        self.rho_b0 = self.cosmo.critical_density0.value * Gram / Centimeter ** 3 * self.cosmo.Ob0  # Baryon density today
        self.n_b0 = self.rho_b0 / m_p  # Baryon number density today
        self.h_Pl = 4.135667662e-15 * eV * Sec  # Planck constant
        self.eta = 6.129e-10  # Baryon-to-photon ratio, from 1912.01132
        self.T_CMB_0 = 2.7255  # CMB temperature today, in K

        self.z_min = z_min
        self.z_max = z_max

        self.f_star_X = f_star_X
        self.f_star_L = f_star_L

        self.data_path = str(Path(__file__).parent / "../data/")

        # Load/create various interpolations
        self.load_L_interpolations()
        self.make_J_interpolation()

    def load_L_interpolations(self):
        """ Create various interpolation tables
        """

        # Probabilities of producing a Ly-a photon after exciting HI to the np configuration
        # From Tab. 1 of astro-ph/0507102
        Pnp_list = np.loadtxt(self.data_path + "/PnpList.dat")
        self.Pnp_ary = np.zeros(int(np.max(Pnp_list[:, 0]) + 1))
        self.Pnp_ary[np.array(Pnp_list[:, 0], dtype=np.int32)] = Pnp_list[:, 1]

        # Fraction of initial electron energy that goes into heating
        # and deposited in HI Ly-a photons, from Fig. 4 of 0910.4410
        # assuming E_e = 3 keV
        f_heat = np.loadtxt(self.data_path + "/f_heat.txt", skiprows=1)
        f_Lya = np.loadtxt(self.data_path + "/f_Lya.txt", skiprows=1)
        f_ion = np.loadtxt(self.data_path + "/f_ion.txt", skiprows=1)

        # Create interpolation
        self.l10_f_heat_int = interp1d((f_heat[:, 0]), (f_heat[:, 1]), bounds_error=False, fill_value="extrapolate")
        self.l10_f_Lya_int = interp1d((f_Lya[:, 0]), (f_Lya[:, 1]), bounds_error=False, fill_value="extrapolate")
        self.l10_f_ion_int = interp1d((f_ion[:, 0]), (f_ion[:, 1]), bounds_error=False, fill_value="extrapolate")

    def J_0(self, z):
        """ Flux scale corresponding to one photon per H atom, from Eq. (6) of 1804.02406
        """
        return 1 / (4 * np.pi) * self.n_H(z) / self.nu_Lya

    def n_H(self, z):
        """ Number density of hydrogen nuclei at redshift `z` in cm^3
        """
        return (1 - self.Y_p) * self.eta * 2 * self.xi_3 / np.pi ** 2 * (self.T_CMB_0 * Kelv * (1 + z)) ** 3

    def n_b(self, z):
        """ Baryon atom number density at redshift `z` in cm^3. 
        """
        return self.eta * 2 * self.xi_3 / np.pi ** 2 * (self.T_CMB_0 * Kelv * (1 + z)) ** 3

    def epsilon_b(self, nu, T, E_per_baryon):
        """ Emissivity spectrum assuming a blackbody (as in astro-ph/0507102)
        """
        norm_spec = E_per_baryon * (np.pi ** 4 * T * k_B / (30 * self.xi_3)) ** -1
        nu_0 = k_B * T / self.h_Pl
        dndnu = (2 * self.xi_3 * nu_0 ** 3) ** -1 * nu ** 2 / (np.exp(nu / nu_0) - 1)
        return norm_spec * dndnu

    def epsilon(self, nu, z, T, E_per_baryon):
        """ Source emissivity. Eq. (55) of  astro-ph/0507102
        """
        return self.epsilon_b(nu, T, E_per_baryon) * self.SFRD(z) / m_p

    def z_maximum(self, n, z):
        """ Maximum redshift from which photon might have been received. Eq. (54) of  astro-ph/0507102
        """
        return (1 - (n + 1) ** -2) / (1 - n ** -2) * (1 + z) - 1

    def nu_p_2(self, z, z_p):
        """ Emission frequency, n = 2 (Ly-A, inferred from Eq. 54 of astro-ph/0507102)
        """
        return self.nu_Lya * (1 + z_p) / (1 + z)

    def nu_p_n(self, n, z, z_p):
        """ Emission frequency, n > 2 (inferred from Eq. 54 of astro-ph/0507102)
        """
        return self.nu_p_2(z, z_p) * (1 - n ** -2) / (1 - 2 ** -2)

    def J_c_integrand(self, z, z_p, T, E_per_baryon):
        """  Continuum Ly-A flux (integrand), Eq. 53 of  astro-ph/0507102
        """
        H = self.cosmo.H(z_p).value * Kmps / Mpc
        return 1 / H * self.epsilon(self.nu_p_2(z, z_p), z_p, T, E_per_baryon)

    def J_c(self, z, T, E_per_baryon):
        """ Continuum Ly-A flux, Eq. (53) of astro-ph/0507102
        """
        return (1 + z) ** 2 / (4 * np.pi) * quad(lambda z_p: self.J_c_integrand(z, z_p, T, E_per_baryon), z, self.z_maximum(2, z))[0]

    def J_c_per_J_0(self, z, T, E_per_baryon):
        """ Continuum Ly-A flux in units of J_0
        """
        return self.J_c(z, T, E_per_baryon) / self.J_0(z)

    def J_i_integrand(self, n, z, z_p, T, E_per_baryon):
        """  Injected Ly-A flux (integrand), Eq. 53 of  astro-ph/0507102
        """
        H = self.cosmo.H(z_p).value * Kmps / Mpc
        return 1 / H * self.epsilon(self.nu_p_n(n, z, z_p), z_p, T, E_per_baryon)

    def J_i_int(self, n, z, T, E_per_baryon):
        """  Injected Ly-A flux (integral part), Eq. 53 of  astro-ph/0507102
        """
        return quad(lambda z_p: self.J_i_integrand(n, z, z_p, T, E_per_baryon), z, self.z_maximum(n, z))[0]

    def J_i(self, z, T, E_per_baryon):
        """  Injected Ly-A flux, Eq. 53 of  astro-ph/0507102
        """
        return (1 + z) ** 2 / (4 * np.pi) * np.sum([self.Pnp_ary[i] * self.J_i_int(i, z, T, E_per_baryon) for i in range(4, 31)])

    def J_i_per_J_0(self, z, T, E_per_baryon):
        """ Injected Ly-A flux in units of J_0
        """
        return self.J_i(z, T, E_per_baryon) / self.J_0(z)

    def J_Ly(self, z, T, E_per_baryon):
        """ Continuum and injected Ly-A flux
        """
        return self.J_c_per_J_0(z, T, E_per_baryon), self.J_i_per_J_0(z, T, E_per_baryon)

    def epsilon_X(self, z):
        """ X-ray heating rate per baryon. Compare Eq. 11 of 1003.3878 and Eq. 56 of astro-ph/0507102.
        """
        return self.Gamma_X(z) / self.n_b0

    def J_X(self, z, x_e):
        """ Lyman-A flux from X-rays. Compare Eqs. 17 and 20 from 1003.3878 to get in terms of epsilon_X.
        """
        H = self.cosmo.H(z).value * Kmps / Mpc
        return self.n_b(z) / (4 * np.pi * H * self.nu_Lya) * self.epsilon_X(z) * 10 ** self.l10_f_Lya_int(np.log10(x_e)) / (10 ** self.l10_f_heat_int(np.log10(x_e))) * 1 / (self.h_Pl * self.nu_Lya)

    def J_X_per_J_0(self, z, x_e):
        """ Ly-A flux from X-rays in units of J_0
        """
        return self.J_X(z, x_e) / self.J_0(z)

    def make_J_interpolation(self):
        """ Create interpolations for Lyman-A heating
        """
        self.z_J_interp_ary = np.linspace(self.z_min, self.z_max, 2000)

        # E_per_baryon = 5.4 MeV for complete H burning to He-4 (page 11 of astro-ph/0507102)
        self.J_c_per_J_0_ary = np.array([self.J_c_per_J_0(z, 1e5 * Kelv, 5.4 * MeV) for z in tqdm(self.z_J_interp_ary)])
        self.J_i_per_J_0_ary = np.array([self.J_i_per_J_0(z, 1e5 * Kelv, 5.4 * MeV) for z in tqdm(self.z_J_interp_ary)])

        self.J_c_per_J_0_interp = interp1d(self.z_J_interp_ary, self.f_star_L / self.f_star_X * self.J_c_per_J_0_ary)
        self.J_i_per_J_0_interp = interp1d(self.z_J_interp_ary, self.f_star_L / self.f_star_X * self.J_i_per_J_0_ary)
