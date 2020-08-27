from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np
from scipy.optimize import root
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.misc import derivative
from colossus.lss import mass_function
from colossus.cosmology import cosmology
from tqdm.notebook import tqdm

from twentyone.units import *


class XRay:
    def __init__(self, z_min, z_max, T_vir_cut=1e4 * Kelv, f_star=0.004, mu=0.59, cosmo=None, sed="PL", M_min=None, sed_kwargs={}, hmf_kwargs={"mdef": "vir", "model": "despali16"}):
        """ Class for calculating X-ray heating

            :param z_min: Minimum redshift (for interpolation tables)
            :param z_max: Maximum redshift (for interpolation tables)
            :param T_vir_cut: Minimum virial temperature of star-forming halos
            :param f_star: Efficiency of star formation (fraction of collapsing gas that turns into stars)
            :param mu: Mean molecular weight. 
                mu = 0.59 for a fully ionized primordial gas (default)
                mu = 1.22 for neutral primordial gas
            :param cosmo: The cosmology, specified as an astropy.cosmology.FlatLambdaCDM instance. If not
                specified, use Planck18 cosmology.
            :param sed: How to normalize X-ray luminosity. "PL" or "Hirata".
        """

        # Set the cosmology
        if cosmo is None:
            self.cosmo = self.get_Planck18_cosmology()
        else:
            self.cosmo = cosmo

        cosmology.setCosmology("planck18")

        self.mu = mu
        self.T_vir_cut = T_vir_cut
        self.M_min_val = M_min
        self.f_star = f_star
        self.z_min = z_min
        self.z_max = z_max
        self.sed = sed

        if sed_kwargs == {}:
            self.sed_kwargs = {"alpha": -1.5, "L_X_per_SFR": 2.6e39 * erg * Sec ** -1 * (M_s * Year ** -1) ** -1, "E_min_calib": 0.5 * KeV, "E_max_calib": 8 * KeV, "E_min": 0.2 * KeV, "E_max": 95 * KeV, "f_X": 1}
        else:
            self.sed_kwargs = sed_kwargs

        self.hmf_kwargs = hmf_kwargs

        self.make_D_int_halo_z_interp()  # Create halo integral interpolation

    def sed_PL(self, E, alpha, E_min_calib, E_max_calib, L_X_per_SFR, f_X):
        """ Power-law X-ray SED
        """
        norm = f_X * L_X_per_SFR * (alpha + 1) / (E_max_calib ** (alpha + 1) - E_min_calib ** (alpha + 1))
        return norm * E ** alpha

    def integ_sed_PL(self, alpha, E_min_calib, E_max_calib, E_min, E_max, L_X_per_SFR, f_X):
        """ Integral of X-ray power-law SED between E_min and E_max
        """
        norm = f_X * L_X_per_SFR * (alpha + 1) / (E_max_calib ** (alpha + 1) - E_min_calib ** (alpha + 1))
        return norm * (E_max ** (alpha + 1) - E_min ** (alpha + 1)) / (alpha + 1)

    def get_Planck18_cosmology(self):
        """
        Stolen from https://github.com/abatten/fruitbat/blob/c074abff432c3b267d00fbb49781a0e0c6eeab75/fruitbat/cosmologies.py
        Planck 2018 paper VI Table 2 Final column (68% confidence interval)
        This is the Planck 2018 cosmology that will be added to Astropy when the
        paper is accepted.

        :return: astropy.cosmology.FlatLambdaCDM instance describing Planck18 cosmology
        """

        planck18_cosmology = {"Oc0": 0.2607, "Ob0": 0.04897, "Om0": 0.3111, "H0": 67.66, "n": 0.9665, "sigma8": 0.8102, "tau": 0.0561, "z_reion": 7.82, "t0": 13.787, "Tcmb0": 2.7255, "Neff": 3.046, "m_nu": [0.0, 0.0, 0.06], "z_recomb": 1089.80, "reference": "Planck 2018 results. VI. Cosmological Parameters, " "A&A, submitted, Table 2 (TT, TE, EE + lowE + lensing + BAO)"}

        Planck18 = FlatLambdaCDM(H0=planck18_cosmology["H0"], Om0=planck18_cosmology["Om0"], Tcmb0=planck18_cosmology["Tcmb0"], Neff=planck18_cosmology["Neff"], Ob0=planck18_cosmology["Ob0"], name="Planck18", m_nu=u.Quantity(planck18_cosmology["m_nu"], u.eV))

        return Planck18

    def delta_vir(self, z):
        """ The virial overdensity in units of the critical density, from Bryan & Norman (1998)
        """
        x = self.cosmo.Om(z) - 1
        return 18 * np.pi ** 2 + 82 * x - 39 * x ** 2

    def _calc_T_vir(self, M, rho_vir):
        """ Return virial temperature for given mass and virial density
        """
        r_vir = ((3 * M) / (4 * np.pi * rho_vir)) ** (1.0 / 3)
        v_circ = np.sqrt(GN * M / r_vir)
        T_vir = self.mu * m_p * v_circ ** 2 / (2 * k_B)
        return T_vir

    def T_vir(self, M, z):
        """ Return virial temperature for given halo mass and redshift
        """
        crit_dens = self.cosmo.critical_density(z).value * Gram / Centimeter ** 3
        rho_vir = self.delta_vir(z) * crit_dens
        T_vir = self._calc_T_vir(M, rho_vir)
        return T_vir

    def M_min(self, T_vir_cut, z):
        """ Solve for minimum halo mass for a give virial temperature cut and redshift
        """
        return 10 ** root(lambda logM: self.T_vir(10 ** logM * (M_s / self.cosmo.h), z) - T_vir_cut, 8).x * (M_s / self.cosmo.h)

    def f_star_T_vir_cut(self, M, z, f_star, T_vir_cut):
        """ Star formation efficiency for a given halo mass and redshift accounting for virial temperature cut
        """
        if self.M_min_val is None:
            return f_star * np.heaviside(M - self.M_min(T_vir_cut, z), 0.0)
        else:
            return f_star * np.heaviside(M - self.M_min_val, 0.0)

    def dndlnm(self, M, z):
        """ Halo mass function for a given halo mass and redshift
            Use Despali (2016) halo mass function to admit any definition of virial overdensity
        """
        return mass_function.massFunction(M / (M_s / self.cosmo.h), z, q_in="M", q_out="dndlnM", **self.hmf_kwargs) * (Mpc / self.cosmo.h) ** -3

    def int_star(self, z, f_star):
        """ Integrate star formation efficiency over the halo mass function
        """
        M_ary = np.logspace(6, 15, 1000) * M_s / self.cosmo.h
        return trapz(self.f_star_T_vir_cut(M_ary, z, f_star, self.T_vir_cut) * self.dndlnm(M_ary, z), M_ary)

    def int_halo(self, f_star, z):
        """ Add f_star dependence to star formation rate integral over halo mass function
        """
        return self.int_star(z, f_star)

    def D_int_halo_z(self, z):
        """ Derivative of star formation rate integral over halo mass function
        """
        return derivative(lambda z: self.int_halo(self.f_star, z), x0=z, dx=z * 1e-2)

    def make_D_int_halo_z_interp(self):
        """ Make interpolation table of derivative of star formation rate integral over halo mass function
        """
        self.z_interp_ary = np.linspace(self.z_min, 450.0, 1000)
        self.D_int_halo_z_ary = [self.D_int_halo_z(z) for z in tqdm(self.z_interp_ary)]
        self.D_int_halo_z_interp = interp1d(self.z_interp_ary, self.D_int_halo_z_ary, bounds_error=False, kind="linear", fill_value=0.0)  # Dodgy interpolation

    def dz_dt(self, z):
        """ dz/dt
        """
        return -self.cosmo.H(z).value * Kmps / Mpc * (1 + z)

    def Gamma_X(self, z):
        """ X-ray heating, using either Eq. (57) of astro-ph/0507102 ("Hirata") or power-law form
        """
        if self.sed == "Hirata":
            # From page 11 of astro-ph/0507102
            f_Gamma = 0.14
            f_Xe_E_X = 27 * KeV

            return f_Gamma * f_Xe_E_X * self.cosmo.Ob0 / (m_p * self.cosmo.Om0) * self.dz_dt(z) * self.D_int_halo_z_interp(z)
        else:
            return self.integ_sed_PL(**self.sed_kwargs) * self.SFRD(z)

    def df_coll_dt_fstar(self, z):
        """ Derivative of fraction of stars in collapsed halos 
        """
        rho_m0 = self.cosmo.critical_density0.value * Gram / Centimeter ** 3 * self.cosmo.Om0
        return self.dz_dt(z) * self.D_int_halo_z_interp(z) / rho_m0

    def df_coll_dz_fstar(self, z):
        """ Derivative of fraction of stars in collapsed halos 
        """
        rho_m0 = self.cosmo.critical_density0.value * Gram / Centimeter ** 3 * self.cosmo.Om0
        return self.D_int_halo_z_interp(z) / rho_m0

    def SFRD(self, z):
        """ Star-formation rate density
        """
        rho_b0 = self.cosmo.critical_density0.value * Gram / Centimeter ** 3 * self.cosmo.Ob0
        return rho_b0 * self.df_coll_dt_fstar(z)

    def heat_coeff(self, z, mu=1.22):
        """ X-ray heating coefficient, Eq. (56) of astro-ph/0507102
            Returned in units of Kelvin
        """

        H = self.cosmo.H(z).value * Kmps / Mpc
        rho_b0 = self.cosmo.critical_density0.value * Gram / Centimeter ** 3 * self.cosmo.Ob0
        return 2 * mu * m_p * self.Gamma_X(z) / (3 * rho_b0 * k_B * H) / Kelv
