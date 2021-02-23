# Scientific libraries
import numpy as np

#astro
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
import astropy.constants as const

#optional 3ML imports

from threeML import load_analysis_results, parallel_computation
from threeML.utils.progress_bar import trange

import multiprocessing as mp

#from synchrotron_models import SynchrotronNumerical, characteristic_synchrotron_energy_fuck_up

#Old synch model
#from pynchrotron.threeML_models import SynchrotronNumericalOld as SynchrotronNumerical
from pynchrotron.threeML_models import SynchrotronNumerical
from glob import glob
import copy
import collections

import os

import h5py

import pandas as pd

# Constants
keV2erg = ((1*u.keV).to('erg')).value

c = const.c.cgs.value  # cm s^-1
m_e = const.m_e.cgs.value  # g
q_e = const.e.esu.value  # g^1/2 cm^3/2 s^-1
sigT = const.sigma_T.cgs.value
a= 16E14 * q_e / (2 * np.pi * m_e * c) #/ u.s
d= (0.5/(3. * np.pi)) * sigT * c
k = (18*np.pi *q_e * m_e * c)/(sigT*sigT)
h = 4.135667662340164e-18 # keV s

def _characteristic_synchrotron_energy(gamma, B, gamma_bulk):
    """
    Calculate the characteristic syncrotron energy
    """

    nu_c = gamma_bulk * gamma**2 * B * q_e / (2 * np.pi * m_e *
                                              c) / u.s

    return (nu_c * const.h).to(u.keV)

class ComputePhysics(object):
    """
    Class to read in the fit results of a pynchrotron fit and calculate
    all the interesting physical quantities.
    """

    def __init__(self,
                 fits_file_path,
                 gamma_bulk=1,
                 gamma_min=1E5,
                 redshift=None,
                 num_samples=-1):

        # Get analysis result
        self._num_samples = num_samples
        self._result = load_analysis_results(fits_file_path)

        # Bulk gamma factor
        self._gamma_bulk = gamma_bulk

        # Gamma min value
        self._gamma_min = gamma_min

        # Redshift
        self._z = redshift
        # luminosity distance
        self._dL = (cosmo.luminosity_distance(self._z).to('cm')).value

        # Read in the neccessary things from the analysis result
        self._read_in()

        # Init synch model with the given gamma min and bulk gamma
        self._synch = SynchrotronNumerical()
        self._synch.gamma_min.bounds = (None, None)
        self._synch.gamma_max.bounds = (None, None)
        self._synch.gamma_cool.bounds = (None, None)
        self._synch.index.bounds = (None, None)
        self._synch.K.bounds = (None, None)
        self._synch.B.bounds = (None, None)
        self._synch.bulk_gamma.bounds = (None, None)

        self._synch.gamma_min.value = self._gamma_min
        self._synch.bulk_gamma.value = self._gamma_bulk
        # Energy grid
        self._ene_grid = np.logspace(-1,4,500)
        # Calculate all spectra
        self._calc_spectra()

        # Compute physics
        #self._compute_gamma()
        #self._compute_B()
        #self._compute_ne()

    def compute_gamma(self, xib, tp = 2):

        q = (self._index-2.)/(self._index-1.)

        top = np.power(2., 1./16.) * np.power(3., 3./4.) * np.power(xib, 3./16.)
        top *= np.power(self._nu_c, 7./32.) * np.power(sigT, 1./4.)
        top *= np.power(self._Fnu_max, 3./16.)
        top *= np.power(self._nu_inj, 3./32.)* np.power(self._dL, 3./8.)
        top *= np.power(np.pi, 1./16.)

        bottom = np.power(m_e, 1./8.)  * np.power(q_e, 1./8.)
        bottom *= np.power(tp, 1./8.) * np.power(c, 11./16.)* np.power(q, 3./16.)

        return (1./3.) * top/bottom

    def compute_B(self,  xib, tp = 2):

        q = (self._index-2.)/(self._index-1.)

        top = np.power(q, 1./16.) * np.power(2., 5./16.) * np.power(m_e, 3./8.)
        top *= np.power(np.pi, 5./16.)
        top *= np.power(q_e, 3./8.) *np.power(c, 9./16.) * np.power(3.,3./4.)

        bottom = np.power(tp, 5./8.) * np.power(xib, 1./16.)
        bottom *= np.power(self._nu_c, 13./32.) * np.power(self._dL, 1./8.)
        bottom *= np.power(sigT, 3./4.) * np.power(self._Fnu_max, 1./16.)
        bottom *= np.power(self._nu_inj, 1./32.)

        return top/bottom

    def compute_ne(self,  xib, tp = 2):

        q = (self._index-2.)/(self._index-1.)

        top = 2 * np.sqrt(3.) * np.power(2, 5./8.) * np.power(q_e, 3./4.)
        top *= np.power(self._Fnu_max, 7./8.)
        top *= np.power(tp, 3./4.) * np.power(self._nu_c, 3./16.) * np.power(q, 1./8.)
        top *= np.power(self._dL, 7./4.) * np.power(np.pi, 5./8.)

        bottom = np.sqrt(sigT) * np.power(xib, 1./8.)
        bottom *= np.power(c, 15./8.)  * np.power(m_e, 5./4.)
        bottom *= np.power(self._nu_inj, 1./16.)

        return top/bottom


    def compute_R(self, xib, tp = 2):

        q = (self._index-2.)/(self._index-1.)

        top = np.power(2, 1./8.) * np.sqrt(3.) *  np.power(xib, 3./8.)
        top *= np.power(self._nu_c, 7./16.) * np.power(tp, 3./4.)
        top *= np.sqrt(sigT) * np.power(self._Fnu_max, 3./8.) *np.power(self._nu_inj, 3./16.)
        top *=  np.power(self._dL, 3./4.)* np.power(np.pi, 1./8.)

        bottom = np.power(m_e, 1./4.)  * np.power(q_e, 1./4.)
        bottom *= np.power(c, 3./8.) * np.power(q, 3./8.)

        return (2./3.) * top/bottom

    def _read_in(self):
        """
        Read in the neccessary things from the analysis result
        """

        # Get the samples for all parameters from the analysis result
        # object. For this we have to check the order of the
        # parameters in the analysis result object
        #
        mask = np.zeros(5, dtype=int)

        B_found = False
        K_found = False
        index_found = False
        gc_found = False
        gmx_found = False
        for i, p in enumerate(self._result._free_parameters.keys()):
            if p[-1:]=="B":
                assert not B_found, "Bug: Two B parameters"
                mask[1] = i
                B_found = True

            if p[-1:]=="K":
                assert not K_found, "Bug: Two B parameters"
                mask[0] = i
                K_found = True

            if p[-5:]=="index":
                assert not index_found, "Bug: Two B parameters"
                mask[2] = i
                index_found = True

            if p[-10:]=="gamma_cool":
                assert not gc_found, "Bug: Two B parameters"
                mask[3] = i
                gc_found = True

            if p[-9:]=="gamma_max":
                assert not gmx_found, "Bug: Two B parameters"
                mask[4] = i
                gmx_found = True

        assert B_found*K_found*index_found*gc_found*gmx_found, \
            "At least one of the needed parameters was not in the analysis result"

        if self._num_samples==-1 or self._num_samples>self._result.samples.shape[1]:
            self._K, self._B, self._index, self._gc, self._gmx = \
                self._result.samples[mask]
        else:
            sample_mask = np.zeros(self._result.samples.shape[1], dtype=bool)
            sample_mask[:self._num_samples] = True
            np.random.shuffle(sample_mask)
            self._K, self._B, self._index, self._gc, self._gmx = \
                self._result.samples[mask][:,sample_mask]

        self._nu_c  = (_characteristic_synchrotron_energy(self._gc,
                                                          self._B,
                                                          self._gamma_bulk).to('keV') /
                       const.h.to('keV s')).value * (1+self._z)

        self._nu_inj = (_characteristic_synchrotron_energy(self._gamma_min,
                                                           self._B,
                                                           self._gamma_bulk).to('keV')/
                        const.h.to('keV s')).value * (1+self._z)

        self._nu_max = (_characteristic_synchrotron_energy(self._gmx,
                                                           self._B,
                                                           self._gamma_bulk).to('keV')/
                        const.h.to('keV s')).value * (1+self._z)

        self._e_c = (self._nu_c *  const.h.to('keV s')).value
        self._e_inj = (self._nu_inj *  const.h.to('keV s')).value
        self._e_max = (self._nu_max *  const.h.to('keV s')).value

    def _calc_spectra(self):
        """
        Calculate the spectrum for all samples and determine the maximum of Fnu
        """
        self._spectra = np.zeros((len(self._K), len(self._ene_grid)))

        # parallize this
        #with parallel_computation():
        for i in trange(len(self._K), desc="Calc spectrum of all samples"):

            B = self._B[i]
            K = self._K[i]
            gc = self._gc[i]
            gmx = self._gmx[i]
            index = self._index[i]
            self._synch.B.value = B
            self._synch.K.value = K
            self._synch.gamma_cool.value = gc
            self._synch.gamma_max.value = gmx
            self._synch.index.value = index
            self._spectra[i] = self._synch(self._ene_grid)

        # Calc max Fnu
        self._Fnu_max = np.max(self._ene_grid*self._spectra, axis=1) * keV2erg * h
