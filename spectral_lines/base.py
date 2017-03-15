import numpy as np
from tqdm import tqdm
from scipy.interpolate import splrep, splev


# Lines are taken from Chotard (in prep)
# Format: [l_0, l_min, l_max, l_bmin, l_bmax, l_rmin, l_rmax]
lines = {'CaII':     [3945., 3450., 4070., 3504., 3687., 3830., 3990.],
         'SiII4131': [4131., 3850., 4150., 3830., 3990., 4030., 4150.],
         'MgII4300': [4481., 4000., 4610., 4030., 4150., 4450., 4650.],
         'FeII4800': [4966., 4350., 5350., 4450., 4650., 5050., 5285.],
         'SIIWL':    [5454., 5060., 5700., 5050., 5285., 5500., 5681.],
         'SIIWR':    [5640., 5060., 5700., 5050., 5285., 5500., 5681.],
         'SIIW':     [5500., 5060., 5700., 5050., 5285., 5500., 5681.],
         'SiII5972': [5972., 5500., 6050., 5550., 5681., 5850., 6015.],
         'SiII6355': [6355., 5600., 6600., 5850., 6015., 6250., 6365.],
         'OICaII':   [8100., 6500., 8800., 7100., 7270., 8300., 8800.]}


class MissingDataError(Exception):
    """Throw an error if data is missing in the IDR"""
    def __init__(self):
        pass


class Measure(object):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False,
                 norm='SNID'):
        """
        Base class for measuring spectral lines in a spectrum from the IDR.

        `spectrum` is an `IDRTools.Spectrum` object if `sim == False`, or a
        list `[wave, flux, flux_var]` if `sim == True`.

        `norm` can be set to `None`, `'SNID'`, or `'deriv'`.
            *If `None`, no flux normalization will be done.
            *If `'SNID'`, a 13-point spline will be fit to the entire spectrum
            and this spline is divided out (see Blondin and Tonry 2007.)
            *If `'line'`, look for local maxima between l_bmin-l_bmax and
            l_rmin-l_rmax, and divide out the line between these maxima.

        """
        self.line = line
        self.l0 = lines[line][0]
        self.l_range = lines[line][1:3]
        self.l_brange = lines[line][3:5]
        self.l_rrange = lines[line][5:]
        self.interp_grid = interp_grid
        if sim:
            self.wave_sn, self.flux_sn, self.var_sn = spectrum
        else:
            self.wave_sn, self.flux_sn, self.var_sn = spectrum.rf_spec()
        if norm == 'SNID':
            self.wave_sn, self.flux_sn, self.var_sn = self.get_snid_norm_spec()
        elif norm == 'line':
            self.wave_sn, self.flux_sn, self.var_sn = self.get_line_norm_spec()
        try:
            self.wave_feat, self.flux_feat, self.var_feat = self.get_feature_spec()
        except MissingDataError:
            raise

    def vel_space(self, wave):
        """
        Returns the feature spectrum in velocity space using the relativistic
        Doppler formula (units are km/s).
        """
        c = 3.e5  # speed of light in km/s
        dl = wave-self.l0
        ddl = dl/self.l0
        v = c*((ddl+1.)**2.-1.)/((ddl+1.)**2.+1.)
        return v

    def wave_space(self, vel):
        """
        Returns the feature spectrum in wavelength space using the relativistic
        Doppler formula (units are km/s).
        """
        c = 3.e5
        return self.l0*np.sqrt((1+vel/c)/(1-vel/c))

    def get_snid_norm_spec(self):
        """
        Does SNID-like (Blondin and Tonry 2007) spectrum normalization.
        Fits a 13 point cubic spline to the spectrum
        """
        w, f, v = self.wave_sn, self.flux_sn, self.var_sn
        knots = np.linspace(2500, 10000, 13)
        knots = knots[(knots>=min(w))&(knots<=max(w))]
        spl = splrep(w, f, k=3, t=knots)
        f_cont = splev(w, spl)
        cont_div = f/f_cont
        v_cont_div = v/f_cont**2
        return w, cont_div, v_cont_div

    def get_line_norm_spec(self):
        """
        Look for local maxima in blue and red ranges (defined by the line)
        and divide out the line connecting these maxima.
        """
        w, f, v = self.wave_sn, self.flux_sn, self.var_sn
        fb = f[(w > self.l_brange[0]) & (w < self.l_brange[1])]
        fr = f[(w > self.l_rrange[0]) & (w < self.l_rrange[1])]
        if len(fb) == 0 or len(fr) == 0:
            raise MissingDataError
        w_bmax = w[f == np.max(fb)]
        w_rmax = w[f == np.max(fr)]
        f_bmax = f[w == w_bmax]
        f_rmax = f[w == w_rmax]
        slope = (f_rmax - f_bmax)/(w_rmax - w_bmax)
        intercept = f_bmax - slope * w_bmax
        line = lambda x: slope * x + intercept
        f_norm = f / line(w)
        v_norm = v / line(w) ** 2
        return w, f_norm, v_norm

    def get_feature_spec(self):
        """
        Returns the spectrum in l_range.
        """  
        w, f, v = self.wave_sn, self.flux_sn, self.var_sn
        f = f[(w >= self.l_range[0]) & (w <= self.l_range[1])]
        v = v[(w >= self.l_range[0]) & (w <= self.l_range[1])]
        w = w[(w >= self.l_range[0]) & (w <= self.l_range[1])]
        if len(w) == 0:
            raise MissingDataError
        # if max(w) < self.l_range[1]:
        #     raise MissingDataError
        return w, f, v
