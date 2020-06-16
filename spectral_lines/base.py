import numpy as np
from tqdm import tqdm
from scipy.interpolate import splrep, splev


# Lines are taken from Chotard (in prep)
extrema_lims = {
    'CaIIHK':   {'b': (3504., 3687.), 'r': (3830., 3990.)},
    'SiII4000': {'b': (3830., 3990.), 'r': (4030., 4150.)},
    'MgII':     {'b': (4030., 4150.), 'r': (4450., 4650.)},
    'Fe4800':   {'b': (4450., 4650.), 'r': (5050., 5285.)},
    'SIIW_L':   {'b': (5050., 5285.), 'r': (5500., 5681.)},
    'SIIW_R':   {'b': (5050., 5285.), 'r': (5500., 5681.)},
    'SIIW':     {'b': (5050., 5285.), 'r': (5500., 5681.)},
    'SiII5972': {'b': (5550., 5681.), 'r': (5850., 6015.)},
    'SiII6355': {'b': (5850., 6015.), 'r': (6250., 6365.)},
    'OI7773':   {'b': (7100., 7270.), 'r': (7720., 8000.)},
}
vel_lims = {
    'SiII4000': (3963, 4034),
    'SIIW_L':   (5200, 5350),
    'SIIW_R':   (5351, 5550),
    'SiII5972': (5700, 5875),
    'SiII6355': (6000, 6210),
}
lambda0 = {
    'CaIIHK':   3945.,
    'SiII4000': 4131.,
    'MgII':     4481.,
    'Fe4800':   4966.,
    'SIIW_L':   5454.,
    'SIIW_R':   5640.,
    'SIIW':     5500.,
    'SiII5972': 5972.,
    'SiII6355': 6355.,
    'OI7773':   8100.,
}

line_names = [
    'CaIIHK',
    'SiII4000',
    'MgII',
    'Fe4800',
    'SIIW_L',
    'SIIW_R',
    'SIIW',
    'SiII5972',
    'SiII6355',
    'OI7773',
]


class MissingDataError(Exception):
    """Throw an error if data is missing in the IDR"""
    def __init__(self):
        pass


class Measure(object):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False,
                 norm='SNID'):
        """Base class for measuring spectral lines in a spectrum.

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
        self.l0 = lambda0[line]
        self.l_range = (extrema_lims[line]['b'][0], extrema_lims[line]['r'][1])
        self.l_brange = extrema_lims[line]['b']
        self.l_rrange = extrema_lims[line]['r']
        self.interp_grid = interp_grid
        self.norm = norm
        if sim:
            self.wave_sn, self.flux_sn, self.var_sn = np.array(spectrum)
        else:
            self.wave_sn, self.flux_sn, self.var_sn = spectrum.rf_spec()
        if self.norm == 'SNID':
            self.wave_sn, self.flux_sn, self.var_sn = self.get_snid_norm_spec()
        elif self.norm == 'line':
            self.wave_sn, self.flux_sn, self.var_sn = self.get_line_norm_spec()
        try:
            wave_feat, flux_feat, var_feat = self.get_feature_spec()
            wave_subfeat, flux_subfeat, var_subfeat = self.get_subfeature_spec()
        except MissingDataError:
            raise
        self.wave_feat = wave_feat
        self.flux_feat = flux_feat
        self.var_feat = var_feat
        self.wave_subfeat = wave_subfeat
        self.flux_subfeat = flux_subfeat
        self.var_subfeat = var_subfeat
        self._maxima = None
        self._minimum = None

    def vel_space(self, wave, rel=True):
        """
        Returns the feature spectrum in velocity space using the relativistic
        Doppler formula (units are km/s).
        """
        c = 3.e5  # speed of light in km/s
        dl = wave - self.l0
        ddl = dl / self.l0
        if rel:
            v = c*((ddl+1.)**2.-1.)/((ddl+1.)**2.+1.)
        else:
            v = c * ddl
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
        Fits a 13 point cubic spline to the spectrum. Knots are evenly spaced
        from 2500 A to 10000 A.
        """
        w, f, v = self.wave_sn, self.flux_sn, self.var_sn
        knots = np.linspace(2500, 10000, 13)
        knots = knots[(knots >= min(w)) & (knots <= max(w))]
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
        wave_cut = (w >= self.l_range[0]) & (w < self.l_range[1])
        f = f[wave_cut]
        v = v[wave_cut]
        w = w[wave_cut]
        if len(w) == 0:
            raise MissingDataError
        return w, f, v
    
    def get_subfeature_spec(self):
        """Returns the spectrum in restricted range between the
        """
        w, f, v = self.wave_sn, self.flux_sn, self.var_sn
        wave_cut = (w >= self.l_brange[0]) & (w < self.l_rrange[1])
        f = f[wave_cut]
        v = v[wave_cut]
        w = w[wave_cut]
        if len(w) == 0:
            raise MissingDataError
        return w, f, v

    def get_smoothed_feature_spec(self):
        raise NotImplementedError

    def get_interp_feature_spec(self):
        raise NotImplementedError

    @property
    def maxima(self):
        if not self._maxima:
            w, f = self.get_interp_feature_spec()
            bw_cut = (w >= self.l_brange[0]) & (w <= self.l_brange[1])
            rw_cut = (w >= self.l_rrange[0]) & (w <= self.l_rrange[1])
            max_b_ind = np.where(f[bw_cut] == max(f[bw_cut]))[0]
            max_r_ind = np.where(f[rw_cut] == max(f[rw_cut]))[0]
            maxima = {'blue_max_wave': w[bw_cut][max_b_ind],
                      'blue_max_flux': f[bw_cut][max_b_ind],
                      'red_max_wave': w[rw_cut][max_r_ind],
                      'red_max_flux': f[rw_cut][max_r_ind]}
            self._maxima = maxima
        return self._maxima

    def get_pseudo_continuum(self):
        """Returns the line connecting the ends of the feature
        """
        wave, _ = self.get_interp_feature_spec()
        flux_diff = self.maxima['red_max_flux'] - self.maxima['blue_max_flux']
        wave_diff = self.maxima['red_max_wave'] - self.maxima['blue_max_wave']
        slope = flux_diff / wave_diff
        inter = self.maxima['red_max_flux'] - slope * self.maxima[
            'red_max_wave']
        return inter + slope * wave

    @property
    def minimum(self):
        if not self._minimum:
            w, f = self.get_interp_feature_spec()
            f_cont = self.get_pseudo_continuum()
            f /= f_cont
            try:
                search_range = (w >= vel_lims[self.line][0])
                search_range &= (w <= vel_lims[self.line][1])
            except KeyError:
                search_range = (w >= self.maxima['blue_max_wave'])
                search_range &= (w <= self.maxima['red_max_wave'])
            min_f = np.min(f[search_range])
            self._minimum = w[np.where(f == min_f)][0]
        return self._minimum

    def get_line_velocity(self):
        """
        Find the velocity of the line (defined to be the velocity that Doppler
        shifts l_0 to the minimum of the smoothed, interpolated spectrum)
        """
        v_abs = self.vel_space(self.minimum)
        return v_abs

    def get_equiv_width(self):
        w, f = self.get_interp_feature_spec()
        f_c = self.get_pseudo_continuum()
        integrand = 1 - f / f_c
        dl = np.diff(w)
        wb_ind = np.where(w == self.maxima['blue_max_wave'])[0][0]
        wr_ind = np.where(w == self.maxima['red_max_wave'])[0][0]
        return np.sum(np.dot(integrand[wb_ind:wr_ind], dl[wb_ind:wr_ind]))
