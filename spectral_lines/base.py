import numpy as np
from tqdm import tqdm
from scipy.interpolate import splrep, splev


# Lines are taken from Chotard (in prep)
extrema_lims = {
    'CaIIHK':   {'b': (3504., 3687.), 'r': (3887., 3990.)},
    'SiII4000': {'b': (3830., 3963.), 'r': (4034., 4150.)},
    'MgII':     {'b': (4034., 4150.), 'r': (4452., 4573.)},
    'Fe4800':   {'b': (4400., 4650.), 'r': (5050., 5300.)},
    'SIIW_L':   {'b': (5085., 5250.), 'r': (5250., 5450.)},
    'SIIW_R':   {'b': (5250., 5450.), 'r': (5500., 5681.)},
    'SIIW':     {'b': (5085., 5250.), 'r': (5500., 5681.)},
    'SiII5972': {'b': (5550., 5681.), 'r': (5850., 6015.)},
    'SiII6355': {'b': (5850., 6015.), 'r': (6250., 6365.)},
    'OI7773':   {'b': (7100., 7270.), 'r': (7720., 8000.)},
}
vel_lims = {
    'SiII4000': (3963, 4034),
    'SIIW_L':   (5200, 5350),
    'SIIW_R':   (5351, 5550),
    'SiII5972': (5700, 5900),
    'SiII6355': (6000, 6210),
}
lambda0 = {
    'CaIIHK':   3945.,
    'SiII4000': 4128.,
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

        `norm` can be set to `'SNID'` or `'None'`
            *If `None`, no flux normalization will be done.
            *If `'SNID'`, a 13-point spline will be fit to the entire spectrum
            and this spline is divided out (see Blondin and Tonry 2007.)


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
        try:
            wave_feat, flux_feat, var_feat = self.get_feature_spec()
        except MissingDataError:
            raise
        self.wave_feat = wave_feat
        self.flux_feat = flux_feat
        self.var_feat = var_feat
        self._interp_wave = None
        self._interp_flux = None
        self._maxima = None
        self._minimum = None
        self._velocity = None
        self._equiv_width = None

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

    def get_smoothed_feature_spec(self):
        raise NotImplementedError

    def get_interp_feature_spec(self):
        raise NotImplementedError

    @property
    def interp_feature_spec(self):
        if self._interp_flux is None:
            self._interp_wave, self._interp_flux = self.get_interp_feature_spec()
        return self._interp_wave, self._interp_flux

    @property
    def maxima(self):
        if not self._maxima:
            w, f = self.interp_feature_spec
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
        wave, _ = self.interp_feature_spec
        flux_diff = self.maxima['red_max_flux'] - self.maxima['blue_max_flux']
        wave_diff = self.maxima['red_max_wave'] - self.maxima['blue_max_wave']
        slope = flux_diff / wave_diff
        inter = self.maxima['red_max_flux'] - slope * self.maxima[
            'red_max_wave']
        return inter + slope * wave

    @property
    def minimum(self):
        if not self._minimum:
            w, f = self.interp_feature_spec
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

    @property
    def velocity(self):
        if not self._velocity:
            self._velocity = self.get_line_velocity()
        return self._velocity

    def get_equiv_width(self):
        w, f = self.get_interp_feature_spec()
        f_c = self.get_pseudo_continuum()
        integrand = 1 - f / f_c
        dl = np.diff(w)
        wb_ind = np.where(w == self.maxima['blue_max_wave'])[0][0]
        wr_ind = np.where(w == self.maxima['red_max_wave'])[0][0]
        return np.sum(np.dot(integrand[wb_ind:wr_ind], dl[wb_ind:wr_ind]))

    @property
    def equiv_width(self):
        if not self._equiv_width:
            self._equiv_width = self.get_equiv_width()
        return self._equiv_width


class MeasureSimErrors(object):

    def __init__(self, spec, kind, n_sims=50, line='SiII6355', sim=False,
                 interp_grid=0.1, norm='SNID', **kwargs):
        if sim:
            wave, flux, var = np.array(spec)
        else:
            wave, flux, var = spec.rf_spec()
        self.sims = []
        for sim_id in range(n_sims):
            noised_flux = flux + np.random.randn(len(wave)) * np.sqrt(var)
            self.sims.append(kind([wave, noised_flux, var], sim=True,
                                  line=line, interp_grid=interp_grid,
                                  norm=norm, **kwargs))
        self._minimum = None
        self._maxima = None
        self._velocity = None
        self._equiv_width = None

    @property
    def minimum(self):
        if not self._minimum:
            sim_minima = [sim.minimum for sim in self.sims]
            self._minimum = {'mean': np.mean(sim_minima),
                             'std': np.std(sim_minima)}
        return self._minimum

    @property
    def velocity(self):
        if not self._velocity:
            sim_vel = [sim.velocity for sim in self.sims]
            self._velocity = {'mean': np.mean(sim_vel),
                              'std': np.std(sim_vel)}
        return self._velocity

    @property
    def maxima(self):
        if not self._maxima:
            sim_maxima = [sim.maxima for sim in self.sims]
            self._maxima = {}
            keys = sim_maxima[0].keys()
            for key in keys:
                vals = [sim[key] for sim in sim_maxima]
                self._maxima[key] = {'mean': np.mean(vals),
                                     'std': np.std(vals)}
        return self._maxima

    @property
    def equiv_width(self):
        if not self._equiv_width:
            sim_ew = [sim.equiv_width for sim in self.sims]
            self._equiv_width = {'mean': np.mean(sim_ew),
                                 'std': np.std(sim_ew)}
        return self._equiv_width

