import numpy as np
from .base import Measure
from scipy.interpolate import UnivariateSpline


class Spl(Measure):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False,
                 norm='SNID', n_l=16, smooth_fac=0.005):
        super(Spl, self).__init__(spectrum, line, interp_grid, sim, norm)
        self.n_l = n_l
        self.smooth_fac = smooth_fac
        self.kind = 'spline'
        self._maxima = None
        self._minimum = None

    def get_smoothed_feature_spec(self):
        """
        Returns the smoothed feature spectrum.
        """
        wave, f, v = self.wave_feat, self.flux_feat, self.var_feat
        f_ts = []
        for i in range(int(self.n_l/2), int(len(f)-self.n_l/2)):
            sig = wave[i]*self.smooth_fac
            sub = range(int(i-self.n_l/2), int(i+self.n_l/2))
            x = wave[i]-wave[sub]
            g = 1/np.sqrt(2*np.pi)*np.exp(-1/sig**2*x**2)
            w = g/v[sub]
            f_ts_i = np.dot(w, f[sub])/np.sum(w)
            f_ts.append(f_ts_i)
        # Cut out wavelengths outside smoothing window
        smooth_wave = wave[int(self.n_l/2):int(-self.n_l/2)]
        f_ts = np.array(f_ts)
        # Check that the variance is such that the spectrum isn't oversmoothed
        flux_diff = f_ts - f[int(self.n_l / 2):int(-self.n_l / 2)]
        flux_err = np.sqrt(v[int(self.n_l / 2):int(-self.n_l / 2)])
        if not np.all(np.abs(flux_diff) <= 6 * flux_err):
            raise ValueError('Spectrum is oversmoothed! Check the variance.')
        return smooth_wave, f_ts

    def get_interp_feature_spec(self, return_spl=False):
        """
        Returns the spline interpolated, smoothed feature spectrum
        """
        w, f = self.get_smoothed_feature_spec()
        spl = UnivariateSpline(w, f, k=4, s=0)
        n_pts = int((max(w)-min(w))/self.interp_grid + 1)
        w_int = np.linspace(min(w), max(w), n_pts)
        f_int = spl(w_int)
        if return_spl:
            return w_int, f_int, spl
        else:
            return w_int, f_int

