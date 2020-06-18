import numpy as np
from .base import Measure
from scipy.interpolate import UnivariateSpline

def gaussian(x, mu, sigma):
    return 1/np.sqrt(2*np.pi) * np.exp(-(x-mu)**2/sigma**2)


class Spl(Measure):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False,
                 norm='SNID', n_l=16, smooth_fac=0.005):
        super(Spl, self).__init__(spectrum, line, interp_grid, sim, norm)
        self.n_l = n_l
        self.smooth_fac = smooth_fac
        self.kind = 'spline'

    def get_smoothed_feature_spec(self):
        """
        Returns the smoothed feature spectrum.
        """
        wave, flux, var = self.wave_feat, self.flux_feat, self.var_feat
        half = int(self.n_l/2)
        windows = np.array([range(i-half, i+half+1)
                            for i in range(half, len(wave)-half)])
        weights = np.array([gaussian(wave[w], wave[w[half]],
                                     wave[w[half]] * self.smooth_fac) /
                            var[w] for w in windows])
        windowed_flux = np.array([flux[w] for w in windows])
        smooth_wave = np.array([wave[w[half]] for w in windows])
        smooth_flux = np.dot(weights, windowed_flux.T).sum(
            axis=0) / weights.sum()
        return smooth_wave, smooth_flux


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


