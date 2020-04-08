import numpy as np
from .base import Measure
from scipy.optimize import curve_fit

def gaussian(x, *params):
    scale, mean, std = params
    return scale * np.exp(-(x-mean)**2/(2 * std**2))

def double_gaussian(x, *params):
    offset, slope = params[:2]
    line = offset + slope * x
    gauss_1_params = params[2:5]
    gauss_2_params = [params[2], *params[5:]]
    return line + gaussian(x, *gauss_1_params) + gaussian(x, *gauss_2_params)

class Doublet(Measure):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False,
                 norm=True):
        super(Doublet, self).__init__(spectrum, line, interp_grid, sim, norm)
        self._double_gauss_params = None
        self.kind = 'doublet'

    @property
    def double_gauss_params(self):
        if self._double_gauss_params is None:
            p0 = [1.0, 0.0,
                  -0.5,
                  6150., 40.,
                  6125., 40.]
            popt, pcov = curve_fit(double_gaussian,
                                   self.wave_feat,
                                   self.flux_feat,
                                   p0=p0,
                                   sigma=np.sqrt(self.var_feat),
                                   maxfev=10000)
            self._double_gauss_params = popt
        return self._double_gauss_params

    def get_smoothed_feature_spec(self):
        flux = double_gaussian(self.wave_feat, *self.double_gauss_params)
        return self.wave_feat, flux


    def get_interp_feature_spec(self):
        wave = np.arange(np.min(self.wave_feat),
                           np.max(self.wave_feat),
                           self.interp_grid)
        flux = double_gaussian(wave, *self.double_gauss_params)
        return wave, flux

    def get_pseudo_continuum(self):
        wave_s = np.arange(np.min(self.wave_feat),
                           np.max(self.wave_feat),
                           self.interp_grid)
        popt = self.double_gauss_params[:2]
        line = lambda x, *p: p[0] + p[1] * x
        return line(wave_s, *popt)

    @property
    def minimum(self):
        if self._minimum is None:
            w, f = self.get_interp_feature_spec()
            f_cont = self.get_pseudo_continuum()
            f /= f_cont
            search_range = (w >= self.l_brange[-1]) & (w <= self.l_rrange[0])
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
        return NotImplementedError

