import numpy as np
from .base import Measure
from iminuit import Minuit


def gaussian(x, *params):
    """Gaussian function with a linear offset"""
    offset, slope, scale, mean, std = params
    line = offset + slope * (x - mean)
    gauss = scale * np.exp(-(x - mean)**2/(2 * std**2))
    return line + gauss


class Gauss(Measure):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False,
                 norm=True):
        super(Gauss, self).__init__(spectrum, line, interp_grid, sim, norm)
        self._gauss_params = None
        self.kind = 'gaussian'


    @property
    def gauss_params(self):
        """Fits a Gaussian to the restricted wavelength range of the feature"""
        if not self._gauss_params:
            wave = self.wave_feat
            flux = self.flux_feat
            var = self.var_feat

            scale_factor = np.median(flux)
            flux /= scale_factor
            var /= scale_factor ** 2

            def chisq(a, b, amp, mu, sig):
                model = gaussian(wave, a, b, amp, mu, sig)
                return np.sum((flux-model)**2/var)

            m = Minuit(chisq,
                       a=1, limit_a=(0, 2),
                       b=0, limit_b=(-0.1, 0.1),
                       amp=-0.5, limit_amp=(-1, 0),
                       mu=6150, limit_mu=(min(wave), max(wave)),
                       sig=50,
                       pedantic=False,
                       print_level=0)
            fval, res = m.migrad()
            popt = [r['value'] for r in res]
            popt[0] *= scale_factor
            popt[1] *= scale_factor
            popt[2] *= scale_factor
            self._gauss_params = popt
        return self._gauss_params

    def get_smoothed_feature_spec(self):
        wave, flux = self.wave_feat, self.flux_feat
        popt = self.gauss_params
        return wave, gaussian(wave, *popt)

    def get_interp_feature_spec(self):
        popt = self.gauss_params
        wave_s = np.arange(np.min(self.wave_feat),
                           np.max(self.wave_feat),
                           self.interp_grid)
        return wave_s, gaussian(wave_s, *popt)
    
    def get_pseudo_continuum(self):
        wave_s = np.arange(np.min(self.wave_feat),
                           np.max(self.wave_feat),
                           self.interp_grid)
        popt = self.gauss_params
        line = lambda x, *p: p[0] + p[1]*(x-p[3])
        return line(wave_s, *popt)

    @property
    def minimum(self):
        if not self._minimum:
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
