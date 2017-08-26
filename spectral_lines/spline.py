import numpy as np
from .base import Measure
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


class Spl(Measure):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False,
                 norm='SNID', n_l=30, smooth_fac=0.005):
        super(Spl, self).__init__(spectrum, line, interp_grid, sim, norm)
        self.n_l = n_l
        self.smooth_fac = smooth_fac

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
        return wave[int(self.n_l/2):int(-self.n_l/2)], np.array(f_ts)

    def get_interp_feature_spec(self, return_spl=False):
        """
        Returns the spline interpolated, smoothed feature spectrum
        """
        w, f = self.get_smoothed_feature_spec()
        spl = UnivariateSpline(w, f, k=4, s=0)
        n_pts = (max(w)-min(w))/self.interp_grid + 1
        w_int = np.linspace(min(w), max(w), n_pts)
        f_int = spl(w_int)
        if return_spl:
            return w_int, f_int, spl
        else:
            return w_int, f_int

    def get_line_velocity(self):
        """
        Find the velocity of the line (defined to be the velocity that Doppler
        shifts l_0 to the minimum of the smoothed, interpolated spectrum)
        """
        w, f = self.get_interp_feature_spec()
        w_abs = w[np.where(f == np.min(f))][0]
        v_abs = self.vel_space(w_abs)
        w_em = w[np.where(f == np.max(f))][0]
        v_em = self.vel_space(w_em)
        return v_abs, v_em
