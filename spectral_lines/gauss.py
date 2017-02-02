import numpy as np
from base import Measure
from scipy.optimize import curve_fit


class Gauss(Measure):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False, norm=True):
        super(Gauss, self).__init__(spectrum, line, interp_grid, sim, norm)

    def get_smoothed_feature_spec(self, return_popt=False):
        w, f, e = self.wave_feat, self.flux_feat, self.var_feat
        g = lambda x, *p: p[0]+p[1]*x+p[2]*np.exp(-(x-p[3])**2/(2*p[4]**2))
        p0 = [1., -0.1, -0.5, np.mean(w), 50.]
        popt, pcov = curve_fit(g, w, f, p0=p0, sigma=e)
        w_s = np.arange(np.min(w), np.max(w), self.interp_grid)
        if return_popt:
            return w_s, g(w_s, *popt), popt
        return w_s, g(w_s, *popt)

    def get_line_velocity(self):
        w, f, popt = self.get_smoothed_feature_spec(return_popt=True)
        w_abs = w[np.where(f == np.min(f))][0]
        v_abs = self.vel_space(w_abs)
        w_em = w[np.where(f == np.max(f))][0]
        v_em = self.vel_space(w_em)
        return v_abs, v_em
