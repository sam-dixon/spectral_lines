import numpy as np
from .base import Measure
from scipy.optimize import curve_fit


class FreeDoublet(Measure):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False, norm=True):
        super(FreeDoublet, self).__init__(spectrum, line, interp_grid, sim, norm)

    def get_smoothed_feature_spec(self, return_popt=False):
        w, f, e = self.wave_feat, self.flux_feat, self.var_feat
        g = lambda x, *p: p[0]*np.exp(-(x-p[1])**2./(2.*p[2]**2.))
        tg = lambda x, *p: p[0]+p[1]*x+g(x, *p[2:5])+g(x, *p[5:])
        p0 = [1.0, 0.0,
              -0.5, 6150., 40.,
              -0.5, 6125., 40.]
        popt, pcov = curve_fit(tg, w, f, p0=p0, sigma=e, maxfev=10000)
        w_s = np.arange(np.min(w), np.max(w), self.interp_grid)
        if return_popt:
            return w_s, tg(w_s, *popt), popt
        return w_s, tg(w_s, *popt)

    def get_line_velocity(self):
        w, f = self.get_smoothed_feature_spec()
        w_abs = w[np.where(f == np.min(f))][0]
        v_abs = self.vel_space(w_abs)
        w_em = w[np.where(f == np.max(f))][0]
        v_em = self.vel_space(w_em)
        return v_abs, v_em


class Doublet(Measure):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False, norm=True):
        super(Doublet, self).__init__(spectrum, line, interp_grid, sim, norm)

    def get_smoothed_feature_spec(self, return_popt=False):
        w, f, e = self.wave_feat, self.flux_feat, self.var_feat
        g = lambda x, *p: p[0]*np.exp(-(x-p[1])**2./(2.*p[2]**2.))
        tg = lambda x, *p: p[0] + p[1]*x + g(x, *p[2:5]) + g(x, p[2], *p[5:])
        p0 = [1.0, 0.0, -0.5,
              6150., 40.,
              6125., 40.]
        popt, pcov = curve_fit(tg, w, f, p0=p0, sigma=e, maxfev=10000)
        w_s = np.arange(np.min(w), np.max(w), self.interp_grid)
        if return_popt:
            return w_s, tg(w_s, *popt), popt
        return w_s, tg(w_s, *popt)

    def get_line_velocity(self):
        w, f = self.get_smoothed_feature_spec()
        w_abs = w[np.where(f == np.min(f))][0]
        v_abs = self.vel_space(w_abs)
        w_em = w[np.where(f == np.max(f))][0]
        v_em = self.vel_space(w_em)
        return v_abs, v_em

    def get_delta_lambda(self):
        _, _, popt = self.get_smoothed_feature_spec(return_popt=True)
        return popt[3]-popt[5]
