import numpy as np
from .base import Measure
from iminuit import Minuit


class Gauss(Measure):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False, norm=True):
        super(Gauss, self).__init__(spectrum, line, interp_grid, sim, norm)

    def get_interp_feature_spec(self, return_popt=False):
        w, f, e = self.wave_subfeat, self.flux_subfeat, self.var_subfeat
        g = lambda x, *p: p[0]+p[1]*(x-p[3])+p[2]*np.exp(-(x-p[3])**2/(2*p[4]**2))
        def chisq(a, b, amp, mu, sig):
            return np.sum((f-g(w, a, b, amp, mu, sig))**2/e)
        m = Minuit(chisq,
                   a=1, limit_a=(0, 2),
                   b=0, limit_b=(-0.1, 0.1),
                   amp=-0.5, limit_amp=(-1, 0),
                   mu=6150, limit_mu=(min(w), max(w)),
                   sig=50,
                   pedantic=False,
                   print_level=0)
        fval, res = m.migrad()
        popt = [r['value'] for r in res]
        w_s = np.arange(np.min(self.wave_feat), np.max(self.wave_feat), self.interp_grid)
        if return_popt:
            return w_s, g(w_s, *popt), popt
        return w_s, g(w_s, *popt)
    
    def get_pseudo_continuum(self):
        w_s, g, popt = self.get_interp_feature_spec(return_popt=True)
        line = lambda x, *p: p[0] + p[1]*x
        return line(w_s, *popt[:2])
    
    def get_line_velocity(self):
        """
        Find the velocity of the line (defined to be the velocity that Doppler
        shifts l_0 to the minimum of the smoothed, interpolated spectrum)
        """
        w, f = self.get_interp_feature_spec()
        f_pc = self.get_pseudo_continuum()
        f /= f_pc
        search_range = (w >= self.l_brange[-1]) & (w <= self.l_rrange[0])
        min_f = np.min(f[search_range])
        w_abs = w[np.where(f == min_f)][0]
        v_abs = self.vel_space(w_abs)
        return v_abs
    
    def get_equiv_width(self):
        w, f = self.get_interp_feature_spec()
        f_cont = self.get_pseudo_continuum()
        return np.dot(1-(f/f_cont)[:-1], np.diff(w))
        