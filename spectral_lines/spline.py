import numpy as np
from .base import Measure
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


class Spl(Measure):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False,
                 norm='SNID', n_l=16, smooth_fac=0.005):
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
        n_pts = int((max(w)-min(w))/self.interp_grid + 1)
        w_int = np.linspace(min(w), max(w), n_pts)
        f_int = spl(w_int)
        if return_spl:
            return w_int, f_int, spl
        else:
            return w_int, f_int
        
    def get_pseudo_continuum(self, return_extrema=False):
        """Returns the line connecting the ends of the feature
        """
        w, f = self.get_interp_feature_spec()
        bw_cut = (w >= self.l_brange[0]) & (w <= self.l_brange[1])
        rw_cut = (w >= self.l_rrange[0]) & (w <= self.l_rrange[1])
        max_b_ind = np.where(f[bw_cut]==max(f[bw_cut]))[0]
        max_r_ind = np.where(f[rw_cut]==max(f[rw_cut]))[0]
        fr = f[rw_cut][max_r_ind]
        fb = f[bw_cut][max_b_ind]
        wr = w[rw_cut][max_r_ind]
        wb = w[bw_cut][max_b_ind]
        slope = (fr-fb)/(wr-wb)
        inter = f[rw_cut][max_r_ind] - slope*w[rw_cut][max_r_ind]
        if return_extrema:
            return inter + slope * w, [wb, wr, fb, fr]
        return inter + slope * w

    def get_line_velocity(self):
        """
        Find the velocity of the line (defined to be the velocity that Doppler
        shifts l_0 to the minimum of the smoothed, interpolated spectrum)
        """
        w, f = self.get_interp_feature_spec()
        f_cont = self.get_pseudo_continuum()
        f /= f_cont
        search_range = (w >= self.l_brange[-1]) & (w <= self.l_rrange[0])
        min_f = np.min(f[search_range])
        w_abs = w[np.where(f == min_f)][0]
        v_abs = self.vel_space(w_abs)
        return v_abs

    def get_equiv_width(self):
        w, f = self.get_interp_feature_spec()
        f_c, extrema = self.get_pseudo_continuum(return_extrema=True)
        wb, wr, _, _ = extrema
        integrand = 1 - f/f_c
        dl = np.diff(w)
        wb_ind = np.where(w==wb)[0][0]
        wr_ind = np.where(w==wr)[0][0]
        return np.sum(np.dot(integrand[wb_ind:wr_ind], dl[wb_ind:wr_ind]))