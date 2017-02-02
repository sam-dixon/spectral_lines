import numpy as np
from base import Measure
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class GP(Measure):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False, norm=True):
        super(GP, self).__init__(spectrum, line, interp_grid, sim, norm)
        self.kernel = 0.1**2 * Matern(length_scale=100)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, optimizer=None)

    def get_smoothed_feature_spec(self):
        """
        Returns the GP smoothed feature spectrum.
        """
        w, f, e = self.wave_feat, self.flux_feat-1, self.var_feat
        self.gp.set_params(alpha=e/f**2)
        self.gp.fit(np.atleast_2d(w).T, f)
        w_s = np.arange(np.min(w), np.max(w), self.interp_grid)
        f_s = self.gp.predict(np.atleast_2d(w_s).T)
        return w_s, f_s+1

    def get_line_velocity(self):
        w, f = self.get_smoothed_feature_spec()
        w_abs = w[np.where(f == np.min(f))][0]
        v_abs = self.vel_space(w_abs)
        w_em = w[np.where(f == np.max(f))][0]
        v_em = self.vel_space(w_em)
        return v_abs, v_em