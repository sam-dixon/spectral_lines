import numpy as np
from .base import Measure
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline


class SG(Measure):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False,
                 norm=True, verbose=False, hsize=None):
        super(SG, self).__init__(spectrum, line, interp_grid, sim, norm)
        self.verbose = verbose
        self.hsize = hsize
        self.kind = 'SG'
        self._maxima = None

    def weight_matrix_corr(self, var, corr):
        # computation of the weight matrix :
        V = np.diag([1.]*len(var))
        V += np.diag([corr]*(len(var)-1), 1)
        V += np.diag([corr]*(len(var)-1), -1)
        W = np.linalg.inv(V*np.transpose([np.sqrt(var)])*np.sqrt(var))
        return W

    def sg_coeff(self, num_points, pol_degree, diff_order=0):
        """ calculates filter coefficients for symmetric savitzky-golay filter.
            see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf

            num_points   means that 2*num_points+1 values contribute to the
                         smoother.
            pol_degree   is degree of fitting polynomial
            diff_order   is degree of implicit differentiation.
                         0 means that filter results in smoothing of function
                         1 means that filter results in smoothing the first
                                                     derivative of function.
                         and so on ...
        """

        """ uses a slow but sure algorithm """
        # setup normal matrix
        r = np.arange(-num_points, num_points+1, dtype=float)
        A = np.array([r**i for i in range(pol_degree+1)]).transpose()
        # calculate diff_order-th row of inv(A^T A)
        ATA = np.dot(A.transpose(), A)
        Am = np.linalg.inv(ATA)
        # calculate filter-coefficients
        coeff = np.dot(Am, np.transpose(A))[diff_order]*np.math.factorial(diff_order)
        return coeff

    def B_matrix_sg(self, num_points, pol_degree, dim):
        coeffs = self.sg_coeff(num_points, pol_degree)
        B = np.diag([coeffs[num_points]]*dim)
        for i in range(num_points):
            B += np.diag([coeffs[num_points-1-i]]*(dim-1-i), i+1)
            B += np.diag([coeffs[num_points-1-i]]*(dim-1-i), -i-1)
        for i in range(dim):
            B[i, :] /= sum(B[i, :])
        return B

    def prediction_error(self, r, B, var, W=None):
        """ r is the vector of residuals
        B is the B matrix (see above)
        furthemore, we assume the error is pure variance (var)
        r is the residual vector """
        # simplification because we have no covariance included
        if W is None:
            pe = np.sum(r**2/var) + 2 * np.sum(np.diag(B)) - len(var)
            if pe < 0 and self.verbose:
                print("WARNING <prediction_error>: pe < 0, variance probably under estimated")
        else:
            pe = np.dot(np.dot(r, W), r) + 2 * np.sum(np.diag(B)) - len(var)
            if pe < 0 and self.verbose:
                print("WARNING <prediction_error>: pe < 0, variance probably under estimated")
        return pe
    
    def sg_find_num_points(self, x, data, var, pol_degree=2, corr=0.):
        W = self.weight_matrix_corr(var, corr)
        e = {}
        for num_pts in range(1, min(int(len(x)/2), 33), 2):
            B = self.B_matrix_sg(num_pts, pol_degree, len(x))
            yy = np.dot(B, data)
            pe = self.prediction_error(yy-data, B, var, W)
            e[num_pts] = pe
        if self.verbose:
            print(e)
        return [k for k, v in e.items() if v==min(e.values())][0]

    def get_smoothed_feature_spec(self, order=2, rho=0.0, return_deriv=False):
        """Use savitzky_golay() to apply a savitzky golay filter on the
        spectrum (from pySnurp)
        Input:
        - hsize : half size of the window (default:15)
        - order : order of the polynome used to smooth the spectrum (default:2)
        Output:
        - self.s : smoothing spectrum
        """
        rc = 1. - 2. * (rho**2)
        hsize = self.hsize
        if hsize is None:
            hsize = int(self.sg_find_num_points(self.wave_feat,
                                                self.flux_feat,
                                                self.var_feat * rc,
                                                corr=(rho**2) / rc))
        if (hsize * 2) + 1 < (order + 2):
            hsize = order/2. + 1
        if self.verbose:
            print('best_w=%i' % hsize)
        s = savgol_filter(self.flux_feat,
                          window_length=(int(hsize) * 2) + 1,
                          polyorder=order,
                          deriv=0)
        self.hsize = hsize
        if return_deriv:
            s_deriv = savgol_filter(self.flux_feat,
                                    window_length=(int(hsize) * 2) + 1,
                                    polyorder=order,
                                    deriv=1)
            return self.wave_feat, s, s_deriv
        return self.wave_feat, s
    
    def get_interp_feature_spec(self, return_deriv=False):
        """Interpolate between smoothed feature spec
        """
        if return_deriv:
            w, f, d = self.get_smoothed_feature_spec(return_deriv=True)
            spl_f = UnivariateSpline(w, f, k=3, s=0)
            spl_d = UnivariateSpline(w, d, k=3, s=0)
            w_int = np.arange(min(w), max(w), self.interp_grid)
            f_int = spl_f(w_int)
            d_int = spl_d(w_int)
            return w_int, f_int, d_int
        else:
            w, f = self.get_smoothed_feature_spec()
            spl = UnivariateSpline(w, f, k=3, s=0)
            w_int = np.arange(min(w), max(w), self.interp_grid)
            f_int = spl(w_int)
            return w_int, f_int

    @property
    def maxima(self):
        if not self._maxima:
            w, f = self.get_interp_feature_spec()
            bw_cut = (w >= self.l_brange[0]) & (w <= self.l_brange[1])
            rw_cut = (w >= self.l_rrange[0]) & (w <= self.l_rrange[1])
            max_b_ind = np.where(f[bw_cut] == max(f[bw_cut]))[0]
            max_r_ind = np.where(f[rw_cut] == max(f[rw_cut]))[0]
            maxima = {'blue_max_wave': w[bw_cut][max_b_ind],
                      'blue_max_flux': f[bw_cut][max_b_ind],
                      'red_max_wave': w[rw_cut][max_r_ind],
                      'red_max_flux': f[rw_cut][max_r_ind]}
            self._maxima = maxima
        return self._maxima

    def get_pseudo_continuum(self):
        """Returns the line connecting the ends of the feature
        """
        wave, _ = self.get_interp_feature_spec()
        flux_diff = self.maxima['red_max_flux'] - self.maxima['blue_max_flux']
        wave_diff = self.maxima['red_max_wave'] - self.maxima['blue_max_wave']
        slope = flux_diff / wave_diff
        inter = self.maxima['red_max_flux'] - slope * self.maxima[
            'red_max_wave']
        return inter + slope * wave

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
        w, f = self.get_interp_feature_spec()
        f_c = self.get_pseudo_continuum()
        integrand = 1 - f / f_c
        dl = np.diff(w)
        wb_ind = np.where(w == self.maxima['blue_max_wave'])[0][0]
        wr_ind = np.where(w == self.maxima['red_max_wave'])[0][0]
        return np.sum(np.dot(integrand[wb_ind:wr_ind], dl[wb_ind:wr_ind]))

