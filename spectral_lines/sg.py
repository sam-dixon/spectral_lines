import numpy as np
from .base import Measure
from scipy.signal import savgol_filter


class SG(Measure):

    def __init__(self, spectrum, line='SiII6355', interp_grid=0.1, sim=False, norm=True):
        super(SG, self).__init__(spectrum, line, interp_grid, sim, norm)

    def weight_matrix_corr(var, corr):
        # computation of the weight matrix :
        V = np.diag([1.]*len(var))
        V += np.diag([corr]*(len(var)-1), 1)
        V += np.diag([corr]*(len(var)-1), -1)
        W = np.linalg.inv(V*np.transpose([np.sqrt(var)])*np.sqrt(var))
        return W

    def sg_coeff(num_points, pol_degree, diff_order=0):
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
        coeff = np.dot(Am, np.transpose(A))[diff_order]*np.factorial(diff_order)
        return coeff

    def B_matrix_sg(self, num_points, pol_degree, dim):
        coeffs = self.sg_coeff(num_points, pol_degree)
        B = np.diag([coeffs[num_points]]*dim)
        for i in xrange(num_points):
            B += np.diag([coeffs[num_points-1-i]]*(dim-1-i), i+1)
            B += np.diag([coeffs[num_points-1-i]]*(dim-1-i), -i-1)
        for i in xrange(dim):
            B[i, :] /= sum(B[i, :])
        return B

    def prediction_error(self, r, B, var, W=None, verbose=False):
        """ r is the vector of residuals
        B is the B matrix (see above)
        furthemore, we assume the error is pure variance (var)
        r is the residual vector """
        # simplification because we have no covariance included
        if W is None:
            pe = np.sum(r**2/var) + 2 * np.sum(np.diag(B)) - len(var)
            if pe < 0 and verbose:
                print("WARNING <prediction_error>: pe < 0, variance probably under estimated")
        else:
            pe = np.dot(np.dot(r, W), r) + 2 * np.sum(np.diag(B)) - len(var)
            if pe < 0 and verbose:
                print("WARNING <prediction_error>: pe < 0, variance probably under estimated")
        return pe

    def sg_find_num_points(self, x, data, var, pol_degree=2, corr=0.0, verbose=False):
        """
        Returns the size of the halfwindow for the best savitzky-Golay approximation.
        """

        def n(i_iteration):
            return (i_iteration-1)+pol_degree/2

        W = self.weight_matrix_corr(var, corr)
        e = {}
        finished = False
        i_iteration = 1
        #- coarse exploration to estimate the number of points
        #- yielding the best smoothing, i.e. yielding the lower "prediction_error"
        #- defined as the error made by the smoothing fonction approximating the data
        #- taking into account the variance of the signal.
        while not finished:
            num_points = n(i_iteration)
            B = self.B_matrix_sg(num_points, pol_degree, len(x))
            yy = np.dot(B, data)
            pe = self.prediction_error(yy-data, B, var, W)
            if verbose:
                print('%d, %f' % (num_points, pe))
            e[num_points] = pe
            if (n(i_iteration*2) > len(x)):
                #- Test to prevent the coarse exploration to end up testing
                #- number of points larger than the total number of data points.
                #- The i_iteration *=2 takes into account that the next finer exporation expects
                #- the coarse exploration to have ended one step *after* the inflection point.
                i_iteration *= 2
                B = self.B_matrix_sg(len(x), pol_degree, len(x))
                yy = np.dot(B, data)
                pe = self.prediction_error(yy-data, B, var, W)
                e[len(x)] = pe
                finished = True
            elif ((pe > np.min(e.values()) and np.min(e.values()) < len(x)*0.9999)):
                #- Test to stop when the prediction error stops decreasing and starts increasing again.
                #- Under the assumption that the prediction error is convex and starts by decreasing, this means
                #- that the inflection point happened just before this step.
                finished = True
            else:
                i_iteration *= 2
        if i_iteration < 3:
            return n(i_iteration)
        #- In the case where the previous exploration was stopped because n(i_iteration) > len(x)
        #- the last key of e is not n(i_iteration) but len(x)
        toler = np.max([e[n(i_iteration/4)]-e[n(i_iteration/2)], e[min(len(x), n(i_iteration))]-e[n(i_iteration/2)]])/2
        #- Finer exploration of the region between where we know the prediction error was decreasing and
        #- either where it was increasing again or the total number of data points
        for num_points in np.arange(n(i_iteration/4), min(n(i_iteration), len(x)), max([1, n(i_iteration/4)/10])):
            B = self.B_matrix_sg(num_points, pol_degree, len(x))
            yy = np.dot(B, data)
            pe = self.prediction_error(yy-data, B, var, W)
            e[num_points] = pe
            if verbose:
                print('%d, %f' % (num_points, pe))
            if num_points > n(i_iteration/2) and (pe-np.min(e.values()) > toler):
                break

        result = e.keys()[e.values().index(np.min(e.values()))]

        if result >= len(x):
            #- This is to avoid the always problematic limit case of calculating an interpolator on all the points available (for example, in the
            #- Savitzky-Golay interpolation (ToolBox.Signal), when all the available points are selected, the convolution by the
            #- kernel chops out the two extreme points).
            result = len(x)-1
        return result

    def get_smoothed_feature_spec(self, hsize=None, order=2, rho=0.0, verbose=False):
        """Use savitzky_golay() to apply a savitzky golay filter on the
        spectrum (from pySnurp)
        Input:
        - hsize : half size of the window (default:15)
        - order : order of the polynome used to smooth the spectrum (default:2)
        Output:
        - self.s : smoothing spectrum
        """
        rc = 1. - 2. * (rho**2)
        if hsize is None:
            try:
                hsize = int(self.sg_find_num_points(self.wave_feat,
                                                    self.flux_feat,
                                                    self.var_feat * rc,
                                                    corr=(rho**2) / rc))
            except:
                if verbose:
                    print('ERROR in computing of best hsize')
                hsize = 15
        if (hsize * 2) + 1 < (order + 2):
            hsize = 10  # order/2.+1
        if verbose:
            print('best_w=%i' % hsize)
        s = savgol_filter(self.flux_feat,
                          window_length=(int(hsize) * 2) + 1,
                          polyorder=order,
                          deriv=0)
        self.hsize = hsize
        self.order = order
        return self.wave_feat, s

    def get_line_velocity(self):
        w, f = self.get_smoothed_feature_spec()
        w_abs = w[np.where(f == np.min(f))][0]
        v_abs = self.vel_space(w_abs)
        w_em = w[np.where(f == np.max(f))][0]
        v_em = self.vel_space(w_em)
        return v_abs, v_em
