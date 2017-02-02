import IDRTools
import matplotlib.pyplot as plt
from spectral_lines import *


if __name__ == '__main__':
    ds = IDRTools.Dataset(subset=['training', 'validation'])
    sn = ds.PTF09dnl
    spec = sn.spec_nearest_max(phase=6)
    print(spec.salt2_phase)
    spl_snid = Spl(spec, norm='SNID')
    spl_line = Spl(spec, norm='line')
    doublet_snid = Doublet(spec, norm='SNID')
    doublet_line = Doublet(spec, norm='line')
    free_doublet_snid = FreeDoublet(spec, norm='SNID')
    free_doublet_line = FreeDoublet(spec, norm='line')
    gauss_snid = Gauss(spec, norm='SNID')
    gauss_line = Gauss(spec, norm='line')
    gp_snid = GP(spec, norm='SNID')
    gp_line = GP(spec, norm='line')
    sg_snid = SG(spec, norm='SNID')
    sg_line = SG(spec, norm='line')

    SNID_fig = plt.figure(figsize=(10, 4))
    plt.plot(*spl_snid.get_feature_spec()[:2], label='Original', color='k', alpha=0.3)
    plt.plot(*spl_snid.get_interp_feature_spec()[:2], label='Spline')
    plt.plot(*doublet_snid.get_smoothed_feature_spec()[:2], label='Doublet')
    plt.plot(*free_doublet_snid.get_smoothed_feature_spec()[:2], label='Double gaussian')
    plt.plot(*gauss_snid.get_smoothed_feature_spec()[:2], label='Gaussian')
    plt.plot(*gp_snid.get_smoothed_feature_spec()[:2], label='Gaussian process')
    plt.plot(*sg_snid.get_smoothed_feature_spec()[:2], label='Savitsky-Golay filter')
    plt.title('PTF09dnl')
    plt.legend()
    plt.show()

    line_fig = plt.figure(figsize=(10, 4))
    plt.plot(*spl_line.get_feature_spec()[:2], label='Original', color='k', alpha=0.3)
    plt.plot(*spl_line.get_interp_feature_spec()[:2], label='Spline')
    plt.plot(*doublet_line.get_smoothed_feature_spec()[:2], label='Doublet')
    plt.plot(*free_doublet_line.get_smoothed_feature_spec()[:2], label='Double gaussian')
    plt.plot(*gauss_line.get_smoothed_feature_spec()[:2], label='Gaussian')
    plt.plot(*gp_line.get_smoothed_feature_spec()[:2], label='Gaussian process')
    plt.plot(*sg_line.get_smoothed_feature_spec()[:2], label='Savitsky-Golay filter')
    plt.title('PTF09dnl')
    plt.legend()
    plt.show()