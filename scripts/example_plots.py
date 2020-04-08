import IDRTools
import numpy as np
import matplotlib.pyplot as plt
from spectral_lines import *

DS = IDRTools.Dataset(subset=['training', 'validation'])
methods = {'spline': Spl,
           'SG filter': SG,
           'gaussian': Gauss,
           'doublet': Doublet}

def plot_norm_spec(meas):
    """Plots the normalized spectrum of the given Measurement object"""
    label = '{} {}'.format(meas.kind, meas.norm).title()
    plt.plot(meas.wave_sn, meas.flux_sn, label=label)
    plt.ylim(0, 3)


def plot_smoothed_spec(meas):
    """Plots the smoothed/modeled spectrum of the given Measurement object"""
    label = '{}_{}'.format(meas.kind, meas.norm).title()
    wave, flux = meas.get_smoothed_feature_spec()
    plt.plot(wave, flux, label=label)
    plt.ylim(0, 3)


def plot_interp_spec(meas):
    """Plots the smoothed/modeled interpolated spectrum of the given
    Measurement object
    """
    label = '{}_{}'.format(meas.kind, meas.norm).title()
    wave, flux = meas.get_interp_feature_spec()
    plt.plot(wave, flux, label=label)
    plt.ylim(0, 3)


def plot_pseudo_continuum(meas):
    """Plots the pseudo continuum determined by the method"""
    label = '{}_{}'.format(meas.kind, meas.norm).title()
    wave, flux = meas.get_interp_feature_spec()
    pc_flux = meas.get_pseudo_continuum()
    plt.plot(wave, flux, label='{} smoothed'.format(label))
    plt.plot(wave, pc_flux, label='{} PC'.format(label))
    plt.ylim(0, 3)


def plot_extrema(meas, **kwargs):
    """Plots the extrema of the spectral feature"""
    wave, flux = meas.get_interp_feature_spec()
    label = '{}_{}'.format(meas.kind, meas.norm).title()
    plt.plot(wave, flux, label=label, **kwargs)
    plt.axvline(meas.minimum, **kwargs)
    plt.axvline(meas.maxima['blue_max_wave'], **kwargs)
    plt.axvline(meas.maxima['red_max_wave'], **kwargs)
    plt.ylim(0, 3)


def main():
    # Set up data set
    # TODO: move the data to a separate data folder
    sn = np.random.choice(DS.sne)
    spec = sn.spec_nearest(phase=0)
    print(spec.target_name, 'phase: {:+0.3f} days'.format(spec.salt2_phase))

    # Initialize measurement objects
    objects = {}
    for method in methods.keys():
        for norm in ['SNID']:
            key = '{}_{}'.format(method, norm)
            objects[key] = methods[method](spec, norm=norm)

    # Check normalization
    plot_norm_spec(objects['spline_SNID'])
    plot_norm_spec(objects['spline_line'])
    plt.legend()
    plt.title('Normalization {}'.format(sn.target_name))
    plt.show()

    # Check smoothed spectra
    for method, obj in objects.items():
        plot_smoothed_spec(obj)
    plt.legend(ncol=2)
    plt.show()

    # Check pseudo-continuum
    for method, obj in objects.items():
        plot_pseudo_continuum(obj)
    plt.legend(ncol=3)
    plt.show()

    # Check extrema
    for i, (method, obj) in enumerate(objects.items()):
        plot_extrema(obj, color='C{}'.format(i%10))
    plt.legend(ncol=2)
    plt.show()


if __name__ == '__main__':
    main()
