# Spectral Lines
Tools for measuring spectral features with a variety of smoothing and modeling methods.

## Methods
The base model defines the wavelength ranges of each of the features, functions for moving between wavelength and velocity space (via the relativistic Doppler effect) and gives two options for normalizing the spectra overall (though some measurement methods further refine the normalization in defining the pseudo-continuum).
* `norm='SNID'` (default) defines a 13-point spline from the entire spectrum and normalizes by this spline (see [Blondin & Tonry 2007](https://ui.adsabs.harvard.edu/abs/2007ApJ...666.1024B/abstract))
* `norm='None'` does no normalization
* `norm='line'` defines a rough pseudo-continuum of the feature by finding local maxima of the raw spectrum in the pre-defined regions at the edge of the feature and dividing out the line connecting these local maxima.

The following objects all inherit from this base measurement class to include additional functionality. Each of these techniques has a particular implementation of the following functions:
* `get_smoothed_feature_spec`
* `get_interp_feature_spec`
* `get_pseudo_continuum`
* `get_extrema`
* `get_line_velocity`
* `get_equiv_width`

### Spline
This method uses an inverse-Gaussian weighted spline to smooth and interpolate the raw spectrum.

### Gaussian process
This method uses a Gaussian process with a Matern kernel to smooth the raw spectrum. The GP is implemented with `scikit-learn`.

### Savitsky-Golay smoothing

### Gaussian

### Gaussian doublet

## Installation
Eventually this will be installable via `pip`. For now, clone this repository, create a conda environment with `conda env create -f environment.yml`, activate this environemnt, and run `python setup.py install`. 
