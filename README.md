# Spectral Lines
Tools for measuring spectral features with a variety of smoothing and modeling methods.

## Methods
The spectral features are defined in 10 different zones.

| Feature name | lambda | Blue lambda_min | Blue lambda_max | Red lambda_min | Red lambda_max |
|--------------|:------:|:---------------:|:---------------:|:--------------:|:--------------:|
CaIIHK         | 3945   | 3504            | 3687            | 3887           | 3990           |
SiII4000       | 4128   | 3830            | 3963            | 4034           | 4150           |
MgII           | 4481   | 4034            | 4150            | 4452           | 4573           |
Fe4800         | 4966   | 4400            | 4650            | 5050           | 5300           |
SIIW_L         | 5454   | 5085            | 5250            | 5250           | 5450           |
SIIW_R         | 5640   | 5250            | 5450            | 5500           | 5681           |
SIIW           | 5500   | 5085            | 5250            | 5500           | 5681           |
SiII5972       | 5972   | 5550            | 5681            | 5850           | 6015           |
SiII6355       | 6355   | 5850            | 6015            | 6250           | 6365           |
OI7773         | 8100   | 7100            | 7270            | 7720           | 8000           |

5 of these zones have specific subregions used to search for minima, which later define the ejecta velocity measured in these zones:

| Feature name | lambda_min | lambda_max |
|--------------|:----------:|:----------:|
SiII4000       |3963        |4034        |
SIIW_L         |5200        |5350        |
SIIW_R         |5351        |5550        |
SiII5972       |5700        |5900        |
SiII6355       |6000        |6210        |

The base class (`Measure`) includes helper functions for moving between wavelength and velocity space (via the relativistic Doppler effect) and gives two options for normalizing a given spectrum:
* `norm='SNID'` (default) defines a 13-point spline from the entire spectrum and normalizes by this spline (see [Blondin & Tonry 2007](https://ui.adsabs.harvard.edu/abs/2007ApJ...666.1024B/abstract))
* `norm='None'` does no normalization

This base class also has methods for finding local extrema once the spectrum is smoothed, using these extrema to define a pseudo-continuum, and finally calculating line velocities and equivalent widths.

**Note**: The velocity calculation in this code uses the relativistic Doppler equation, unlike most spectral feature measurements (including the SNfactory pipeline).

The remaining objects all inherit from this base measurement class and  implement particular smoothing methods. The interface is through:
* `get_smoothed_feature_spec`: returns the flux of the feature smoothed by the method. The wavelengths that the flux model is evaluated at are the same as the input spectrum
* `get_interp_feature_spec`: same as `get_smoothed_feature_spec` but interpolated on a finer wavelength grid (determined by the `interp_grid` parameter)

### Spline (`spline.py`)
An inverse-Gaussian weighted spline is used to smooth and interpolate the raw spectrum. This is the method presented in [Blondin et al. 2006](https://ui.adsabs.harvard.edu/abs/2006AJ....131.1648B/abstract). First, we define a Gaussian window with a width corresponding to the approximate scale of Doppler broadening of the spectral lines at each wavelength. This Gaussian filter is then weighted by the inverse of the variance of the observed spectrum, and then applied to the spectrum signal.

This method requires an accurate variance spectrum to be provided. If the variance is overestimated, the resulting smoothed spectrum will be oversmoothed, which can bias some of the spectral feature measurements. 

### Savitsky-Golay smoothing (`sg.py`)
This is the method laid out in Chotard et al. (in prep)., and matches the behavior of the code in the SNfactory pipeline. This method uses a [Savitsky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) with a smoothing window width determined optimally (Appendix A of the paper).

### Gaussian and Gaussian doublet (`gauss.py` and `doublet.py`)
These use simple maximum likelihood estimation to fit a Gaussian/double Gaussian profile to the zone.

## Scripts and notebooks
Each of these scripts makes use of the internal release of the Supernova Factory data (and related [IDRTools package](https://github.com/sam-dixon/snfidrtools)). `example_plots.py` can be easily modified to use other data sets.
* `example_plots.py` contains functions to make sample plots comparing the different smoothing/normalization methods.
* `compare_snf.py` calculates all measurements with errors for all at-max spectra in the SNfactory data set and compiles the same measurements in the output of the SNfactory pipeline code.
* `debug_plots.ipynb` contains some code to visualize the differences between the SNfactory pipeline measurement codes and this code based on the results of `compare_snf.py`.
* `method_comparison.ipynb` contains code to directly compare the results of the SNf pipeline measurement to this code.

## Installation
1. Clone this repository
```git clone https://github.com/sam-dixon/spectral_lines.git```

2. Create a conda environment. 
```conda env create -f environment.yml```

3. Activate this environment
```conda activate spectral_lines```

4. Install
```pip install .```
