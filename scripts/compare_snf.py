"""Compare measurements between SNfactory pipeline and this code"""
import os
import click
import pickle
import IDRTools
import spectral_lines
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


SNF_PATH = '/Users/samdixon/data/BLACKSTON_spec_feat_py3.pkl'
with open(SNF_PATH, 'rb') as f:
    SNF = pickle.load(f)
DS = IDRTools.Dataset(subset=['training', 'validation'],
                      data_dir='/Users/samdixon/data/BLACKSTONE')

LINE_NAMES = spectral_lines.base.line_names
V_LINE_NAME_MAP = {'SiII4000': 'SiII_4128',
                   'SIIW_L': 'SiII_5454',
                   'SIIW_R': 'SiII_5640',
                   'SiII5972': 'SiII_5972',
                   'SiII6355': 'SiII_6355'}

DATA_DIR = 'data'
if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)


def measure_all_spec_lines(spec):
    """Measures all lines of a given spectrum without simulating errors"""
    snf_spec = SNF[spec.target_name]['spectra'][spec.obs_exp]
    spec_spl_data = {}
    spec_snf_data = {}
    for line_name in LINE_NAMES:
        try:
            spl = spectral_lines.Spl(spec, line=line_name)
            spec_spl_data['EW{}'.format(line_name)] = spl.equiv_width
            spec_spl_data['EW{}_b_lbd'.format(line_name)] = spl.maxima['blue_max_wave']
            spec_spl_data['EW{}_r_lbd'.format(line_name)] = spl.maxima['red_max_wave']
            if line_name in V_LINE_NAME_MAP.keys():
                spec_spl_data['v{}'.format(line_name)] = spl.velocity
                spec_spl_data['v{}_lbd'.format(line_name)] = spl.minimum
        except (ValueError, spectral_lines.MissingDataError):
            spec_spl_data['EW{}'.format(line_name)] = np.nan
            spec_spl_data['EW{}_b_lbd'.format(line_name)] = np.nan
            spec_spl_data['EW{}_r_lbd'.format(line_name)] = np.nan
            if line_name in V_LINE_NAME_MAP.keys():
                spec_spl_data['v{}'.format(line_name)] = np.nan
                spec_spl_data['v{}_lbd'.format(line_name)] = np.nan

        ew_snf = snf_spec['phrenology.EW{}'.format(line_name)]
        ew_snf_b_lbd, ew_snf_r_lbd = snf_spec['phrenology.lbd_EW{}'.format(line_name)]
        spec_snf_data['EW{}'.format(line_name)] = ew_snf
        spec_snf_data['EW{}_b_lbd'.format(line_name)] = ew_snf_b_lbd
        spec_snf_data['EW{}_r_lbd'.format(line_name)] = ew_snf_r_lbd
        if line_name in V_LINE_NAME_MAP.keys():
            v_key_name = V_LINE_NAME_MAP[line_name]
            v_snf = snf_spec['phrenology.v{}'.format(v_key_name)]
            l_snf = snf_spec['phrenology.v{}_lbd'.format(v_key_name)]
            spec_snf_data['v{}'.format(line_name)] = v_snf
            spec_snf_data['v{}_lbd'.format(line_name)] = l_snf
    return spec_spl_data, spec_snf_data


def measure_all_spec_lines_with_errors(spec):
    """Measures all lines of a given spectrum with Monte Carlo errors"""
    snf_spec = SNF[spec.target_name]['spectra'][spec.obs_exp]
    spec_spl_data = {}
    spec_snf_data = {}
    for line_name in LINE_NAMES:
        try:
            spl = spectral_lines.MeasureSimErrors(spec,
                                                  spectral_lines.Spl,
                                                  line=line_name)
            spec_spl_data['EW{}'.format(line_name)] = spl.equiv_width['mean']
            spec_spl_data['EW{}_err'.format(line_name)] = spl.equiv_width['std']
            spec_spl_data['EW{}_b_lbd'.format(line_name)] = spl.maxima['blue_max_wave']['mean']
            spec_spl_data['EW{}_r_lbd'.format(line_name)] = spl.maxima['red_max_wave']['mean']
            if line_name in V_LINE_NAME_MAP.keys():
                spec_spl_data['v{}'.format(line_name)] = spl.velocity['mean']
                spec_spl_data['v{}_err'.format(line_name)] = spl.velocity['std']
                spec_spl_data['v{}_lbd'.format(line_name)] = spl.minimum['mean']
                spec_spl_data['v{}_lbd_err'.format(line_name)] = spl.minimum['std']
        except (ValueError, spectral_lines.MissingDataError):
            spec_spl_data['EW{}'.format(line_name)] = np.nan
            spec_spl_data['EW{}_err'.format(line_name)] = np.nan
            spec_spl_data['EW{}_b_lbd'.format(line_name)] = np.nan
            spec_spl_data['EW{}_r_lbd'.format(line_name)] = np.nan
            if line_name in V_LINE_NAME_MAP.keys():
                spec_spl_data['v{}'.format(line_name)] = np.nan
                spec_spl_data['v{}_err'.format(line_name)] = np.nan
                spec_spl_data['v{}_lbd'.format(line_name)] = np.nan
                spec_spl_data['v{}_lbd_err'.format(line_name)] = np.nan

        ew_snf = snf_spec['phrenology.EW{}'.format(line_name)]
        ew_snf_err = snf_spec['phrenology.EW{}.err'.format(line_name)]
        ew_snf_b_lbd, ew_snf_r_lbd = snf_spec['phrenology.lbd_EW{}'.format(line_name)]
        spec_snf_data['EW{}'.format(line_name)] = ew_snf
        spec_snf_data['EW{}_err'.format(line_name)] = ew_snf_err
        spec_snf_data['EW{}_b_lbd'.format(line_name)] = ew_snf_b_lbd
        spec_snf_data['EW{}_r_lbd'.format(line_name)] = ew_snf_r_lbd
        if line_name in V_LINE_NAME_MAP.keys():
            v_key_name = V_LINE_NAME_MAP[line_name]
            v_snf = snf_spec['phrenology.v{}'.format(v_key_name)]
            v_snf_err = snf_spec['phrenology.v{}.err'.format(v_key_name)]
            l_snf = snf_spec['phrenology.v{}_lbd'.format(v_key_name)]
            l_snf_err = snf_spec['phrenology.v{}_lbd.err'.format(v_key_name)]
            spec_snf_data['v{}'.format(line_name)] = v_snf
            spec_snf_data['v{}_err'.format(line_name)] = v_snf_err
            spec_snf_data['v{}_lbd'.format(line_name)] = l_snf
            spec_snf_data['v{}_lbd_err'.format(line_name)] = l_snf_err
    return spec_spl_data, spec_snf_data


def measure_all_at_max_lines(sn, no_errors=True):
    """Measures all lines for the spectrum closest to maximum brightness of the
    given supernova"""
    path = os.path.join(DATA_DIR, '{}_at_max.csv'.format(sn.target_name))
    try:
        df = pd.read_csv(path)
        return sn.target_name, df
    except FileNotFoundError:
        spec = sn.spec_nearest()
        if no_errors:
            spec_spl_data, spec_snf_data = measure_all_spec_lines(spec)
        else:
            spec_spl_data, spec_snf_data = measure_all_spec_lines_with_errors(spec)
        spec_data = {'phase': spec.salt2_phase,
                     'obs_exp': spec.obs_exp}
        for k, v in spec_spl_data.items():
            spec_data['spl_{}'.format(k)] = v
        for k, v in spec_snf_data.items():
            spec_data['snf_{}'.format(k)] = v
        df = pd.DataFrame({sn.target_name: spec_data})
        df.to_csv(path)
        return sn.target_name, df


def measure_all_sn_lines(sn, no_errors=True):
    """Measures all lines for all spectra of the
    given supernova"""
    path = os.path.join(DATA_DIR, '{}.csv'.format(sn.target_name))
    try:
        df = pd.read_csv(path)
        return sn.target_name, df
    except FileNotFoundError:
        sn_data = {}
        for spec in sn.spectra_noflags:
            if no_errors:
                spec_spl_data, spec_snf_data = measure_all_spec_lines(spec)
            else:
                spec_spl_data, spec_snf_data = measure_all_spec_lines_with_errors(spec)
            spec_data = {'phase': spec.salt2_phase,
                         'obs_exp': spec.obs_exp}
            for k, v in spec_spl_data.items():
                spec_data['spl_{}'.format(k)] = v
            for k, v in spec_snf_data.items():
                spec_data['snf_{}'.format(k)] = v
            sn_data[sn.obs_exp] = spec_data
        df = pd.DataFrame(sn_data)
        df.to_csv(path)
        return sn.target_name, df


@click.command(help='Measure spectral features for SNfactory data set and'
                    'compare to pipeline results.')
@click.option('--at_max', is_flag=True, help='Run only at-max')
@click.option('--no_errors', is_flag=True, help='Run without errors.')
def main(at_max, no_errors):
    p = Pool()
    if at_max:
        func = partial(measure_all_at_max_lines, no_errors=no_errors)
    else:
        func = partial(measure_all_sn_lines, no_errors=no_errors)
    for _ in tqdm(p.imap_unordered(func, DS.sne), total=len(DS.sne)):
        pass


if __name__ == '__main__':
    main()
