# import pytest
# import sncosmo
# import IDRTools
# import numpy as np
# from spectral_lines import SG
#
# np.random.seed(0)
# DS = IDRTools.Dataset(subset=['training', 'validation'])
# HSIAO = sncosmo.Model(source='hsiao')
# HSIAO.set(z=0.05, amplitude=1)
#
#
# class TestSG:
#     @pytest.fixture(scope='class')
#     def state(self):
#         sn = np.random.choice(DS.sne)
#         idr_spec = sn.spec_nearest(0)
#         self.idr_si = SG(idr_spec)
#         self.idr_ca = SG(idr_spec, line='CaII')
#         wave, flux, var = idr_spec.rf_spec()
#         self.idr_sim_si = SG([wave, flux, var], sim=True)
#         self.idr_sim_ca = SG([wave, flux, var], line='CaII', sim=True)
#         wrong_var = np.mean(flux)/1e6 * np.ones(flux.shape)
#         self.idr_sim_wrong_var = SG([wave, flux, wrong_var], sim=True)
#         guess_var = (0.05 * np.quantile(flux, 0.99)) ** 2 * np.ones(flux.shape)
#         self.idr_sim_guess_var = SG([wave, flux, guess_var], sim=True)
#         sim_wave = np.arange(3000., 9000., 2.)
#         sim_flux = HSIAO.flux(0, sim_wave)
#         sim_wave /= 1.05
#         sim_var = (0.05 * np.quantile(sim_flux, 0.99)) ** 2
#         sim_var *= np.ones(sim_flux.shape)
#         sim_flux += np.sqrt(sim_var) * np.random.randn(len(sim_flux))
#         self.sim_si = SG([sim_wave, sim_flux, sim_var], sim=True)
#         self.sim_ca = SG([sim_wave, sim_flux, sim_var], line='CaII', sim=True)
#         self.sim_si_tiny = SG([sim_wave, sim_flux/1e16, sim_var/1e32],
#                                sim=True)
#         self.measurements = {'idr_si': self.idr_si,
#                              'idr_ca': self.idr_ca,
#                              'idr_sim_si': self.idr_sim_si,
#                              'idr_sim_ca': self.idr_sim_ca,
#                              'idr_sim_guess_var': self.idr_sim_guess_var,
#                              'sim_si': self.sim_si,
#                              'sim_ca': self.sim_ca,
#                              'sim_si_tiny': self.sim_si_tiny
#                              }
#         return self
#
#     def test_smooth(self, state):
#         for name, meas in state.measurements.items():
#             wave, flux = meas.get_smoothed_feature_spec()
#             wave_feat = meas.wave_feat
#             flux_diff = flux - meas.flux_feat[np.in1d(wave_feat, wave)]
#             flux_err = np.sqrt(meas.var_feat[np.in1d(wave_feat, wave)])
#             assert np.all(np.abs(flux_diff) <= 5 * flux_err)
#
#     def test_interp(self, state):
#         for name, meas in state.measurements.items():
#             wave, flux = meas.get_interp_feature_spec()
#             assert np.allclose(np.diff(wave), meas.interp_grid, atol=1e-4)
#
#     def test_idr_match_sim(self, state):
#         v_idr = state.idr_si.get_line_velocity()
#         v_sim = state.idr_sim_si.get_line_velocity()
#         ew_idr = state.idr_si.get_equiv_width()
#         ew_sim = state.idr_sim_si.get_equiv_width()
#         assert v_idr == v_sim
#         assert ew_idr == ew_sim
#         v_idr = state.idr_ca.get_line_velocity()
#         v_sim = state.idr_sim_ca.get_line_velocity()
#         ew_idr = state.idr_ca.get_equiv_width()
#         ew_sim = state.idr_sim_ca.get_equiv_width()
#         assert v_idr == v_sim
#         assert ew_idr == ew_sim
#
#     def test_idr_match_guess_var(self, state):
#         v_idr = state.idr_si.get_line_velocity()
#         v_sim = state.idr_sim_guess_var.get_line_velocity()
#         assert np.abs(v_idr - v_sim) < 300
#
#     def test_sim_match_sim_tiny(self, state):
#         v_idr = state.sim_si.get_line_velocity()
#         v_sim = state.sim_si_tiny.get_line_velocity()
#         assert np.abs(v_idr - v_sim) < 300
