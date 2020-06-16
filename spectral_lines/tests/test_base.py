import pytest
import sncosmo
import IDRTools
import numpy as np
from spectral_lines import Measure

np.random.seed(0)
DS = IDRTools.Dataset(subset=['training', 'validation'])
HSIAO = sncosmo.Model(source='hsiao')
HSIAO.set(z=0.05, amplitude=1)


class TestBase:
    @pytest.fixture(scope='class')
    def state(self):
        sn = np.random.choice(DS.sne)
        idr_spec = sn.spec_nearest(0)
        self.idr_meas = Measure(idr_spec)
        wave, flux, var = idr_spec.rf_spec()
        self.idr_sim_meas = Measure([wave, flux, var], sim=True)
        self.sim_wave = np.arange(3000., 9000., 2.)
        sim_flux = HSIAO.flux(0, self.sim_wave)
        sim_var = (0.01 * np.max(sim_flux)) ** 2
        self.sim_meas = Measure([self.sim_wave, sim_flux, sim_var], sim=True)
        return self

    def test_load_idr_sn(self, state):
        assert state.idr_meas.wave_sn.shape[0] >= 0

    def test_load_sim_sn(self, state):
        assert state.sim_meas.wave_sn.shape[0] == state.sim_wave.shape[0]
        assert state.sim_meas.wave_feat.shape[0] == 258

    def test_raise_not_implemented(self, state):
        with pytest.raises(NotImplementedError):
            state.idr_sim_meas.get_smoothed_feature_spec()
            state.idr_meas.get_smoothed_feature_spec()
            state.idr_sim.get_smoothed_feature_spec()
            state.idr_sim_meas.get_interp_feature_spec()
            state.idr_meas.get_interp_feature_spec()
            state.idr_sim.get_interp_feature_spec()

    def test_idr_match_sim(self, state):
        """Ensure an object initialized with an IDR spectrum matches an object
        initialized as a simulated spectrum.
        """
        for attr_name, attr in state.idr_meas.__dict__.items():
            if type(attr) is np.ndarray:
                assert np.all(getattr(state.idr_sim_meas, attr_name) == attr)
            else:
                assert getattr(state.idr_sim_meas, attr_name) == attr
