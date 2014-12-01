#!/usr/bin/env python
# -*- coding: utf-8 -*-

# builtin
import os

# external
import numpy as np
from numpy import testing
import pandas
from pandas.util.testing import assert_frame_equal
from nose.tools import assert_raises

# local
from ..gait import find_constant_speed, interpolate, GaitData
from dtk.process import time_vector

# debugging
try:
    from IPython.core.debugger import Tracer
except ImportError:
    pass
else:
    set_trace = Tracer()


def test_find_constant_speed():

    speed_array = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                          'data/treadmill-speed.csv'),
                             delimiter=',')
    time = speed_array[:, 0]
    speed = speed_array[:, 1]

    indice, constant_speed_time = find_constant_speed(time, speed, plot=False)

    assert 6.5 < constant_speed_time < 7.5


def test_interpolate():

    df = pandas.DataFrame({'a': [np.nan, 3.0, 5.0, 7.0],
                           'b': [5.0, np.nan, 9.0, 11.0],
                           'c': [2.0, 4.0, 6.0, 8.0],
                           'd': [0.5, 1.0, 1.5, np.nan]},
                          index=[0.0, 2.0, 4.0, 6.0])

    time = [0.0, 1.0, 3.0, 5.0]

    interpolated = interpolate(df, time)

    # NOTE : pandas.Series.interpolate does not extrapolate (because
    # np.interp doesn't.

    df_expected = pandas.DataFrame({'a': [4.0, 4.0, 4.0, 6.0],
                                    'b': [5.0, 6.0, 8.0, 10.0],
                                    'c': [2.0, 3.0, 5.0, 7.0],
                                    'd': [0.5, 0.75, 1.25, 1.5]},
                                   index=time)

    testing.assert_allclose(interpolated.values, df_expected.values)

    testing.assert_allclose(interpolated.values, df_expected.values)
    testing.assert_allclose(interpolated.index.values.astype(float),
                            df_expected.index.values.astype(float))


class TestGaitData():

    def setup(self):

        time = time_vector(1000, 100)
        cortex_time = time
        dflow_time = time

        omega = 2 * np.pi

        right_grf = 1000 * (0.75 + np.sin(omega * time))
        right_grf[right_grf < 0.0] = 0.0
        right_grf += 2.0 * np.random.normal(size=right_grf.shape)

        left_grf = 1000 * (0.75 + np.cos(omega * time))
        left_grf[left_grf < 0.0] = 0.0
        left_grf += 2.0 * np.random.normal(size=left_grf.shape)

        right_knee_angle = np.arange(len(time))
        right_knee_moment = np.arange(len(time))

        self.data_frame = \
            pandas.DataFrame({'Right Vertical GRF': right_grf,
                              'Left Vertical GRF': left_grf,
                              'Right Knee Angle': right_knee_angle,
                              'Right Knee Moment': right_knee_moment,
                              'Cortex Time': cortex_time,
                              'D-Flow Time': dflow_time},
                             index=time)

        self.threshold = 10.0

    def test_init(self):

        gait_data = GaitData(self.data_frame)

        assert gait_data.data is self.data_frame

    def test_inverse_dynamics_2d(self):
        # This only tests to make sure new columns were inserted after the
        # command. There is a test for the underlying leg2d Octave program
        # that actually tests the computed values.

        # Add some columns for the data we need.
        lmark = ['LSHO.PosX', 'LSHO.PosY',
                 'LGTRO.PosX', 'LGTRO.PosY',
                 'LLEK.PosX', 'LLEK.PosY',
                 'LLM.PosX', 'LLM.PosY',
                 'LHEE.PosX', 'LHEE.PosY',
                 'LMT5.PosX', 'LMT5.PosY']

        rmark = ['RSHO.PosX', 'RSHO.PosY',
                 'RGTRO.PosX', 'RGTRO.PosY',
                 'RLEK.PosX', 'RLEK.PosY',
                 'RLM.PosX', 'RLM.PosY',
                 'RHEE.PosX', 'RHEE.PosY',
                 'RMT5.PosX', 'RMT5.PosY']

        lforce = ['FP1.ForX', 'FP1.ForY', 'FP1.MomZ']
        rforce = ['FP2.ForX', 'FP2.ForY', 'FP2.MomZ']

        columns = lmark + rmark + lforce + rforce
        rand = np.random.random((len(self.data_frame), len(columns)))
        new_data = pandas.DataFrame(rand, index=self.data_frame.index,
                                    columns=columns)

        data_frame = self.data_frame.join(new_data)

        gait_data = GaitData(data_frame)

        data_frame = gait_data.inverse_dynamics_2d(lmark, rmark, lforce,
                                                      rforce, 72.0, 6.0)

        # The new columns that should be created.
        new_columns = ['Left.Hip.Flexion.Angle',
                       'Left.Hip.Flexion.Rate',
                       'Left.Hip.Flexion.Moment',
                       'Left.Hip.X.Force',
                       'Left.Hip.Y.Force',
                       'Left.Knee.Flexion.Angle',
                       'Left.Knee.Flexion.Rate',
                       'Left.Knee.Flexion.Moment',
                       'Left.Knee.X.Force',
                       'Left.Knee.Y.Force',
                       'Left.Ankle.PlantarFlexion.Angle',
                       'Left.Ankle.PlantarFlexion.Rate',
                       'Left.Ankle.PlantarFlexion.Moment',
                       'Left.Ankle.X.Force',
                       'Left.Ankle.Y.Force',
                       'Right.Hip.Flexion.Angle',
                       'Right.Hip.Flexion.Rate',
                       'Right.Hip.Flexion.Moment',
                       'Right.Hip.X.Force',
                       'Right.Hip.Y.Force',
                       'Right.Knee.Flexion.Angle',
                       'Right.Knee.Flexion.Rate',
                       'Right.Knee.Flexion.Moment',
                       'Right.Knee.X.Force',
                       'Right.Knee.Y.Force',
                       'Right.Ankle.PlantarFlexion.Angle',
                       'Right.Ankle.PlantarFlexion.Rate',
                       'Right.Ankle.PlantarFlexion.Moment',
                       'Right.Ankle.X.Force',
                       'Right.Ankle.Y.Force']

        for col in new_columns:
            assert col in gait_data.data.columns

    def test_grf_landmarks(self, plot=False):
        # Test for force plate version
        gait_data = GaitData(self.data_frame)

        min_idx = len(self.data_frame) / 3
        max_idx = 2*len(self.data_frame) / 3

        min_time = self.data_frame.index.astype(float)[min_idx]
        max_time = self.data_frame.index.astype(float)[max_idx]

        right_strikes, left_strikes, right_offs, left_offs = \
            gait_data.grf_landmarks('Right Vertical GRF',
                                    'Left Vertical GRF',
                                    min_time=min_time,
                                    max_time=max_time,
                                    threshold=self.threshold,
                                    do_plot=plot)

        right_zero = self.data_frame['Right Vertical GRF'].iloc[min_idx:max_idx] \
                        < self.threshold
        instances = right_zero.apply(lambda x: 1 if x else 0).diff()
        expected_right_offs = \
            instances[instances == 1].index.values.astype(float)
        expected_right_strikes = \
            instances[instances == -1].index.values.astype(float)

        left_zero = self.data_frame['Left Vertical GRF'].iloc[min_idx:max_idx] \
                        < self.threshold
        instances = left_zero.apply(lambda x: 1 if x else 0).diff()
        expected_left_offs = \
            instances[instances == 1].index.values.astype(float)
        expected_left_strikes = \
            instances[instances == -1].index.values.astype(float)

        testing.assert_allclose(expected_right_offs, right_offs)
        testing.assert_allclose(expected_right_strikes, right_strikes)

        testing.assert_allclose(expected_left_offs, left_offs)
        testing.assert_allclose(expected_left_strikes, left_strikes)

        # TODO : Add test for accelerometer based gait landmarks

    def test_plot_landmarks(self):
        gait_data = GaitData(self.data_frame)
        gait_data.grf_landmarks('Right Vertical GRF',
                                   'Left Vertical GRF',
                                   threshold=self.threshold)
        side = 'right'
        col_names = ['Right Vertical GRF','Right Knee Angle','Right Knee Moment']
        time = gait_data.data.index.values.astype(float)

        assert_raises(ValueError, gait_data.plot_landmarks, [], side)
        assert_raises(ValueError, gait_data.plot_landmarks, col_names, '')
        # TODO: Test to see if user wants heelstrikes or toeoffs
        # assert_raises(ValueError, gait_data.plot_landmarks, col_names, side, event='')

    def test_split_at(self, plot=False):

        gait_data = GaitData(self.data_frame)
        gait_data.grf_landmarks('Right Vertical GRF',
                                   'Left Vertical GRF',
                                   threshold=self.threshold)

        side = 'right'
        series = 'Right Vertical GRF'

        gait_cycles = gait_data.split_at(side)

        for i, cycle in gait_cycles.iteritems():
            start_heelstrike_time = gait_data.strikes[side][i]
            end_heelstrike_time = gait_data.strikes[side][i + 1]
            hs_to_hs = gait_data.data[series][start_heelstrike_time:end_heelstrike_time]
            num_samples = len(cycle[series])
            new_time = np.linspace(0.0, end_heelstrike_time,
                                   num=num_samples + 1)
            old_time = np.linspace(0.0, end_heelstrike_time, num=num_samples)
            new_values = np.interp(new_time, old_time, hs_to_hs.values)
            testing.assert_allclose(cycle[series], new_values[:-1])

        if plot is True:
            gait_data.plot_gait_cycles(series, 'Left Vertical GRF')

        gait_cycles = gait_data.split_at(side, 'stance')

        for i, cycle in gait_cycles.iteritems():
            start_heelstrike_time = gait_data.strikes[side][i]
            end_toeoff_time = gait_data.offs[side][i + 1]
            hs_to_toeoff = gait_data.data[series][start_heelstrike_time:end_toeoff_time]
            num_samples = len(cycle[series])
            new_time = np.linspace(0.0, end_toeoff_time,
                                   num=num_samples + 1)
            old_time = np.linspace(0.0, end_toeoff_time, num=num_samples)
            new_values = np.interp(new_time, old_time, hs_to_toeoff.values)
            testing.assert_allclose(cycle[series], new_values[:-1])

        if plot is True:
            gait_data.plot_gait_cycles(series, 'Left Vertical GRF')

        gait_cycles = gait_data.split_at(side, 'swing')

        for i, cycle in gait_cycles.iteritems():
            start_toeoff_time = gait_data.offs[side][i]
            end_heelstrike_time = gait_data.strikes[side][i]
            toeoff_to_heelstrike = gait_data.data[series][start_toeoff_time:end_heelstrike_time]
            num_samples = len(cycle[series])
            new_time = np.linspace(0.0, end_heelstrike_time,
                                   num=num_samples + 1)
            old_time = np.linspace(0.0, end_heelstrike_time, num=num_samples)
            new_values = np.interp(new_time, old_time,
                                   toeoff_to_heelstrike.values)
            testing.assert_allclose(cycle[series], new_values[:-1])

        if plot is True:
            gait_data.plot_gait_cycles(series, 'Left Vertical GRF')
            import matplotlib.pyplot as plt
            plt.show()

        # TODO : Add tests for gait cycle statistics, i.e. stride frequency,
        # etc.

    def test_plot_gait_cycles(self):

        gait_data = GaitData(self.data_frame)
        gait_data.grf_landmarks('Right Vertical GRF',
                                   'Left Vertical GRF',
                                   threshold=self.threshold)
        gait_data.split_at('right')

        assert_raises(ValueError, gait_data.plot_gait_cycles)

    def test_save_load(self):

        gait_data = GaitData(self.data_frame)
        gait_data.grf_landmarks('Right Vertical GRF',
                                   'Left Vertical GRF',
                                   threshold=self.threshold)
        gait_data.split_at('right')

        gait_data.save('some_data.h5')

        gait_data_from_file = GaitData('some_data.h5')

        assert_frame_equal(gait_data.data, gait_data_from_file.data)
        for key, cycle in gait_data.gait_cycles.iteritems():
            assert_frame_equal(cycle, gait_data_from_file.gait_cycles[key])
        assert_frame_equal(gait_data.gait_cycle_stats,
                           gait_data_from_file.gait_cycle_stats)
        assert all(gait_data.strikes['right'] ==
                   gait_data_from_file.strikes['right'])
        assert all(gait_data.strikes['left'] ==
                   gait_data_from_file.strikes['left'])
        assert all(gait_data.offs['right'] ==
                   gait_data_from_file.offs['right'])
        assert all(gait_data.offs['left'] ==
                   gait_data_from_file.offs['left'])

    def teardown(self):
        try:
            open('some_data.h5')
        except IOError:
            pass
        else:
            os.remove('some_data.h5')
