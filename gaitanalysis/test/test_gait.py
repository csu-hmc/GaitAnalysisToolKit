#!/usr/bin/env python
# -*- coding: utf-8 -*-

# builtin
import os

# external
import numpy as np
from numpy import testing
import pandas
from nose.tools import assert_raises

# local
from ..gait import find_constant_speed, interpolate, WalkingData
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


class TestWalkingData():

    def setup(self):

        time = time_vector(1000, 100)

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
                              'Right Knee Moment': right_knee_moment},
                             index=time)

        self.threshold = 10.0

    def test_init(self):

        walking_data = WalkingData(self.data_frame)

        assert walking_data.raw_data is self.data_frame

    def test_grf_landmarks(self, plot=False):

        walking_data = WalkingData(self.data_frame)

        right_strikes, left_strikes, right_offs, left_offs = \
            walking_data.grf_landmarks('Right Vertical GRF',
                                       'Left Vertical GRF',
                                       threshold=self.threshold,
                                       do_plot=plot)

        right_zero = self.data_frame['Right Vertical GRF'] < self.threshold
        instances = right_zero.apply(lambda x: 1 if x else 0).diff()
        expected_right_offs = \
            instances[instances == 1].index.values.astype(float)
        expected_right_strikes = \
            instances[instances == -1].index.values.astype(float)

        left_zero = self.data_frame['Left Vertical GRF'] < self.threshold
        instances = left_zero.apply(lambda x: 1 if x else 0).diff()
        expected_left_offs = \
            instances[instances == 1].index.values.astype(float)
        expected_left_strikes = \
            instances[instances == -1].index.values.astype(float)

        testing.assert_allclose(expected_right_offs, right_offs)
        testing.assert_allclose(expected_right_strikes, right_strikes)

        testing.assert_allclose(expected_left_offs, left_offs)
        testing.assert_allclose(expected_left_strikes, left_strikes)

    def test_split_at(self, plot=False):

        walking_data = WalkingData(self.data_frame)
        walking_data.grf_landmarks('Right Vertical GRF',
                                   'Left Vertical GRF',
                                   threshold=self.threshold)

        side = 'right'
        series = 'Right Vertical GRF'

        steps = walking_data.split_at('right')

        for i, step in steps.iteritems():
            start_step = walking_data.strikes[side][i]
            end_step = walking_data.strikes[side][i + 1]
            testing.assert_allclose(step[series],
                walking_data.raw_data[series][start_step:end_step])

        if plot is True:
            walking_data.plot_steps(series, 'Left Vertical GRF')

        steps = walking_data.split_at(side, 'stance')

        for i, step in steps.iteritems():
            start_step = walking_data.strikes[side][i]
            end_step = walking_data.offs[side][i + 1]
            testing.assert_allclose(step[series],
                walking_data.raw_data[series][start_step:end_step])

        if plot is True:
            walking_data.plot_steps(series, 'Left Vertical GRF')

        steps = walking_data.split_at(side, 'swing')

        for i, step in steps.iteritems():
            start_step = walking_data.offs[side][i]
            end_step = walking_data.strikes[side][i]
            testing.assert_allclose(step[series],
                walking_data.raw_data[series][start_step:end_step])

        if plot is True:
            walking_data.plot_steps(series, 'Left Vertical GRF')
            import matplotlib.pyplot as plt
            plt.show()

    def test_plot_steps(self):

        walking_data = WalkingData(self.data_frame)
        walking_data.grf_landmarks('Right Vertical GRF',
                                   'Left Vertical GRF',
                                   threshold=self.threshold)
        walking_data.split_at('right')

        assert_raises(ValueError, walking_data.plot_steps)


