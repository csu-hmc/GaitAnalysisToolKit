#!/usr/bin/env python
# -*- coding: utf-8 -*-

# builtin
import os
from time import strptime
from distutils.version import LooseVersion
from random import sample, choice

# external
import numpy as np
from numpy import testing
from scipy import __version__ as scipy_version
import pandas
from pandas.util.testing import assert_frame_equal
from nose.tools import assert_raises
import yaml
from dtk.process import time_vector

# local
from ..motek import (DFlowData, spline_interpolate_over_missing,
                     low_pass_filter, MissingMarkerIdentifier,
                     markers_for_2D_inverse_dynamics)
from .utils import compare_data_frames

# debugging
try:
    from IPython.core.debugger import Tracer
except ImportError:
    pass
else:
    set_trace = Tracer()


def test_markers_for_2D_inverse_dynamics():

    lmark, rmark, lforce, rforce = markers_for_2D_inverse_dynamics()

    expected_lmark = ['STRN.PosX', 'STRN.PosY',
                      'LGTRO.PosX', 'LGTRO.PosY',
                      'LLEK.PosX', 'LLEK.PosY',
                      'LLM.PosX', 'LLM.PosY',
                      'LHEE.PosX', 'LHEE.PosY',
                      'LMT5.PosX', 'LMT5.PosY']

    assert lmark == expected_lmark

    expected_rmark = ['STRN.PosX', 'STRN.PosY',
                      'RGTRO.PosX', 'RGTRO.PosY',
                      'RLEK.PosX', 'RLEK.PosY',
                      'RLM.PosX', 'RLM.PosY',
                      'RHEE.PosX', 'RHEE.PosY',
                      'RMT5.PosX', 'RMT5.PosY']

    assert rmark == expected_rmark

    assert lforce == ['FP1.ForX', 'FP1.ForY', 'FP1.MomZ']
    assert rforce == ['FP2.ForX', 'FP2.ForY', 'FP2.MomZ']

    lmark, rmark, lforce, rforce = markers_for_2D_inverse_dynamics('full')

    assert lmark == ['LSHO.PosX', 'LSHO.PosY',
                     'LGTRO.PosX', 'LGTRO.PosY',
                     'LLEK.PosX', 'LLEK.PosY',
                     'LLM.PosX', 'LLM.PosY',
                     'LHEE.PosX', 'LHEE.PosY',
                     'LMT5.PosX', 'LMT5.PosY']

    assert rmark == ['RSHO.PosX', 'RSHO.PosY',
                     'RGTRO.PosX', 'RGTRO.PosY',
                     'RLEK.PosX', 'RLEK.PosY',
                     'RLM.PosX', 'RLM.PosY',
                     'RHEE.PosX', 'RHEE.PosY',
                     'RMT5.PosX', 'RMT5.PosY']


class TestMissingMarkerIdentfier:

    def setup(self):

        self.with_constant = pandas.DataFrame({
            'TimeStamp': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'FP1.ForX': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'T10.PosX': [1.0, 1.0, 1.0, 1.0, 5.0, 6.0, 7.0],
            'T10.PosY': [1.0, 1.0, 1.0, 1.0, 5.0, 6.0, 7.0],
            'T10.PosZ': [1.0, 1.0, 1.0, 1.0, 5.0, 6.0, 6.0]})

        self.with_missing = pandas.DataFrame({
            'TimeStamp': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'FP1.ForX': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'T10.PosX': [1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0],
            'T10.PosY': [1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0],
            'T10.PosZ': [1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 6.0]})

        self.statistics = pandas.DataFrame({'T10.PosX': [3, 3],
                                            'T10.PosY': [3, 3],
                                            'T10.PosZ': [3, 3]},
                                           index=['# NA', 'Largest Gap']).T

    def test_identify(self):
        identifier = MissingMarkerIdentifier(self.with_constant)
        assert_raises(ValueError, identifier.identify)

        marker_columns = ['T10.PosX', 'T10.PosY', 'T10.PosZ']
        identifier = \
            MissingMarkerIdentifier(self.with_constant[marker_columns].copy())
        identified = identifier.identify()
        compare_data_frames(identified, self.with_missing[marker_columns])

        identifier = MissingMarkerIdentifier(self.with_constant)
        identified = identifier.identify(columns=marker_columns)
        compare_data_frames(identified, self.with_missing)

    def test_statistics(self):

        marker_columns = ['T10.PosX', 'T10.PosY', 'T10.PosZ']
        identifier = \
            MissingMarkerIdentifier(self.with_constant[marker_columns].copy())
        identifier.identify()
        statistics = identifier.statistics()
        compare_data_frames(statistics, self.statistics)

        identifier = \
            MissingMarkerIdentifier(self.with_constant[marker_columns])
        assert_raises(StandardError, identifier.statistics)


class TestSplineInterpolateOverMissing:

    def setup(self):

        self.with_missing = pandas.DataFrame({
            'TimeStamp': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'FP1.ForX': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'T10.PosX': [1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0],
            'T10.PosY': [1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0],
            'T10.PosZ': [1.0, np.nan, np.nan, np.nan, 5.0, 6.0, np.nan]})

        self.without_missing = pandas.DataFrame({
            'TimeStamp': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'FP1.ForX': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'T10.PosX': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'T10.PosY': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'T10.PosZ': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})

    def test_basic(self):

        interpolated = spline_interpolate_over_missing(self.with_missing,
                                                       'TimeStamp')

        assert not pandas.isnull(interpolated).any().any()
        assert_frame_equal(interpolated, self.without_missing)

    def test_columns(self):

        columns_to_interpolate = ['T10.PosX', 'T10.PosY']

        interpolated = \
            spline_interpolate_over_missing(self.with_missing, 'TimeStamp',
                                            columns=columns_to_interpolate)

        testing.assert_allclose(interpolated['T10.PosZ'],
                                self.with_missing['T10.PosZ'])
        assert_frame_equal(interpolated[columns_to_interpolate],
                           self.without_missing[columns_to_interpolate])


def test_low_pass_filter():

    time = np.linspace(0.0, 1.0, 2001)
    sample_rate = 1.0 / np.diff(time).mean()  # 2000 Hz

    low_freq = np.sin(5.0 * 2.0 * np.pi * time)  # 5 Hz * 2 * pi
    high_freq = np.sin(250.0 * 2.0 * np.pi * time)  # 250 Hz * 2 * pi

    df = pandas.DataFrame({'T': time,
                           'A': low_freq + high_freq,
                           'B': low_freq,
                           'C': high_freq,
                           'D': low_freq + high_freq})

    filtered = low_pass_filter(df, ['A', 'D'], 125.0, sample_rate, order=8,
                               padlen=150)

    testing.assert_allclose(filtered['T'].values, time)
    testing.assert_allclose(filtered['B'].values, low_freq)
    testing.assert_allclose(filtered['C'].values, high_freq)

    nine = LooseVersion('0.9.0')
    ten = LooseVersion('0.10.0')
    current = LooseVersion(scipy_version)

    if current >= nine and current < ten:
        # SciPy 0.9.0 can't handle the end points.
        testing.assert_allclose(filtered['A'].values[50:-50],
                                low_freq[50:-50], rtol=0.01, atol=0.01)
        testing.assert_allclose(filtered['D'].values[50:-50],
                                low_freq[50:-50], rtol=0.01, atol=0.01)
    else:
        testing.assert_allclose(filtered['A'].values, low_freq, rtol=1e-5,
                                atol=1e-5)
        testing.assert_allclose(filtered['D'].values, low_freq, rtol=1e-5,
                                atol=1e-5)


class TestDFlowData():

    cortex_start_frame = 2375
    cortex_sample_period = 1.0 / 100.0
    cortex_number_of_samples = 2001
    missing_marker_start_indices = [78, 158, 213, 401, 478]
    length_missing = [2, 5, 8, 10, 12]

    dflow_start_time = 51.687
    dflow_mocap_max_period_deviation = 0.0012
    dflow_record_max_sample_period = 1.0 / 30.0
    dflow_record_min_sample_period = 1.0 / 300.0
    dflow_record_number_of_samples = 1000
    delsys_time_delay = DFlowData.delsys_time_delay

    # TODO: Ensure that the Cortex and D-Flow time series span the same
    # amount of time.

    cortex_marker_labels = ['T10.PosX',
                            'T10.PosY',
                            'T10.PosZ']

    cortex_force_labels = ['FP1.ForX', 'FP1.ForY', 'FP1.ForZ',
                           'FP1.MomX', 'FP1.MomY', 'FP1.MomZ',
                           'FP1.CopX', 'FP1.CopY', 'FP1.CopZ',
                           'FP2.ForX', 'FP2.ForY', 'FP2.ForZ',
                           'FP2.MomX', 'FP2.MomY', 'FP2.MomZ',
                           'FP2.CopX', 'FP2.CopY', 'FP2.CopZ']

    cortex_analog_labels = ['Channel1.Anlg', 'Channel2.Anlg']
    relabeled_cortex_analog_labels = ["F1Y1", "F1Y2"]
    default_cortex_analog_labels = relabeled_cortex_analog_labels

    dflow_hbm_labels = ['RKneeFlexion.Ang',
                        'RKneeFlexion.Mom',
                        'RKneeFlexion.Pow',
                        'R_PectoralisMajorTH1',
                        'L_RectusFemoris']

    compensation_treadmill_markers = ['ROT_REF.PosX', 'ROT_REF.PosY',
                                      'ROT_REF.PosZ', 'ROT_C1.PosX',
                                      'ROT_C1.PosY', 'ROT_C1.PosZ',
                                      'ROT_C2.PosX', 'ROT_C2.PosY',
                                      'ROT_C2.PosZ', 'ROT_C3.PosX',
                                      'ROT_C3.PosY', 'ROT_C3.PosZ',
                                      'ROT_C4.PosX', 'ROT_C4.PosY',
                                      'ROT_C4.PosZ']

    all_marker_labels = compensation_treadmill_markers + cortex_marker_labels

    # These are these are the XYZ components of the first 4 accelerometers.
    delsys_labels = ["Channel13.Anlg", "Channel14.Anlg",
                     "Channel15.Anlg", "Channel16.Anlg",
                     "Channel17.Anlg", "Channel18.Anlg",
                     "Channel19.Anlg", "Channel20.Anlg",
                     "Channel21.Anlg", "Channel22.Anlg",
                     "Channel23.Anlg", "Channel24.Anlg",
                     "Channel25.Anlg", "Channel26.Anlg",
                     "Channel27.Anlg", "Channel28.Anlg"]

    relabeled_delsys_labels = ["Front_Left_EMG", "Front_Left_AccX",
                               "Front_Left_AccY", "Front_Left_AccZ",
                               "Back_Left_EMG", "Back_Left_AccX",
                               "Back_Left_AccY", "Back_Left_AccZ",
                               "Front_Right_EMG", "Front_Right_AccX",
                               "Front_Right_AccY", "Front_Right_AccZ",
                               "Back_Right_EMG", "Back_Right_AccX",
                               "Back_Right_AccY", "Back_Right_AccZ"]

    default_delsys_labels = ["Sensor01_EMG", "Sensor01_AccX",
                             "Sensor01_AccY", "Sensor01_AccZ",
                             "Sensor02_EMG", "Sensor02_AccX",
                             "Sensor02_AccY", "Sensor02_AccZ",
                             "Sensor03_EMG", "Sensor03_AccX",
                             "Sensor03_AccY", "Sensor03_AccZ",
                             "Sensor04_EMG", "Sensor04_AccX",
                             "Sensor04_AccY", "Sensor04_AccZ"]

    mocap_labels_without_hbm = (['TimeStamp', 'FrameNumber'] +
                                all_marker_labels +
                                cortex_force_labels +
                                cortex_analog_labels +
                                delsys_labels)
    relabeled_mocap_labels_without_hbm = (['TimeStamp', 'FrameNumber'] +
                                          all_marker_labels +
                                          cortex_force_labels +
                                          relabeled_cortex_analog_labels +
                                          relabeled_delsys_labels)
    default_mocap_labels_without_hbm = (['TimeStamp', 'FrameNumber'] +
                                        all_marker_labels +
                                        cortex_force_labels +
                                        default_cortex_analog_labels +
                                        default_delsys_labels)

    mocap_labels_with_hbm = mocap_labels_without_hbm + dflow_hbm_labels
    relabeled_mocap_labels_with_hbm = (relabeled_mocap_labels_without_hbm +
                                       dflow_hbm_labels)
    default_mocap_labels_with_hbm = (default_mocap_labels_without_hbm +
                                     dflow_hbm_labels)

    record_labels = ['Time', 'RightBeltSpeed', 'LeftBeltSpeed']

    path_to_mocap_data_file = 'example_mocap_tsv_file.txt'
    path_to_record_data_file = 'example_record_tsv_file.txt'
    path_to_meta_data_file = 'example_meta_data_file.yml'
    path_to_compensation_data_file = 'example_compensation_file.txt'

    #with open(os.path.join(__file__, 'data', 'meta-sample-full.yml'), 'r') as f:
        #meta_data = yaml.load(f)

    number_of_events = choice(range(2, 7))
    possible_event_names = ['A', 'B', 'C', 'D', 'E', 'F']
    possible_event_descriptions = ['Zeroing', 'Walking', 'Running',
                                   'Sitting', 'Standing', 'Relaxing']

    meta_data = {'trial': {'id': 5,
                           'datetime': strptime('2013-10-03', "%Y-%m-%d"),
                           'notes': 'All about this trial.',
                           'nominal-speed': 5.0,
                           'nominal-speed-units': 'meters per second',
                           'stationary-platform': False,
                           'pitch': False,
                           'sway': True,
                           'marker-set': 'lower',
                           'dflow-version': '3.16.1',
                           'files': {
                                     'mocap': path_to_mocap_data_file,
                                     'record': path_to_record_data_file,
                                     'meta': path_to_meta_data_file,
                                     'compensation': path_to_compensation_data_file,
                                    },
                           'events': dict(zip(possible_event_names[:number_of_events],
                                              possible_event_descriptions[:number_of_events])),
                           'analog-channel-names':
                               {
                                    "Channel1.Anlg": "F1Y1",
                                    "Channel2.Anlg": "F1Y2",
                                    "Channel3.Anlg": "F1Y3",
                                    "Channel4.Anlg": "F1X1",
                                    "Channel5.Anlg": "F1X2",
                                    "Channel6.Anlg": "F1Z1",
                                    "Channel7.Anlg": "F2Y1",
                                    "Channel8.Anlg": "F2Y2",
                                    "Channel9.Anlg": "F2Y3",
                                    "Channel10.Anlg": "F2X1",
                                    "Channel11.Anlg": "F2X2",
                                    "Channel12.Anlg": "F2Z1",
                                    "Channel13.Anlg": "Front_Left_EMG",
                                    "Channel14.Anlg": "Front_Left_AccX",
                                    "Channel15.Anlg": "Front_Left_AccY",
                                    "Channel16.Anlg": "Front_Left_AccZ",
                                    "Channel17.Anlg": "Back_Left_EMG",
                                    "Channel18.Anlg": "Back_Left_AccX",
                                    "Channel19.Anlg": "Back_Left_AccY",
                                    "Channel20.Anlg": "Back_Left_AccZ",
                                    "Channel21.Anlg": "Front_Right_EMG",
                                    "Channel22.Anlg": "Front_Right_AccX",
                                    "Channel23.Anlg": "Front_Right_AccY",
                                    "Channel24.Anlg": "Front_Right_AccZ",
                                    "Channel25.Anlg": "Back_Right_EMG",
                                    "Channel26.Anlg": "Back_Right_AccX",
                                    "Channel27.Anlg": "Back_Right_AccY",
                                    "Channel28.Anlg": "Back_Right_AccZ",
                                },
                            'sensor-orientation':
                                {
                                    "Front_Left": [[0, 1, 0],
                                                   [1, 0, 0],
                                                   [0, 0, -1]],
                                    "Back_Left": [[0, -1, 0],
                                                  [1, 0, 0],
                                                  [0, 0, 1]],
                                    "Front_Right": [[0, 0, 1],
                                                    [1, 0, 0],
                                                    [0, 1, 0]],
                                    "Back_Right": [[0, 1, 0],
                                                   [0, 0, 1],
                                                   [1, 0, 0]],
                                },
                           'data-description':
                               {
                                    "ROT_REF.PosX": "A marker place on the rigid structure of the treadmill.",
                               },
                           },
                 'subject': {
                             'id': 234,
                             'age': 28,
                             'mass': 70,
                             'mass-units': 'kilograms',
                             'height': 1.82,
                             'height-units': 'meters',
                             'gender': 'male',
                             },
                 'study': {
                           'id': 12,
                           'name': 'Human Locomotion Control Identification',
                           'description': 'Perturbations during walking and running.',
                          },
                 }

    def create_sample_mocap_file(self):
        """
        The text file output from the mocap module in DFlow is a tab
        delimited file. The first line is the header. The header contains
        the `TimeStamp` column which is the system time on the DFlow
        computer when it receives the Cortex frame and is thus not exactly
        at 100 hz, it has a light variable sample rate. The next column is
        the `FrameNumber` column which is the Cortex frame number. Cortex
        samples at 100hz and the frame numbers start at some positive
        integer value. The remaining columns are samples of the computed
        marker positions at each Cortex frame and the analog signals (force
        plate forces/moments, EMG, accelerometers, etc). The analog signals
        are simply voltages that have been scaled by some calibration
        function and they should have a reading at each frame. The markers
        sometimes go missing (i.e. can't been seen by the cameras. When a
        marker goes missing DFlow outputs the last non-missing value in all
        three axes until the marker is visible again. The mocap file can
        also contain variables computed by the real time implementation of
        the Human Body Model (HBM). If the HBM computation fails at a D-Flow
        sample period, strings of zeros, '0.000000', are inserted for
        missing values. Note that the order of the "essential" measurements
        in the file must be retained if you expect to run the file back into
        D-Flow for playback.

        """
        # This generates the slightly variable sampling periods seen in the
        # time stamp column.
        deviations = (self.dflow_mocap_max_period_deviation *
                      np.random.uniform(-1.0, 1.0,
                                        self.cortex_number_of_samples))

        variable_periods = (self.cortex_sample_period *
                            np.ones(self.cortex_number_of_samples) +
                            deviations)

        mocap_data = {'TimeStamp': self.dflow_start_time +
                      np.cumsum(variable_periods),
                      'FrameNumber': range(self.cortex_start_frame,
                                           self.cortex_start_frame +
                                           self.cortex_number_of_samples, 1),
                      }

        for label in self.mocap_labels_with_hbm[2:]:  # skip TimeStamp & FrameNumber
            if label in self.delsys_labels:
                mocap_data[label] = np.sin(mocap_data['TimeStamp'] -
                                           self.delsys_time_delay)
            else:
                mocap_data[label] = np.sin(mocap_data['TimeStamp'])

        self.mocap_data_frame = pandas.DataFrame(mocap_data)

        # TODO : This adds missing values to the data that represents the
        # output of D-Flow versions <= 3.16.1. But we need to also create
        # some data created which represents the missing values as zeros for
        # the newer D-Flow versions and test that as well.

        # Add "missing" values to the marker data.
        for j, index in enumerate(self.missing_marker_start_indices):
            for signal in (self.cortex_marker_labels +
                           self.compensation_treadmill_markers):
                s = self.mocap_data_frame.loc[:, signal]
                self.mocap_data_frame.ix[index:index + self.length_missing[j],
                                         signal] = s.iloc[index]

        # The kwarg 'cols' is being deprecated and a warning is issued in
        # 0.15, but in 0.12 columns is not supported. I haven't checked 0.13
        # or 0.14.
        kwargs = {'sep': '\t', 'float_format': '%1.6f', 'index': False}
        if LooseVersion(pandas.__version__) >= LooseVersion('0.15.0'):
            kwargs['columns'] = self.mocap_labels_with_hbm
        else:
            kwargs['cols'] = self.mocap_labels_with_hbm
        self.mocap_data_frame.to_csv(self.path_to_mocap_data_file, **kwargs)

    def create_sample_compensation_file(self):

        # This generates the slightly variable sampling periods seen in the
        # time stamp column.
        deviations = (self.dflow_mocap_max_period_deviation *
                      np.random.uniform(-1.0, 1.0,
                                        self.cortex_number_of_samples))

        variable_periods = (self.cortex_sample_period *
                            np.ones(self.cortex_number_of_samples) +
                            deviations)

        compensation_data = {'TimeStamp': self.dflow_start_time +
                      np.cumsum(variable_periods),
                      'FrameNumber': range(self.cortex_start_frame,
                                           self.cortex_start_frame +
                                           self.cortex_number_of_samples, 1),
                      }

        for label in self.cortex_force_labels:
            compensation_data[label] = 0.5 * np.cos(compensation_data['TimeStamp'])

        for label in self.delsys_labels:
            compensation_data[label] = np.cos(compensation_data['TimeStamp']
                                              - self.delsys_time_delay)

        self.compensation_data_frame = pandas.DataFrame(compensation_data)

        cols = (["TimeStamp", 'FrameNumber'] +
                self.cortex_force_labels +
                self.delsys_labels
                )

        # The kwarg 'cols' is being deprecated and a warning is issued in
        # 0.15, but in 0.12 columns is not supported. I haven't checked 0.13
        # or 0.14.
        kwargs = {'sep': '\t', 'float_format': '%1.6f', 'index': False}
        if LooseVersion(pandas.__version__) >= LooseVersion('0.15.0'):
            kwargs['columns'] = cols
        else:
            kwargs['cols'] = cols
        self.compensation_data_frame.to_csv(self.path_to_compensation_data_file,
                                            **kwargs)

    def create_sample_meta_data_file(self):
        """We will have an optional YAML file containing meta data for
        each trial in the same directory as the trials time series data
        files."""

        with open(self.path_to_meta_data_file, 'w') as f:
            yaml.dump(self.meta_data, f)

    def create_sample_record_file(self):
        """The record module output file is a tab delimited file. The
        first list is the header and the first column is a `Time` column
        which records the Dflow system time. The period between each
        time sample is variable depending on DFlow's processing load.
        The sample rate can be as high as 300 hz. The file also records
        events as hashed comments between lines. A count of all of the
        events is listed at the end of the file.

        Note: This requires the sample mocap file to be already created.
        """

        self.record_time = [0.0]
        final_mocap_time = self.mocap_data_frame['TimeStamp'].iloc[-1]

        while self.record_time[-1] < final_mocap_time:
            variable_periods = (self.dflow_record_min_sample_period +
                                (self.dflow_record_max_sample_period -
                                self.dflow_record_min_sample_period) *
                                np.random.random(self.dflow_record_number_of_samples))
            self.record_time = self.dflow_start_time + np.cumsum(variable_periods)
            self.dflow_record_number_of_samples += 100

        if self.record_time[-1] > final_mocap_time:
            i = np.argmin(np.abs(self.record_time - final_mocap_time))
            self.record_time = self.record_time[:i]
            self.record_time[-1] = final_mocap_time

        record_data = {}
        record_data['Time'] = self.record_time
        record_data['LeftBeltSpeed'] = np.random.random(len(self.record_time))
        record_data['RightBeltSpeed'] = np.random.random(len(self.record_time))

        self.record_data_frame = pandas.DataFrame(record_data)

        # The kwarg 'cols' is being deprecated and a warning is issued in
        # 0.15, but in 0.12 columns is not supported. I haven't checked 0.13
        # or 0.14.
        kwargs = {'sep': '\t', 'float_format': '%1.6f', 'index': False}
        if LooseVersion(pandas.__version__) >= LooseVersion('0.15.0'):
            kwargs['columns'] = ['Time', 'LeftBeltSpeed', 'RightBeltSpeed']
        else:
            kwargs['cols'] = ['Time', 'LeftBeltSpeed', 'RightBeltSpeed']
        self.record_data_frame.to_csv(self.path_to_record_data_file,
                                      **kwargs)

        event_template = "#\n# EVENT {} - COUNT {}\n#\n"

        time = self.record_data_frame['Time']

        self.n_event_times = sorted(sample(time, self.number_of_events))
        event_letters = self.possible_event_names[:self.number_of_events]
        self.event_times = dict(zip(event_letters, self.n_event_times))

        # This loops through the record file and inserts the events.
        new_lines = ''
        with open(self.path_to_record_data_file, 'r') as f:
            for line in f:
                new_lines += line

                if 'Time' in line:
                    time_col_index = line.strip().split('\t').index('Time')

                time_string = line.strip().split('\t')[time_col_index]

                for key, value in self.event_times.items():
                    if '{:1.6f}'.format(value) == time_string:
                        new_lines += event_template.format(key, '1')

        new_lines += "\n".join("# EVENT {} occured 1 time".format(letter)
                               for letter in event_letters)

        with open(self.path_to_record_data_file, 'w') as f:
            f.write(new_lines)

    def setup(self):
        self.create_sample_mocap_file()
        self.create_sample_compensation_file()
        self.create_sample_record_file()
        self.create_sample_meta_data_file()

    def test_init(self):

        data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                         record_tsv_path=self.path_to_record_data_file,
                         meta_yml_path=self.path_to_meta_data_file)

        assert data.mocap_tsv_path == self.path_to_mocap_data_file
        assert data.record_tsv_path == self.path_to_record_data_file
        assert data.meta_yml_path == self.path_to_meta_data_file

        for attr in ['meta', 'mocap_column_labels', 'marker_column_labels',
                     'hbm_column_labels', 'hbm_column_indices',
                     'non_hbm_column_indices', 'analog_column_labels',
                     'analog_column_indices', 'emg_column_labels',
                     'accel_column_labels']:
            try:
                getattr(data, attr)
            except AttributeError:
                assert False

        data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                         meta_yml_path=self.path_to_meta_data_file)

        assert data.mocap_tsv_path == self.path_to_mocap_data_file
        assert data.record_tsv_path is None
        assert data.meta_yml_path == self.path_to_meta_data_file

        for attr in ['meta', 'mocap_column_labels', 'marker_column_labels',
                     'hbm_column_labels', 'hbm_column_indices',
                     'non_hbm_column_indices', 'analog_column_labels',
                     'analog_column_indices', 'emg_column_labels',
                     'accel_column_labels']:
            try:
                getattr(data, attr)
            except AttributeError:
                assert False

        data = DFlowData(record_tsv_path=self.path_to_record_data_file,
                         meta_yml_path=self.path_to_meta_data_file)

        assert data.mocap_tsv_path is None
        assert data.record_tsv_path == self.path_to_record_data_file
        assert data.meta_yml_path == self.path_to_meta_data_file

        for attr in ['meta']:
            try:
                getattr(data, attr)
            except AttributeError:
                assert False

        data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file)

        assert data.mocap_tsv_path == self.path_to_mocap_data_file
        assert data.record_tsv_path is None
        assert data.meta_yml_path is None

        for attr in ['mocap_column_labels', 'marker_column_labels',
                     'hbm_column_labels', 'hbm_column_indices',
                     'non_hbm_column_indices', 'analog_column_labels',
                     'analog_column_indices', 'emg_column_labels',
                     'accel_column_labels']:
            try:
                getattr(data, attr)
            except AttributeError:
                assert False

        data = DFlowData(record_tsv_path=self.path_to_record_data_file)

        assert data.mocap_tsv_path is None
        assert data.record_tsv_path == self.path_to_record_data_file
        assert data.meta_yml_path is None

        assert_raises(ValueError, DFlowData)

    def test_parse_meta_data_file(self):

        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                               meta_yml_path=self.path_to_meta_data_file)

        assert self.meta_data == dflow_data._parse_meta_data_file()

    def test_compensation_needed(self):

        # Default in the example meta file is that compensation is needed.
        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                               meta_yml_path=self.path_to_meta_data_file)
        assert dflow_data._compensation_needed()

        # Test if the platform is specified as stationary.
        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                               meta_yml_path=self.path_to_meta_data_file)
        dflow_data.meta['trial']['stationary-platform'] = True
        assert not dflow_data._compensation_needed()

        # Test if it wasn't in the meta file at all.
        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                               meta_yml_path=self.path_to_meta_data_file)
        dflow_data.meta['trial'].pop('stationary-platform')
        assert not dflow_data._compensation_needed()

        # Test if no meta file is provided.
        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file)
        assert not dflow_data._compensation_needed()

    def test_store_compensation_data_path(self):
        # TODO : Not sure if this test is a good one. It needs to test
        # whether the path builds correctly if you have the path to the main
        # mocap file and the path to the compensation file relative to the
        # main mocap file.
        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                               meta_yml_path=self.path_to_meta_data_file)
        dflow_data._store_compensation_data_path()
        assert (dflow_data.compensation_tsv_path ==
                self.meta_data['trial']['files']['compensation'])

    def test_load_compensation_data(self):
        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                               meta_yml_path=self.path_to_meta_data_file)
        dflow_data._store_compensation_data_path()
        unloaded_trial = dflow_data._load_compensation_data()

        # TODO : Implement a test.

        #assert len(set(unloaded_trial.columns).diff() == 0

    def test_mocap_column_labels(self):

        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file)

        assert self.mocap_labels_with_hbm == dflow_data._mocap_column_labels()

    def test_marker_column_labels(self):

        dflow_data = DFlowData(self.path_to_mocap_data_file)
        all_labels = dflow_data.mocap_column_labels
        labels = dflow_data._marker_column_labels(all_labels)

        assert labels == (self.compensation_treadmill_markers +
                          self.cortex_marker_labels)

    def test_hbm_column_labels(self):

        dflow_data = DFlowData(self.path_to_mocap_data_file)
        all_labels = dflow_data.mocap_column_labels

        hbm_lab, hbm_i, non_hbm_i = dflow_data._hbm_column_labels(all_labels)

        assert self.dflow_hbm_labels == hbm_lab
        assert hbm_i == range(len(self.mocap_labels_without_hbm),
                              len(self.mocap_labels_with_hbm))
        assert non_hbm_i == range(len(self.mocap_labels_without_hbm))

    def test_analog_channel_labels(self):
        dflow_data = DFlowData(self.path_to_mocap_data_file)
        all_labels = dflow_data.mocap_column_labels

        anal_lab, anal_ind, emg_lab, accel_lab = \
            dflow_data._analog_column_labels(all_labels)

        assert anal_lab == self.cortex_analog_labels + self.delsys_labels

        for label in emg_lab + accel_lab:
            assert label in self.delsys_labels

    def test_force_column_labels(self):
        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                               meta_yml_path=self.path_to_meta_data_file)
        force_plate_labels = dflow_data._force_column_labels(without_center_of_pressure=False)
        assert sorted(self.cortex_force_labels) == sorted(force_plate_labels)

        force_plate_labels = dflow_data._force_column_labels(without_center_of_pressure=True)
        for label in force_plate_labels:
            assert label in self.cortex_force_labels
            assert 'Cop' not in label

    def test_relabel_analog_column(self):

        # Test if analog columns are relabeled to what is indicated in
        # meta file
        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                               meta_yml_path=self.path_to_meta_data_file)

        relabeled_data = dflow_data._relabel_analog_columns(self.mocap_data_frame.copy())

        relabeled_columns = relabeled_data.columns
        anal_lab = dflow_data.analog_column_labels
        emg_lab = dflow_data.emg_column_labels
        accel_lab = dflow_data.accel_column_labels


        for col in self.relabeled_cortex_analog_labels + \
                              self.relabeled_delsys_labels:
            assert col in relabeled_columns

        for col in self.relabeled_cortex_analog_labels + \
                              self.relabeled_delsys_labels:
            assert col in anal_lab

        for col in self.relabeled_delsys_labels:
            assert col in emg_lab + accel_lab

        # Test if analog channels are relabeled to default names in absence
        # of a meta file
        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file)

        relabeled_data = dflow_data._relabel_analog_columns(self.mocap_data_frame.copy())

        relabeled_columns = relabeled_data.columns
        anal_lab = dflow_data.analog_column_labels
        emg_lab = dflow_data.emg_column_labels
        accel_lab = dflow_data.accel_column_labels


        for col in self.default_cortex_analog_labels + \
                              self.default_delsys_labels:
            assert col in relabeled_columns

        for col in self.default_cortex_analog_labels + \
                              self.default_delsys_labels:
            assert col in anal_lab

        for col in self.default_delsys_labels:
            assert col in emg_lab + accel_lab

    def test_relabel_markers(self):

        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                               meta_yml_path=self.path_to_meta_data_file)
        mocap_data_frame = dflow_data._load_mocap_data(ignore_hbm=True)

        # If there is no marker map then this method should do nothing.
        relabeled_data_frame = dflow_data._relabel_markers(mocap_data_frame)
        compare_data_frames(relabeled_data_frame, mocap_data_frame)

        # If there is a marker map then the labels should be changed.
        dflow_data.meta['trial']['marker-map'] = {'T10': 'STRN'}
        relabeled_data_frame = dflow_data._relabel_markers(mocap_data_frame)

        assert 'T10.PosX' not in relabeled_data_frame.columns
        assert 'T10.PosY' not in relabeled_data_frame.columns
        assert 'T10.PosZ' not in relabeled_data_frame.columns

        assert 'T10.PosX' not in dflow_data.marker_column_labels
        assert 'T10.PosY' not in dflow_data.marker_column_labels
        assert 'T10.PosZ' not in dflow_data.marker_column_labels

        assert 'T10.PosX' not in dflow_data.mocap_column_labels
        assert 'T10.PosY' not in dflow_data.mocap_column_labels
        assert 'T10.PosZ' not in dflow_data.mocap_column_labels

        assert 'STRN.PosX' in relabeled_data_frame.columns
        assert 'STRN.PosY' in relabeled_data_frame.columns
        assert 'STRN.PosZ' in relabeled_data_frame.columns

        assert 'STRN.PosX' in dflow_data.marker_column_labels
        assert 'STRN.PosY' in dflow_data.marker_column_labels
        assert 'STRN.PosZ' in dflow_data.marker_column_labels

        assert 'STRN.PosX' in dflow_data.mocap_column_labels
        assert 'STRN.PosY' in dflow_data.mocap_column_labels
        assert 'STRN.PosZ' in dflow_data.mocap_column_labels

    def test_shift_delsys_signals(self):
        dflow_data = DFlowData(self.path_to_mocap_data_file)
        mocap_data_frame = dflow_data._load_mocap_data(ignore_hbm=True)
        shifted_mocap_data_frame = \
            dflow_data._shift_delsys_signals(mocap_data_frame)

        # TODO: The last 10 points don't match well. Is probably due to the spline
        # extrapolation. Maybe better to check this out at some point.
        testing.assert_allclose(shifted_mocap_data_frame['Channel13.Anlg'][:1990],
                                np.sin(shifted_mocap_data_frame['TimeStamp'])[:1990],
                                atol=1e-5, rtol=1e-5)

    def test_identify_missing_markers(self):
        dflow_data = DFlowData(self.path_to_mocap_data_file)
        data_frame = dflow_data._load_mocap_data(ignore_hbm=True)
        identified = dflow_data._identify_missing_markers(data_frame)

        for i, index in enumerate(self.missing_marker_start_indices):
            for label in (self.compensation_treadmill_markers +
                          self.cortex_marker_labels):
                assert all(identified[label][index + 1:index +
                                             self.length_missing[i]].isnull())

    def test_generate_cortex_time_stamp(self):
        dflow_data = DFlowData(self.path_to_mocap_data_file)
        data = dflow_data._generate_cortex_time_stamp(self.mocap_data_frame)
        expected_time = time_vector(self.cortex_number_of_samples, 1.0 /
                                    self.cortex_sample_period)
        testing.assert_allclose(data['Cortex Time'], expected_time)

    def test_interpolate_missing_markers(self):

        dflow_data = DFlowData(self.path_to_mocap_data_file)
        mocap_data_frame = dflow_data._load_mocap_data(ignore_hbm=True)
        identified = dflow_data._identify_missing_markers(mocap_data_frame)
        interpolated = spline_interpolate_over_missing(identified,
                                                       'TimeStamp',
                                                       columns=dflow_data.marker_column_labels)

        # There should be no nans in the interpolated data.
        assert not pandas.isnull(interpolated).any().any()

        for label in (self.compensation_treadmill_markers +
                      self.cortex_marker_labels):
            testing.assert_allclose(interpolated[label].values,
                                    np.sin(interpolated['TimeStamp']).values,
                                    rtol=1e-3, atol=1e-3)

    def test_missing_markers_are_zeros(self):

        # no meta data should return True
        dflow_data = DFlowData(self.path_to_mocap_data_file)
        assert dflow_data._missing_markers_are_zeros()

        # meta data with dflow version less or equal to 3.16.1 should return
        # False
        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                               meta_yml_path=self.path_to_meta_data_file)
        assert not dflow_data._missing_markers_are_zeros()

        # meta data with dflow version above 3.16.2rc4 should return True
        dflow_data.meta['trial']['dflow-version'] = '3.16.3'
        assert dflow_data._missing_markers_are_zeros()

        # meta data with dflow version in between 3.16.1 and 3.16.1rc4 should
        # return True
        dflow_data.meta['trial']['dflow-version'] = '3.16.2rc1'
        assert dflow_data._missing_markers_are_zeros()

        # meta data without dflow version should return True
        dflow_data.meta['trial'].pop('dflow-version', None)
        assert dflow_data._missing_markers_are_zeros()

    def test_load_mocap_data(self):
        dflow_data = DFlowData(self.path_to_mocap_data_file)
        raw_mocap_data = dflow_data._load_mocap_data()

        compare_data_frames(raw_mocap_data, self.mocap_data_frame, atol=1e-6)

        # TODO : Add some missing values into the HBM columns of
        # self.mocap_data_frame and make sure they get replaced with NaN.

        raw_mocap_data = dflow_data._load_mocap_data(ignore_hbm=True)

        expected = self.mocap_data_frame[self.mocap_labels_without_hbm]

        compare_data_frames(raw_mocap_data, expected, atol=1e-6)

    def test_extract_events_from_record_file(self):
        #This checks with a meta.yml file
        directory = os.path.split(__file__)[0]
        data = DFlowData(record_tsv_path=directory + '/data/example_extract_events_from_record_module.txt',
                         meta_yml_path=directory + '/data/meta_events_example.yml')
        data._extract_events_from_record_file()

        expected_time_map = {'ForcePlateZeroing': (208.038250, 228.046535),
                             'NormalWalking': (228.046535, 288.051670),
                             'TreadmillPerturbation': (288.051670, 528.056120),
                             'Both': (528.056120, 768.060985),
                             'Normal': (768.060985, 768.064318)}

        for key, (start, end) in expected_time_map.items():
            assert abs(data.events[key][0] - start) < 1e-16
            assert abs(data.events[key][1] - end) < 1e-16

        #This checks without a meta.yml file
        data = DFlowData(record_tsv_path=directory + '/data/example_extract_events_from_record_module.txt')
        data._extract_events_from_record_file()

        expected_time_map = {'A': (208.038250, 228.046535),
                             'B': (228.046535, 288.051670),
                             'C': (288.051670, 528.056120),
                             'D': (528.056120, 768.060985),
                             'E': (768.060985, 768.064318)}

        for key, (start, end) in expected_time_map.items():
            assert abs(data.events[key][0] - start) < 1e-16
            assert abs(data.events[key][1] - end) < 1e-16

        dflow_data = DFlowData(self.path_to_mocap_data_file,
                               self.path_to_record_data_file)
        dflow_data._extract_events_from_record_file()

        event_descriptions = self.possible_event_names[:self.number_of_events]
        event_times = dict(zip(event_descriptions, self.n_event_times))
        for k, v in dflow_data.events.items():
            testing.assert_allclose(v[0], event_times[k], rtol=1e-5, atol=1e-5)

    def test_load_record_data(self):
        dflow_data = DFlowData(record_tsv_path=self.path_to_record_data_file)
        raw_record_data = dflow_data._load_record_data()

        compare_data_frames(raw_record_data, self.record_data_frame,
                            atol=1e-6)

    def test_resample_record_data(self):
        dflow_data = DFlowData(self.path_to_mocap_data_file,
                               self.path_to_record_data_file)
        dflow_data.mocap_data = self.mocap_data_frame
        dflow_data._generate_cortex_time_stamp(self.mocap_data_frame)
        record_data = dflow_data._resample_record_data(self.record_data_frame)
        expected_time = time_vector(self.cortex_number_of_samples, 1.0 /
                                    self.cortex_sample_period)

        testing.assert_allclose(record_data['Time'], expected_time)
        # TODO : Test that the values are correct. How?

    def test_compensate_forces(self):
        data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                         record_tsv_path=self.path_to_record_data_file,
                         meta_yml_path=self.path_to_meta_data_file)
        # TODO : Add test.

    def test_calibrate_accel_data(self):
        data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file)

        # TODO : Add test.

    def test_orient_accelerometers(self):
        data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                         meta_yml_path=self.path_to_meta_data_file)

        relabeled_data = data._relabel_analog_columns(self.mocap_data_frame.copy())
        reoriented_data = data._orient_accelerometers(relabeled_data.copy())

        testing.assert_allclose(reoriented_data['Front_Left_EMG'],
                relabeled_data['Front_Left_EMG'])
        testing.assert_allclose(reoriented_data['Back_Left_AccX'],
                -relabeled_data['Back_Left_AccY'])
        testing.assert_allclose(reoriented_data['Front_Right_AccY'],
                relabeled_data['Front_Right_AccX'])
        testing.assert_allclose(reoriented_data['Back_Right_AccZ'],
                relabeled_data['Back_Right_AccX'])

    def test_express(self):
        data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                         record_tsv_path=self.path_to_record_data_file,
                         meta_yml_path=self.path_to_meta_data_file)
        data.clean_data()

        x_motek_labels = ['T10.PosX', 'FP1.ForX', 'FP2.ForX', 'FP1.MomX',
                          'FP2.MomX', 'FP1.CopX', 'FP2.CopX']
        y_motek_labels = ['T10.PosY', 'FP1.ForY', 'FP2.ForY', 'FP1.MomY',
                          'FP2.MomY', 'FP1.CopY', 'FP2.CopY']
        z_motek_labels = ['T10.PosZ', 'FP1.ForZ', 'FP2.ForZ', 'FP1.MomZ',
                          'FP2.MomZ', 'FP1.CopZ', 'FP2.CopZ']

        # This changes the values in the data frame so that the columns are
        # recognizable from each other.
        changed = self.cortex_force_labels + self.all_marker_labels
        data.data[changed] = \
            np.random.random(len(changed)) * data.data[changed]

        # The Motion Analysis reference frame is x to the right, y up, and z
        # backwards. The ISB standard (Wu and Cavanagh 1995) is x forward, y
        # up, and z to the right. The following rotation matrix can be
        # multiplied by a vector expressed in the Motion Analysis
        # reference frame to get the same vector expressed in the ISB
        # standard frame (i.e. isb = R * motion analysis).

        R = np.array([[0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0]])

        # This should be a new data frame.
        rotated = data._express(data.data, R)

        for label in x_motek_labels:
            testing.assert_allclose(data.data[label].values,
                                    rotated[label[:-1] + 'Z'].values)

        for label in y_motek_labels:
            testing.assert_allclose(data.data[label].values,
                                    rotated[label].values)

        for label in z_motek_labels:
            testing.assert_allclose(data.data[label].values,
                                    -rotated[label[:-1] + 'X'].values)

        rotated_to_standard = data._express_in_isb_standard_coordinates(data.data)
        compare_data_frames(rotated_to_standard, rotated)

    def test_clean_data(self):
        data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                         record_tsv_path=self.path_to_record_data_file,
                         meta_yml_path=self.path_to_meta_data_file)

        data.clean_data(ignore_hbm=True)

        # TODO : Check for an events dictionary if the record file included
        # events.

        assert (data._marker_column_labels(data.mocap_column_labels) ==
                self.all_marker_labels)
        expected_columns = self.relabeled_mocap_labels_without_hbm + \
            self.record_labels + ['Cortex Time', 'D-Flow Time']
        for col in data.data.columns:
            assert col in expected_columns
            assert col not in self.dflow_hbm_labels
        expected_time = time_vector(self.cortex_number_of_samples, 1.0 /
                                    self.cortex_sample_period)
        testing.assert_allclose(data.data['Cortex Time'], expected_time)

        try:
            data.meta
        except AttributeError:
            assert False

        # Without the record file.
        data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                         meta_yml_path=self.path_to_meta_data_file)
        data.clean_data(ignore_hbm=True)

        assert (data._marker_column_labels(data.mocap_column_labels) ==
                self.all_marker_labels)
        expected_columns = self.relabeled_mocap_labels_without_hbm + ['Cortex Time',
                                                            'D-Flow Time']
        for col in data.data.columns:
            assert col in expected_columns
            assert col not in self.dflow_hbm_labels
        expected_time = time_vector(self.cortex_number_of_samples, 1.0 /
                                    self.cortex_sample_period)
        testing.assert_allclose(data.data['Cortex Time'], expected_time)

        try:
            data.meta
        except AttributeError:
            assert False

        assert_raises(AttributeError, lambda: data.record_data)

        # Without the mocap file.
        data = DFlowData(record_tsv_path=self.path_to_record_data_file,
                         meta_yml_path=self.path_to_meta_data_file)
        data.clean_data(ignore_hbm=True)

        assert_raises(AttributeError, lambda: data.mocap_column_labels)

        expected_columns = self.record_labels
        for col in data.data.columns:
            assert col in expected_columns
            assert col not in ['TimeStamp', 'Cortex Time', 'D-Flow Time',
                               'FrameNumber']
            assert col not in self.dflow_hbm_labels
            assert col not in self.mocap_labels_with_hbm
            assert col not in self.cortex_analog_labels
        expected_time = time_vector(self.cortex_number_of_samples, 1.0 /
                                    self.cortex_sample_period)
        testing.assert_allclose(data.data['Time'], self.record_time)

        try:
            data.meta
        except AttributeError:
            assert False

        assert_raises(AttributeError, lambda: data.mocap_data)

        # Without record file and meta data.
        data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file)
        data.clean_data(ignore_hbm=True)

        assert (data._marker_column_labels(data.mocap_column_labels) ==
                self.all_marker_labels)
        expected_columns = self.default_mocap_labels_without_hbm + ['Cortex Time',
                                                            'D-Flow Time']
        for col in data.data.columns:
            assert col in expected_columns
            assert col not in self.dflow_hbm_labels
            assert col not in ['Time', 'RightBeltSpeed', 'LeftBeltSpeed',
                               self.dflow_hbm_labels]
        expected_time = time_vector(self.cortex_number_of_samples, 1.0 /
                                    self.cortex_sample_period)
        testing.assert_allclose(data.data['Cortex Time'], expected_time)

        assert_raises(AttributeError, lambda: data.meta)
        assert_raises(AttributeError, lambda: data.record_data)

        # Without mocap file and meta data.
        data = DFlowData(record_tsv_path=self.path_to_record_data_file)
        data.clean_data(ignore_hbm=True)

        assert_raises(AttributeError, lambda: data.mocap_column_labels)

        expected_columns = self.record_labels
        for col in data.data.columns:
            assert col in expected_columns
            assert col not in ['Cortex Time', 'D-Flow Time']
            assert col not in self.mocap_labels_with_hbm
        expected_time = time_vector(self.cortex_number_of_samples, 1.0 /
                                    self.cortex_sample_period)
        testing.assert_allclose(data.data['Time'], self.record_time)

        assert_raises(AttributeError, lambda: data.meta)
        assert_raises(AttributeError, lambda: data.mocap_data)

    def test_clean_data_options(self):

        data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                         record_tsv_path=self.path_to_record_data_file,
                         meta_yml_path=self.path_to_meta_data_file)

        # Default options
        data.clean_data()

        # No nans should be present.
        assert not pandas.isnull(data.data).any().any()
        # HBM columns should be present
        assert 'RKneeFlexion.Ang' in data.data.columns

        # Don't identify missing markers or missing values in HBM data.
        data.clean_data(id_na=False)

        # No nans should be present.
        assert not pandas.isnull(data.data).any().any()
        # HBM columns should be present
        assert 'RKneeFlexion.Ang' in data.data.columns

        # Identify missing markers but don't interpolate.
        data.clean_data(interpolate=False)

        # Nans should be present.
        assert pandas.isnull(data.data).any().any()
        # HBM columns should be present
        assert 'RKneeFlexion.Ang' in data.data.columns

        # Don't load HBM data.
        data.clean_data(ignore_hbm=True)

        # Nans should not be present.
        assert not pandas.isnull(data.data).any().any()
        # HBM columns should be present
        assert 'RKneeFlexion.Ang' not in data.data.columns

    def test_extract_processed_data(self):
        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                               record_tsv_path=self.path_to_record_data_file,
                               meta_yml_path=self.path_to_meta_data_file)
        dflow_data.clean_data()

        zeroing_data_frame = dflow_data.extract_processed_data(event='Zeroing')

        start_i = np.argmin(np.abs(dflow_data.data['TimeStamp'] -
                                   self.n_event_times[0]))

        stop_i = np.argmin(np.abs(dflow_data.data['TimeStamp'] -
                                  self.n_event_times[1]))

        compare_data_frames(zeroing_data_frame,
                            dflow_data.data.iloc[start_i:stop_i, :])

        # The next two statements test issue #57, i.e. extracting multiple
        # copies of the processed data with index column rewrite.
        timestamp_index_data_frame = dflow_data.extract_processed_data(index_col='TimeStamp')

        timestamp_index_data_frame = \
            dflow_data.extract_processed_data(event='Zeroing',
                                              index_col='TimeStamp')

        timestamp_index_data_frame = \
            dflow_data.extract_processed_data(event='Zeroing',
                                              index_col='TimeStamp',
                                              isb_coordinates=True)

        # TODO : Make some assertions!

    def teardown(self):
        os.remove(self.path_to_mocap_data_file)
        os.remove(self.path_to_record_data_file)
        os.remove(self.path_to_meta_data_file)
        os.remove(self.path_to_compensation_data_file)
