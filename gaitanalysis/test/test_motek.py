#!/usr/bin/env python
# -*- coding: utf-8 -*-

# builtin
import os
from time import strptime

# external
import numpy as np
from numpy import testing
import pandas
from pandas.util.testing import assert_frame_equal
from nose.tools import assert_raises
import yaml

# local
from ..motek import DFlowData
from utils import compare_data_frames
from dtk.process import time_vector

# debugging
try:
    from IPython.core.debugger import Tracer
except ImportError:
    pass
else:
    set_trace = Tracer()


class TestDFlowData():

    cortex_start_frame = 2375
    cortex_sample_period = 0.01
    cortex_number_of_samples = 501
    missing_marker_start_indices = [78, 158, 213, 401, 478]
    length_missing = [2, 5, 8, 10, 12]

    dflow_start_time = 51.687
    dflow_mocap_max_period_deviation = 0.0012
    dflow_record_max_sample_period = 1.0 / 10.0
    dflow_record_min_sample_period = 1.0 / 300.0
    dflow_record_number_of_samples = 1000

    cortex_marker_labels = ['T10.PosX',
                            'T10.PosY',
                            'T10.PosZ']
    cortex_analog_labels = ['FP1.ForX',
                            'FP1.MomX',
                            'Channel1.Anlg',
                            'Channel2.Anlg']
    dflow_hbm_labels = ['RKneeFlexion.Ang',
                        'RKneeFlexion.Mom',
                        'RKneeFlexion.Pow',
                        'R_PectoralisMajorTH1',
                        'L_RectusFemoris']
    mocap_labels_without_hbm = (['TimeStamp', 'FrameNumber'] +
                                cortex_marker_labels + cortex_analog_labels)
    mocap_labels_with_hbm = mocap_labels_without_hbm + dflow_hbm_labels
    record_labels = ['Time', 'RightBeltSpeed', 'LeftBeltSpeed']

    path_to_mocap_data_file = 'example_mocap_tsv_file.txt'
    path_to_record_data_file = 'example_record_tsv_file.txt'
    path_to_meta_data_file = 'example_meta_data_file.yml'

    meta_data = {'trial': {'id': 5,
                           'datetime': strptime('2013-10-03', "%Y-%m-%d")},
                 'subject': {'id': 234,
                             'age': 28,
                             'mass': 70,
                             'mass-units': 'kilogram'},
                 'study': {'id': 12,
                           'name': 'Human Locomotion Control Identification',
                           'description': 'Perturbations during walking and running.'},
                 'files': [path_to_mocap_data_file,
                           path_to_record_data_file],
                 'events': {'A': 'Zeroing',
                            'B': 'Walking',
                            'C': 'Relaxing'},
                 'units': {
                     '.*\.Pos[XYZ]$': 'meters',
                     '^[LR]_.*': 'newtons',
                     '.*\.Mom$': 'newton-meters',
                     '.*\.Ang$': 'degrees',
                     '.*\.Pow$': 'watts'
                     },
                 'analog-channel-names': {
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
                    }
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

        for label in self.cortex_marker_labels:
            mocap_data[label] = np.sin(mocap_data['TimeStamp'])

        for label in self.cortex_analog_labels:
            mocap_data[label] = np.cos(mocap_data['TimeStamp'])

        for label in self.dflow_hbm_labels:
            mocap_data[label] = np.cos(mocap_data['TimeStamp'])

        self.mocap_data_frame = pandas.DataFrame(mocap_data)

        for j, index in enumerate(self.missing_marker_start_indices):
            for signal in self.cortex_marker_labels:
                self.mocap_data_frame[signal][index:index + self.length_missing[j]] = \
                    self.mocap_data_frame[signal][index]

        self.mocap_data_frame.to_csv(self.path_to_mocap_data_file, sep='\t',
                                     float_format='%1.6f', index=False,
                                     cols=self.mocap_labels_with_hbm)

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
        """

        variable_periods = (self.dflow_record_min_sample_period +
                            (self.dflow_record_max_sample_period -
                             self.dflow_record_min_sample_period) *
                            np.random.random(self.dflow_record_number_of_samples))

        record_data = {}
        self.record_time = self.dflow_start_time + np.cumsum(variable_periods)
        record_data['Time'] = self.record_time
        record_data['LeftBeltSpeed'] = \
            np.random.random(self.dflow_record_number_of_samples)
        record_data['RightBeltSpeed'] = \
            np.random.random(self.dflow_record_number_of_samples)

        self.record_data_frame = pandas.DataFrame(record_data)
        # TODO : Pandas 0.11.0 does not have a cols argument.
        # http://pandas.pydata.org/pandas-docs/version/0.10.1/generated/pandas.Series.to_csv.html
        self.record_data_frame.to_csv(self.path_to_record_data_file,
                                      sep='\t', float_format='%1.6f',
                                      index=False, cols=['Time',
                                                         'LeftBeltSpeed',
                                                         'RightBeltSpeed'])
        event_template = "#\n# EVENT {} - COUNT {}\n#\n"

        time = self.record_data_frame['Time']

        event_times = {'A': time[333],
                       'B': time[784],
                       'C': time[955]}

        # This loops through the record file and inserts the events.
        new_lines = ''
        with open(self.path_to_record_data_file, 'r') as f:
            for line in f:
                new_lines += line

                if 'Time' in line:
                    time_col_index = line.strip().split('\t').index('Time')

                time_string = line.strip().split('\t')[time_col_index]

                for key, value in event_times.items():
                    if '{:1.6f}'.format(value) == time_string:
                        new_lines += event_template.format(key, '1')

        new_lines += "\n".join("# EVENT {} occured 1 time".format(letter)
                               for letter in ['A', 'B', 'C'])

        with open(self.path_to_record_data_file, 'w') as f:
            f.write(new_lines)

    def setup(self):
        self.create_sample_mocap_file()
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
                     'non_hbm_column_indices']:
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
                     'non_hbm_column_indices']:
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
                     'non_hbm_column_indices']:
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

    def test_mocap_column_labels(self):

        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file)

        assert self.mocap_labels_with_hbm == dflow_data._mocap_column_labels()

    def test_marker_column_labels(self):

        dflow_data = DFlowData(self.path_to_mocap_data_file)
        all_labels = dflow_data.mocap_column_labels
        labels = dflow_data._marker_column_labels(all_labels)

        assert labels == self.cortex_marker_labels

    def test_hbm_column_labels(self):

        dflow_data = DFlowData(self.path_to_mocap_data_file)
        all_labels = dflow_data.mocap_column_labels

        hbm_lab, hbm_i, non_hbm_i = dflow_data._hbm_column_labels(all_labels)

        assert self.dflow_hbm_labels == hbm_lab
        assert hbm_i == range(len(self.mocap_labels_without_hbm),
                              len(self.mocap_labels_with_hbm))
        assert non_hbm_i == range(len(self.mocap_labels_without_hbm))

    def test_identify_missing_markers(self):
        dflow_data = DFlowData(self.path_to_mocap_data_file)
        data_frame = dflow_data._load_mocap_data(ignore_hbm=True)
        identified = dflow_data._identify_missing_markers(data_frame)

        for i, index in enumerate(self.missing_marker_start_indices):
            for suffix in ['.PosX', '.PosY', '.PosZ']:
                assert all(identified['T10' + suffix][index + 1:index + self.length_missing[i]].isnull())

    def test_generate_cortex_time_stamp(self):
        dflow_data = DFlowData(self.path_to_mocap_data_file)
        data = dflow_data._generate_cortex_time_stamp(self.mocap_data_frame)
        expected_time = time_vector(self.cortex_number_of_samples, 1.0 /
                                    self.cortex_sample_period)
        testing.assert_allclose(data['Cortex Time'], expected_time)

    def test_interpolate_missing_markers(self):
        dflow_data = DFlowData(self.path_to_mocap_data_file)

        with_missing = pandas.DataFrame({
            'TimeStamp': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'FP1.ForX': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'T10.PosX': [1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0],
            'T10.PosY': [1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0],
            'T10.PosZ': [1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0]})

        interpolated = dflow_data._interpolate_missing_markers(with_missing)

        without_missing = pandas.DataFrame({
            'TimeStamp': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'FP1.ForX': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'T10.PosX': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'T10.PosY': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'T10.PosZ': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})

        assert not pandas.isnull(interpolated).any().any()
        assert_frame_equal(interpolated, without_missing)

        dflow_data = DFlowData(self.path_to_mocap_data_file)
        mocap_data_frame = dflow_data._load_mocap_data(ignore_hbm=True)
        identified = dflow_data._identify_missing_markers(mocap_data_frame)
        interpolated = dflow_data._interpolate_missing_markers(identified)

        assert not pandas.isnull(interpolated).any().any()

        testing.assert_allclose(interpolated['T10.PosX'].values,
                                np.sin(interpolated['TimeStamp']).values,
                                atol=1e-3)
        testing.assert_allclose(interpolated['T10.PosY'].values,
                                np.sin(interpolated['TimeStamp']).values,
                                atol=1e-3)
        testing.assert_allclose(interpolated['T10.PosZ'].values,
                                np.sin(interpolated['TimeStamp']).values,
                                atol=1e-3)

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
        pass

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

    def test_clean_data(self):
        data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                         record_tsv_path=self.path_to_record_data_file,
                         meta_yml_path=self.path_to_meta_data_file)

        data.clean_data()

        # TODO : Check for an events dictionary if the record file included
        # events.

        assert not pandas.isnull(data.data).any().any()
        assert (data._marker_column_labels(data.mocap_column_labels) ==
                self.cortex_marker_labels)
        expected_columns = self.mocap_labels_without_hbm + \
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
        data.clean_data()

        assert not pandas.isnull(data.data).any().any()
        assert (data._marker_column_labels(data.mocap_column_labels) ==
                self.cortex_marker_labels)
        expected_columns = self.mocap_labels_without_hbm + ['Cortex Time',
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
        data.clean_data()

        assert not pandas.isnull(data.data).any().any()
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
        data.clean_data()

        assert not pandas.isnull(data.data).any().any()
        assert (data._marker_column_labels(data.mocap_column_labels) ==
                self.cortex_marker_labels)
        expected_columns = self.mocap_labels_without_hbm + ['Cortex Time',
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
        data.clean_data()

        assert not pandas.isnull(data.data).any().any()
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

    def test_extract_data(self):
        dflow_data = DFlowData(mocap_tsv_path=self.path_to_mocap_data_file,
                               record_tsv_path=self.path_to_record_data_file,
                               meta_yml_path=self.path_to_meta_data_file)
        dflow_data.clean_data()

        # returns all signals with time stamp as the index for the whole
        # measurement
        full_run_data_frame = dflow_data.extract_data()

    def teardown(self):
        os.remove(self.path_to_mocap_data_file)
        os.remove(self.path_to_record_data_file)
        os.remove(self.path_to_meta_data_file)

