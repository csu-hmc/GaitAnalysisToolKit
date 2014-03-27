#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import re
import os

# external libraries
import numpy as np
import pandas
from scipy.interpolate import InterpolatedUnivariateSpline
import yaml
from dtk import process
from oct2py import octave

# debugging
try:
    from IPython.core.debugger import Tracer
except ImportError:
    pass
else:
    set_trace = Tracer()


def markers_for_2D_inverse_dynamics(marker_set='lower'):
    """Returns lists of markers from the D-Flow human body model (lower or
    full), that should be used with leg2d.m.

    Parameters
    ----------
    marker_set : string, optional, default='lower'
        Specify either 'lower' or 'full' depending on which marker set you
        used.

    Returns
    -------
    left_marker_coords : list of strings, len(12)
        The Z and Y coordinates for the 6 left markers.
    right_marker_coords : list of strings, len(12)
        The Z and Y coordinates for the 6 right markers.
    left_forces : list of strings, len(3)
        The Z and Y ground reaction forces and the X ground reaction moment
        of the left leg.
    right_forces : list of strings, len(3)
        The Z and Y ground reaction forces and the X ground reaction moment
        of the left leg.

    Notes
    -----
    The are the correct markers to use in the 2D inverse dynamics
    calculations and in the correct order but the signs must be changed for
    the Z values when used:

    - The D-Flow X unit vector is equal to the leg2d Z unit vector.
    - The D-Flow Y unit vector is equal to the leg2d Y unit vector.
    - The D-FLow Z unit vector is equal to the leg2d -X unit vector.

    The forces and moments must be normalized by body mass before using with
    leg2d.m.

    """

    # NOTE : leg2D requires the shoulder marker, but we only have the
    # sternum and the xyphoid process available in the lower marker set.

    # Use the actual shoulder marker if available.
    if marker_set == 'lower':
        shoulder = 'STRN'
    elif marker_set == 'full':
        shoulder = 'SHO'

    six_markers = [shoulder, 'GTRO', 'LEK', 'LM', 'HEE', 'MT5']

    left_markers = ['L' + m if m != 'STRN' else m for m in six_markers]
    right_markers = ['R' + m if m != 'STRN' else m for m in six_markers]

    left_marker_coords = []
    right_marker_coords = []
    for ml, mr in zip(left_markers, right_markers):
        left_marker_coords.append(ml + '.PosZ')
        left_marker_coords.append(ml + '.PosY')
        right_marker_coords.append(mr + '.PosZ')
        right_marker_coords.append(mr + '.PosY')

    left_forces = ['FP1.ForZ', 'FP1.ForY', 'FP1.MomX']
    right_forces = ['FP2.ForZ', 'FP2.ForY', 'FP2.MomX']

    return left_marker_coords, right_marker_coords, left_forces, right_forces


class MissingMarkerIdentifier(object):

    marker_coordinate_suffixes = ['.Pos' + _c for _c in ['X', 'Y', 'Z']]

    constant_marker_tolerance = 1e-16

    def __init__(self, data_frame):
        """Instantiates the class with a data frame attribute.

        Parameters
        ----------
        data_frame : pandas.DataFrame, size(n, m)
            A data frame which contains at least some columns of marker
            position time histories. The marker time histories should
            contain periods of constant values. The column names should have
            one of three suffixes: `.PosX`, `.PosY`, or `.PosZ`.

        """

        self.data_frame = data_frame

    def identify(self, columns=None):
        """Returns the data frame in which all or the specified columns have
        had constant values replaced with NaN.

        Returns
        -------
        data_frame : pandas.DataFrame, size(n, m)
            The same data frame which was supplied with constant values
            replaced with NaN.
        columns : list of strings, optional, default=None
            The specific list of columns in the data frame that should be
            analyzed. This is typically a list of all marker columns.

        Notes
        -----
        D-Flow replaces missing marker values with the last available
        measurement in time. This method is used to properly replace them
        with a unique identifier, NaN. If two adjacent measurements in time
        were actually the same value, then this method will replace the
        subsequent ones with NaNs, and is not correct, but the likelihood of
        this happening is low.

        """
        if columns is None:
            columns = self.data_frame.columns

        self.columns = columns

        for col in columns:
            if not any(col.endswith(suffix) for suffix in
                       self.marker_coordinate_suffixes):
                raise ValueError('Please pass in a list of only marker columns. ' +
                                 'The columns should have .Pos[XYZ] as a suffixes.')

        # A list of unique markers in the data set (i.e. without the
        # suffixes).
        unique_marker_names = list(set([c.split('.')[0] for c in columns]))

        # Create a boolean array that labels all constant values (wrt to
        # tolerance) as True.
        are_constant = (self.data_frame[columns].diff().abs() <
                        self.constant_marker_tolerance)

        # Now make sure that the marker is constant in all three coordinates
        # before setting it to NaN.
        for marker in unique_marker_names:
            single_marker_cols = [marker + pos for pos in
                                  self.marker_coordinate_suffixes]
            for col in single_marker_cols:
                are_constant[col] = \
                    are_constant[single_marker_cols].all(axis=1)

        self.data_frame[are_constant] = np.nan

        return self.data_frame

    def statistics(self):
        """Returns a data frame containing the number of missing samples and
        maximum number of consecutive missing samples for each column."""

        # TODO : Maybe just move this into
        # DFlowData.missing_marker_statistics and make this class a
        # function.

        # TODO : This would be nicer if it gave length of each gap for each
        # marker. Then you could count the number of gaps, find the max gap,
        # and total the missing samples easily. The np.diff(index) - 1 gives
        # length of each gap.

        try:
            self.columns
        except AttributeError:
            raise StandardError("You must run the `identify()` method " +
                                "before computing the statistics.")

        df = self.data_frame[self.columns]

        number_nan = len(df) - df.count()

        not_missing = df.unstack().dropna()
        max_missing = []
        for col in df.columns:
            index = not_missing[col].index.values
            max_missing.append(np.max(np.diff(index)) - 1)

        max_missing = pandas.Series(max_missing, index=df.columns)

        return pandas.DataFrame([number_nan, max_missing],
                                index=['# NA', 'Largest Gap']).T


def spline_interpolate_over_missing(data_frame, abscissa_column, order=1,
                                    columns=None):
    """Returns the data frame with all missing values replaced by some
    interpolated or extrapolated values derived from a spline.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        A data frame which contains a column for the abscissa and other
        columns which may or may not have missing values, i.e. NaN.
    abscissa_column : string
        The column name which represents the abscissa.
    order : integer, optional, default=1
        The order of the spline. Can be 1 through 5 for linear through
        quintic splines. The default is a linear spline. See documentation
        for scipy.interpolate.InterpolatedUnivariateSpline.
    columns : list of strings, optional, default=None
        If only a particular set of columns need interpolation, they can be
        specified here.

    Returns
    -------
    data_frame : pandas.DataFrame
        The same data frame passed in with all NaNs in the specified columns
        replaced with interpolated or extrapolated values.

    """

    # Pandas 0.13.0 will have all the SciPy interpolation functions built
    # in. But for now, we've got to do this manually.

    # TODO : DataFrame.apply() might clean this code up.

    if columns is None:
        columns = list(data_frame.columns)

    try:
        columns.remove(abscissa_column)
    except ValueError:
        pass

    for column in columns:
        time_series = data_frame[column]
        is_null = time_series.isnull()
        if any(is_null):
            time = data_frame[abscissa_column]
            without_na = time_series.dropna().values
            time_at_valid = time[time_series.notnull()].values
            interpolate = InterpolatedUnivariateSpline(time_at_valid,
                                                       without_na, k=order)
            interpolated_values = interpolate(time[is_null].values)
            data_frame[column][is_null] = interpolated_values

    return data_frame


def low_pass_filter(data_frame, columns, cutoff, sample_rate):
    """Returns the data frame with indicated columns filtered with a low
    pass second order forward/backward Butterworth filter.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        A data frame with time series columns.
    columns : sequence of strings
        The columns that should be filtered.
    cutoff : float
        The low pass cutoff frequency in Hz.
    sample_rate : float
        The sample rate of the time series in Hz.

    Returns
    -------
    data_frame : pandas.DataFrame
        The same data frame which was passed in with the specified columns
        replaced by filtered versions.

    """

    data_frame[columns] = process.butterworth(data_frame[columns].values,
                                              cutoff, sample_rate, axis=0)
    return data_frame


class DFlowData(object):
    """A class to store and manipulate the data outputs from Motek Medical's
    D-Flow software."""

    marker_coordinate_suffixes = ['.Pos' + _c for _c in ['X', 'Y', 'Z']]
    marker_coordinate_regex = '.*\.Pos[XYZ]$'

    # In the motek output file there is a space preceding only these two
    # column names: ' L_Psoas' ' R_Psoas'     

    hbm_column_regexes = ['^\s?[LR]_.*', '.*\.Mom$', '.*\.Ang$', '.*\.Pow$',
                          '.*\.COM.[XYZ]$']
    analog_channel_regex = '^Channel[0-9]+\.Anlg$'

    force_plate_names = ['FP1', 'FP2']  # FP1 : left, FP2 : right
    force_plate_suffix = [_suffix_beg + _end for _end in ['X', 'Y', 'Z'] for
                          _suffix_beg in ['.For', '.Mom', '.Cop']]
    # TODO : Check if this is correct.
    force_plate_regex = '^FP[12]\.[For|Mom|Cop][XYZ]$'

    # TODO: There are surely more segment names for the full body. Need to
    # get those.
    dflow_segments = ['pelvis', 'thorax', 'spine', 'pelvislegs', 'lfemur',
                      'ltibia', 'lfoot', 'toes', 'rfemur', 'rtibia',
                      'rfoot', 'rtoes']

    rotation_suffixes = ['.Rot' + c for c in ['X', 'Y', 'Z']]
    segment_labels = [_segment + _suffix for _segment in dflow_segments for
                      _suffix in marker_coordinate_suffixes +
                      rotation_suffixes]

    # TODO: These should be stored in the meta data for each trial because
    # the names could theorectically change, as they are selected by the
    # user. They are used to express the forces in the global reference
    # frame.
    treadmill_markers = ['ROT_REF.PosX', 'ROT_REF.PosY', 'ROT_REF.PosZ',
                         'ROT_C1.PosX', 'ROT_C1.PosY', 'ROT_C1.PosZ',
                         'ROT_C2.PosX', 'ROT_C2.PosY', 'ROT_C2.PosZ',
                         'ROT_C3.PosX', 'ROT_C3.PosY', 'ROT_C3.PosZ',
                         'ROT_C4.PosX', 'ROT_C4.PosY', 'ROT_C4.PosZ']

    cortex_sample_rate = 100  # Hz
    constant_marker_tolerance = 1e-16  # meters
    low_pass_cutoff = 6.0  # Hz
    delsys_time_delay = 0.096  # seconds
    hbm_na = ['0.000000', '-0.000000']

    def __init__(self, mocap_tsv_path=None, record_tsv_path=None,
                 meta_yml_path=None):
        """Sets the data file paths, loads the meta data, if present, and
        generates lists of the columns in the mocap and record files.

        Parameters
        ----------
        mocap_tsv_path : string, optional, default=None
            The path to a tab delimited file generated from D-Flow's mocap
            module.
        record_tsv_path : string, optional, default=None
            The path to a tab delimited file generated from D-Flow's record
            module.
        meta_yml_path : string, optional, default=None
            The path to a yaml file.

        Notes
        -----
        You must supply at least either a mocap or record file. If you
        supply both, they should be from the same run. The meta data file is
        always optional, but without it some class methods or options will
        be disabled.

        """

        # TODO : Support passing only a meta data file or directory with a
        # meta data file, so long at the metadata file has all the files
        # specified in it.

        if mocap_tsv_path is None and record_tsv_path is None:
            raise ValueError("You must supply at least a D-Flow mocap file "
                             + "or a D-Flow record file.")

        self.mocap_tsv_path = mocap_tsv_path
        self.record_tsv_path = record_tsv_path
        self.meta_yml_path = meta_yml_path

        if self.meta_yml_path is not None:
            self.meta = self._parse_meta_data_file()

        if self.mocap_tsv_path is not None:

            self.mocap_column_labels = self._mocap_column_labels()

            self.marker_column_labels = \
                self._marker_column_labels(self.mocap_column_labels)

            (self.hbm_column_labels, self.hbm_column_indices,
             self.non_hbm_column_indices) = \
                self._hbm_column_labels(self.mocap_column_labels)

            (self.analog_column_labels, self.analog_column_indices,
                self.emg_column_labels, self.accel_column_labels) = \
                self._analog_column_labels(self.mocap_column_labels)

    def _parse_meta_data_file(self):
        """Returns a dictionary containing the meta data stored in the
        optional meta data file."""

        with open(self.meta_yml_path, 'r') as f:
            meta = yaml.load(f)

        return meta

    def _compensation_needed(self):
        """Returns true if the meta data includes:

           'trial: stationary-platform: False'

        """

        # TODO : This looks ridiculous, I must be doing something wrong
        # here.

        if self.meta_yml_path is not None:
            try:
                if self.meta['trial']['stationary-platform'] is False:
                    return True
                else:
                    return False
            except KeyError:
                return False
        else:
            return False

    def _store_compensation_data_path(self):
        """Stores the path to the compensation data file.

        Notes
        -----

        The meta data yaml file must include a relative file path to a mocap
        file that contains time series data appropriate for computing the
        force inertial and rotational compensations. The yaml declaration
        should look like this example:

        files:
            mocap: mocap-378.txt
            record: record-378.txt
            meta: meta-378.yml
            compensation: ../path/to/mocap/file.txt

        """

        trial_directory = os.path.split(self.mocap_tsv_path or
                                        self.record_tsv_path)[0]

        try:
            relative_path_to_unloaded_mocap_file = \
                self.meta['trial']['files']['compensation']
        except KeyError:
            raise Exception('You must include relative file path to the ' +
                            'compensation file in {}.'.format(self.meta_yml_path))
        except AttributeError:  # no meta
            raise Exception('You must include a meta data file with a ' +
                            'relative file path to the compensation file.')
        else:
            self.compensation_tsv_path = \
                os.path.join(trial_directory,
                             relative_path_to_unloaded_mocap_file)

    def _load_compensation_data(self):
        """Returns a data frame which includes the treadmill forces/moments,
        and the accelerometer signals as time series with respect to the
        D-Flow time stamp."""

        compensation_data = pandas.read_csv(self.compensation_tsv_path, delimiter='\t')
        compensation_data = self._relabel_analog_columns(compensation_data)

        force_labels = \
            self._force_column_labels(without_center_of_pressure=True)
        accel_labels = self.accel_column_labels
        necessary_columns = ['TimeStamp'] + force_labels + accel_labels

        #all_columns = self._header_labels(self.compensation_tsv_path)
        #indices = []
        #for i, label in enumerate(all_columns):
        #    if label in necessary_columns:
        #        indices.append(i)

        return compensation_data[necessary_columns]

    def _clean_compensation_data(self, data_frame):
        """Returns a the data frame with Delsys signals shifted and all
        signals low pass filtered.

        Parameters
        ----------
        data_frame : pandas.DataFrame
            This data frame should contain only the columns needed for the
            compensation calculations: accelerometers and forces/moments.

        Returns
        -------
        data_frame : pandas.DataFrame
            The cleaned compensation data.

        """

        data_frame = self._shift_delsys_signals(data_frame)

        columns = list(data_frame.columns)
        columns.remove('TimeStamp')

        data_frame = low_pass_filter(data_frame, columns,
                                     self.low_pass_cutoff,
                                     self.cortex_sample_rate)

        return data_frame

    def _shift_delsys_signals(self, data_frame, time_col='TimeStamp'):
        """Returns a data frame in which the Delsys columns are linearly
        interpolated (and extrapolated) at the time they were actually
        measured."""

        delsys_time = data_frame[time_col] - self.delsys_time_delay
        delsys_labels = self.emg_column_labels + self.accel_column_labels

        for delsys_label in set(data_frame.columns).intersection(delsys_labels):
            interpolate = InterpolatedUnivariateSpline(delsys_time,
                                                       data_frame[delsys_label],
                                                       k=1)
            data_frame[delsys_label] = interpolate(data_frame[time_col])

        return data_frame

    @staticmethod
    def _header_labels(path_to_file, delimiter='\t'):
        """Returns a list of labels from the header, i.e. the first line of
        a delimited text file.

        Parameters
        ----------
        path_to_file : string
            Path to the delimited text file with a header on the first line.
        delimiter : string, optional, default='\t'
            The delimiter used in the file.

        Returns
        -------
        header_labels : list of strings
            A list of the headers in order as included from the file.

        """

        with open(path_to_file, 'r') as f:
            header_labels = f.readline().strip().split(delimiter)

        return header_labels

    def _mocap_column_labels(self):
        """Returns a list of strings containing the motion capture file's
        column labels. The list is in the same order as in the mocap tsv
        file."""

        return self._header_labels(self.mocap_tsv_path)

    def _marker_column_labels(self, labels):
        """Returns a list of column labels that correpsond to markers, i.e.
        ones that end in '.PosX', '.PosY', or '.PosZ', given a master list.

        Parameters
        ----------
        labels : list of strings
            This should be a superset of column labels, some of which may be
            marker column labels.

        Returns
        -------
        marker_labels : list of strings
            The labels of columns of marker time series in the order found
            in `labels`.

        """

        reg_exp = re.compile(self.marker_coordinate_regex)

        marker_labels = []
        for i, label in enumerate(labels):
            if reg_exp.match(label) and label not in self.segment_labels:
                marker_labels.append(label)

        return marker_labels

    def _hbm_column_labels(self, labels):
        """Returns a list of human body model column labels, the indices of
        the labels, and the indices of the non-hbm labels in relation to the
        rest of the header.

        Parameters
        ----------
        labels : list of strings
            This should be a superset of column labels, some of which may be
            human body model results.

        Returns
        -------
        hbm_labels : list of strings
            The labels of columns of HBM data time series in the order found
            in `labels`.
        hbm_indices : list of integers
            The indices of the HBM columns with respect to the indices of
            `labels`.
        non_hbm_indices : list of integers
            The indices of the non-HBM columns with respect to the indices
            of `labels`.

        """

        hbm_labels = []
        hbm_indices = []
        non_hbm_indices = []

        reg_exps = [re.compile(regex) for regex in self.hbm_column_regexes]

        for i, label in enumerate(labels):
            if any(exp.match(label) for exp in reg_exps):
                hbm_indices.append(i)
                hbm_labels.append(label)
            else:
                non_hbm_indices.append(i)

        return hbm_labels, hbm_indices, non_hbm_indices

    def _analog_column_labels(self, labels):
        """Returns a list of analog channel column labels and the indices of
        the labels.

        Parameters
        ----------
        labels : list of strings
            This should be a superset of column labels, some of which may be
            human body model results.

        Returns
        -------
        analog_labels : list of strings
            The labels of analog channels in the order found
            in `labels`.
        analog_indices : list of integers
            The indices of the analog columns with respect to the indices of
            `labels`.
        emg_column_labels : list of strings
            The labels of emg channels in the order found in `labels`
        accel_column_labels : list of strings
            The labels of accelerometer channels in the order found
            in `labels`
        """

        def delsys_column_labels():
            """Returns the default EMG and Accelerometer column labels in which
            the Delsys system is connected."""
    
            number_delsys_sensors = 16
    
            emg_analog_numbers = [4 * n + 13 for n in
                                  range(number_delsys_sensors)]
    
            accel_analog_numbers = [4 * n + m + 14 for n in
                                    range(number_delsys_sensors) for m in
                                    range(3)]
    
            emg_column_labels = ['Channel{}.Anlg'.format(4 * n + 13) for n in
                                 range(number_delsys_sensors)]
    
            accel_column_labels = ['Channel{}.Anlg'.format(4 * n + m + 14) for n
                                   in range(number_delsys_sensors) for m in
                                   range(3)]
    
            return emg_column_labels, accel_column_labels

        # All analog channels
        analog_labels = []
        analog_indices = []

        reg_exp = re.compile(self.analog_channel_regex)

        for i, label in enumerate(labels):
            if reg_exp.match(label):
                analog_labels.append(label)
                analog_indices.append(i)

        # EMG and Accelerometer channels
        emg_column_labels, accel_column_labels = delsys_column_labels()

        # Remove channels from the default list that are not in `analog_labels`
        emg_column_labels = [label for label in emg_column_labels if label in analog_labels]
        accel_column_labels = [label for label in accel_column_labels if label in analog_labels]

        return analog_labels, analog_indices, emg_column_labels, accel_column_labels

    def _force_column_labels(self, without_center_of_pressure=False):
        """Returns a list of force column labels.

        Parameters
        ----------
        without_center_of_pressure: boolean, optional, default=False
            If true, the center of pressure labels will not be included in
            the list.

        Returns
        -------
        labels : list of strings
            A list of the force plate related signals.

        """

        if without_center_of_pressure is True:
            f = lambda suffix: 'Cop' in suffix
        else:
            f = lambda suffix: True

        return [side + suffix for side in self.force_plate_names for suffix
                in self.force_plate_suffix if not f(suffix)]

    def _relabel_analog_columns(self, data_frame):
        """
        Relabels analog channels in data frame to names defined in the
        yml meta file. Channels not specified in the meta file are keep
        their original names.
        self.analog_column_labels, self.emg_column_labels, and 
        self.accel_column_labels are updated with the new names.

        Parameters
        ==========
        data_frame : pandas.DataFrame, size(n, m)

        Returns
        =======
        data_frame : pandas.DataFrame, size(n, m
            The same data frame with columns relabeled.

        """

        # default channel names
        force_channels = {'Channel1.Anlg': 'F1Y1',
                'Channel2.Anlg': 'F1Y2',
                'Channel3.Anlg': 'F1Y3',
                'Channel4.Anlg': 'F1X1',
                'Channel5.Anlg': 'F1X2',
                'Channel6.Anlg': 'F1Z1',
                'Channel7.Anlg': 'F2Y1',
                'Channel8.Anlg': 'F2Y2',
                'Channel9.Anlg': 'F2Y3',
                'Channel10.Anlg': 'F2X1',
                'Channel11.Anlg': 'F2X2',
                'Channel12.Anlg': 'F2Z1'}

        # default name format: SensorXX_AccX
        num_force_plate_channels = 12
        signals = ['EMG', 'AccX','AccY','AccZ']
        num_sensors = 16
        sensor_channels = {'Channel' + str(4*i+(j+1)+num_force_plate_channels) +
            '.Anlg': 'Sensor' + str(i+1).zfill(2) + '_' +
            signal for i in range(num_sensors) for j,signal in enumerate(signals)}
        
        channel_names = dict(force_channels.items() + sensor_channels.items())
        
        # update labels from meta file
        try:
            channel_names.update(self.meta['trial']['analog-channel-names'])
        except:
            pass

        data_frame.rename(columns=channel_names, inplace=True)

        for column_label_list in (self.analog_column_labels, self.emg_column_labels, self.accel_column_labels):
            for i,label in enumerate(column_label_list):
                if label in channel_names:
                    column_label_list[i] = channel_names[label]
        
        return data_frame

    def _identify_missing_markers(self, data_frame):
        """Returns the data frame in which all marker columns have had
        constant marker values replaced with NaN.

        Parameters
        ----------
        data_frame : pandas.DataFrame, size(n, m)
            A data frame which contains columns of marker position time
            histories. The marker time histories may contain periods of
            constant values.

        Returns
        -------
        data_frame : pandas.DataFrame, size(n, m)
            The same data frame which was supplied expect that constant
            values in the marker columns have been replaced with NaN.

        Notes
        -----
        D-Flow replaces missing marker values with the last available
        measurement in time. This method is used to properly replace them
        with a unique idnetifier, NaN. If two adjacent measurements in time
        were actually the same value, then this method will replace the
        subsequent ones with NaNs, and is not correct, but the likelihood of
        this happening is low.

        """

        # For each marker column we need to identify the constants values
        # and replace with NaN, only if the values are constant in all
        # coordinates of a marker.

        identifier = MissingMarkerIdentifier(data_frame)
        data_frame = identifier.identify(columns=self.marker_column_labels)

        return data_frame

    def _generate_cortex_time_stamp(self, data_frame):
        """Returns the data frame with a new index based on the constant
        sample rate from Cortex."""

        # It doesn't seem that cortex frames are ever dropped (i.e. missing
        # frame number in the sequence). But if that is ever the case, this
        # function needs to be modified to deal with that and to generate
        # the new time stamp with the frame number column instead of a
        # generic call to the time_vector function.

        self.cortex_num_samples = len(data_frame)
        self.cortex_time = process.time_vector(self.cortex_num_samples,
                                               self.cortex_sample_rate)
        data_frame['Cortex Time'] = self.cortex_time
        data_frame['D-Flow Time'] = data_frame['TimeStamp']

        return data_frame

    def _interpolate_missing_markers(self, data_frame, time_col="TimeStamp",
                                     order=1):
        """Returns the data frame with all missing markers replaced by some
        interpolated value."""

        data_frame = \
            spline_interpolate_over_missing(data_frame, time_col,
                                            order=order,
                                            columns=self.marker_column_labels)

        return data_frame

    def _load_mocap_data(self, ignore_hbm=False, id_hbm_na=False):
        """Returns a data frame generated from the tsv mocap file.

        Parameters
        ----------
        ignore_hbm : boolean, optional, default=False
            If true, the columns associated with D-Flow's real time human
            body model computations will not be loaded.
        id_hbm_na : boolean, optional, default=False
            If true and `ignore_hbm` is false, then the HBM columns will be
            loaded with all '0.000000' and '-0.000000' strings in the HBM
            columns replaced with NaN.

        Returns
        -------
        data_frame : pandas.DataFrame

        """

        if ignore_hbm is True:
            data_frame = pandas.read_csv(self.mocap_tsv_path, delimiter='\t',
                                   usecols=self.non_hbm_column_indices)
        else:
            if id_hbm_na is True:
                hbm_na_values = {k: self.hbm_na for k in
                                 self.hbm_column_labels}
                data_frame = pandas.read_csv(self.mocap_tsv_path, delimiter='\t',
                                       na_values=hbm_na_values)
            else:
                data_frame =  pandas.read_csv(self.mocap_tsv_path, delimiter='\t')

        return data_frame
        

    def missing_value_statistics(self, data_frame):
        """Returns a report of missing values in the data frame."""

        identifier = MissingMarkerIdentifier(data_frame)
        identifier._identified = True

        return identifier.statistics()

    def _extract_events_from_record_file(self):
        """Returns a dictionary of events and times. The event names will be
        the default A-F which is output by D-Flow unless you specify unique
        names in the meta data file. If there are no events in the record
        file, this will return nothing."""

        f = open(self.record_tsv_path, 'r')
        filecontents = f.readlines()
        f.close()

        end = filecontents[-6]
        end_value = end.split()
        end_value1 = end_value[0]
        end_time = float(end_value1)

        if 'EVENT' in ''.join(filecontents):
            event_time1 = []
            event_labels = []
            for i in range(len(filecontents)):
                if 'COUNT' in filecontents[i]:
                    event_labels.append(filecontents[i].split(' ')[2])
                    event = filecontents[i - 2]
                    event_data = event.split()
                    event_time1.append(float(event_data[0]))
        else:
            return

        event_time1.append(end_time)
        self.events = {}

        for i, label in enumerate(event_labels):
            self.events[label] = (event_time1[i], event_time1[i + 1])

        if self.meta_yml_path is not None:
            if 'events' in self.meta['trial']:
                new_events = {}
                event_dictionary = self.meta['trial']['events']
                for key, value in event_dictionary.items():
                    new_events[value] = self.events[key]
                self.events = new_events

    def _load_record_data(self):
        """Returns a data frame containing the data from the record
        module."""

        # The record module file is tab delimited and may have events
        # interspersed in between the rows which are commenting out by
        # hashes. We must dropna to remove commented lines from the
        # resutling data frame, only if all values in a row are NA. The
        # comment keyword argument only ingores comments at the end of each
        # line, not comments that take up an entire line, it is acutally
        # probably not even needed in this case.
        return pandas.read_csv(self.record_tsv_path, delimiter='\t',
                               comment='#').dropna(how='all').reset_index(drop=True)

    def _resample_record_data(self, data_frame):
        """Resamples the raw data from the record file at the sample rate of
        the mocap file."""

        # The 'TimeStamp' column in the mocap data is the time at which
        # D-Flow recieves the Cortex data. Each of which corresponds to a
        # Cortex time stamp. The 'Time' column from the record module is the
        # D-Flow time which corresponds to the D-Flow variable frame rate.
        # The purpose of this code is to find interpolated values from each
        # column in the record data at the Cortex time stamp.

        # Combine the D-Flow times from each data frame and sort them.
        all_times = np.hstack((data_frame['Time'],
                               self.mocap_data['TimeStamp']))
        all_times_sort_indices = np.argsort(all_times)

        # Create a dictionary which has each column from the record data
        # frame, but NaNs in the rows corresponding to the mocap 'TimeStamp'
        # in all columns but the new 'Time' column.
        total = {}
        for label, series in data_frame.iteritems():
            all_values = np.hstack((series, np.nan *
                                    np.ones(len(self.mocap_data))))
            total[label] = all_values[all_times_sort_indices]
        total['Time'] = np.sort(all_times)
        total = pandas.DataFrame(total)

        def linear_time_interpolate(series):
            # Note : scipy.interpolate.interp1d does not extrapolate, so it
            # failed here, but the general spline class does extropolate so
            # the following seems to work.
            f = InterpolatedUnivariateSpline(data_frame['Time'].values,
                                             series[series.notnull()].values,
                                             k=1)
            return f(self.mocap_data['TimeStamp'])

        new_record = {}
        for label, series in total.iteritems():
            if label != 'Time':
                new_record[label] = linear_time_interpolate(series)
        new_record['Time'] = self.cortex_time

        return pandas.DataFrame(new_record)

    def _compensate_forces(self, calibration_data_frame, data_frame):
        """Computes the forces and moments which are due to the lateral and
        pitching motions of the treadmill and subtracts them from the
        measured forces and moments based on linear acceleration
        measurements of the treadmill."""
        """Re-expresses the forces and moments measured by the treadmill to
        the earth inertial reference frame."""

        # If you accelerate the treadmill there will be forces and moments
        # measured by the force plates that simply come from the motion.
        # When external loads are applied to the force plates, you must
        # subtract these inertial forces from the measured forces to get
        # correct estimates of the body fixed externally applied forces.

        # The markers are measured with respect to the camera's inertial
        # reference frame, earth, but the treadmill forces are measured with
        # respect to the treadmill's laterally and rotationaly moving
        # reference frame. We need both to be expressed in the same inertial
        # reference frame for ease of future computations.

        mfile = os.path.abspath(os.path.join(os.path.split(__file__)[0],
                                             '..', 'Octave-Matlab-Codes',
                                             'Inertial-Compensation'))
        octave.addpath(mfile)

        forces = ['FP1.ForX', 'FP1.ForY', 'FP1.ForZ', 'FP1.MomX',
                  'FP1.MomY', 'FP1.MomZ', 'FP2.ForX', 'FP2.ForY',
                  'FP2.ForZ', 'FP2.MomX', 'FP2.MomY', 'FP2.MomZ']

        # First four accelerometers.
        accelerometers = self.accel_column_labels[:4 * 3]
        
        compensated_forces = \
                        octave.inertial_compensation(calibration_data_frame[forces].values,
                        calibration_data_frame[accelerometers].values,
                        data_frame[self.treadmill_markers].values,
                        data_frame[forces].values,
                        data_frame[accelerometers].values)

        data_frame[forces] = compensated_forces

        return data_frame

    def _compensate(self, mocap_data_frame, markers_missing=False):
        """Returns the data frame with the forces compensated.

        Parameters
        ----------
        mocap_data_frame : pandas.DataFrame
            A data frame that contains the force plate forces and moments to
            be compensated, along with the measurements of the markers and
            accelerometers that were attached to the treadmill.
        markers_missing : pandas.DataFrame
            If the treadmill markers have missing markers, this should be
            true so that they are fixed before the compensation.

        Returns
        -------
        mocap

        Notes
        -----

        This method does the following:

        1. Pulls the compensation file path from meta data.
        2. Loads the compensation file (only the necessary columns).
        3. Identifies the missing markers in the compensation file and
           interpolates to fill them.
        4. Shifts the Delsys signals to correct time.
        5. Filter the forces, accelerometer, and treadmill markers at 6 hz low pass.
        6. Compute the compensated forces (apply inertial compensation and express
            in global reference frame)
        7. Replace the force/moment measurements in the mocap data file with the
            compensated forces/moments.

        """
        if self._compensation_needed() is True:

            self._store_compensation_data_path()

            unloaded_trial = self._load_compensation_data()

            unloaded_trial = self._clean_compensation_data(unloaded_trial)

            if markers_missing is True:
                identifier = MissingMarkerIdentifier(mocap_data_frame)
                mocap_data_frame = \
                    identifier.identify(columns=self.treadmill_markers)

            spline_interpolate_over_missing(mocap_data_frame, 'TimeStamp',
                                            columns=self.treadmill_markers)

            return self._compensate_forces(unloaded_trial, mocap_data_frame)
        else:
            return mocap_data_frame

    def _orient_accelerometers(self, data_frame):
        """
        Parameters
        ==========
        mocap_data_frame : pandas.DataFrame
            DataFrame containing accelerometer signals to be placed in treadmill
            reference frame.

        Returns
        =======
        mocap_data_frame : pandas.DataFrame
            DataFrame containing accelerometer signals in treadmill
            reference frame.
        """

        def orient_accelerometer(orientation, x_prime, y_prime, z_prime):
            if np.shape(orientation) != (3, 3):
                raise ValueError("Bad orientation matrix.")
            if ((abs(orientation) != 1) * (orientation != 0)).any():
                raise ValueError("Bad orientation matrix.")
            for row in range(3):
                if abs(orientation[row,:].sum()) != 1:
                    raise ValueError("Bad orientation matrix.")
    
            if not (np.shape(x_prime) == np.shape(y_prime) == np.shape(z_prime)):
                raise ValueError("X, Y, Z vectors not the same length.")
    
            x_inertial = np.zeros(np.shape(x_prime))
            y_inertial = np.zeros(np.shape(y_prime))
            z_inertial = np.zeros(np.shape(z_prime))
            
            for row, world in enumerate([x_inertial, y_inertial, z_inertial]):
                for col, local in enumerate([x_prime, y_prime, z_prime]):
                    world += orientation[row,col] * local
    
            return x_inertial, y_inertial, z_inertial

        for sensor, rot_matrix in self.meta['trial']['sensor-orientation'].iteritems():
            data_frame[sensor + '_AccX'], data_frame[sensor + '_AccY'], \
                data_frame[sensor + '_AccZ'] = orient_accelerometer(
                        np.array(rot_matrix), data_frame[sensor + '_AccX'],
                        data_frame[sensor + '_AccY'], data_frame[sensor + '_AccZ'])

        return data_frame

    def _calibrate_accel_data(self, data_frame, y1=0, y2=-9.81):
        """Two-point calibration of accelerometer signals.
        Converts from voltage to meters/second^2

        Parameters
        ==========
        data_frame : pandas.DataFrame
            Accelerometer data  in volts to be calibrated
        y1 : float, optional
        y2 : float, optional

        Returns
        =======
        data_frame : pandas.DataFrame
            Calibrated accelerometer data in m/s^2

        Notes
        =====
        A calibration file must be specified in the meta file and its
        structure is as follows:
        There must be a column for each accelerometer signal to be calibrated,
        so three columns per sensor. There must be three rows of accelerometer
        readings. The first row is the reading when the sensors are placed with
        z-axis pointing straight up. The second row is the reading when the
        x-axis is pointing straight up. The third row is the reading when the
        y-axis is pointing straight up.
        (xyz)
        -----
        (001)
        (100)
        (010)
        """

        trial_directory = os.path.split(self.meta_yml_path)[0]
        try:
            calib_file = self.meta['trial']['files']['accel-calibration']
        except KeyError:
            raise Exception('Accelerometer calibration file not specified ' + 
                            'in {}.'.format(self.meta_yml_path))
        except AttributeError:
            raise Exception('You must include a meta data file with a path ' +
                            'to the accelerometer calibration file.')
        else:
            calib_file_path = os.path.join(trial_directory, calib_file)

        accel_channels = self.accel_column_labels       

        cal = pandas.read_csv(calib_file_path)

        cal = cal.drop('Time', 1)

        if len(accel_channels) != cal.shape[1]:
            raise ValueError("Calibration file doesn't match mocap data.")

        def twopointcalibration(x, x1, x2):
            m = (y2-y1)/(x2-x1)
            return m*(x-x1)+y1

        for i in range(0, len(accel_channels),3):
            x1 = cal.icol(i)[0]; x2 = cal.icol(i)[1]
            data_frame[accel_channels[i]] = twopointcalibration(data_frame[accel_channels[i]], x1, x2)

        for i in range(1, len(accel_channels),3):
            x1 = cal.icol(i)[0]; x2 = cal.icol(i)[2]
            data_frame[accel_channels[i]] = twopointcalibration(data_frame[accel_channels[i]], x1, x2)

        for i in range(2, len(accel_channels),3):
            x1 = cal.icol(i)[1]; x2 = cal.icol(i)[0]
            data_frame[accel_channels[i]] = twopointcalibration(data_frame[accel_channels[i]], x1, x2)

        return data_frame

    def clean_data(self, interpolate_markers=False):
        """Returns the processed, "cleaned", data.

        Parameters
        ----------
        interpolate_markers : boolean, optional, default=False
            If true the missing markers will be interpolated. Note that if
            force compensation is needed, the treadmill missing markers will
            always be fixed.

        Notes
        -----

        1. Loads the mocap and record modules into Pandas ``DataFrame``\s.
        2. Relabels of columns of Pandas ``DataFrame`` to more meaningful
           names.
        3. Shifts the Delsys signals in the mocap module data to accomodate
           for the wireless time delay, ~96ms.
        4. Identifies the missing values in the mocap marker data and
           replaces with NaN.
        5. Returns statistics on how many missing values in the marker time
           series are present, the max consecutive missing values, etc.
        6. Optionally, interpolates the missing marker values and replaces them with
           interpolated estimates.
        7. Compensates the force measurments for the motion of the treadmill
           base, if needed.

            1. Pulls the compensation file path from meta data.
            2. Loads the compensation file (only the necessary columns).
            3. Identifies the missing markers and interpolates to fill them.
            4. Shifts the Delsys signals to correct time.
            5. Filter the forces, accelerometer, and treadmill markers at 6 hz low pass.
            6. Compute the compensated forces (apply inertial compensation and express
                in global reference frame)
            7. Replace the force/moment measurements in the mocap data file with the
                compensated forces/moments.

        8. Optionally, low pass filter all human related data. (If there
           wasn't a stationary platform, then these should always be
           filtered with the same low pass filter as the compensation
           algorithm used.)
        9. Merges the data from the mocap module and record module into one
           ``DataFrame``.

        """
        if self.mocap_tsv_path is not None:

            raw_mocap_data_frame = self._load_mocap_data(ignore_hbm=True)

            relabeled_mocap_data_frame = self._relabel_analog_columns(raw_mocap_data_frame)

            shifted_mocap_data_frame = self._shift_delsys_signals(relabeled_mocap_data_frame)

            identified_mocap_data_frame = \
                self._identify_missing_markers(shifted_mocap_data_frame)

            mocap_data_frame = \
                self._generate_cortex_time_stamp(identified_mocap_data_frame)

            if interpolate_markers is True:
                mocap_data_frame = \
                    self._interpolate_missing_markers(mocap_data_frame)

            mocap_data_frame = self._compensate(mocap_data_frame,
                                                markers_missing=not interpolate_markers)

            self.mocap_data = mocap_data_frame

        if self.record_tsv_path is not None:

            self._extract_events_from_record_file()
            self.raw_record_data_frame = self._load_record_data()

        self.data = self._merge_mocap_record()


    def _merge_mocap_record(self):
        """Returns a data frame that is a merger of the mocap and record
        data, if needed."""

        mocap = self.mocap_tsv_path
        record = self.record_tsv_path

        if mocap is not None and record is not None:

            self.record_data = \
                self._resample_record_data(self.raw_record_data_frame)

            data = self.mocap_data.join(self.record_data)

        elif mocap is None and record is not None:

            data = self.raw_record_data_frame

        elif mocap is not None and record is None:

            data = self.mocap_data

        return data

    def extract_processed_data(self, event=None, index_col=None):
        """Returns the processed data in a data frame. If an event name is
        provided, then a data frame with only that event is returned.

        Parameters
        ----------
        event : string, optional, default=None
            A name of a detected event. Must be a valid key in self.events.
            This will be either the D-Flow auto-named events (A, B, C, D, E,
            F) or the names specified in the meta data file.
        index_col : string, optional, default=None
            A name of a column in the data frame. If provided the the column
            will be removed from the data frame and used as the index. This
            is useful for assigning one of the time columns as the index.

        Returns
        -------
        data_frame : pandas.DataFrame
            The processed data.

        """

        try:
            self.data
        except AttributeError:
            raise AttributeError('You must run clean_data first.')

        if event is None:
            data_frame = self.data
        else:
            try:
                start, stop = self.events[event]
            except AttributeError:
                raise AttributeError('No events have been initialized.')
            except KeyError:
                raise KeyError('{} is not a valid event. Valid events are: {}.'.format(event, ','.join(self.events.keys())))
            else:
                start_i = np.argmin(np.abs(self.data['TimeStamp'] - start))
                stop_i = np.argmin(np.abs(self.data['TimeStamp'] - stop))
                data_frame = self.data.iloc[start_i:stop_i, :]

        if index_col is not None:
            data_frame.index = data_frame[index_col]
            del data_frame[index_col]

        return data_frame

    def write_dflow_tsv(self, filename, na_rep='NA'):

        # This must preserve the mocap column order and can only append the
        # record to the right most columns.

        self.data.to_csv(filename, sep='\t', float_format='%1.6f',
                         na_rep=na_rep, index=False,
                         cols=self.mocap_column_labels)
