#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import re

# external libraries
import numpy as np
import pandas
from scipy.interpolate import InterpolatedUnivariateSpline
import yaml
from dtk import process

# debugging
try:
    from IPython.core.debugger import Tracer
except ImportError:
    pass
else:
    set_trace = Tracer()


class DFlowData(object):
    """A class to store and manipulate the data outputs from Motek Medical's
    D-Flow software."""

    coordinate_labels = ['X', 'Y', 'Z']
    marker_coordinate_suffixes = ['.Pos' + c for c in coordinate_labels]
    marker_coordinate_regex = '.*\.Pos[XYZ]$'
    # In the motek output file there is a space preceding only these two
    # column names: ' L_Psoas' ' R_Psoas'
    hbm_column_regexes = ['^\s?[LR]_.*', '.*\.Mom$', '.*\.Ang$', '.*\.Pow$']
    analog_channel_regex = '^Channel[0-9]+\.Anlg$' 
    # TODO: There are also .Rot[XYZ] and .COM.X values. Need to determine
    # what they are.
    force_plate_names = ['FP1', 'FP2']  # FP1 : left, FP2 : right
    force_plate_suffix = [suffix_beg + end for end in coordinate_labels for
                          suffix_beg in ['.For', '.Mom', '.Cop']]

    cortex_sample_rate = 100  # Hz
    constant_marker_tolerance = 1e-16
    hbm_na = ['0.000000', '-0.000000']

    def __init__(self, mocap_tsv_path=None, record_tsv_path=None,
                 meta_yml_path=None):
        """Sets the data file paths."""

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

            (self.analog_column_labels, self.analog_column_indices) = self._analog_column_labels(self.mocap_column_labels)

    def _parse_meta_data_file(self):
        """Returns a dictionary containing the meta data stored in the
        optional meta data file."""

        with open(self.meta_yml_path, 'r') as f:
            meta = yaml.load(f)

        return meta

    def _mocap_column_labels(self):
        """Returns a list of strings containing the motion capture file's
        column labels.  The list is in the same order as in the mocap tsv
        file."""
        with open(self.mocap_tsv_path, 'r') as f:
            mocap_column_labels = f.readline().strip().split('\t')

        return mocap_column_labels

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
            if reg_exp.match(label):
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
        """

        analog_labels = []
        analog_indices = []

        reg_exp = re.compile(self.analog_channel_regex)

        for i, label in enumerate(labels):
            if reg_exp.match(label):
                analog_labels.append(label)
                analog_indices.append(i)
        
        if self.meta_yml_path is not None:
            if 'analog-channel-map' in self.meta['trial']:
                channel_dictionary = self.meta['trial']['analog-channel-map']
                for i,channel in enumerate(analog_labels):
                    if channel in channel_dictionary:   
                        analog_labels[i] = channel_dictionary[channel]

        return analog_labels, analog_indices

    def _identify_missing_markers(self, data_frame):
        """Returns the data frame in which all marker columns (ends with
        '.PosX', '.PosY', '.PosZ') have had constant marker values replaced
        with NaN."""

        # For each marker column we need to identify the constants values
        # and replace with NaN, only if the values are constant in all
        # coordinates of a marker.

        # Get a list of all columns that give marker coordinate data, i.e.
        # ones that end in '.PosX', '.PosY', or '.PosZ'.
        marker_coordinate_col_names = \
            self._marker_column_labels(self.mocap_column_labels)

        # A list of unique markers in the data set (i.e. without the
        # suffixes).
        unique_marker_names = list(set([c.split('.')[0] for c in
                                        marker_coordinate_col_names]))

        # Create a boolean array that labels all constant values (wrt to
        # tolerance) as True.
        are_constant = data_frame[marker_coordinate_col_names].diff().abs() < \
            self.constant_marker_tolerance

        # Now make sure that the marker is constant in all three
        # coordinates before setting it to NaN.
        for marker in unique_marker_names:
            single_marker_cols = [marker + pos for pos in
                                  self.marker_coordinate_suffixes]
            for col in single_marker_cols:
                are_constant[col] = \
                    are_constant[single_marker_cols].all(axis=1)

        data_frame[are_constant] = np.nan

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

        # Pandas 0.13.0 will have all the SciPy interpolation functions
        # built in. But for now, we've got to do this manually.

        # TODO : Interpolate the HBM columns if they are loaded from file.

        # TODO : DataFrame.apply() might clean this code up.

        markers = self._marker_column_labels(self.mocap_column_labels)
        for marker_label in markers:
            time_series = data_frame[marker_label]
            is_null = time_series.isnull()
            if any(is_null):
                time = data_frame[time_col]
                without_na = time_series.dropna().values
                time_at_valid = time[time_series.notnull()].values

                interpolate = InterpolatedUnivariateSpline(time_at_valid,
                                                           without_na,
                                                           k=order)
                interpolated_values = interpolate(time[is_null].values)
                data_frame[marker_label][is_null] = interpolated_values

        return data_frame

    def _load_mocap_data(self, ignore_hbm=False, id_hbm_na=False, rename_analog_channels=True):
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

        if rename_analog_channels is True:
            data_frame.rename(columns=self.meta['trial']['analog-channel-map'], inplace=True)

        return data_frame
        

    def missing_value_statistics(self, data_frame):
        """Returns a report of missing values in the marker and/or HBM
        columns."""
        pass

    def _extract_events_from_record_file(self):
        """Returns a dictionary of events and times. The event names will be
        the default A-F which is output by D-Flow unless you specify unique
        names in the meta data file. If there are no events in the record
        file, this will return nothing."""

        f=open(self.record_tsv_path,'r')
        filecontents=f.readlines()
        f.close()
        end=filecontents[-6]
        end_value=end.split()
        end_value1=end_value[0]
        end_time=float(end_value1)

        if 'EVENT' in ''.join(filecontents):
            event_time1=[]
            event_labels=[]
            for i in range(len(filecontents)):
                if 'COUNT' in filecontents[i]:
                    event_labels.append(filecontents[i].split(' ')[2])
                    event=filecontents[i-2]
                    event_data=event.split()
                    event_time1.append(float(event_data[0]))
        else: return

        event_time1.append(end_time)
        self.events={}

        for i,label in enumerate(event_labels):
            self.events[label]=(event_time1[i],event_time1[i+1])

        if self.meta_yml_path is not None:
            if 'event' in self.meta:
                new_events={}
                event_dictionary=self.meta['event']
                for key,value in event_dictionary.items():
                    new_events[value]=self.events[key]
                self.events=new_events

    def _load_record_data(self):
        """Returns a data frame containing the data from the record
        module."""

        # The record module file is tab delimited and may have events
        # interspersed in between the rows which are commenting out by
        # hashes. We must dropna to remove commented lines from the
        # resutling data frame, only if all values in a row are NA.
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

    def _(self, data_frame):
        """
        There is a time delay between accelerometer and force plate signals.
        Assuming the accelerometer signals are delayed 72ms
        """
        def nearest_index(array, val):
            return np.abs(array - val).argmin()

        delay_time = .072
        data_frame['D-Flow Time'][0] + delay_time

        delay_index = nearest_index(data_frame['D-Flow Time'][0].values,
                                    data_frame['D-Flow Time'][0].values + delay_time)

        for channel in dataframe.columns:
            if ('Acc' in channel) or ('EMG' in channel):
                dataframe[channel] = dataframe[channel].shift(delay_index)

        return

    def _compensate_forces(self):
        """Computes the forces and moments which are due to the lateral and
        pitching motions of the treadmill and subtracts them from the
        measured forces and moments based on linear acceleration
        measurements of the treadmill."""
        # If you accelerate the treadmill there will be forces and moments
        # measured by the force plates that simply come from the motion.
        # When external loads are applied to the force plates, you must
        # subtract these inertial forces from the measured forces to get
        # correct estimates of the body fixed externally applied forces.
        # TODO : Implement this.
        raise NotImplementedError()

    def _express_forces_in_treadmill_reference_frame(self):
        """Re-expresses the forces and moments measured by the treadmill to
        the earth inertial reference frame."""
        # The markers are measured with respect to the camera's inertial
        # reference frame, earth, but the treadmill forces are measured with
        # respect to the treadmill's laterally and rotationaly moving
        # reference frame. We need both to be expressed in the same inertial
        # reference frame for ease of future computations.
        # TODO : Implement this.
        raise NotImplementedError()

    def _inverse_dynamics(self):
        """Returns a data frame with joint angles, rates, and torques based
        on the measured marker positions and force plate forces."""
        # TODO : Add some method of generating joint angles, rates, and
        # torques. Note that if the treadmill is in motion (laterally,
        # pitch), then one must compensate for the interial forces and deal
        # with reexpressing in the treadmill reference frame before
        # computing the inverse dynamics.
        raise NotImplementedError()

    def _calibrate_accel_data(self, data_frame, y1=0, y2=9.81):
        """Two-point calibration of accelerometer signals.
        Converts from voltage to meters/second^2
        Assuming a triaxial accelerometer
        (xyz)
        -----
        (001)
        (100)
        (010)
        """

        accel_channels = []
        for i,channel in enumerate(self.analog_column_labels):
            if 'Acc' in channel:
                accel_channels.append(self.analog_column_labels[i])        

        cal = pandas.read_csv(self.meta['trial']['files']['accel-calibration'])

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

    def clean_data(self):
        """Loads and processes the data."""

        if self.mocap_tsv_path is not None:
            raw_mocap_data_frame = self._load_mocap_data(ignore_hbm=False)
            mocap_data_frame = self._calibrate_accel_data(raw_mocap_data_frame)
            mocap_data_frame = self._identify_missing_markers(raw_mocap_data_frame)
            mocap_data_frame = \
                self._generate_cortex_time_stamp(mocap_data_frame)
            mocap_data_frame = \
                self._interpolate_missing_markers(mocap_data_frame)
            self.mocap_data = mocap_data_frame

        if self.record_tsv_path is not None:
            # TODO : A record file that has events but no event mapping in
            # given in a meta file should do some default event handling
            # behavior. Keep in mind that D-Flow only allows a certain
            # number of events (A through F) and multiple counts for the
            # events.
            self._extract_events_from_record_file()
            self.raw_record_data_frame = self._load_record_data()

        if self.mocap_tsv_path is not None and self.record_tsv_path is not None:
            self.record_data = \
                self._resample_record_data(self.raw_record_data_frame)
            self.data = self.mocap_data.join(self.record_data)
        elif self.mocap_tsv_path is None and self.record_tsv_path is not None:
            self.data = self.raw_record_data_frame
        elif self.mocap_tsv_path is not None and self.record_tsv_path is None:
            self.data = self.mocap_data

        return self.data

    def extract_data(self, event=None, columns=None, **kwargs):
        """Returns a data frame which may be a subject of the master data
        frame."""
        if columns is None:
            return self.data
        else:
            return self.data[columns]

    def write_dflow_tsv(self, filename):

        # This must preserve the mocap column order and can only append the
        # record or inverse dynamics stuff to the right most columns.

        self.data.to_csv(filename, sep='\t', float_format='%1.6f',
                         index=False, cols=self.mocap_column_labels)
