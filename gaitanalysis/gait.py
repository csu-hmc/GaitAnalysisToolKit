#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
from collections import namedtuple
import warnings

# external libraries
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.signal import firwin, filtfilt
import matplotlib.pyplot as plt
import pandas
from dtk import process

# local
from .utils import _percent_formatter

# debugging
try:
    from IPython.core.debugger import Tracer
except ImportError:
    pass
else:
    set_trace = Tracer()


def find_constant_speed(time, speed, plot=False, filter_cutoff=1.0):
    """Returns the indice at which the treadmill speed becomes constant and
    the time series when the treadmill speed is constant.

    Parameters
    ==========
    time : array_like, shape(n,)
        A monotonically increasing array.
    speed : array_like, shape(n,)
        A speed array, one sample for each time. Should ramp up and then
        stablize at a speed.
    plot : boolean, optional
        If true a plot will be displayed with the results.
    filter_cutoff : float, optional
        The filter cutoff frequency for filtering the speed in Hertz.

    Returns
    =======
    indice : integer
        The indice at which the speed is consider constant thereafter.
    new_time : ndarray, shape(n-indice,)
        The new time array for the constant speed section.

    """

    sample_rate = 1.0 / (time[1] - time[0])

    filtered_speed = process.butterworth(speed, filter_cutoff, sample_rate)

    acceleration = process.derivative(time, filtered_speed,
                                      method='central',
                                      padding='second order')
    last = acceleration[int(0.2 * len(acceleration)):]
    noise_level = np.max(np.abs(last - np.mean(last)))

    reversed_acceleration = acceleration[::-1]

    indice = np.argmax(reversed_acceleration > noise_level)

    additional_samples = sample_rate * 0.65

    new_indice = indice - additional_samples

    if plot is True:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(time, speed, '.', time, filtered_speed, 'g-')
        ax[0].plot(np.ones(2) * (time[len(time) - new_indice]),
                   np.hstack((np.max(speed), np.min(speed))))
        ax[1].plot(time, acceleration, '.')
        fig.show()

    return len(time) - (new_indice), time[len(time) - new_indice]


def interpolate(data_frame, time):
    """Returns a new data frame with a index based on the provided time
    array and linear interpolation.

    Parameters
    ==========
    data_frame : pandas.DataFrame
        A data frame with time series columns. The index should be in same
        units as the provided time array.
    time : array_like, shape(n,)
        A monotonically increasing array of time in seconds at which the
        data frame should be interpolated at.

    Returns
    =======
    interpolated_data_frame : pandas.DataFrame
        The data frame with an index matching `time_vector` and interpolated
        values based on `data_frame`.

    """

    column_names = data_frame.columns
    old_time = data_frame.index.values
    vals = data_frame.values

    f = interp1d(old_time, vals, axis=0)
    new_vals = f(time)

    return pandas.DataFrame(new_vals, index=time, columns=column_names)


class GaitData(object):
    """A class to store typical gait data."""

    attrs_to_store = ['data', 'gait_cycles', 'gait_cycle_stats',
                      'strikes', 'offs']

    def __init__(self, data):
        """Initializes the data structure.

        Parameters
        ==========
        data : pandas.DataFrame or string
            A data frame with an index of time and columns for each variable
            measured during a gait or the path to a HDF5 file created from
            ``GaitData.save()``.

        """
        # Could have a real time index:
        # new_index = [pandas.Timestamp(x, unit='s') for x in data.index]
        # data_frame.index = new_index
        # data.index.values.astype(float)*1e-9

        try:
            f = open(data)
        except TypeError:
            self.data = data
        else:
            f.close()
            self.load(data)

    def _leg2d(self, time, marker_pos, normalized_force_plate_values,
               cutoff, sample_rate):
        """This method effectively does the same thing that the Octave
        routine does."""

        # differentiate to get the marker velocities and accelerations
        marker_vel = process.derivative(time, marker_pos,
                                        method='combination')
        marker_acc = process.derivative(time, marker_vel,
                                        method='combination')

        # filter all the input data with the same filter
        marker_pos = process.butterworth(marker_pos, cutoff, sample_rate,
                                         axis=0)
        marker_vel = process.butterworth(marker_vel, cutoff, sample_rate,
                                         axis=0)
        marker_acc = process.butterworth(marker_acc, cutoff, sample_rate,
                                         axis=0)
        force_array = process.butterworth(normalized_force_plate_values,
                                          cutoff, sample_rate, axis=0)

        # compute the inverse dynamics
        inv_dyn = lower_extremity_2d_inverse_dynamics
        dynamics = inv_dyn(time, marker_pos, marker_vel, marker_acc,
                           force_array)

        return dynamics

    def inverse_dynamics_2d(self, left_leg_markers, right_leg_markers,
                            left_leg_forces, right_leg_forces, body_mass,
                            low_pass_cutoff):
        """Computes the hip, knee, and ankle angles, angular rates, joint
        moments, and joint forces and adds them as columns to the data
        frame.

        Parameters
        ----------
        left_leg_markers : list of strings, len(12)
            The names of the columns that give the X and Y marker
            coordinates for six markers.
        right_leg_markers : list of strings, len(12)
            The names of the columns that give the X and Y marker
            coordinates for six markers.
        left_leg_forces : list of strings, len(3)
            The names of the columns of the ground reaction forces and
            moments (Fx, Fy, Mz).
        right_leg_forces : list of strings, len(3)
            The names of the columns of the ground reaction forces and
            moments (Fx, Fy, Mz).
        body_mass : float
            The mass, in kilograms, of the subject.
        low_pass_cutoff : float
            The cutoff frequency in hertz.

        Returns
        -------
        data_frame : pandas.DataFrame
            The main data frame now with columns for the new variables. Note
            that the force coordinates labels (X, Y) are relative to the
            coordinate system described herein.

        Notes
        ------

        This computation assumes the following coordinate system::

           Y
            ^ _ o _
            |   |   ---> v
            |  / \
            -----> x

        where X is forward (direction of walking) and Y is up.

        Make sure the sign conventions of the columns you pass in are
        correct!

        The markers should be in the following order:
            1. Shoulder
            2. Greater trochanter
            3. Lateral epicondyle of knee
            4. Lateral malleolus
            5. Heel (placed at same height as marker 6)
            6. Head of 5th metatarsal

        The underlying function low pass filters the data before computing
        the inverse dynamics. You should pass in unfiltered data.

        """

        time = self.data.index.values.astype(float)
        sample_rate = 1.0 / np.mean(np.diff(time))

        marker_sets = [left_leg_markers, right_leg_markers]
        force_sets = [left_leg_forces, right_leg_forces]

        side_labels = ['Left', 'Right']
        joint_labels = ['Hip', 'Knee', 'Ankle']
        sign_labels = ['Flexion', 'Flexion', 'PlantarFlexion']
        dynamic_labels = ['Angle', 'Rate', 'Moment', 'Force']
        scale_factors = [1.0, 1.0, body_mass, body_mass]

        for side_label, markers, forces in zip(side_labels, marker_sets,
                                               force_sets):

            marker_vals = self.data[markers].values.copy()
            force_vals = self.data[forces].values.copy() / body_mass

            dynamics = self._leg2d(time, marker_vals, force_vals,
                                   low_pass_cutoff, sample_rate)

            fours = zip(dynamics, dynamic_labels, scale_factors)

            for array, dynamic_label, scale_factor in fours:

                if dynamic_label == 'Force':

                    # array is N x 6, (Fx, Fy) for each joint

                    a = array[:, :2], array[:, 2:4], array[:, 4:]

                    for joint_label, vectors in zip(joint_labels, a):

                        for slab, vector in zip(('X', 'Y'), vectors.T):

                            label = '.'.join([side_label, joint_label, slab,
                                              dynamic_label])

                            self.data[label] = scale_factor * vector

                else:

                    for joint_label, sign_label, vector in zip(joint_labels,
                                                               sign_labels,
                                                               array.T):

                        label = '.'.join([side_label, joint_label,
                                          sign_label, dynamic_label])

                        self.data[label] = scale_factor * vector

        return self.data

    def tpose(self, data_frame):
        """
        Computes the mass of the subject.
        Computes to orientation of accelerometers on a subject during quiet
        standing relative to treadmill Y-axis
        """
        self.bodymass = np.mean(data_frame['FP1.ForY'] + data_frame['FP2.ForY']) / 9.81
        self.sensor_angle = {}
        for column in data_frame.columns:
            if '_AccX' in column:
                self.sensor_angle[column] = np.arcsin(-data_frame[column].mean()/9.81)
            if '_AccY' in column:
                self.sensor_angle[column] = np.arccos(-data_frame[column].mean()/9.81)
            if '_AccZ' in column:
                self.sensor_angle[column] = np.arcsin(-data_frame[column].mean()/9.81)

        return self.bodymass, self.sensor_angle

    def grf_landmarks(self, right_vertical_signal_col_name,
                      left_vertical_signal_col_name, method='force',
                      do_plot=False, min_time=None,
                      max_time=None, **kwargs):

        """Returns the times at which heel strikes and toe offs happen in
        the raw data.

        Parameters
        ==========
        right_vertical_signal_col_name : string
            The name of the column in the raw data frame which corresponds
            to the right foot vertical ground reaction force.
        left_vertical_signal_col_name : string
            The name of the column in the raw data frame which corresponds
            to the left foot vertical ground reaction force.
        method: string {force|accel}
            Whether to use force plate data or accelerometer data to
            calculate landmarks

        Returns
        =======
        right_strikes : np.array
            All indices at which right_grfy is non-zero and it was 0 at the
            preceding time index.
        left_strikes : np.array
            Same as above, but for the left foot.
        right_offs : np.array
            All indices at which left_grfy is 0 and it was non-zero at the
            preceding time index.
        left_offs : np.array
            Same as above, but for the left foot.

        Notes
        =====
        This is a simple wrapper to gait_landmarks_from_grf and supports all
        the optional keyword arguments that it does.

        """

        def nearest_index(array, val):
            return np.abs(array - val).argmin()

        time = self.data.index.values.astype(float)

        # Time range to consider.

        if max_time is None:
            self.max_idx = len(time)
        elif max_time > time[0]:
            self.max_idx = min(len(time), nearest_index(time, max_time))
        else:
            raise ValueError('max_time out of range.')

        if min_time is None:
            self.min_idx = 0
        elif min_time < time[-1]:
            self.min_idx = max(0, nearest_index(time, min_time))
        else:
            raise ValueError('min_time out of range.')

        if method is not 'accel' and method is not 'force':
            raise ValueError('{} is not a valid method'.format(method))

        func = {'force': gait_landmarks_from_grf,
                'accel': gait_landmarks_from_accel}

        right_strikes, left_strikes, right_offs, left_offs = \
            func[method](time[self.min_idx:self.max_idx],
                         self.data[right_vertical_signal_col_name].values[self.min_idx:self.max_idx],
                         self.data[left_vertical_signal_col_name].values[self.min_idx:self.max_idx],
                         **kwargs)

        self.strikes = {}
        self.offs = {}

        self.strikes['right'] = right_strikes
        self.strikes['left'] = left_strikes
        self.offs['right'] = right_offs
        self.offs['left'] = left_offs

        if do_plot:
            try:
                right_col_names = kwargs.pop('right_col_names')
            except KeyError:
                right_col_names = [right_vertical_signal_col_name]

            try:
                left_col_names = kwargs.pop('left_col_names')
            except KeyError:
                left_col_names = [left_vertical_signal_col_name]
            try:
                num_cycles_to_plot = kwargs.pop('num_cycles_to_plot')
            except KeyError:
                num_cycles_to_plot = None

            self.plot_landmarks(col_names=right_col_names, side='right',
                                num_cycles_to_plot=num_cycles_to_plot)

            self.plot_landmarks(col_names=left_col_names, side='left',
                                num_cycles_to_plot=num_cycles_to_plot)

        return right_strikes, left_strikes, right_offs, left_offs

    def plot_landmarks(self, col_names, side, event='both', index=0,
                       window=None, num_cycles_to_plot=None,
                       curve_kwargs=None, heel_kwargs=None,
                       toe_kwargs=None):
        """Creates a plot of the desired signal(s) with the gait event times
        overlaid on top of the signal.

        Parameters
        ==========
        col_names : sequence of strings
            A variable number of strings naming the columns to plot.
        side : string, {right|left}
            Whether to plot the gait landmarks from the right or left leg.
        event : string, {heelstrikes|toeoffs|both|none}
            Which gait landmarks to plot.
        index : integer, optional, default=0
            The index of the first time sample in the plot. This is useful
            if you want to plot the cycles starting at an arbitrary point in
            time in the data.
        window : integer, optional, default=None
            The number of time samples to plot. This is useful when a trial
            has many cycles and you only want to view some of them in the
            plot.
        num_cycles_to_plot : integer, optional, default=None
            This is an alternative way to specify the window. If this is
            provided, the window argment is ignored and the window is
            estimated by the desired number of cycles.
        curve_kwargs : dictionary, optional
            Valid matplotlib kwargs that will be used for the signal curves.
        heel_kwargs : dictionary, optional
            Valid matplotlib kwargs that will be used for the heel-strike
            lines.
        toe_kwargs : dictionary, optional
            Valid matplotlib kwargs that will be used for the toe-off lines.

        Returns
        =======
        axes : matplotlib.Axes
            The list of axes for the subplots or a single axes if only one
            column was supplied. Same as `matplotlib.pyplot.subplots`
            returns.

        Notes
        =====

        The `index`, `window` and `num_cycles_to_plot` arguments do not
        simply set the x limit to bound the data of interest, they do not
        plot any data outside the desired range (and is thus faster).

        """

        if len(col_names) == 0:
            raise ValueError('Please supply some column names to plot.')

        if event not in ['heelstrikes', 'toeoffs', 'both', 'none']:
            raise ValueError('{} is not a valid event to plot'.format(event))

        if side != 'right' and side != 'left':
            raise ValueError("Please indicate the 'right' or 'left' side.")

        if curve_kwargs is None:
            curve_kwargs = {'color': 'black'}

        if heel_kwargs is None:
            heel_kwargs = {'color': 'red'}

        if toe_kwargs is None:
            toe_kwargs = {'color': 'blue'}

        fig, axes = plt.subplots(len(col_names), sharex=True)

        time = self.data.index.values.astype(float)

        if num_cycles_to_plot is not None:
            # Estimate number of samples in window from the first registered
            # strikes. This will always overwrite anything supplied for
            # `window` by the user.
            cycle_times = self.strikes['right'][:num_cycles_to_plot + 1]
            window = (np.argmin(np.abs(time - cycle_times[-1])) -
                      np.argmin(np.abs(time - cycle_times[0])))

        if window is None:
            time_window = time[index:-1]
        else:
            time_window = time[index:index + window]

        for i, col_name in enumerate(col_names):
            try:
                ax = axes[i]
            except TypeError:  # if only one column
                ax = axes

            signal = self.data[col_name]
            if window is None:
                signal_window = signal.iloc[index:-1]
            else:
                signal_window = signal.iloc[index:index + window]

            ax.plot(time_window, signal_window, label="_nolegend_",
                    **curve_kwargs)

            heel_labels, toe_labels = [], []
            heel_lines, toe_lines = [], []

            if event == 'heelstrikes' or event == 'both':
                idx_in_window = ((time_window[0] < self.strikes[side]) &
                                 (self.strikes[side] < time_window[-1]))
                strikes_in_window = self.strikes[side][idx_in_window]
                heel_labels = (['Heel Strikes'] +
                               (len(strikes_in_window) - 1) * ["_nolegend_"])
                heel_lines = ax.plot(strikes_in_window *
                                     np.ones((2, strikes_in_window.shape[0])),
                                     ax.get_ylim(), **heel_kwargs)

            if event == 'toeoffs' or event == 'both':
                idx_in_window = ((time_window[0] < self.offs[side]) &
                                 (self.offs[side] < time_window[-1]))
                offs_in_window = self.offs[side][idx_in_window]
                toe_labels = (['Toe Offs'] + (len(offs_in_window) - 1) *
                              ["_nolegend_"])
                toe_lines = ax.plot(offs_in_window *
                                    np.ones((2, offs_in_window.shape[0])),
                                    ax.get_ylim(), **toe_kwargs)

            ax.set_ylabel(col_name)
            ax.set_xlim((time_window[0], time_window[-1]))
            if event != 'none':
                for line, label in zip(heel_lines + toe_lines,
                                       heel_labels + toe_labels):
                    line.set_label(label)
                ax.legend()

        # draw only on the last axes
        ax.set_xlabel('Time [s]')

        if event != 'none':
            title = '{} Gait Events:'.format(side.capitalize())

            if event == 'heelstrikes' or event == 'both':
                title += ' {} heel strikes'.format(len(strikes_in_window))

            if event == 'both':
                title += ','

            if event == 'toeoffs' or event == 'both':
                title += ' {} toeoffs'.format(len(strikes_in_window))

            fig.suptitle(title)

        return axes

    def plot_gait_cycles(self, *col_names, **kwargs):
        """Plots the time histories of each gait cycle.

        Parameters
        ==========
        col_names : string
            A variable number of strings naming the columns to plot.
        mean : boolean, optional
            If true the mean and standard deviation of the cycles will be
            plotted.
        kwargs : key value pairs
            Any extra kwargs to pass to the matplotlib plot command.

        """
        return plot_gait_cycles(self.gait_cycles, *col_names, **kwargs)

    def split_at(self, side, section='both', num_samples=None,
                 belt_speed_column=None):
        """Forms a pandas.Panel which has an item for each cycle. The index
        of each cycle data frame will be a percentage of gait cycle.

        Parameters
        ==========
        side : string {right|left}
            Split with respect to the right or left side heel strikes and/or
            toe-offs.
        section : string {both|stance|swing}
            Whether to split around the stance phase, swing phase, or both.
        num_samples : integer, optional
            If provided, the time series in each gait cycle will be
            interpolated at values evenly spaced at num_sample in time
            across the gait cycle. If None, the maximum number of possible
            samples per gait cycle will be used.
        belt_speed_column : string, optional
            The column name corresponding to the belt speed on the
            corresponding side.

        Returns
        =======
        gait_cycles : pandas.Panel
            A panel where each item is a gait cycle. Each cycle has the same
            number of time samples and the index is set to the  percent of
            the gait cycle.

        """

        if section == 'stance':
            lead = self.strikes[side]
            trail = self.offs[side]
        elif section == 'swing':
            lead = self.offs[side]
            trail = self.strikes[side]
        elif section == 'both':
            lead = self.strikes[side]
            trail = self.strikes[side][1:]
        else:
            raise ValueError('{} is not a valid section name'.format(section))

        if lead[0] > trail[0]:
            trail = trail[1:]

        # TODO: If there are short gait cycles, and num_samples=None then
        # max_num_samples will be low. This needs to be fixed. See issue
        # #71.
        samples = []
        for i, lead_val in enumerate(lead):
            try:
                gait_cycle_slice = self.data[lead_val:trail[i]]
            except IndexError:
                pass
            else:
                samples.append(len(gait_cycle_slice))

        max_num_samples = min(samples)
        if num_samples is None:
            num_samples = max_num_samples
        # TODO: The percent of the gait cycle should always be computed with
        # respect to heel strike to next heel strike, i.e.:
        # stance: 0.0 to percent stance
        # swing: percent stance to 1.0 - 1.0 / n
        # both: 0.0 to 1.0 - 1.0 / n
        # but right now this only works correctly for section='both'. It
        # currently generates the percent of the phase of the portion of the
        # gait cycle.
        percent_gait = np.linspace(0.0, 1.0 - 1.0 / num_samples,
                                   num=num_samples)

        gait_cycles = {}
        gait_cycle_stats = {'Number of Samples': [],
                            'Stride Duration': [],
                            'Stride Frequency': [],
                            }

        if belt_speed_column is not None:
            gait_cycle_stats['Stride Length'] = []
            gait_cycle_stats['Average Belt Speed'] = []

        for i, lead_val in enumerate(lead):
            try:
                data_frame = self.data[lead_val:trail[i]]
            except IndexError:
                pass
            else:
                # create a time vector index which has the correct number of
                # samples over the gait cycle
                duration = data_frame.index[-1] - data_frame.index[0]
                # note that this does not include the ending heelstrike
                last_sample_time = (data_frame.index[0] +
                                    duration * (1.0 - 1.0 / num_samples))
                sub_sample_index = np.linspace(data_frame.index[0],
                                               last_sample_time,
                                               num=num_samples)
                interpolated_data_frame = interpolate(data_frame,
                                                      sub_sample_index)
                # keep the interpolation of the original time around for
                # future use
                interpolated_data_frame['Original Time'] = \
                    interpolated_data_frame.index.values.astype(float)
                interpolated_data_frame['Percent Gait Cycle'] = percent_gait
                # change the index to percent of gait cycle
                interpolated_data_frame.index = percent_gait
                gait_cycles[i] = interpolated_data_frame
                # compute some gait cycle stats
                gait_cycle_stats['Number of Samples'].append(len(data_frame))
                gait_cycle_stats['Stride Duration'].append(duration)
                gait_cycle_stats['Stride Frequency'].append(1.0 / duration)
                if belt_speed_column is not None:
                    stride_len = simps(data_frame[belt_speed_column].values,
                                       data_frame.index.values.astype(float))
                    gait_cycle_stats['Stride Length'].append(stride_len)
                    avg_speed = data_frame[belt_speed_column].mean()
                    gait_cycle_stats['Average Belt Speed'].append(avg_speed)

        self.gait_cycles = pandas.Panel(gait_cycles)
        self.gait_cycle_stats = pandas.DataFrame(gait_cycle_stats)

        return self.gait_cycles

    def time_derivative(self, col_names, new_col_names=None):
        """Numerically differentiates the specified columns with respect to
        the time index and adds the new columns to `self.data`.

        Parameters
        ==========
        col_names : list of strings
            The column names for the time series which should be numerically
            time differentiated.
        new_col_names : list of strings, optional
            The desired new column name(s) for the time differentiated
            series. If None, then a default name of `Time derivative of
            <origin column name>` will be used.

        """

        if new_col_names is None:
            new_col_names = ['Time derivative of {}'.format(c) for c in
                             col_names]

        for col_name, new_col_name in zip(col_names, new_col_names):
            self.data[new_col_name] = \
                process.derivative(self.data.index.values.astype(float),
                                   self.data[col_name].values,
                                   method='combination')

    def low_pass_filter(self, col_names, cutoff, new_col_names=None,
                        order=2):
        """Low pass filters the specified columns with a Butterworth filter.

        Parameters
        ==========
        col_names : list of strings
            The column names for the time series which should be numerically
            time differentiated.
        cutoff : float
            The desired low pass cutoff frequency in Hertz.
        new_col_names : list of strings, optional
            The desired new column name(s) for the filtered series. If None,
            then a default name of `Filtered <origin column name>` will be
            used.
        order : int
            The order of the Butterworth filter.

        """

        if new_col_names is None:
            new_col_names = ['Filtered {}'.format(c) for c in col_names]

        time = self.data.index.values.astype(float)
        sample_rate = 1.0 / np.mean(np.diff(time))

        filtered_data = process.butterworth(self.data[col_names].values,
                                            cutoff, sample_rate,
                                            order=order, axis=0)

        # TODO : Ideally these could be added to the DataFrame in one
        # command.

        for i, col in enumerate(new_col_names):
            self.data[col] = filtered_data[:, i]

    def save(self, filename):
        """Saves data to disk via HDF5 (PyTables).

        Parameters
        ==========
        filename : string
            Path to an HDF5 file.

        """

        with pandas.get_store(filename) as store:
            for item in self.attrs_to_store:
                try:
                    data = getattr(self, item)
                except AttributeError:
                    pass
                else:
                    if item in ['strikes', 'offs']:
                        store[item + '_right'] = pandas.Series(data['right'])
                        store[item + '_left'] = pandas.Series(data['left'])
                    else:
                        store[item] = data

    def load(self, filename):
        """Loads data from disk via HDF5 (PyTables).

        Parameters
        ==========
        filename : string
            Path to an HDF5 file.

        """
        with pandas.get_store(filename) as store:
            for item in self.attrs_to_store:
                try:
                    if item in ['strikes', 'offs']:
                        data = {}
                        data['right'] = store[item + '_right'].values
                        data['left'] = store[item + '_left'].values
                    else:
                        data = store[item]
                except KeyError:
                    pass
                else:
                    setattr(self, item, data)


def gait_landmarks_from_grf(time, right_grf, left_grf,
                            threshold=1e-5, filter_frequency=None, **kwargs):
    """
    Obtain gait landmarks (right and left foot strike & toe-off) from ground
    reaction force (GRF) time series data.

    Parameters
    ----------
    time : array_like, shape(n,)
        A monotonically increasing time array.
    right_grf : array_like, shape(n,)
        The vertical component of GRF data for the right leg.
    left_grf : str, shape(n,)
        Same as above, but for the left leg.
    threshold : float, optional
        Below this value, the force is considered to be zero (and the
        corresponding foot is not touching the ground).
    filter_frequency : float, optional, default=None
        If a filter frequency is provided, in Hz, the right and left ground
        reaction forces will be filtered with a 2nd order low pass filter
        before the landmarks are identified. This method assumes that there
        is a constant (or close to constant) sample rate.

    Returns
    -------
    right_foot_strikes : np.array
        All times at which right_grfy is non-zero and it was 0 at the
        preceding time index.
    left_foot_strikes : np.array
        Same as above, but for the left foot.
    right_toe_offs : np.array
        All times at which left_grfy is 0 and it was non-zero at the
        preceding time index.
    left_toe_offs : np.array
        Same as above, but for the left foot.

    Notes
    -----
    Source modifed from:

    https://github.com/fitze/epimysium/blob/master/epimysium/postprocessing.py

    """

    # Helper functions
    # ----------------
    def zero(number):
        return abs(number) < threshold

    def birth_times(ordinate):
        births = list()
        for i in range(len(ordinate) - 1):
            # 'Skip' first value because we're going to peak back at previous
            # index.
            if zero(ordinate[i]) and (not zero(ordinate[i+1])):
                births.append(time[i + 1])
        return np.array(births)

    def death_times(ordinate):
        deaths = list()
        for i in range(len(ordinate) - 1):
            if (not zero(ordinate[i])) and zero(ordinate[i+1]):
                deaths.append(time[i + 1])
        return np.array(deaths)

    # If the ground reaction forces are very noisy, it may help to low pass
    # filter the signals before searching for the strikes and offs.

    if filter_frequency is not None:
        average_sample_rate = 1.0 / np.mean(np.diff(time))
        right_grf = process.butterworth(right_grf, filter_frequency,
                                        average_sample_rate)
        left_grf = process.butterworth(left_grf, filter_frequency,
                                       average_sample_rate)

    right_foot_strikes = birth_times(right_grf)
    left_foot_strikes = birth_times(left_grf)
    right_toe_offs = death_times(right_grf)
    left_toe_offs = death_times(left_grf)

    return right_foot_strikes, left_foot_strikes, right_toe_offs, left_toe_offs


def gait_landmarks_from_accel(time, right_accel, left_accel, threshold=0.33, **kwargs):
    """
    Obtain right and left foot strikes from the time series data of accelerometers placed on the heel.

    Parameters
    ==========
    time : array_like, shape(n,)
        A monotonically increasing time array.
    right_accel : array_like, shape(n,)
        The vertical component of accel data for the right foot.
    left_accel : str, shape(n,)
        Same as above, but for the left foot.
    threshold : float, between 0 and 1
        Increase if heelstrikes/toe-offs are falsly detected

    Returns
    =======
    right_foot_strikes : np.array
        All times at which a right foot heelstrike is determined
    left_foot_strikes : np.array
        Same as above, but for the left foot.
    right_toe_offs : np.array
        All times at which a right foot toeoff is determined
    left_toe_offs : np.array
        Same as above, but for the left foot.
    """

    sample_rate = 1.0 / np.mean(np.diff(time))

    # Helper functions
    # ----------------

    def filter(data):

        a = np.array([1])

        # 10 Hz highpass
        n = 127  # filter order
        Wn = 10.0 / (sample_rate / 2)  # cut-off frequency
        b = firwin(n, Wn, window='blackman', pass_zero=False)
        data = filtfilt(b, a, data)

        data = abs(data)  # rectify signal

        # 5 Hz lowpass
        Wn = 5 / (sample_rate/2)
        b = firwin(n, Wn, window='blackman')
        data = filtfilt(b, a, data)

        return data

    def peak_detection(x):

        dx = process.derivative(time, x, method="combination")
        dx[dx > 0] = 1
        dx[dx < 0] = -1
        ddx = process.derivative(time, dx, method="combination")

        peaks = []
        for i, spike in enumerate(ddx < 0):
            if spike:
                peaks.append(i)

        peaks = peaks[::2]

        threshold_value = (max(x) - min(x))*threshold + min(x)

        peak_indices = []
        for i in peaks:
            if x[i] > threshold_value:
                peak_indices.append(i)

        return peak_indices

    def determine_foot_event(foot_spikes):
        heelstrikes = []
        toeoffs = []

        spike_time_diff = np.diff(foot_spikes)

        for i, spike in enumerate(foot_spikes):
            if spike_time_diff[i] > spike_time_diff[i+1]:
                heelstrikes.append(time[spike])
            else:
                toeoffs.append(time[spike])
            if i == len(foot_spikes) - 3:
                if spike_time_diff[i] > spike_time_diff[i+1]:
                    toeoffs.append(time[foot_spikes[i+1]])
                    heelstrikes.append(time[foot_spikes[i+2]])
                else:
                    toeoffs.append(time[foot_spikes[i+2]])
                    heelstrikes.append(time[foot_spikes[i+1]])
                break

        return np.array(heelstrikes), np.array(toeoffs)

    # ----------------

    right_accel_filtered = filter(right_accel)
    right_spikes = peak_detection(right_accel_filtered)
    (right_foot_strikes, right_toe_offs) = \
        determine_foot_event(right_spikes)

    left_accel_filtered = filter(left_accel)
    left_spikes = peak_detection(left_accel_filtered)
    (left_foot_strikes, left_toe_offs) = \
        determine_foot_event(left_spikes)

    return right_foot_strikes, left_foot_strikes, right_toe_offs, left_toe_offs


def plot_gait_cycles(gait_cycles, *col_names, **kwargs):
    """Plots the time histories from each gait cycle on one graph.

    Parameters
    ==========
    gait_cycles : pandas.Panel
        A panel of gait cycles. Each item should be a cycle DataFrame with
        time histories of variables. The index should be the percent gait
        cycle.
    col_names : string
        A variable number of strings naming the columns to plot.
    mean : boolean, optional, default=False
        If true the mean and standard deviation of the gait cycles will be
        plotted instead of the individual lines.
    kwargs : key value pairs
        Any extra kwargs to pass to the matplotlib plot command.

    """

    if len(col_names) == 0:
        raise ValueError('Please supply some column names to plot.')

    try:
        mean = kwargs.pop('mean')
    except KeyError:
        mean = False

    try:
        axes = kwargs.pop('axes')
        fig = plt.gcf()
    except KeyError:
        axes = None

    try:
        marker = kwargs.pop('marker')
    except KeyError:
        marker = 'o'

    try:
        alpha = kwargs.pop('alpha')
    except KeyError:
        alpha = 0.5

    if axes is None:
        fig, axes = plt.subplots(len(col_names), sharex=True)

    if mean is True:
        fig.suptitle('Mean and standard deviation of ' +
                     '{} gait cycles.'.format(gait_cycles.shape[0]))
        mean_of_cycles = gait_cycles.mean(axis='items')
        std_of_cycles = gait_cycles.std(axis='items')
    else:
        fig.suptitle('{} gait cycles'.format(gait_cycles.shape[0]))

    for i, col_name in enumerate(col_names):
        try:
            ax = axes[i]
        except TypeError:
            ax = axes
        if mean is True:
            ax.fill_between(mean_of_cycles.index.values.astype(float),
                            (mean_of_cycles[col_name] -
                                std_of_cycles[col_name]).values,
                            (mean_of_cycles[col_name] +
                                std_of_cycles[col_name]).values,
                            alpha=alpha, **kwargs)
            ax.plot(mean_of_cycles.index.values.astype(float),
                    mean_of_cycles[col_name].values, marker=marker, **kwargs)
        else:
            for key, value in gait_cycles.iteritems():
                ax.plot(value[col_name].index, value[col_name], **kwargs)

        ax.xaxis.set_major_formatter(_percent_formatter)

        ax.set_ylabel(col_name)

    # plot only on the last axes
    ax.set_xlabel('Percent of Gait Cycle [%]')

    return axes


def lower_extremity_2d_inverse_dynamics(time, marker_pos, marker_vel,
                                        marker_acc, force_plate_values,
                                        g=9.81):
    """Returns the 2D inverse dynamics of a single lower limb.

    Parameters
    ==========
    time: array_like, shape(N,)
        Time stamps for the marker and force plate data in seconds.
    marker_pos: array_like, shape(N, 12)
        The X and Y coordinates of the six markers in meters given as
        alternating columns: [X0, Y0, X1, Y1, ..., X5, Y5].
    marker_vel: array_like, shape(N, 12)
        The rate of change of the X and Y coordinates of the six markers in
        meters per second.
    marker_acc: array_like, shape(N, 12)
        The rate of change of the X and Y velocities of the six markers in
        meters per second per second.
    force_plate_values: array_like, shape(N, 3)
        Normalized loads applied to foot [Fx, Fy, Mz] in N/kg (normalized to
        body mass).
    g : float
        Acceleration due to gravity in meters per second per second.

    Returns
    =======
    joint_angles: ndarray, shape(N, 3)
        Joint angles in three joints: hip, knee, ankle in radians.
    joint_angular_rates: ndarray, shape(N, 3)
        Angular velocities in three joints: hip, knee, ankle in radians per
        second.
    joint_torques: ndarray, shape(N, 3)
        Torques in three joints: hip, knee, ankle in  Nm per kg body mass.
    joint_forces:  (Nsamples x 6)
        Forces (Fx, Fy) in three joints, (N per kg body mass)

    Notes
    =====

    Coordinate system:
      X is forward (direction of walking), Y is up

    Markers:
      0: Shoulder
      1: Greater trochanter
      2: Lateral epicondyle of knee
      3: Lateral malleolus
      4: Heel (placed at same height as marker 6)
      5: Head of 5th metatarsal

    Joints:
      hip, knee, ankle
      sign convention for angles and moments: hip flexion, knee flexion,
      ankle plantarflexion are positive

    References
    ==========

    Method:
       Winter, DA (2005) Biomechanics of Human Movement.


    """

    # TODO : Remove the time variable, it is not needed.

    num_markers = 6
    num_coordinates = 2 * num_markers
    num_force_plate_channels = 3
    num_segments = 4
    num_samples = time.shape[0]

    # define the body segments with these properties:
    # - proximal marker
    # - distal marker
    # - joint marker (at proximal end)
    # - mass as fraction of body mass (from Winter book)
    # - center of mass as fraction of length (from Winter book)
    # - radius of gyration as fraction of length (from Winter book)
    # - +1(-1) if positive angle/moment in prox, joint corresponds to
    #   counterclockwise(clockwise) rotation of segment

    # TODO : Create a way for the user to pass in the body segment values.

    Segment = namedtuple('Segment', ['name',
                                     'num',
                                     'prox_marker_idx',
                                     'dist_marker_idx',
                                     'prox_joint_marker_idx',
                                     'normalized_mass',
                                     'mass_center_fraction',
                                     'radius_of_gyration_fraction',
                                     'sign'])

    segments = [Segment('torso', 0, 0, 1, np.nan, np.nan, np.nan, np.nan, np.nan),
                Segment('thigh', 1, 1, 2, 1, 0.1000, 0.433, 0.323,  1),
                Segment('shank', 2, 2, 3, 2, 0.0465, 0.433, 0.302, -1),
                Segment('foot',  3, 4, 5, 3, 0.0145, 0.500, 0.475, -1)]

    if time.shape[0] != marker_pos.shape[0]:
        msg = ('The number of samples in marker data is not the same as '
               'number of time stamps.')
        raise ValueError(msg)

    if time.shape[0] != marker_vel.shape[0]:
        msg = ('The number of samples in marker data is not the same as '
               'number of time stamps.')
        raise ValueError(msg)

    if time.shape[0] != marker_acc.shape[0]:
        msg = ('The number of samples in marker data is not the same as '
               'number of time stamps.')
        raise ValueError(msg)

    if time.shape[0] != force_plate_values.shape[0]:
        msg = ('The number of samples in force plate data is not the same as '
               'number of time stamps.')
        raise ValueError(msg)

    if marker_pos.shape[1] != num_coordinates:
        msg = 'The number of columns in mocap data is not correct.'
        raise ValueError(msg)

    if force_plate_values.shape[1] != num_force_plate_channels:
        msg = 'The number of columns in force plate data is not correct.'
        raise ValueError(msg)

    seg_lengths = np.zeros(num_segments)

    seg_com_x_pos = np.zeros((num_samples, num_segments))
    seg_com_y_pos = np.zeros((num_samples, num_segments))
    seg_com_x_acc = np.zeros((num_samples, num_segments))
    seg_com_y_acc = np.zeros((num_samples, num_segments))

    seg_theta = np.zeros((num_samples, num_segments))
    seg_omega = np.zeros((num_samples, num_segments))
    seg_alpha = np.zeros((num_samples, num_segments))

    for i, segment in enumerate(segments):

        prox_x_idx = 2 * segment.prox_marker_idx
        prox_y_idx = prox_x_idx + 1

        prox_pos = marker_pos[:, prox_x_idx:prox_y_idx + 1]
        prox_vel = marker_vel[:, prox_x_idx:prox_y_idx + 1]
        prox_acc = marker_acc[:, prox_x_idx:prox_y_idx + 1]

        dist_x_idx = 2 * segment.dist_marker_idx
        dist_y_idx = dist_x_idx + 1

        dist_pos = marker_pos[:, dist_x_idx:dist_y_idx + 1]
        dist_vel = marker_vel[:, dist_x_idx:dist_y_idx + 1]
        dist_acc = marker_acc[:, dist_x_idx:dist_y_idx + 1]

        # vector R points from proximal to distal marker
        R_pos = dist_pos - prox_pos
        R_vel = dist_vel - prox_vel
        R_acc = dist_acc - prox_acc

        # calculate segment center of mass position and segment orientation
        # angle, and 1st and 2nd derivatives

        seg_com_pos = prox_pos + segment.mass_center_fraction * R_pos

        seg_com_x_pos[:, i] = seg_com_pos[:, 0]
        seg_com_y_pos[:, i] = seg_com_pos[:, 1]

        seg_com_acc = prox_acc + segment.mass_center_fraction * R_acc

        seg_com_x_acc[:, i] = seg_com_acc[:, 0]
        seg_com_y_acc[:, i] = seg_com_acc[:, 1]

        # orientation of the vector R, unwrap removes -pi to pi discontinuities
        seg_theta[:, i] = np.unwrap(np.arctan2(R_pos[:, 1], R_pos[:, 0]))

        # analytical time derivative of segment angle
        seg_omega[:, i] = ((R_pos[:, 0] * R_vel[:, 1] -
                            R_pos[:, 1] * R_vel[:, 0]) /
                           (R_pos[:, 1]**2 + R_pos[:, 0]**2))

        # analytical time derivative of segment angular velocity
        a_0 = R_pos[:, 0] * R_acc[:, 1] - R_pos[:, 1] * R_acc[:, 0]
        a_1 = R_pos[:, 0] * R_vel[:, 1] - R_pos[:, 1] * R_vel[:, 0]
        a_2 = R_pos[:, 1] * R_vel[:, 1] + R_pos[:, 0] * R_vel[:, 0]
        a_3 = R_pos[:, 1]**2 + R_pos[:, 0]**2

        seg_alpha[:, i] = a_0 / a_3 - 2.0 * a_1 * a_2 / a_3**2

        seg_length = np.sqrt(R_pos[:, 0]**2 + R_pos[:, 1]**2)
        seg_lengths[i] = seg_length.mean()

        if np.max(np.abs(seg_length - seg_lengths[i])) > 0.1:
            msg = ('Error detected while processing segment {}\n'
                   'Segment length changed by more than 0.1 meters')
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.warn(msg.format(i), Warning)

    joint_angles = np.zeros((num_samples, num_segments - 1))
    joint_angular_rates = np.zeros((num_samples, num_segments - 1))
    joint_moments = np.zeros((num_samples, num_segments - 1))
    joint_forces = np.zeros((num_samples, 2 * (num_segments - 1)))

    # do the inverse dynamics, starting at the foot, but not for the torso
    # segment
    for segment in reversed(segments[1:]):

        i = segment.num

        # compute vectors P and D from center of mass to distal and proximal
        # joint

        prox_joint_x_idx = 2 * segment.prox_joint_marker_idx
        prox_joint_y_idx = prox_joint_x_idx + 1

        Px = marker_pos[:, prox_joint_x_idx] - seg_com_x_pos[:, i]
        Py = marker_pos[:, prox_joint_y_idx] - seg_com_y_pos[:, i]

        if segment.name == 'foot':
            # for the last segment, distal joint is the force plate data,
            # applied to foot at global origin

            Dx = -seg_com_x_pos[:, i]
            Dy = -seg_com_y_pos[:, i]

            dist_force_x = force_plate_values[:, 0]
            dist_force_y = force_plate_values[:, 1]
            dist_moment = force_plate_values[:, 2]
        else:
            # The marker at the distal joint of this segment is the same as
            # the marker at the proximal joint of next the segment.
            dist_joint_x_idx = 2 * segments[i + 1].prox_joint_marker_idx
            dist_joint_y_idx = dist_joint_x_idx + 1

            Dx = marker_pos[:, dist_joint_x_idx] - seg_com_x_pos[:, i]
            Dy = marker_pos[:, dist_joint_y_idx] - seg_com_y_pos[:, i]

            # loads at the distal joint are the opposite of the proximal
            # loads in the previous segment
            dist_force_x = -prox_force_x
            dist_force_y = -prox_force_y
            dist_moment = -prox_moment

        # solve force and moment at proximal joint from the Newton-Euler
        # equations
        mass = segment.normalized_mass

        prox_force_x = mass * seg_com_x_acc[:, i] - dist_force_x
        prox_force_y = mass * seg_com_y_acc[:, i] - dist_force_y + mass * g

        radius_of_gyration = (segment.radius_of_gyration_fraction *
                              seg_lengths[i])
        inertia = mass * radius_of_gyration**2

        prox_moment = (inertia * seg_alpha[:, i] - dist_moment
                       - (Dx * dist_force_y - Dy * dist_force_x)
                       - (Px * prox_force_y - Py * prox_force_x))

        # and store proximal joint motions and loads in the output variables

        # joint index (hip, knee, ankle) for the proximal joint of segment i
        j = i - 1

        joint_angles[:, j] = segment.sign * (seg_theta[:, i] -
                                             seg_theta[:, i - 1])

        joint_angular_rates[:, j] = segment.sign * (seg_omega[:, i] -
                                                    seg_omega[:, i - 1])

        joint_moments[:, j] = segment.sign * prox_moment

        joint_forces[:, 2 * j] = prox_force_x

        joint_forces[:, 2 * j - 1] = prox_force_y

    return joint_angles, joint_angular_rates, joint_moments, joint_forces
