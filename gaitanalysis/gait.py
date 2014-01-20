#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import os

# external libraries
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import pandas
from dtk import process
from oct2py import octave

# local
from .utils import _percent_formatter

# debugging
try:
    from IPython.core.debugger import Tracer
except ImportError:
    pass
else:
    set_trace = Tracer()


def find_constant_speed(time, speed, plot=False):
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

    Returns
    =======
    indice : integer
        The indice at which the speed is consider constant thereafter.
    new_time : ndarray, shape(n-indice,)
        The new time array for the constant speed section.

    """

    sample_rate = 1.0 / (time[1] - time[0])

    filtered_speed = process.butterworth(speed, 3.0, sample_rate)

    acceleration = np.hstack((0.0, np.diff(filtered_speed)))

    noise_level = np.max(np.abs(acceleration[int(0.2 * len(acceleration)):-1]))

    reversed_acceleration = acceleration[::-1]

    indice = np.argmax(reversed_acceleration > noise_level)

    additional_samples = sample_rate * 0.65

    new_indice = indice - additional_samples

    if plot is True:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(time, speed, '.', time, filtered_speed, 'g-')
        ax[0].plot(np.ones(2) * (time[len(time) - new_indice]),
                   np.hstack((np.max(speed), np.min(speed))))
        ax[1].plot(time, np.hstack((0.0, np.diff(filtered_speed))), '.')
        fig.show()

    return len(time) - (new_indice), time[len(time) - new_indice]


def interpolate(data_frame, time):
    """Returns a data frame with a index based on the provided time
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

    total_index = np.sort(np.hstack((data_frame.index.values, time)))
    reindexed_data_frame = data_frame.reindex(total_index)
    interpolated_data_frame = \
        reindexed_data_frame.apply(pandas.Series.interpolate,
                                   method='values').loc[time]

    # If the first or last value of a series is NA then the interpolate
    # function leaves it as an NA value, so use backfill to take care of
    # those.
    interpolated_data_frame = \
        interpolated_data_frame.fillna(method='backfill')
    # Because the time vector may have matching indices as the original
    # index (i.e. always the zero indice), drop any duplicates so the len()
    # stays consistent
    return interpolated_data_frame.drop_duplicates()


class WalkingData(object):
    """A class to store typical walking data."""

    def __init__(self, data_frame):
        """Initializes the data structure.

        Parameters
        ==========
        data_frame : pandas.DataFrame
            A data frame with an index of time and columns for each variable
            measured during a walking run.

        """
        # Could have a real time index:
        # new_index = [pandas.Timestamp(x, unit='s') for x in data.index]
        # data_frame.index = new_index
        # data.index.values.astype(float)*1e-9

        self.raw_data = data_frame

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
            The mass in kilograms of the subject.
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
        mfile = os.path.abspath(os.path.join(os.path.split(__file__)[0],
                                             '..', 'Octave-Matlab-Codes',
                                             '2D-Inverse-Dynamics'))
        octave.addpath(mfile)

        options = {'freq': low_pass_cutoff}

        time = self.raw_data.index.values.astype(float)
        time = time.reshape((len(time), 1))  # octave wants a column vector

        marker_sets = [left_leg_markers, right_leg_markers]
        force_sets = [left_leg_forces, right_leg_forces]

        side_labels = ['Left', 'Right']
        joint_labels = ['Hip', 'Knee', 'Ankle']
        sign_labels = ['Flexion', 'Flexion', 'PlantarFlexion', ('X', 'Y')]
        dynamic_labels = ['Angle', 'Rate', 'Moment', 'Force']
        scale_factors = [1.0, 1.0, body_mass, body_mass]

        for side_label, markers, forces in zip(side_labels, marker_sets,
                                               force_sets):

            marker_array = self.raw_data[markers].values.copy()
            normalized_force_array = \
                self.raw_data[forces].values.copy() / body_mass

            # oct2py doesn't allow multiple outputs to be stored in a tuple
            # like python, so you have to output each variable
            # independently
            angles, velocities, moments, forces = \
                octave.leg2d(time, marker_array, normalized_force_array,
                             options)

            dynamics = angles, velocities, moments, forces

            fours = zip(dynamics, sign_labels, dynamic_labels,
                        scale_factors)

            for array, sign_label, dynamic_label, scale_factor in fours:

                if isinstance(sign_label, tuple):

                    # array is N x 6, (Fx, Fy) for each joint

                    a = array[:, :2], array[:, 2:4], array[:, 4:]

                    for joint_label, vectors in zip(joint_labels, a):

                        for slab, vector in zip(sign_label, vectors.T):

                            label = '.'.join([side_label, joint_label, slab,
                                              dynamic_label])

                            self.raw_data[label] = scale_factor * vector

                else:

                    for joint_label, vector in zip(joint_labels, array.T):

                        label = '.'.join([side_label, joint_label,
                                          sign_label, dynamic_label])

                        self.raw_data[label] = scale_factor * vector

        return self.raw_data

    def grf_landmarks(self, right_vertical_grf_col_name,
                      left_vertical_grf_col_name, **kwargs):
        """Returns the times at which heel strikes and toe offs happen in
        the raw data.

        Parameters
        ==========
        right_vertical_grf_column_name : string
            The name of the column in the raw data frame which corresponds
            to the right foot vertical ground reaction force.
        left_vertical_grf_column_name : string
            The name of the column in the raw data frame which corresponds
            to the left foot vertical ground reaction force.

        Returns
        =======
        right_strikes : np.array
            All times at which right_grfy is non-zero and it was 0 at the
            preceding time index.
        left_strikes : np.array
            Same as above, but for the left foot.
        right_offs : np.array
            All times at which left_grfy is 0 and it was non-zero at the
            preceding time index.
        left_offs : np.array
            Same as above, but for the left foot.

        Notes
        =====
        This is a simple wrapper to gait_landmarks_from_grf and supports all
        the optional keyword arguments that it does.

        """

        right_strikes, left_strikes, right_offs, left_offs = \
            gait_landmarks_from_grf(self.raw_data.index.values.astype(float),
                                    self.raw_data[right_vertical_grf_col_name].values,
                                    self.raw_data[left_vertical_grf_col_name].values,
                                    **kwargs)

        self.strikes = {}
        self.offs = {}

        self.strikes['right'] = right_strikes
        self.strikes['left'] = left_strikes
        self.offs['right'] = right_offs
        self.offs['left'] = left_offs

        return right_strikes, left_strikes, right_offs, left_offs

    def plot_steps(self, *col_names, **kwargs):
        """Plots the steps.

        Parameters
        ==========
        col_names : string
            A variable number of strings naming the columns to plot.
        mean : boolean, optional
            If true the mean and standard deviation of the steps will be
            plotted.
        kwargs : key value pairs
            Any extra kwargs to pass to the matplotlib plot command.

        """

        if len(col_names) == 0:
            raise ValueError('Please supply some column names to plot.')

        try:
            mean = kwargs.pop('mean')
        except KeyError:
            mean = False

        fig, axes = plt.subplots(len(col_names), sharex=True)

        if mean is True:
            fig.suptitle('Mean and standard deviation of ' +
                         '{} steps.'.format(self.steps.shape[0]))
            mean_of_steps = self.steps.mean(axis='items')
            std_of_steps = self.steps.std(axis='items')
        else:
            fig.suptitle('Gait cycle for ' +
                         '{} steps.'.format(self.steps.shape[0]))

        for i, col_name in enumerate(col_names):
            try:
                ax = axes[i]
            except TypeError:
                ax = axes
            if mean is True:
                ax.fill_between(mean_of_steps.index.values.astype(float),
                                (mean_of_steps[col_name] -
                                    std_of_steps[col_name]).values,
                                (mean_of_steps[col_name] +
                                    std_of_steps[col_name]).values,
                                alpha=0.5)
                ax.plot(mean_of_steps.index.values.astype(float),
                        mean_of_steps[col_name].values, marker='o')
            else:
                for key, value in self.steps.iteritems():
                    ax.plot(value[col_name].index, value[col_name], **kwargs)

            ax.xaxis.set_major_formatter(_percent_formatter)

            ax.set_ylabel(col_name)

        # plot only on the last axes
        ax.set_xlabel('Time [s]')

        return axes

    def split_at(self, side, section='both', num_samples=None,
                 belt_speed_column=None):
        """Forms a pandas.Panel which has an item for each step. The index
        of each step data frame will be a percentage of gait cycle.

        Parameters
        ==========
        side : string {right|left}
            Split with respect to the right or left side heel strikes and/or
            toe-offs.
        section : string {both|stance|swing}
            Whether to split around the stance phase, swing phase, or both.
        num_samples : integer, optional
            If provided the time series in each step will be interpolated at
            values evenly spaced at num_sample in time across the step. If
            none, the minimum number of samples per step will be used.
        belt_speed_column : string, optional
            The column name corresponding to the belt speed on the
            corresponding side.

        Returns
        =======
        steps : pandas.Panel

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

        samples = []
        for i, lead_val in enumerate(lead):
            try:
                step_slice = self.raw_data[lead_val:trail[i]]
            except IndexError:
                pass
            else:
                samples.append(len(step_slice))

        max_num_samples = min(samples)
        if num_samples is None:
            num_samples = max_num_samples
        # TODO: This percent should always be computed with respect to heel
        # strike to next heel strike, i.e.:
        # stance: 0.0 to percent stance
        # swing: percent stance to 1.0
        # both: 0.0 to 1.0
        percent_gait = np.linspace(0.0, 1.0, num=num_samples)

        steps = {}
        step_data = {'Number of Samples': [],
                     'Step Duration': [],
                     'Cadence': [],  # step / time
                     }
        if belt_speed_column is not None:
            step_data['Stride Length'] = []
            step_data['Average Belt Speed'] = []
        # TODO : Compute average treadmill speed over step.
        for i, lead_val in enumerate(lead):
            try:
                data_frame = self.raw_data[lead_val:trail[i]]
            except IndexError:
                pass
            else:
                # make a new time index starting from zero for each step
                original_time = data_frame.index.values.astype(float)
                new_index = original_time - data_frame.index[0]
                data_frame.index = new_index
                # keep the original time around for future use
                data_frame['Original Time'] = original_time
                # create a time vector index which has a specific number
                # of samples over the period of time
                sub_sample_index = np.linspace(0.0, new_index[-1],
                                               num=num_samples)
                interpolated_data_frame = interpolate(data_frame,
                                                      sub_sample_index)
                # change the index to percent of gait cycle
                interpolated_data_frame.index = percent_gait
                steps[i] = interpolated_data_frame
                # compute some step stats
                step_data['Number of Samples'].append(len(data_frame))
                step_data['Step Duration'].append(new_index[-1])
                step_data['Cadence'].append(1.0 / new_index[-1])
                if belt_speed_column is not None:
                    step_data['Stride Length'] = \
                        cumtrapz(data_frame[belt_speed_column].values,
                                 new_index)
                    step_data['Average Belt Speed'] = \
                        data_frame[belt_speed_column].mean()

        self.steps = pandas.Panel(steps)
        self.step_data = pandas.DataFrame(step_data)

        return self.steps

    def time_derivative(self, col_names, new_col_names=None):
        """Numerically differentiates the specified columns with respect to
        the time index and adds the new columns to `self.raw_data`.

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
            self.raw_data[new_col_name] = \
                process.derivative(self.raw_data.index.values.astype(float),
                                   self.raw_data[col_name],
                                   method='combination')


def gait_landmarks_from_grf(time, right_grf, left_grf,
                            threshold=1e-5, do_plot=False, min_time=None,
                            max_time=None, num_steps_to_plot=None,
                            filter_frequency=None):
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
    do_plot : bool, optional (default: False)
        Create plots of the detected gait landmarks on top of the vertical
        ground reaction forces.
    min_time : float, optional
        If set, only consider times greater than `min_time`.
    max_time : float, optional
        If set, only consider times greater than `max_time`.
    num_steps_to_plot : integer, optional, default=None
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
    # TODO : Have an option to low pass filter the grf signals first so that
    # there is less noise in the swing phase.

    # Helper functions
    # ----------------
    def zero(number):
        return abs(number) < threshold

    def birth_times(ordinate):
        births = list()
        for i in index_range:
            # 'Skip' first value because we're going to peak back at previous
            # index.
            if zero(ordinate[i - 1]) and (not zero(ordinate[i])):
                births.append(time[i])
        return np.array(births)

    def death_times(ordinate):
        deaths = list()
        for i in index_range:
            if (not zero(ordinate[i - 1])) and zero(ordinate[i]):
                deaths.append(time[i])
        return np.array(deaths)

    def nearest_index(array, val):
        return np.abs(array - val).argmin()

    # Time range to consider.
    if max_time is None:
        max_idx = len(time)
    else:
        max_idx = nearest_index(time, max_time)

    if min_time is None:
        min_idx = 1
    else:
        min_idx = max(1, nearest_index(time, min_time))

    # If the ground reaction forces are very noisy, it may help to low pass
    # filter the signals before searching for the strikes and offs.

    if filter_frequency is not None:
        average_sample_rate = 1.0 / np.mean(np.diff(time))
        right_grf = process.butterworth(right_grf, filter_frequency,
                                        average_sample_rate)
        left_grf = process.butterworth(left_grf, filter_frequency,
                                       average_sample_rate)

    index_range = range(min_idx, max_idx)

    right_foot_strikes = birth_times(right_grf)
    left_foot_strikes = birth_times(left_grf)
    right_toe_offs = death_times(right_grf)
    left_toe_offs = death_times(left_grf)

    if do_plot:

        plt.figure()
        ones = np.array([1, 1])

        def myplot(index, label, ordinate, foot_strikes, toe_offs):
            ax = plt.subplot(2, 1, index)
            plt.plot(time[min_idx:max_idx], ordinate[min_idx:max_idx], '.k')
            plt.ylabel('vertical ground reaction force (N)')
            plt.title('%s (%i foot strikes, %i toe-offs)' % (
                label, len(foot_strikes), len(toe_offs)))

            for i, strike in enumerate(foot_strikes):
                if i == 0:
                    kwargs = {'label': 'foot strikes'}
                else:
                    kwargs = dict()
                plt.plot(strike * ones, ax.get_ylim(), 'r', **kwargs)

            for i, off in enumerate(toe_offs):
                if i == 0:
                    kwargs = {'label': 'toe-offs'}
                else:
                    kwargs = dict()
                plt.plot(off * ones, ax.get_ylim(), 'b', **kwargs)

            if num_steps_to_plot is not None:
                try:
                    ax.set_xlim((ax.get_xlim()[0],
                                foot_strikes[num_steps_to_plot]))
                except IndexError:
                    raise IndexError('{} is not a valid number of steps to plot'.format(num_steps_to_plot))

        myplot(1, 'left foot', left_grf, left_foot_strikes, left_toe_offs)
        plt.legend(loc='best')

        myplot(2, 'right foot', right_grf, right_foot_strikes, right_toe_offs)

        plt.xlabel('time (s)')
        plt.show()

    return right_foot_strikes, left_foot_strikes, right_toe_offs, left_toe_offs
