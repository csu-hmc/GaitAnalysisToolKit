#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import warnings

# external libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy import sparse
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


class SimpleControlSolver(object):
    """This assumes a simple linear control structure at each time instance
    in a gait cycle.

    The measured joint torques equal some limit cycle joint torque plus a
    matrix of gains multiplied by the error in the sensors and the nominal
    value of the sensors.

    m_measured(t) = m_nominal + K(t) [s_nominal(t) - s(t)] = m*(t) - K(t) s(t)

    This class solves for the time dependent gains and the "commanded"
    controls using a simple linear least squares.

    """

    def __init__(self, data, sensors, controls, validation_data=None):
        """Initializes the solver.

        Parameters
        ==========
        data : pandas.Panel, shape(r, n, p + q)
            A panel in which each item is a data frame of the time series of
            various measurements with time as the index. This should not
            have any missing values.
        sensors : sequence of strings
            A sequence of p strings which match column names in the data
            panel for the sensors.
        controls : sequence of strings
            A sequence of q strings which match column names in the data
            panel for the controls.
        validation_data : pandas.Panel, shape(w, n, p + q), optional
            A panel in which each item is a data frame of the time series of
            various measured sensors with time as the index. This should not
            have any missing values.

        Notes
        =====
        r : number of gait cycles
        n : number of time samples in each gait cycle
        p + q : number of measurements in the gait cycle

        m : number of gait cycles used in identification

        If no validation data is supplied then the last half of the gait
        cycles will be used for validation.
        m = r / 2 (integer division)
        else
        m = r

        """
        self._gain_inclusion_matrix = None

        self.sensors = sensors
        self.controls = controls

        if validation_data is None:
            num_gait_cycles = data.shape[0] / 2
            self.identification_data = data.iloc[:num_gait_cycles]
            self.validation_data = data.iloc[num_gait_cycles:]
        else:
            self.identification_data = data
            self.validation_data = validation_data

    @property
    def controls(self):
        return self._controls

    @controls.setter
    def controls(self, value):
        self._controls = value
        self.q = len(value)

    @property
    def identification_data(self):
        return self._identification_data

    @identification_data.setter
    def identification_data(self, value):
        self._identification_data = value
        self.m = value.shape[0]
        self.n = value.shape[1]

    @property
    def gain_inclusion_matrix(self):
        return self._gain_inclusion_matrix

    @gain_inclusion_matrix.setter
    def gain_inclusion_matrix(self, value):
        if value is not None:
            if value.shape != (self.q, self.p):
                raise ValueError('The gain omission matrix should be of ' +
                                 'shape({}, {})'.format(self.q, self.p))
        self._gain_inclusion_matrix = value

    @property
    def sensors(self):
        return self._sensors

    @sensors.setter
    def sensors(self, value):
        self._sensors = value
        self.p = len(value)

    @property
    def validation_data(self):
        return self._validation_data

    @validation_data.setter
    def validation_data(self, value):
        if value.shape[1] != self.identification_data.shape[1]:
            raise ValueError('The validation data must have the same '
                             'number of time gait cycles as the '
                             'identification data.')

        self._validation_data = value

    def compute_estimated_controls(self, gain_matrices, nominal_controls):
        """Returns the predicted values of the controls and the
        contributions to the controls given gains, K(t), and nominal
        controls, m*(t), for each point in the gait cycle.

        Parameters
        ==========
        gain_matrices : ndarray, shape(n, q, p)
            The estimated gain matrices for each time step.
        control_vectors : ndarray, shape(n, q)
            The nominal control vector plus the gains multiplied by the
            reference sensors at each time step.

        Returns
        =======
        panel : pandas.Panel, shape(m, n, q)
            There is one data frame to correspond to each gait cycle in
            self.validation_data. Each data frame has columns of time series
            which store m(t), m*(t), and the individual components due to
            K(t) * se(t).

        Notes
        =====

        m(t) = m0(t) + K(t) * [ s0(t) - s(t) ] = m0(t) + K(t) * se(t)
        m(t) = m*(t) - K(t) * s(t)

        This function returns m(t), m0(t), m*(t) for each control and K(t) *
        [s0(t) - s(t)] for each sensor affecting each control. Where s0(t)
        is estimated by taking the mean with respect to the gait cycles.

        """
        # generate all of the column names
        contributions = []
        for control in self.controls:
            for sensor in self.sensors:
                contributions.append(control + '-' + sensor)

        control_star = [control + '*' for control in self.controls]
        control_0 = [control + '0' for control in self.controls]

        col_names = self.controls + contributions + control_star + control_0

        panel = {}

        # The mean of the sensors which we consider to be the commanded
        # motion. This may be a bad assumption, but is the best we can do.
        mean_sensors = self.validation_data.mean(axis='items')[self.sensors]

        for i, df in self.validation_data.iteritems():

            blank = np.zeros((self.n, self.q * 3 + self.p * self.q))
            results = pandas.DataFrame(blank, index=df.index,
                                       columns=col_names)

            sensor_error = mean_sensors - df[self.sensors]

            for j in range(self.n):

                # m(t) = m*(t) - K(t) * s(t)
                m = nominal_controls[j] - np.dot(gain_matrices[j],
                                                 df[self.sensors].iloc[j])
                # m0(t) = m(t) - K(t) * se(t)
                m0 = m - np.dot(gain_matrices[j],
                                sensor_error[self.sensors].iloc[j])

                # these assignments don't work if I do:
                # results[self.controls].iloc[j] = m
                # but this seems to work
                # results.iloc[j][self.controls] = m
                # this is explained here:
                # https://github.com/pydata/pandas/issues/5093
                row_label = results.index[j]
                results.loc[row_label, self.controls] = m
                results.loc[row_label, control_0] = m0
                results.loc[row_label, control_star] = nominal_controls[j]

                for k, sensor in enumerate(self.sensors):
                    # control contribution due to the kth sensor
                    names = [c + '-' + sensor for c in self.controls]
                    results.loc[row_label, names] = gain_matrices[j, :, k] * \
                        sensor_error.iloc[j, k]

            results['Original Time'] = df['Original Time']

            panel[i] = results

        return pandas.Panel(panel)

    def deconstruct_solution(self, x, covariance):
        """Returns the gain matrices, K(t), and m*(t) for each time step in
        the gait cycle given the solution vector and the covariance matrix
        of the solution.

        m(t) = m*(t) - K(t) s(t)

        Parameters
        ==========
        x : array_like, shape(n * q * (p + 1),)
            The solution matrix containing the gains and the commanded
            controls.
        covariance : array_like, shape(n * q * (p + 1), n * q * (p + 1))
            The covariance of x with respect to the variance in the fit.

        Returns
        =======
        gain_matrices : ndarray,  shape(n, q, p)
            The gain matrices at each time step, K(t).
        control_vectors : ndarray, shape(n, q)
            The nominal control vector plus the gains multiplied by the
            reference sensors at each time step.
        gain_matrices_variance : ndarray, shape(n, q, p)
            The variance of the found gains (covariance is neglected).
        control_vectors_variance : ndarray, shape(n, q)
            The variance of the found commanded controls (covariance is
            neglected).

        Notes
        =====
        x looks like:
            [k11(0), k12(0), ..., kqp(0), m1*(0), ..., mq*(0), ...,
             k11(n), k12(0), ..., kqp(n), m1*(n), ..., mq*(n)]

        If there is a gain omission matrix then nan's are substituted for
        all gains that were set to zero.

        """
        # TODO : Fix the doc string to reflect the fact that x will be
        # smaller when there is a gain omission matrix.

        # If there is a gain omission matrix then augment the x vector and
        # covariance matrix with nans for the missing values.
        if self.gain_inclusion_matrix is not None:
            x1 = self.gain_inclusion_matrix.flatten()
            x2 = np.array(self.q * [True])
            for i in range(self.n):
                try:
                    x_total = np.hstack((x_total, x1, x2))
                except NameError:
                    x_total = np.hstack((x1, x2))
            x_total = x_total.astype(object)
            x_total[x_total == True] = x
            x_total[x_total == False] = np.nan
            x = x_total.astype(float)

            cov_total = np.nan * np.ones((len(x), len(x)))
            cov_total[np.outer(~np.isnan(x), ~np.isnan(x))] = \
                covariance.flatten()
            covariance = cov_total

            x[np.isnan(x)] = 0.0
            covariance[np.isnan(covariance)] = 0.0

        gain_matrices = np.zeros((self.n, self.q, self.p))
        control_vectors = np.zeros((self.n, self.q))

        gain_matrices_variance = np.zeros((self.n, self.q, self.p))
        control_vectors_variance = np.zeros((self.n, self.q))

        parameter_variance = np.diag(covariance)

        for i in range(self.n):

            k_start = i * self.q * (self.p + 1)
            k_end = self.q * ((i + 1) * self.p + i)
            m_end = (i + 1) * self.q * (self.p + 1)

            gain_matrices[i] = x[k_start:k_end].reshape(self.q, self.p)
            control_vectors[i] = x[k_end:m_end]

            gain_matrices_variance[i] = \
                parameter_variance[k_start:k_end].reshape(self.q, self.p)
            control_vectors_variance[i] = parameter_variance[k_end:m_end]

        return (gain_matrices, control_vectors, gain_matrices_variance,
                control_vectors_variance)

    def form_a_b(self):
        """Returns the A matrix and the b vector for the linear least
        squares fit.

        Returns
        =======
        A : ndarray, shape(n * q, n * q * (p + 1))
            The A matrix which is sparse and contains the sensor
            measurements and ones.
        b : ndarray, shape(n * q,)
            The b vector which constaints the measured controls.

        Notes
        =====

        In the simplest fashion, you can put::

            m(t) = m*(t) - K * s(t)

        into the form::

            Ax = b

        with::

            b = m(t)
            A = [-s(t) 1]
            x = [K(t) m*(t)]^T

            [-s(t) 1] * [K(t) m*(t)]^T = m(t)

        """
        control_vectors = self.form_control_vectors()

        b = np.array([])
        for cycle in control_vectors:
            for time_step in cycle:
                b = np.hstack((b, time_step))

        sensor_vectors = self.form_sensor_vectors()

        A = np.zeros((self.m * self.n * self.q,
                      self.n * self.q * (self.p + 1)))

        for i in range(self.m):

            Am = np.zeros((self.n * self.q,
                           self.n * self.q * (self.p + 1)))

            for j in range(self.n):

                An = np.zeros((self.q, self.q * self.p))

                for row in range(self.q):

                    An[row, row * self.p:(row + 1) * self.p] = \
                        -sensor_vectors[i, j]

                An = np.hstack((An, np.eye(self.q)))

                num_rows, num_cols = An.shape

                Am[j * num_rows:(j + 1) * num_rows, j * num_cols:(j + 1) *
                    num_cols] = An

            A[i * self.n * self.q:i * self.n * self.q + self.n * self.q] = Am

        # If there are nans in the gain omission matrix, then delete the
        # columns in A associated with gains that are set to zero.
        # TODO : Turn this into a method because I use it at least twice.
        if self.gain_inclusion_matrix is not None:
            x1 = self.gain_inclusion_matrix.flatten()
            x2 = np.array(self.q * [True])
            for i in range(self.n):
                try:
                    x_total = np.hstack((x_total, x1, x2))
                except NameError:
                    x_total = np.hstack((x1, x2))

            A = A[:, x_total]

        # TODO : Matrix rank computations for large A matrices can take
        # quite some time.
        rank_of_A = np.linalg.matrix_rank(A)
        if rank_of_A < A.shape[1] or rank_of_A > A.shape[0]:
            warnings.warn('The rank of A is {} and x is of length {}.'.format(
                rank_of_A, A.shape[1]))

        return A, b

    def form_control_vectors(self):
        """Returns an array of control vectors for each cycle and each time
        step in the identification data.

        Returns
        =======
        control_vectors : ndarray, shape(m, n, q)
            The sensor vector form the i'th cycle and the j'th time step
            will look like [control_0, ..., control_(q-1)].

        """
        control_vectors = np.zeros((self.m, self.n, self.q))
        for i, (panel_name, data_frame) in \
                enumerate(self.identification_data.iteritems()):
            for j, (index, values) in \
                    enumerate(data_frame[self.controls].iterrows()):
                control_vectors[i, j] = values.values

        return control_vectors

    def form_sensor_vectors(self):
        """Returns an array of sensor vectors for each cycle and each time
        step in the identification data.

        Returns
        =======
        sensor_vectors : ndarray, shape(m, n, p)
            The sensor vector form the i'th cycle and the j'th time step
            will look like [sensor_0, ..., sensor_(p-1)].
        """
        sensor_vectors = np.zeros((self.m, self.n, self.p))
        for i, (panel_name, data_frame) in \
                enumerate(self.identification_data.iteritems()):
            for j, (index, values) in \
                    enumerate(data_frame[self.sensors].iterrows()):
                sensor_vectors[i, j] = values.values

        return sensor_vectors

    def least_squares(self, A, b, ignore_cov=False):
        """Returns the solution to the linear least squares and the
        covariance matrix of the solution.

        Parameters
        ==========
        A : array_like, shape(n, m)
            The coefficient matrix of Ax = b.
        b : array_like, shape(n,)
            The right hand side of Ax = b.
        ignore_cov: boolean, optional, default=False
            The covariance computation for a very large A matrix can be
            extremely slow. If this is set to True, then the computation is
            skipped and the covariance of the identified parameters is set
            to zero.

        Returns
        =======
        x : ndarray, shape(m,)
            The best fit solution.
        variance : float
            The variance of the fit.
        covariance : ndarray, shape(m, m)
            The covariance of the solution.

        """

        num_equations, num_unknowns = A.shape

        if num_equations < num_unknowns:
            raise Exception('Please add some walking cycles. There is ' +
                            'not enough data to solve for the number of ' +
                            'unknowns.')

        if sparse.issparse(A):
            # scipy.sparse.linalg.lsmr is also an option
            x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = \
                sparse.linalg.lsqr(A, b)
            sum_of_residuals = r1norm  # this may should be the r2norm
            # TODO : Make sure I'm doing the correct norm here.
        else:
            x, sum_of_residuals, rank, s = np.linalg.lstsq(A, b)
            # Also this is potentially a faster implementation:
            # http://graal.ift.ulaval.ca/alexandredrouin/2013/06/29/linear-least-squares-solver/

            # TODO : compute the rank of the sparse matrix
            if rank < A.shape[1] or rank > A.shape[0]:
                print("After lstsq")
                warnings.warn('The rank of A is {} and the shape is {}.'.format(
                    rank, A.shape))

        if ignore_cov is True:
            degrees_of_freedom = (A.shape[0] - A.shape[1])
            variance = sum_of_residuals / degrees_of_freedom
            covariance = np.zeros((len(x), len(x)))
        else:
            variance, covariance = \
                process.least_squares_variance(A, sum_of_residuals)

        return x, variance, covariance

    def plot_control_contributions(self, estimated_panel,
                                   max_num_gait_cycles=4):
        """Plots two graphs for each control and each gait cycle showing
        contributions from the linear portions. The first set of graphs
        shows the first few gait cycles and the contributions to the control
        moments. The second set of graph shows the mean contributions to the
        control moment over all gait cycles.

        Parameters
        ----------
        panel : pandas.Panel, shape(m, n, q)
            There is one data frame to correspond to each gait cycle. Each
            data frame has columns of time series which store m(t), m*(t),
            and the individual components due to K(t) * se(t).

        """

        num_gait_cycles = estimated_panel.shape[0]
        if num_gait_cycles > max_num_gait_cycles:
            num_gait_cycles = max_num_gait_cycles

        column_names = estimated_panel.iloc[0].columns

        for control in self.controls:
            fig, axes = plt.subplots(int(round(num_gait_cycles / 2.0)), 2,
                                     sharex=True, sharey=True)
            fig.suptitle('Contributions to the {} control'.format(control))
            contribs = [name for name in column_names if '-' in name and
                        name.startswith(control)]
            contribs += [control + '0']

            for ax, (gait_cycle_num, cycle) in zip(axes.flatten()[:num_gait_cycles],
                                             estimated_panel.iteritems()):
                # here we want to plot each component of this:
                # m0 + k11 * se1 + k12 se2
                cycle[contribs].plot(kind='bar', stacked=True, ax=ax,
                                     title='Gait Cycle {}'.format(gait_cycle_num),
                                     colormap='jet')
                # TODO: Figure out why the xtick formatting doesn't work
                # this formating method seems to make the whole plot blank
                #formatter = FuncFormatter(lambda l, p: '{1.2f}'.format(l))
                #ax.xaxis.set_major_formatter(formatter)
                # this formatter doesn't seem to work with this plot as it
                # operates on the xtick values instead of the already
                # overidden labels
                #ax.xaxis.set_major_formatter(_percent_formatter)
                # This doesn't seem to actually overwrite the labels:
                #for label in ax.get_xticklabels():
                    #current = label.get_text()
                    #label.set_text('{:1.0%}'.format(float(current)))

                for t in ax.get_legend().get_texts():
                    t.set_fontsize(6)
                    # only show the contribution in the legend
                    try:
                        t.set_text(t.get_text().split('-')[1])
                    except IndexError:
                        t.set_text(t.get_text().split('.')[1])

            for axis in axes[-1]:
                axis.set_xlabel('Time [s]')

        # snatch the colors from the last axes
        contrib_colors = [patch.get_facecolor() for patch in
                          ax.get_legend().get_patches()]

        mean = estimated_panel.mean(axis='items')
        std = estimated_panel.std(axis='items')

        for control in self.controls:
            fig, ax = plt.subplots()
            fig.suptitle('Contributions to the {} control'.format(control))
            contribs = [control + '0']
            contribs += [name for name in column_names if '-' in name and
                         name.startswith(control)]
            for col, color in zip(contribs, contrib_colors):
                ax.errorbar(mean.index.values, mean[col].values,
                            yerr=std[col].values, color=color)

            labels = []
            for contrib in contribs:
                try:
                    labels.append(contrib.split('-')[1])
                except IndexError:
                    labels.append(contrib.split('.')[1])
            ax.legend(labels, fontsize=10)
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(_percent_formatter)

    def plot_estimated_vs_measure_controls(self, estimated_panel, variance):
        """Plots a figure for each control where the measured control is
        shown compared to the estimated along with a plot of the error.

        Parameters
        ==========
        estimated_panel : pandas.Panel
            A panel where each item is a gait cycle.
        variance : float
            The variance of the fit.

        Returns
        =======
        axes : array of matplotlib.axes.Axes, shape(q,)
            The plot axes.

        """

        # TODO : Construct the original time vector for the index.
        # TODO : Plot the estimated controls versus the full actual
        # measurement curve so that the measurement curve is very smooth.

        estimated_walking = pandas.concat([df for k, df in
                                           estimated_panel.iteritems()],
                                          ignore_index=True)

        actual_walking = pandas.concat([df for k, df in
                                        self.validation_data.iteritems()],
                                       ignore_index=True)

        fig, axes = plt.subplots(self.q * 2, sharex=True)

        for i, control in enumerate(self.controls):

            compare_axes = axes[i * 2]
            error_axes = axes[i * 2 + 1]

            sample_number = actual_walking.index.values
            measured = actual_walking[control].values
            predicted = estimated_walking[control].values
            std_of_predicted = np.sqrt(variance) * np.ones_like(predicted)
            error = measured - predicted
            rms = np.sqrt(np.linalg.norm(error).mean())
            r_squared = process.coefficient_of_determination(measured,
                                                             predicted)

            compare_axes.plot(sample_number, measured, color='black',
                              marker='.')
            compare_axes.errorbar(sample_number, predicted,
                                  yerr=std_of_predicted, fmt='.')
            compare_axes.set_ylabel(control)
            compare_axes.legend(('Measured',
                                 'Estimated {:1.1%}'.format(r_squared)))

            if i == len(self.controls) - 1:
                error_axes.set_xlabel('Sample Number')

            error_axes.plot(sample_number, error, color='black')
            error_axes.legend(['RMS = {:1.2f}'.format(rms)])
            error_axes.set_ylabel('Error in\n{}'.format(control))

        return axes

    def plot_gains(self, gains, gain_variance, y_scale_function=None):
        """Plots the identified gains versus percentage of the gait cycle.

        Parameters
        ==========
        gain_matrix : ndarray, shape(n, q, p)
            The estimated gain matrices for each time step.
        gain_variance : ndarray, shape(n, q, p)
            The variance of the estimated gain matrices for each time step.
        y_scale_function : function, optional, default=None
            A function that returns the portion of a control and sensor
            label that can be used for scaling the y axes.

        Returns
        =======
        axes : ndarray of matplotlib.axis, shape(q, p)

        """

        # TODO : Make plots have the same scale if they share the same units
        # or figure out how to normalize these.

        n, q, p = gains.shape

        fig, axes = plt.subplots(q, p, sharex=True)

        percent_of_gait_cycle = \
            self.identification_data.iloc[0].index.values.astype(float)
        xlim = (percent_of_gait_cycle[0], percent_of_gait_cycle[-1])

        for i in range(q):
            for j in range(p):
                try:
                    ax = axes[i, j]
                except TypeError:
                    ax = axes
                sigma = np.sqrt(gain_variance[:, i, j])
                ax.fill_between(percent_of_gait_cycle,
                                gains[:, i, j] - sigma,
                                gains[:, i, j] + sigma,
                                alpha=0.5)
                ax.plot(percent_of_gait_cycle, gains[:, i, j], marker='o')
                ax.xaxis.set_major_formatter(_percent_formatter)
                ax.set_xlim(xlim)

                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(6)
                if j == 0:
                    ax.set_ylabel('{}\nGain'.format(self.controls[i]),
                                  fontsize=10)
                if i == 0:
                    ax.set_title(self.sensors[j])
                if i == q - 1:
                    ax.set_xlabel('Percent of gait cycle', fontsize=10)

        plt.tight_layout()

        return axes

    def solve(self, sparse_a=False, gain_inclusion_matrix=None,
              ignore_cov=False):
        """Returns the estimated gains and sensor limit cycles along with
        their variance.

        Parameters
        ==========
        sparse_a : boolean, optional, default=False
            If true a sparse A matrix will be used along with a sparse
            linear least squares solver.
        gain_inclusion_matrix : boolean array_like, shape(q, p)
            A matrix which is the same shape as the identified gain matrices
            which has False in place of gains that should be assumed to be
            zero and True for gains that should be identified.
        ignore_cov: boolean, optional, default=False
            The covariance computation for a very large A matrix can be
            extremely slow. If this is set to True, then the computation is
            skipped and the covariance of the identified parameters is set
            to zero.

        Returns
        =======
        gain_matrices : ndarray, shape(n, q, p)
            The estimated gain matrices for each time step.
        control_vectors : ndarray, shape(n, q)
            The nominal control vector plus the gains multiplied by the
            reference sensors at each time step.
        variance : float
            The variance in the fitted curve.
        gain_matrices_variance : ndarray, shape(n, q, p)
            The variance of the found gains (covariance is neglected).
        control_vectors_variance : ndarray, shape(n, q)
            The variance of the found commanded controls (covariance is
            neglected).
        estimated_controls : pandas.Panel

        """
        self.gain_inclusion_matrix = gain_inclusion_matrix

        A, b = self.form_a_b()

        # TODO : To actually get some memory reduction I should construct
        # the A matrix with a scipy.sparse.lil_matrix in self.form_a_b
        # instead of simply converting the dense matrix after I build it.

        if sparse_a is True:
            A = sparse.csr_matrix(A)

        x, variance, covariance = \
            self.least_squares(A, b, ignore_cov=ignore_cov)

        deconstructed_solution = self.deconstruct_solution(x, covariance)

        gain_matrices = deconstructed_solution[0]
        gain_matrices_variance = deconstructed_solution[2]

        nominal_controls = deconstructed_solution[1]
        nominal_controls_variance = deconstructed_solution[3]

        estimated_controls = \
            self.compute_estimated_controls(gain_matrices, nominal_controls)

        return (gain_matrices, nominal_controls, variance,
                gain_matrices_variance, nominal_controls_variance,
                estimated_controls)
