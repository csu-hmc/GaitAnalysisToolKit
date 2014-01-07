#!/usr/bin/env python

# external
import numpy as np
from numpy import testing

# local
from ..markers import soderkvist


def test_soderkvist():

    num_frames = 50  # n
    num_markers = 5  # m

    time = np.linspace(0, 10, num_frames)
    frequency = 0.1 * 2.0 * np.pi

    # Create three Euler angles which vary with time.
    # n x 3
    euler_angles = np.array([[0.5], [1.0], [2.0]]) * np.sin(frequency * time)

    # These are the rotation matrices we are trying to identify, size: 3 x 3
    # x n. They represent the 123 Euler rotation of the second reference
    # frame, B, with respect to the first reference frame, A, where va = R *
    # vb. This is the direction cosine matrix for a 1-2-3 Euler rotation
    # (body fixed rotation). See Kane, Likins, Levinson (1983) Page 423.
    s = np.sin(euler_angles)
    c = np.cos(euler_angles)
    expected_rotation = np.zeros((num_frames, 3, 3))
    for i in range(num_frames):
        expected_rotation[i] = np.array(
            [[c[1, i] * c[2, i],
              -c[1, i] * s[2, i],
              s[1, i]],
             [s[0, i] * s[1, i] * c[2, i] + s[2, i] * c[0, i],
              -s[0, i] * s[1, i] * s[2, i] + c[2, i] * c[0, i],
              -s[0, i] * c[1, i]],
             [-c[0, i] * s[1, i] * c[2, i] + s[2, i] * s[0, i],
              c[0, i] * s[1, i] * s[2, i] + c[2, i] * s[0, i],
              c[0, i] * c[1, i]]])

    # We assume that there are five markers located in the B body fixed
    # reference frame. Thus, the markers' relative positions do not vary
    # through time.
    marker_initial_vectors_in_local_frame = \
        np.random.random((3, num_markers))  # 3 x m

    # The location of the marker set in the global reference frame varies with
    # time in a simple linear fashion, size: 3 x n.
    expected_translation = np.array([[1.0], [2.0], [3.0]]) * time

    # We now can express the vector to each marker in the global reference
    # frame as the markers translate and rotate through time.
    marker_trajectories = np.zeros((num_frames, num_markers, 3))
    for i in range(num_frames):
        for j in range(num_markers):
            # 3 x 1 = 3 x 1 + 3 x 3 * 3 x 1
            marker_trajectories[i, j] = (expected_translation[:, i] +
                np.dot(expected_rotation[i],
                       marker_initial_vectors_in_local_frame[:, j]))

    # Now use the soder function to compute the rotation matrices and
    # translations vectors given the marker trajectories.
    rotation, translation = soderkvist(marker_trajectories[0], marker_trajectories)

    testing.assert_allclose(rotation, expected_rotation)
    testing.assert_allclose(translation, expected_translation)
