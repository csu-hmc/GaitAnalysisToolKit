#!/usr/bin/env python

# builtin
from distutils.version import LooseVersion

# external
import numpy as np
from numpy.core.umath_tests import matrix_multiply


def det3(ar):
    """Returns the determinants of an array of 3 x 3 matrices.

    Parameters
    ----------
    ar : array_like, shape(n, 3, 3)
        A array of 3 x 3 arrays.

    Returns
    -------
    tot : ndarray, shape(n, )
        An array of determinants.

    Notes
    -----

    This is extremely faster than calling numpy.linalg.det for 3 x 3
    matrices and is adopted from:

    http://mail.scipy.org/pipermail/numpy-discussion/2008-August/036928.html

    """
    a = ar[..., 0, 0]; b = ar[..., 0, 1]; c = ar[..., 0, 2]
    d = ar[..., 1, 0]; e = ar[..., 1, 1]; f = ar[..., 1, 2]
    g = ar[..., 2, 0]; h = ar[..., 2, 1]; i = ar[..., 2, 2]

    t = a.copy(); t *= e; t *= i; tot = t
    t = b.copy(); t *= f; t *= g; tot += t
    t = c.copy(); t *= d; t *= h; tot += t
    t = g.copy(); t *= e; t *= c; tot -= t
    t = h.copy(); t *= f; t *= a; tot -= t
    t = i.copy(); t *= d; t *= b; tot -= t

    return tot


def soederkvist(first_positions, second_positions):
    """Returns the rotation matrix and translation vector that relates two
    sets of markers in 3D space that are assumed to be attached to the same
    rigid body in two different positions and orientations given noisy
    measurements of the marker sets' global positions.

    Parameters
    ----------
    first_positions : array_like, shape(n, m, 3) or shape(1, m, 3)
        The x, y, and z coordinates of m markers in n positions in a global
        reference frame.
    second_positions : array_like, shape(n, m, 3)
        The x, y, and z coordinates of the same m markers in n positions in
        a global reference frame.

    Returns
    -------
    rotation : ndarray, shape(n, 3, 3)
        These rotation matrices is defined such that v1 = R * v2 where v1 is
        the vector, v, expressed in a reference frame associated with the
        first position and v2 is the same vector expressed in a reference
        frame associated with the second position.
    translation : ndarray, shape(n, 3)
        The translation vector from the first position to the second
        position expressed in the same frame as the x and y values.

    Notes
    -----

    The relationship between x, y, R and d is defined as:

    yi = R * xi + d

    This alogrithm is explicitly taken from:

    I. Soederkvist and P.A. Wedin (1993) Determining the movement of the
    skeleton using well-configured markers. J. Biomech. 26:1473-1477.

    But the same algorithm is described in:

    J.H. Challis (1995) A prodecure for determining rigid body transformation
    parameters, J. Biomech. 28, 733-737.

    with the latter also includes possibilities for scaling, reflection, and
    weighting of marker data.

    """
    num_frames, num_markers, num_coordinates = first_positions.shape

    # TODO : This may be an uneccesary memory increase and broadcasting may
    # deal with this properly without having to do this explicitly.
    if num_frames == 1:
        first_positions = np.repeat(first_positions,
                                    second_positions.shape[0], 0)
        num_frames = first_positions.shape[0]

    if num_markers != first_positions.shape[1]:
        raise ValueError('The first and second positions must have the ' +
                         'same number of markers.')

    if num_coordinates != 3 or second_positions.shape[2] != 3:
        raise ValueError('You must have three coordinates for each marker.')

    if num_frames != second_positions.shape[0]:
        raise ValueError('The first and second positions must have the ' +
                         'same number of frames.')

    # This is the mean location of the markers at each position.
    # n x 3
    mx = first_positions.mean(1)
    my = second_positions.mean(1)

    # Subtract the mean location of the markers to remove the translation
    # and leave the markers that have only been rotated with respect to one
    # another (about their mean).
    # n x m x 3 = n x m x 3 - n x 1 x 3
    A = first_positions - np.expand_dims(mx, 1)
    B = second_positions - np.expand_dims(my, 1)

    # n x 3 x m
    B_T = B.transpose((0, 2, 1))

    # n x 3 x 3 = n x 3 x m * n x m x 3
    C = matrix_multiply(B_T, A)

    # TODO : The svd of a 3 x 3 may involve simple math and it would be more
    # efficient to hard code it like the `det3` function for determinants.
    # Note that svd in NumPy svd returns the transpose of Q as compared to
    # Matlab/Octave.
    # n x 3 x 3, n x 3, n x 3 x 3 = svd(n x 3 x 3)
    if LooseVersion(np.__version__) < LooseVersion('1.8.0'):
        P = np.zeros_like(C)
        Q = np.zeros_like(C)
        for i, c in enumerate(C):
            P[i], T, Q[i] = np.linalg.svd(c)
    else:
        P, T, Q = np.linalg.svd(C)

    # n x 3 x 3 = n x 3 x 3 * n x 3 x 3
    rotations = matrix_multiply(P, Q)

    # n determinants
    det_P_Q = det3(rotations)

    # I think this construction of an identity matrix is here because the
    # determinants can sometimes be -1 instead of 1. This may represent a
    # reflection and plugging in the determinant deals with that. If the
    # determinants are all positive 1's then we can skip this operation.
    if (np.abs(det_P_Q - 1.0) < 1e-16).all():
        # n x 3 x 3
        I = np.zeros((num_frames, 3, 3))
        I[:, 0, 0] = 1.0
        I[:, 1, 1] = 1.0
        I[:, 2, 2] = det_P_Q

        # n x 3 x 3
        rotations = matrix_multiply(matrix_multiply(P, I), Q)

    # n x 3 = squeeze(n x 3 x 1 - n x 3 x 3 * n x 3 x 1)
    translations = np.squeeze(np.expand_dims(my, 2) -
        matrix_multiply(rotations, np.expand_dims(mx, 2)))

    return rotations, translations
