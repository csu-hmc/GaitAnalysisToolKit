#!/usr/bin/env python
# -*- coding: utf-8 -*-

# external
from numpy import testing


def compare_data_frames(actual, expected, rtol=1e-7, atol=0.0):
    """Compares two data frames column by column for numerical
    equivalence."""

    # Make sure all columns are present.
    assert sorted(list(expected.columns)) == sorted(list(actual.columns))

    for col in expected.columns:
        testing.assert_allclose(actual[col], expected[col], rtol, atol)
