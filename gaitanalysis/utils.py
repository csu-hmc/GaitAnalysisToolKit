#!/usr/bin/env python
# -*- coding: utf-8 -*-

# external libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def _to_percent(value, position):
    """Returns a string representation of a percentage for matplotlib
    tick labels."""
    if plt.rcParams['text.usetex'] is True:
        return '{:1.0%}'.format(value).replace('%', r'$\%$')
    else:
        return '{:1.0%}'.format(value)

# tick label formatter
_percent_formatter = FuncFormatter(_to_percent)
