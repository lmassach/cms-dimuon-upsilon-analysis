# Copyright (C) 2021 Ludovico Massaccesi
#
# This file is part of upsilon_analysis.
#
# upsilon_analysis is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Utility functions for the upsilon analysis."""
import collections

__all__ = ["y_bin_edges", "pt_bin_edges", "bins", "static_variables",
           "GausParameters", "LineParameters", "FitResults",
           "get_gaus_parameters", "print_fit_results", "Namespace"]


def y_bin_edges(y_min, y_max, n_bins):
    """A generator that yields bin edges.

    Guarantees that ``y_min`` is the first item yielded and ``y_max`` is
    the last.
    """
    bin_width = (y_max - y_min) / n_bins
    for i in range(n_bins):
        yield y_min + i * bin_width
    yield y_max


def pt_bin_edges(pt_min, pt_max, bin_width):
    """A generator that makes the bins larger above a certain threshold.

    Yields bin edges from ``pt_min`` to ``pt_max`` (both end are always
    present). Above 40 GeV/c, bins are made progressively wider:

    *  above 40 GeV/c they are 1.5x as wide;
    *  above 45 GeV/c they are 2x as wide;
    *  above 50 GeV/c they are 2.5x as wide;
    *  above 60 GeV/c they are 5x as wide;
    *  above 70 GeV/c they are 15x as wide.
    """
    edge = pt_min
    while edge < pt_max:
        yield edge
        if edge < 40:
            new_edge = edge + bin_width
        elif edge < 45:
            new_edge = edge + bin_width * 1.5
        elif edge < 50:
            new_edge = edge + bin_width * 2
        elif edge < 60:
            new_edge = edge + bin_width * 2.5
        elif edge < 70:
            new_edge = edge + bin_width * 5
        else:
            new_edge = edge + bin_width * 15
        if pt_max - new_edge < new_edge - edge:
            break
        edge = new_edge
    yield pt_max


def bins(edges):
    """Roughly equivalent to ``zip(edges[:-1], edges[1:])``.

    Actually ``edges`` can also be an iterable that does not support
    slicing, so it is a more general implementation (altough probably
    less efficient).
    """
    last = None
    for item in edges:
        if last is not None:
            yield (last, item)
        last = item


def static_variables(**kwargs):
    """Decorator to define C-like static variables for a function.

    Using the decorator like in::

       @static_variables(variable=value)
       def func(...):
           ...

    is equivalent to::

       def func(...):
           ...
       func.variable = value

    so the "static variable" can be accessed from within the function by
    writing ``func.variable``.
    """
    def decorator(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorator


GausParameters = collections.namedtuple("GausParameters", "a m sigma")


LineParameters = collections.namedtuple("LineParameters", "q m")


FitResults = collections.namedtuple("FitResults", "y1 y2 y3 bkg chi2 ndf")


def get_gaus_parameters(ga, bin_width, hist_range=(8.5, 11.5)):
    """Gets a ``GausParameters`` named tuple from a ``TF1``.

    This function takes a ``TF1`` defined by the formula "gaus(0)" as
    input and returns a ``GausParameters`` where the field ``a`` is the
    total number of occurrences given by the fit (while the other fields
    are the usual mean and sigma). This function is necessary because
    "gaus(0)" is *not* normalized.

    The argument ``ga`` is the ``TF1`` with the parameters set by the
    fit.

    Assuming the histogram being fitted has fixed-width bins, the
    integral of the gaussian will be equal to the integral of the
    histogram, which is the total number of entries times the bin width.

    Note that this also works if the fit option "I" (using the integral
    of the function in the bin rather than its value at the center) is
    used, since ROOT normalizes histogram integrals to the bin width.

    Do not forget to provide the range if the default ``hist_range`` is
    not correct.
    """
    p0 = ga.Integral(*hist_range) / bin_width
    return GausParameters(p0, ga.GetParameter(1), ga.GetParameter(2))


def print_fit_results(results, file=None):
    """Print fit results in CSV format to ``file`` (default stdout)."""
    print("y_min,y_max,pt_min,pt_max,n_y1,m_y1,sigma_y1,n_y2,m_y2,sigma_y2,"
          "n_y3,m_y3,sigma_y3,q,m,chi2,ndf", file=file)
    for (y_low, y_high), pt_bins in results.items():
        for (pt_low, pt_high), res in pt_bins.items():
            cols = (y_low, y_high, pt_low, pt_high, *res.y1, *res.y2, *res.y3,
                    *res.bkg, res.chi2, res.ndf)
            print(",".join(str(x) for x in cols), file=file)


class Namespace:
    """A class for holding options similar to ``argparse.Namespace``.

    ``kwargs`` is used to set the initial attributes of the object.
    """
    def __init__(self, **kwargs):
        self.update(kwargs)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        for name, value in self._get_kwargs():
            arg_strings.append(f"{name}={value!r}")
        return f"{type_name}({', '.join(arg_strings)})"

    def _get_kwargs(self):
        return sorted(self.__dict__.items())

    def update(self, dictionary):
        """Same as the constuctor, but takes a dictionary."""
        for name, value in dictionary.items():
            setattr(self, name, value)
