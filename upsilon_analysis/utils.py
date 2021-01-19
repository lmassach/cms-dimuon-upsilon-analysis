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
from collections import namedtuple

__all__ = ["y_bin_edges", "pt_bin_edges", "bins", "static_variables",
           "GausParameters", "LineParameters", "FitResults",
           "get_gaus_parameters", "print_fit_results",
           "sort_bins", "uniques"]


def y_bin_edges(y_min, y_max, n_bins):
    """A generator that yields bin edges.

    :param y_min: The lower edge of the first bin.
    :type y_min: float
    :param y_max: The upper edge of the last bin.
    :type y_max: float
    :param n_bins: The number of bins.
    :type n_bins: int
    :rtype: float

    Guarantees that ``y_min`` is the first item yielded and ``y_max`` is
    the last, without roundoff errors.
    """
    bin_width = (y_max - y_min) / n_bins
    for i in range(n_bins):
        yield y_min + i * bin_width
    yield y_max


def pt_bin_edges(pt_min, pt_max, bin_width):
    """A generator that makes the bins larger above a certain threshold.

    :param pt_min: The lower edge of the first bin.
    :type pt_min: float
    :param pt_max: The upper edge of the last bin.
    :type pt_max: float
    :param bin_width: The width bins (but see below).
    :type bin_width: float
    :rtype: float

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

    :param edges: The bins' edges.
    :type edges: :class:`Iterable[float]`
    :rtype: :class:`tuple[float, float]`

    Actually ``edges`` can also be an iterable that does not support
    slicing, so it is a more general implementation than that with
    ``zip`` (altough probably less efficient).
    """
    last = None
    for item in edges:
        if last is not None:
            yield (last, item)
        last = item


def static_variables(**kwargs):
    """Decorator to define C-like static variables for a function.

    Using the decorator like in:

    .. code-block:: python

       @static_variables(variable=value)
       def func(...):
           ...

    is equivalent to:

    .. code-block:: python

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


class GausParameters(namedtuple("GausParameters", "a m sigma")):
    """A :class:`namedtuple` for the results of a fit with a Gaussian.

    :ivar a: Number of occurrences fitted (see
       :any:`get_gaus_parameters`).
    :vartype a: float
    :ivar m: Fitted pole mass (Gaussian's mean).
    :vartype m: float
    :ivar sigma: Fitted width (Gaussian's standard deviation).
    :vartype sigma: float
    """


class LineParameters(namedtuple("LineParameters", "q m")):
    """A named tuple for the results of a fit with a line.

    :ivar q: Constant / y-intercept.
    :vartype q: float
    :ivar m: Slope / coefficient of x.
    :vartype m: float
    """


class FitResults(namedtuple("FitResults", "y1 y2 y3 bkg chi2 ndf nevt")):
    """A named tuple for the results of the global fit.

    :ivar y1: First resonance.
    :vartype y1: GausParameters
    :ivar y2: Second resonance.
    :vartype y2: GausParameters
    :ivar y3: Third resonance.
    :vartype y3: GausParameters
    :ivar bkg: Background.
    :vartype bkg: LineParameters
    :ivar chi2: Chi-squared of the fit.
    :vartype chi2: float
    :ivar ndf: Number of degrees of freedom of the fit.
    :vartype ndf: float
    :ivar nevt: Total number of events in the fitted histogram.
    :vartype nevt: int
    """


def get_gaus_parameters(ga, bin_width, hist_range=(8.5, 11.5)):
    """Gets a :any:`GausParameters` from a ``TF1``.

    :param ga: The ``TF1`` with the parameters to get.
    :type ga: ROOT.TF1
    :param bin_width: The width of the bin of the fitted histogram.
    :type bin_width: float
    :param hist_range: The range of the fitted histogram.
    :type hist_range: :class:`tuple[float, float]`, optional
    :rtype: GausParameters

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

    Note that this also works if the fit option ``I`` (using the
    integral of the function in the bin rather than its value at the
    center) is used, since ROOT normalizes histogram integrals to the
    bin width.

    .. warning::
       Do not forget to provide the range if the default ``hist_range`` is
       not correct.
    """
    p0 = ga.Integral(*hist_range) / bin_width
    return GausParameters(p0, ga.GetParameter(1), ga.GetParameter(2))


def print_fit_results(results, file=None):
    """Print fit results in CSV format to ``file`` (default ``stdout``).

    :param results: A dictionary like that returned by
       :any:`core.fit_histograms`
    :type results: :class:`dict[tuple[float, float],
       dict[tuple[float, float], FitResults]]`
    :param file: The file to write to. Default is stdout.
    :return: None
    """
    print("y_min,y_max,pt_min,pt_max,n_y1,m_y1,sigma_y1,n_y2,m_y2,sigma_y2,"
          "n_y3,m_y3,sigma_y3,q,m,chi2,ndf", file=file)
    for (y_low, y_high), pt_bins in results.items():
        for (pt_low, pt_high), res in pt_bins.items():
            cols = (y_low, y_high, pt_low, pt_high, *res.y1, *res.y2, *res.y3,
                    *res.bkg, res.chi2, res.ndf)
            print(",".join(str(x) for x in cols), file=file)


def sort_bins(iterable):
    """Takes an iterable of bins and returns a sorted list of them.

    :param iterable: The iterable with the bins to sort.
    :type iterable: :class:`Iterable[tuple[float, float]]`
    :raises RuntimeError: If the bins are not valid or overlap.
    :rtype: :class:`list[tuple[float, float]]`

    This function takes an iterable of bins, given as tuples
    ``(bin_low, bin_high)``, and returns a list of the same bins, but
    in ascending order.

    The following requirements must be met:

    *  all bins must be well defined, i.e. the lower limit must be
       strictly less than the upper limit;
    *  bins must not overlap;

    if they are not, ``RuntimeError`` will be raised.
    """
    iterable = list(iterable)
    if any(b[0] >= b[1] for b in iterable):
        raise RuntimeError("Invalid bin found (lower limit > upper limit).")
    iterable.sort(key=lambda b: b[0] + b[1])  # Sort by central value
    if any(a[1] > b[0] for a, b in zip(iterable[:-1], iterable[1:])):
        raise RuntimeError("Overlapping bins found.")
    return iterable


def uniques(iterable):
    """Yields only uniques values out of a *sorted* iterable.

    :param iterable: A *sorted* iterable of objects that support the
       ``!=`` operator.

    Loops over ``iterable`` (which is assumed to be sorted) and yields
    its items only if they are unique (i.e. different from the previous,
    since they are sorted).

    .. warning::
       This function is a very simple and fast implementation for the
       specific case of sorted iterables.

       If ``iterable`` is not sorted, the same item may be yielded
       multiple times.
    """
    prev = None
    for item in iterable:
        if item != prev:
            yield item
        prev = item
