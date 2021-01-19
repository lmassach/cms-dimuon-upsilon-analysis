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
"""Core functions for the upsilon analysis."""
import argparse
import os
import logging
import array
import itertools
import math
import ROOT
from . import utils

__all__ = ["make_argument_parser", "build_dataframe", "book_histograms",
           "fit_histograms", "build_cross_section_hist",
           "build_cross_section_graph"]


def make_argument_parser():
    """Prepares an :class:`ArgumentParser` with some analysis arguments.

    :rtype: argparse.ArgumentParser

    Builds an :class`argparse.ArgumentParser` with all the appropriate
    command-line arguments for the analysis already added; these are:

    *  ``--input-file`` or ``-i``: optional, default input file is a
       ``root://`` link to CMS Open Data;
    *  ``--pt-min``: in GeV/c, optional, default 10;
    *  ``--pt-max``: in GeV/c, optional, default 100;
    *  ``--pt-bin-width``: in GeV/c, optional, default 2;
    *  ``--y-min``: in absolute value, optional, default 0;
    *  ``--y-max``: in absolute value, optional, default 1.2;
    *  ``--y-bins``: number of y bins, optional, default 2;
    *  ``--mass-bins``: number of mass bins, optional, default 100 (the
       range is fixed to 8.5-11.5 GeV/c^2);
    *  ``--no-quality``: suppress quality cuts;
    *  ``-v``: verbose mode, can be used for logging setup; default
       ``False``;
    *  ``--vv``: very verbose mode, see above; default ``False``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", "-i", metavar="PATH",
                        default=("root://eospublic.cern.ch//eos/opendata/cms"
                                 "/derived-data/AOD2NanoAODOutreachTool"
                                 "/Run2012BC_DoubleMuParked_Muons.root"),
                        help=("Input file to be used, optional; the default "
                              "file is opened from root://eospublic.cern.ch; "
                              "any URL supported by RDataFrame can be used."))
    parser.add_argument("--pt-min", type=float, default=10., metavar="MIN",
                        help="Minimum resonance pt (GeV/c); default 10.")
    parser.add_argument("--pt-max", type=float, default=100., metavar="MAX",
                        help="Maximum resonance pt (GeV/c); default 100.")
    parser.add_argument("--pt-bin-width", type=float, default=2., metavar="W",
                        help=("Width of the pt bins (GeV/c); above 40 GeV/c "
                              "bins will be made larger; default 2."))
    parser.add_argument("--y-min", type=float, default=0., metavar="MIN",
                        help=("Minimum resonance rapidity (absolute value); "
                              "default 0."))
    parser.add_argument("--y-max", type=float, default=1.2, metavar="MAX",
                        help=("Maximum resonance rapidity (absolute value); "
                              "default 1.2"))
    parser.add_argument("--y-bins", type=float, default=2, metavar="N",
                        help=("Number of resonance rapidity (absolute value) "
                              "bins; default 2."))
    parser.add_argument("--mass-bins", type=int, default=100, metavar="N",
                        help=("Number of invariant mass bins; default 100; "
                              "histograms with too few events are rebinned."))
    parser.add_argument("--no-quality", action="store_true",
                        help="Skip muon quality cuts.")
    parser.add_argument("-v", action="store_true", help="Verbose mode.")
    parser.add_argument("--vv", action="store_true", help="Very verbose mode.")
    return parser


@utils.static_variables(c_functions_defined=False)
def build_dataframe(input_file, no_quality=False, y_min=0, y_max=1.2,
                    pt_min=10, pt_max=100, **kwargs):
    """Opens the given file in an ``RDataFrame``.

    :param input_file: The file from which to take the Events `TTree`.
    :type input_file: str
    :param no_quality: Wether to skip muon quality cuts (see readme).
    :type no_quality: bool, optional
    :param y_min: Minimum abs(y) for the dimuon system.
    :type y_min: float, optional
    :param y_max: Maximum abs(y) for the dimuon system.
    :type y_max: float, optional
    :param pt_min: Minimum pt in GeV/c for the dimuon system.
    :type pt_min: float, optional
    :param pt_max: Maximum pt in GeV/c for the dimuon system.
    :type pt_max: float, optional
    :param \**kwargs: Only used to avoid errors when the result of
       :any:`make_argument_parser` is given.
    :rtype: ROOT.RDataFrame

    Makes a ``ROOT.RDataFrame`` with the given input, applies cuts
    common to all analysis tasks by calling ``Filter`` and returns the
    result. Nothing will be run as only lazy action are requested.

    The cuts performed (actually "booked") are:

    *  selection of only events with exactly two opposite charge muons;
    *  quality cuts (see readme) unless ``no_quality`` is ``True``;
    *  invariant mass cut between 8.5 and 11.5 GeV/c^2;
    *  dimuon system rapidity and transverse momentum cuts defined by
       ``y_min``, ``y_max``, ``pt_min``, ``pt_max``.

    In the process, the columns ``dimuon_mass``, ``dimuon_pt`` and
    ``dimuon_y`` are created and populated.

    This function uses ``ROOT.gInterpreter.Declare`` to define some
    global C++ functions, namely:

    *  ``float m12(RVec<float>, RVec<float>, RVec<float>, RVec<float>)``
    *  ``float pt12(RVec<float>, RVec<float>, RVec<float>, RVec<float>)``
    *  ``float y12(RVec<float>, RVec<float>, RVec<float>, RVec<float>)``

    If the function is run more than once, the C++ functions are *not*
    redefined.
    """
    # C++ functions to compute the invariant mass, the transverse
    # momentum and the rapidity of the dimuon system
    if not build_dataframe.c_functions_defined:
        build_dataframe.c_functions_defined = True
        ROOT.gInterpreter.Declare("""
        float m12(ROOT::VecOps::RVec<float> pt, ROOT::VecOps::RVec<float> eta,
                  ROOT::VecOps::RVec<float> phi, ROOT::VecOps::RVec<float> m){
            ROOT::Math::PtEtaPhiMVector p1(pt[0], eta[0], phi[0], m[0]);
            ROOT::Math::PtEtaPhiMVector p2(pt[1], eta[1], phi[1], m[1]);
            return (p1 + p2).M();
        }
        float pt12(ROOT::VecOps::RVec<float> pt, ROOT::VecOps::RVec<float> eta,
                   ROOT::VecOps::RVec<float> phi, ROOT::VecOps::RVec<float> m){
            ROOT::Math::PtEtaPhiMVector p1(pt[0], eta[0], phi[0], m[0]);
            ROOT::Math::PtEtaPhiMVector p2(pt[1], eta[1], phi[1], m[1]);
            return (p1 + p2).Pt();
        }
        float y12(ROOT::VecOps::RVec<float> pt, ROOT::VecOps::RVec<float> eta,
                  ROOT::VecOps::RVec<float> phi, ROOT::VecOps::RVec<float> m){
            ROOT::Math::PtEtaPhiMVector p1(pt[0], eta[0], phi[0], m[0]);
            ROOT::Math::PtEtaPhiMVector p2(pt[1], eta[1], phi[1], m[1]);
            return (p1 + p2).Rapidity();
        }
        """)

    # Lazy calls: here the analysis tasks are "booked", but non yet run
    logger = logging.getLogger(__name__)
    logger.info("Opening input file %s", input_file)
    df = ROOT.RDataFrame("Events", input_file)
    df = df.Filter("nMuon==2", "Two muons")
    df = df.Filter("Muon_charge[0]!=Muon_charge[1]", "Opposite charge")
    if not no_quality:
        df = df.Filter("abs(Muon_eta[0])<1.6&&abs(Muon_eta[1])<1.6"
                       "&&Muon_pt[0]>3&&Muon_pt[1]>3"
                       "&&(Muon_pt[0]>3.5||abs(Muon_eta[0])>1.4)"
                       "&&(Muon_pt[1]>3.5||abs(Muon_eta[1])>1.4)"
                       "&&(Muon_pt[0]>4.5||abs(Muon_eta[0])>1.2)"
                       "&&(Muon_pt[1]>4.5||abs(Muon_eta[1])>1.2)",
                       "Quality cuts")
    df = df.Define("dimuon_mass", "m12(Muon_pt,Muon_eta,Muon_phi,Muon_mass)")
    df = df.Filter("8.5<dimuon_mass&&dimuon_mass<11.5", "Mass limits")
    df = df.Define("dimuon_y", "y12(Muon_pt,Muon_eta,Muon_phi,Muon_mass)")
    df = df.Filter(f"{y_min}<=abs(dimuon_y)&&abs(dimuon_y)<{y_max}",
                   "Rapidity limits")
    df = df.Define("dimuon_pt", "pt12(Muon_pt,Muon_eta,Muon_phi,Muon_mass)")
    df = df.Filter(f"{pt_min}<dimuon_pt&&dimuon_pt<{pt_max}",
                   "pt limits")
    return df


def book_histograms(df, mass_bins=100, y_min=0, y_max=1.2, y_bins=2,
                    pt_min=10, pt_max=100, pt_bin_width=2, **kwargs):
    """Pepares mass histograms to be made by the ``RDataFrame``.

    :param df: The ``RDataFrame`` to work on.
    :type df: ROOT.RDataFrame
    :param mass_bins: The number of bins for the mass histograms.
    :type mass_bins: int, optional
    :param y_min: Minimum for abs(y) binning.
    :type y_min: float, optional
    :param y_max: Maximum for abs(y) binning.
    :type y_max: float, optional
    :param y_bins: Number of abs(y) bins.
    :type y_bins: int, optional
    :param pt_min: Minimum, in GeV/c, for pt binning.
    :type pt_min: float, optional
    :param pt_max: Maximum, in GeV/c, for pt binning.
    :type pt_max: float, optional
    :param pt_bin_width: Width, in GeV/c, of the pt bins (see
       :any:`utils.pt_bin_edges`)
    :type pt_bin_width: float, optional
    :param \**kwargs: Only used to avoid errors when the result of
       :any:`make_argument_parser` is given.
    :rtype: :class:`dict[tuple[float, float], dict[tuple[float, float],
       TH1D]]`

    Builds a dictionary whose keys are tuples ``(y_low, y_high)``, i.e.
    the edges of a rapidity bin, and the values are dictionaries whose
    keys are tuples ``(pt_low, pt_high)``, i.e. the edges of a
    transverse momentum bin, and the values are histograms of the
    invariant mass distribution within that rapidity and transverse
    momentum bin. In other words this function returns an object like

    .. code-block:: python

       {(y_low_1, y_high_1): {(pt_low_1, pt_high_1): TH1D, ...}, ...}

    The histograms are actually ``RResultPtr<TH1D>``, this means they
    are "booked" using the ``Histo1D`` method of the given
    ``RDataFrame df``, but are not actually built because that is a lazy
    action.

    The bins are chosen by the parameters; beside those, a pt
    bin that includes all pt values is present for each y bin, and a y
    bin that includes all y values is also present.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Booking histograms")
    histo_model = ("dimuon_mass",
                   "m_{#mu#mu};m_{#mu#mu} [GeV/c^{2}];Occurrences / bin",
                   mass_bins, 8.5, 11.5)
    edges = utils.y_bin_edges(y_min, y_max, y_bins)
    y_bins = {bin: {} for bin in utils.bins(edges)}
    y_bins[(y_min, y_max)] = {}  # Bin with all rapidities
    for (y_low, y_high), pt_bins in y_bins.items():
        y_filtered = df.Filter(
            f"{y_low}<=abs(dimuon_y)&&abs(dimuon_y)<{y_high}")
        edges = utils.pt_bin_edges(pt_min, pt_max, pt_bin_width)
        for pt_low, pt_high in utils.bins(edges):
            pt_filtered = y_filtered.Filter(
                f"{pt_low}<dimuon_pt&&dimuon_pt<{pt_high}")
            pt_bins[(pt_low, pt_high)] = pt_filtered.Histo1D(
                histo_model, "dimuon_mass")
        pt_bins[(pt_min, pt_max)] = y_filtered.Histo1D(
            histo_model, "dimuon_mass")
    return y_bins


def fit_histograms(histos, output_dir=None, vv=False, **kwargs):
    """Automatic mass histograms fitting.

    :param histos: Dictionary of dictionaries of histograms (see below).
    :type histos: dict
    :param output_dir: Output directory for the PDF with the histograms;
       if not given or ``None`` no PDF is produced.
    :type output_dir: str
    :param vv: Very verbose mode.
    :type vv: bool
    :param \**kwargs: Only used to avoid errors when the result of
       :any:`make_argument_parser` is given.
    :rtype: :class:`dict[tuple[float, float], dict[tuple[float, float],
       utils.FitResults]]`

    Fits the histograms ``histos`` returned by :any:`book_histograms`,
    saves the plots to "mass_histos.pdf" and returns the results of the
    fits in a dictionary of dictionaries similar to ``histos``.

    The fit function used is the sum of three Gaussian functions for the
    resonances plus a linear parametrization for the background.

    The results of each fit are returned as a :any:`utils.FitResults`.
    """
    logger = logging.getLogger(__name__)
    logger.info("Fitting histograms")
    # Fit function
    ff = ROOT.TF1("ff", "gaus(0)+gaus(3)+gaus(6)+pol1(9)", 8.5, 11.5)
    # Partial functions for plotting
    ga1 = ROOT.TF1("ga1", "gaus(0)", 8.5, 11.5)
    ga2 = ROOT.TF1("ga2", "gaus(0)", 8.5, 11.5)
    ga3 = ROOT.TF1("ga3", "gaus(0)", 8.5, 11.5)
    bkg = ROOT.TF1("bkg", "pol1(0)", 8.5, 11.5)

    results = {}
    if output_dir is not None:
        canvas = ROOT.TCanvas("", "", 640, 480)
        ROOT.gStyle.SetOptStat(10)
        ROOT.gStyle.SetOptFit(100)
        ROOT.gStyle.SetStatX(0.91)
        ROOT.gStyle.SetStatY(0.91)
        ROOT.gStyle.SetStatH(0.05)
        ROOT.gStyle.SetStatW(0.15)
        out_pdf = os.path.join(output_dir, "mass_histos.pdf")
        logger.info("Saving mass plots to %s", out_pdf)
        canvas.Print(f"{out_pdf}[")  # Open PDF to plot multiple pages
        ff.SetLineColor(ROOT.kBlack)
        ga1.SetLineColor(ROOT.kRed)
        ga2.SetLineColor(ROOT.kGreen + 3)
        ga3.SetLineColor(ROOT.kMagenta + 2)
        bkg.SetLineColor(ROOT.kBlue)

    for (y_low, y_high), pt_bins in histos.items():
        pt_results = {}
        for (pt_low, pt_high), histo in pt_bins.items():
            if output_dir is not None:
                histo.SetTitle(f"|y| #in [{y_low:g},{y_high:g}), "
                               f"p_{{T}} #in [{pt_low:g},{pt_high:g}) GeV/c")
                histo.SetMinimum(0)
                histo.Draw("E")
            # Initial parameters
            a = histo.GetMaximum()  # Y(1s) peak height estimate
            b = histo.GetBinContent(1)  # Bkg height estimate
            a -= b  # Corrected peak height estimate
            ff.SetParameters(a, 9.46, 0.1,
                             0.5 * a, 10.023, 0.1,  # Thumb rule here
                             0.35 * a, 10.355, 0.1,
                             b, 0)
            logger.info("Fitting histogram %s", histo.GetTitle())
            histo.Fit(ff, "B" + ("" if vv else "Q"))
            ga1.SetParameters(*(ff.GetParameter(i) for i in range(3)))
            ga2.SetParameters(*(ff.GetParameter(i) for i in range(3, 6)))
            ga3.SetParameters(*(ff.GetParameter(i) for i in range(6, 9)))
            bkg.SetParameters(ff.GetParameter(9), ff.GetParameter(10))
            if output_dir is not None:
                ga1.Draw("L SAME")
                ga2.Draw("L SAME")
                ga3.Draw("L SAME")
                bkg.Draw("L SAME")
                canvas.SetGrid()
                canvas.Print(out_pdf, f"Title:y ({y_low:g},{y_high:g}), "
                             f"pt ({pt_low:g},{pt_high:g})")
            pt_results[(pt_low, pt_high)] = utils.FitResults(
                utils.get_gaus_parameters(ga1, histo.GetBinWidth(1)),
                utils.get_gaus_parameters(ga2, histo.GetBinWidth(1)),
                utils.get_gaus_parameters(ga3, histo.GetBinWidth(1)),
                utils.LineParameters(bkg.GetParameter(0), bkg.GetParameter(1)),
                ff.GetChisquare(), ff.GetNDF()
            )
        results[(y_low, y_high)] = pt_results

    if output_dir is not None:
        canvas.Print(f"{out_pdf}]")  # Close PDF
    return results


def build_cross_section_hist(fit_results):
    """Builds a dσ/dpt vs pt histogram from a collection of fit results.

    :param fit_results: Dictionary of the occurrences in each bin.
    :type fit_results: :class:`dict[tuple[float, float], int]`
    :rtype: ROOT.TH1F

    The argument ``fit_results`` should be a dictionary whose keys are
    the pt bin limits, as a tuple, and whose values are the occurrences
    of the resonance in that pt bin. In other words, it should be like

    .. code-block:: python

       {(pt_low_1, pt_high_1): n_occ_1, ... }

    The results will be a ``TH1F`` whose bins are given by the keys of
    ``fit_results``, and whose bins' contents are the respective numbers
    of occurrences divided by the bin width. This will be proportional
    to dσ/dpt: to get the real value one needs to ``TH1F::Scale`` the
    histogram by ``1 / int_luminosity / efficiency / acceptance``.

    If the given bins overlap, a ``RuntimeError`` will be raised.

    If the bins are non-contiguos (i.e. there is a "hole"), a bin with
    zero content will be made (this is important if you intend to fit
    the histogram).
    """
    fit_bins = utils.sort_bins(fit_results.keys())
    # TH1 requires contiguos bins
    fit_bins_edges = utils.uniques(itertools.chain.from_iterable(fit_bins))
    hist_bins = list(utils.bins(fit_bins_edges))
    hist = ROOT.TH1F(
        "xsecbr",
        "Cross section;p_{T} [GeV/c];d#sigma/dp_{T}#times#it{Br} [arb. un.]",
        len(hits_bins), array.array('d', fit_bins_edges)
    )
    for bin, n in fit_results.items():
        bin_idx = hist_bins.index(bin) + 1
        delta_pt = bin[1] - bin[0]
        hist.SetBinContent(bin_idx, n / delta_pt)
        hist.SetBinError(bin_idx, math.sqrt(n) / delta_pt)
    return hist


def build_cross_section_graph(fit_results):
    """Builds a dσ/dpt vs pt graph from a collection of fit results.

    :param fit_results: Dictionary of the occurrences in each bin.
    :type fit_results: :class:`dict[tuple[float, float], int]`
    :rtype: ROOT.TGraphErrors

    The argument ``fit_results`` should be a dictionary whose keys are
    the pt bin limits, as a tuple, and whose values are the occurrences
    of the resonance in that pt bin. In other words, it should be like

    .. code-block:: python

       {(pt_low_1, pt_high_1): n_occ_1, ... }

    The results will be a ``TGraphErrors`` where the x values are
    defined by the centers of the pt bins, the x errors are defined by
    the pt bins' width, the y values are the numbers of occurrences
    divided by the pt bin width and the y error is estimated as the
    square root of the number of occurrences divided by the pt bin
    width. This way the y values will be proportional to dσ/dpt: to get
    the real value one needs to scale the graph (with a custom function)
    by ``1 / int_luminosity / efficiency / acceptance``.

    If the given bins overlap, a ``RuntimeError`` will be raised.
    """
    fit_bins = utils.sort_bins(fit_results.keys())
    graph = ROOT.TGraphErrors(
        len(fit_bins),
        array.array('d', ((b[0] + b[1]) / 2 for b in fit_bins)),
        array.array('d', (fit_results[b] / (b[1] - b[0]) for b in fit_bins)),
        array.array('d', ((b[1] - b[0]) / 2 for b in fit_bins)),
        array.array('d', (math.sqrt(fit_results[b]) / (b[1] - b[0])
                          for b in fit_bins))
    )
    graph.SetTitle("Cross section;p_{T} [GeV/c];"
                   "d#sigma/dp_{T}#times#it{Br} [arb. un.]")
    return graph
