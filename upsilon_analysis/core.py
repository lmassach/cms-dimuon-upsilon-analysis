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
import itertools
import math
import logging
import ROOT
from .utils import *

__all__ = ["parse_args", "make_args", "build_dataframe", "book_histograms",
           "fit_histograms"]


def parse_args(args=None):
    """Command-line arguments parsing function.

    Builds an ``argparse.ArgumentParser`` reading all the appropriate
    command-line arguments for the analysis, calls its ``parse_args``
    method with the given ``args`` (so it uses ``sys.argv`` as default)
    and returns its result.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", "-i", metavar="PATH",
                        default=("root://eospublic.cern.ch//eos/opendata/cms"
                                 "/derived-data/AOD2NanoAODOutreachTool"
                                 "/Run2012BC_DoubleMuParked_Muons.root"),
                        help=("Input file to be used, optional; the default "
                              "file is opened from root://eospublic.cern.ch; "
                              "any URL supported by RDataFrame can be used."))
    parser.add_argument("--threads", "-j", type=int, default=0, metavar="N",
                        help=("Number of threads, see ROOT::EnableImplicitMT; "
                              "chosen automatically by ROOT by default; if "
                              "set to 1, MT is not enabled at all."))
    parser.add_argument("--output-dir", "-o", default=".", metavar="DIR",
                        help="Output directory for the plots; default is cd.")
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
    return parser.parse_args(args)


def make_args(**kwargs):
    """Makes an object like the one returned by ``parse_args``.

    The object returned has all the required fields with their default
    values, unless the keyword arguments are used to change them. This
    allows to use a more Pythonic syntax to pass options to the core
    functions.
    """
    ns = Namespace(**vars(parse_args([])))
    ns.update(kwargs)
    return ns


@static_variables(c_functions_defined=False)
def build_dataframe(args):
    """
    Makes a ``ROOT.RDataFrame`` with the given input, applies cuts
    common to all analysis tasks by calling ``Filter`` and returns the
    result. Nothing will be run as only lazy action are requested.

    The cuts performed (actually "booked") are:

    *  selection of only events with exactly two opposite charge muons;
    *  quality cuts (see readme) unless ``args.no_quality`` is ``True``;
    *  invariant mass cut between 8.5 and 11.5 GeV/c^2;
    *  dimuon system rapidity and transverse momentum cuts defined by
       ``args.{pt|y}_{min|max}``.

    In the process, the columns ``dimuon_mass``, ``dimuon_pt`` and
    ``dimuon_y`` are created and populated.

    The argument ``args`` must be the result of ``parse_args`` or
    equivalent object; of its fields, these are used in this function:

    *  ``input_file``
    *  ``no_quality``
    *  ``y_min`` and ``y_max``
    *  ``pt_min`` and ``pt_max``

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
    logger.info("Opening input file %s", args.input_file)
    df = ROOT.RDataFrame("Events", args.input_file)
    df = df.Filter("nMuon==2", "Two muons")
    df = df.Filter("Muon_charge[0]!=Muon_charge[1]", "Opposite charge")
    if not args.no_quality:
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
    df = df.Filter(f"{args.y_min}<=abs(dimuon_y)&&abs(dimuon_y)<{args.y_max}",
                   "Rapidity limits")
    df = df.Define("dimuon_pt", "pt12(Muon_pt,Muon_eta,Muon_phi,Muon_mass)")
    df = df.Filter(f"{args.pt_min}<dimuon_pt&&dimuon_pt<{args.pt_max}",
                   "pt limits")
    return df


def book_histograms(df, args):
    """
    Builds a dictionary whose keys are tuples ``(y_low, y_high)``, i.e.
    the edges of a rapidity bin, and the values are dictionaries whose
    keys are tuples ``(pt_low, pt_high)``, i.e. the edges of a
    transverse momentum bin, and the values are histograms of the
    invariant mass distribution within that rapidity and transverse
    momentum bin. In other words this function returns an object like
    ``{(y_low_1, y_high_1): {(pt_low_1, pt_high_1): TH1D, ...}, ...}``.

    The histograms are actually ``RResultPtr<TH1D>``, this means they
    are "booked" using the ``Histo1D`` method of the given
    ``RDataFrame df``, but are not actually built because that is a lazy
    action.

    The argument ``args`` must be the result of ``parse_args`` or
    equivalent object; of its fields, these are used in this function:

    *  ``mass_bins``
    *  ``y_min``, ``y_max`` and ``y_bins``
    *  ``pt_min``, ``pt_max`` and ``pt_bin_width``

    The bins are chosen by the fields of ``args``; beside those, a pt
    bin that includes all pt values is present for each y bin, and a y
    bin that includes all y values is also present.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Booking histograms")
    histo_model = ("dimuon_mass",
                   "m_{#mu#mu};m_{#mu#mu} [GeV/c^{2}];Occurrences / bin",
                   args.mass_bins, 8.5, 11.5)
    edges = y_bin_edges(args.y_min, args.y_max, args.y_bins)
    y_bins = {bin: {} for bin in bins(edges)}
    y_bins[(args.y_min, args.y_max)] = {}  # Bin with all rapidities
    for (y_low, y_high), pt_bins in y_bins.items():
        y_filtered = df.Filter(f"{y_low}<=abs(dimuon_y)&&"
                               f"abs(dimuon_y)<{y_high}")
        edges = pt_bin_edges(args.pt_min, args.pt_max, args.pt_bin_width)
        for pt_low, pt_high in bins(edges):
            pt_filtered = y_filtered.Filter(f"{pt_low}<dimuon_pt&&"
                                            f"dimuon_pt<{pt_high}")
            pt_bins[(pt_low, pt_high)] = pt_filtered.Histo1D(histo_model,
                                                             "dimuon_mass")
        pt_bins[(args.pt_min, args.pt_max)] = y_filtered.Histo1D(histo_model,
                                                                 "dimuon_mass")
    return y_bins


def fit_histograms(histos, args):
    """
    Fits the histograms ``histos`` returned by ``book_histograms``,
    saves the plots to "mass_histos.pdf" and returns the results of the
    fits in a dictionary of dictionaries similar to ``histos``.

    The fit function used is the sum of three Gaussian functions for the
    resonances plus a linear parametrization for the background.

    The results of each fit are returned as a ``FitResults`` named
    tuple, which has six fields:

    *  ``y1``, ``y2`` and ``y3`` are ``GausParameters`` named tuples,
       whose fields are:

       *  ``a`` the factor multiplying the normalized distribution,
          a.k.a. number of occurrences of the resonance;
       *  ``m`` the mean of the resonance, a.k.a. pole mass;
       *  ``sigma`` the resonance width parameter;

    *  ``bkg`` is a ``LineParameters`` named tuple, whose fields are:

       *  ``q`` the vertical intercept of the line;
       *  ``m`` the slope of the line;

    *  ``chi2`` and ``ndf`` are the chi squared and the number of
       degrees of freedom, for goodness of fit tests.

    The argument ``args`` must be the result of ``parse_args`` or
    equivalent object; of its fields only ``output_dir`` and `vv` are
    used.
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
    canvas = ROOT.TCanvas("", "", 640, 480)
    ROOT.gStyle.SetOptStat(10)
    ROOT.gStyle.SetOptFit(100)
    ROOT.gStyle.SetStatX(0.91)
    ROOT.gStyle.SetStatY(0.91)
    ROOT.gStyle.SetStatH(0.05)
    ROOT.gStyle.SetStatW(0.15)
    out_pdf = os.path.join(args.output_dir, "mass_histos.pdf")
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
            histo.SetTitle(f"y #in [{y_low:g},{y_high:g}), "
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
            histo.Fit(ff, "B" + ("" if args.vv else "Q"))
            ga1.SetParameters(*(ff.GetParameter(i) for i in range(3)))
            ga2.SetParameters(*(ff.GetParameter(i) for i in range(3, 6)))
            ga3.SetParameters(*(ff.GetParameter(i) for i in range(6, 9)))
            bkg.SetParameters(ff.GetParameter(9), ff.GetParameter(10))
            ga1.Draw("L SAME")
            ga2.Draw("L SAME")
            ga3.Draw("L SAME")
            bkg.Draw("L SAME")
            canvas.SetGrid()
            canvas.Print(out_pdf, f"Title:y ({y_low:g},{y_high:g}), "
                         f"pt ({pt_low:g},{pt_high:g})")
            pt_results[(pt_low, pt_high)] = FitResults(
                get_gaus_parameters(ga1, histo.GetBinWidth(1)),
                get_gaus_parameters(ga2, histo.GetBinWidth(1)),
                get_gaus_parameters(ga3, histo.GetBinWidth(1)),
                LineParameters(bkg.GetParameter(0), bkg.GetParameter(1)),
                ff.GetChisquare(), ff.GetNDF()
            )
        results[(y_low, y_high)] = pt_results

    canvas.Print(f"{out_pdf}]")  # Close PDF
    return results