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
"""Example analysis using the upsilon_analysis package."""
import logging
import argparse
import os
import ROOT
from . import *
from .utils import print_fit_results


def args_for(func, kwargs):
    """Strips ``kwargs`` of the names that func does not require."""
    return {k: v for k, v in kwargs.items() if k in func.__code__.co_varnames}


if __name__ == "__main__":
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
    parser.add_argument("--threads", "-j", type=int, default=0, metavar="N",
                        help=("Number of threads, see ROOT::EnableImplicitMT; "
                              "chosen automatically by ROOT by default; if "
                              "set to 1, MT is not enabled at all."))
    parser.add_argument("--output-dir", "-o", default=".", metavar="DIR",
                        help="Output directory for the plots; default is cd.")
    parser.add_argument("--max-mass-delta", type=float, default=0.025,
                        help=("Max delta (in GeV) between fitted and known "
                              "resonance mass to consider the fit good; "
                              "default 0.025; known masses are from PDG."))
    args = parser.parse_args()
    kwargs = vars(args)
    logging.basicConfig(level=(logging.DEBUG if args.vv else
                               (logging.INFO if args.v else logging.WARNING)))

    # Data reading, reconstruction, cuts, mass histograms
    if args.threads != 1:
        logging.debug("Enabling MT")
        ROOT.EnableImplicitMT(args.threads)
    df = build_dataframe(**args_for(build_dataframe, kwargs))
    mass_histos = book_histograms(df, **args_for(book_histograms, kwargs))
    logging.info("Actually running the analysis with the RDataFrame")
    df.Report().Print()  # Here all the booked actions are actually run

    # Fitting, plotting, saving to disk plots and fit results
    if not os.path.isdir(args.output_dir):
        logging.info("Creating output directory %s", args.output_dir)
        os.mkdir(args.output_dir)
    fits = fit_histograms(mass_histos, **args_for(fit_histograms, kwargs))
    # Save fit results to a CSV file, for importing use TTree::ReadFile
    out_csv = os.path.join(args.output_dir, "fit_results.csv")
    logging.info("Saving fit results to %s", out_csv)
    with open(out_csv, "w") as ofs:
        print_fit_results(fits, file=ofs)

    # Differential cross sections
    canvas = ROOT.TCanvas("", "", 640, 480)
    ROOT.gStyle.SetOptStat(10)
    ROOT.gStyle.SetOptFit(100)
    ROOT.gStyle.SetStatX(0.91)
    ROOT.gStyle.SetStatY(0.91)
    ROOT.gStyle.SetStatH(0.05)
    ROOT.gStyle.SetStatW(0.15)
    out_pdf = os.path.join(args.output_dir, "cross_section_plots.pdf")
    logging.info("Saving xsec plots to %s", out_pdf)
    canvas.Print(f"{out_pdf}[")  # Open PDF to plot multiple pages
    for (y_low, y_high), pt_bins in fits.items():
        # Cut the bin with all the data and those with bad fits
        del pt_bins[(args.pt_min, args.pt_max)]
        ok_bins = {k: (v.y1.a, v.y2.a, v.y3.a) for k, v in pt_bins.items()
                   if abs(v.y1.m - 9.4603) <= args.max_mass_delta
                   and abs(v.y2.m - 10.0232) <= args.max_mass_delta
                   and abs(v.y3.m - 10.3552) <= args.max_mass_delta}
        logging.debug("Discarded %d bins due to wrong fitted mass(es) "
                      "(xsec plots %g<y<%g)", len(pt_bins) - len(ok_bins),
                      y_low, y_high)
        if len(ok_bins) < 2:
            logging.warning("Not enough pt bins for xsec plots (%g<y<%g)",
                            y_low, y_high)
            continue
        # Graphs for the three Ys' cross sections
        graphs = [
            build_cross_section_graph({k: v[n] for k, v in ok_bins.items()})
            for n in range(3)
        ]
        for n, graph in enumerate(graphs):
            graph.SetTitle(f"#Upsilon({n+1}s), |y| #in [{y_low:g},{y_high:g})")
            graph.SetLineWidth(2)
            graph.Draw("APZ")
            canvas.SetGrid()
            canvas.SetLogy()
            canvas.Print(out_pdf, f"Title:Y({n+1}s), y ({y_low:g},{y_high:g})")
    canvas.Print(f"{out_pdf}]")  # Close PDF

    del canvas
    # TODO use acceptance tables from article, adding a command-line
    # argument for loading that table from a csv
