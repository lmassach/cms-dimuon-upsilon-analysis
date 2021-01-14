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
"""Module for executing the analysis from the command line."""
import logging
import os
import ROOT
from . import *
from .utils import print_fit_results


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=(logging.DEBUG if args.vv else
                               (logging.INFO if args.v else logging.WARNING)))
    if args.threads != 1:
        logging.debug("Enabling MT")
        ROOT.EnableImplicitMT(args.threads)
    df = build_dataframe(args)
    mass_histos = book_histograms(df, args)
    logging.info("Actually running the analysis with the RDataFrame")
    df.Report().Print()  # Here all the booked actions are actually run

    if not os.path.isdir(args.output_dir):
        logging.info("Creating output directory %s", args.output_dir)
        os.mkdir(args.output_dir)
    fits = fit_histograms(mass_histos, args)
    # Save fit results to a CSV file, for importing use TTree::ReadFile
    out_csv = os.path.join(args.output_dir, "fit_results.csv")
    logging.info("Saving fit results to %s", out_csv)
    with open(out_csv, "w") as ofs:
        print_fit_results(fits, file=ofs)


    # TODO take into account failing fits when peaks are too small
    # TODO remember to keep into account that the fit function returns the
    # number of occurrences of resonances N, but we need N/ΔptΔy
