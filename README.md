# cms-dimuon-upsilon-analysis
[![Documentation Status](https://readthedocs.org/projects/cms-dimuon-upsilon-analysis/badge/?version=latest)](https://cms-dimuon-upsilon-analysis.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/lmassach/cms-dimuon-upsilon-analysis.svg?branch=main)](https://travis-ci.com/lmassach/cms-dimuon-upsilon-analysis)
<!--- Note: docs/readme.rst includes this markdown file from the fifht line on, so the title and the badges (and also this comment) are conveniently excluded. -->

Analysis of the di-muon events from CMS open data aimed at resolving and
fitting the upsilon resonances' masses as a function of their transverse
momentum.

The software is distributed under the GNU GPLv3 license and requires Python 3
and PyROOT.

This analysis is based on
[this](https://twiki.cern.ch/twiki/bin/view/CMSPublic/PhysicsResultsBPH12006)
article ([arXiv](https://arxiv.org/abs/1501.07750)) and is originally intended
to be used with [this](http://opendata.web.cern.ch/record/12341) dataset.

## Usage
The simplest way is to run the package. Command-line options can be listed by
```bash
python3 -m upsilon_analysis --help
```

## Event selection
The events in the dataset are selected by applying the following cuts:
 - the event must have exactly two opposite-charge muons;
 - each muon must must pass the quality cuts (can be skipped):
    - eta must be less than 1.6, pt more than 3 GeV;
    - if eta is less than 1.4, pt must be more than 3.5 GeV;
    - if eta is less than 1.2, pt must be more than 4.5 GeV;
 - the invariant mass of the dimuon system must be between 8.5 and 11.5 GeV;
 - the rapidity of the dimuon system must be less than 1.2 (configurable);
 - the pt of the dimuon system must be between 10 and 100 GeV (configurable);

## Binning
The selected events are binned by the resonance's rapidity y; y limits and
number of bins can be selected via command-line arguments. An extra bin
including all the events within y limits is always produced.

Within each y bin, the events are binned by the resonance's transverse momentum
pt; again, limits and bin width can be selected via command-line arguments. An
extra bin including all the events within pt limits is always produced. At
higher pt the bins are automatically grown wider to account for the smaller
number of occurrences (see the documentation of `pt_bin_edges()` for details).

## Fitting
In the end a 2D histogram in y and pt is made; for each of its bins an
histogram of the invariant mass distribution is made (number of bins can be
set via command-line arguments; range is fixed to 8.5-11.5 GeV). This
distribution is fitted with the sum of three gaussians (for the resonances)
plus a 1st order polynomial (for the background). The number of occurrences of
each resonance is computed from the gaussians' parameters (see
`get_gaus_parameters()`).

Since the pole masses of the upsilon resonances are known, their fitted masses
can be used as goodness-of-fit statistics besides chi-squared / degrees of
freedom. It is expected that, in some bins, the resonances will be impossible
to resolve due to the lower signal to noise ratio (particularly at high pt).

The histograms and the fitted functions are plotted and saved in
`mass_histos.pdf` in the output directory. The results of the fits are saved in
the `fit_results.csv` file. The easiest way to reopen it in ROOT is
```C++
TTree fitResults;
fitResults.ReadFile("fit_results.csv");
```
ROOT will deduce the column names from the first line, all columns will be
floats (which is the intended result).

If you use the API yourself, saving the PDF is optional, and saving the CSV has
to be done manually (see `print_fit_results()`).

## Differential cross sections
For each y bin the differential cross section dσ/dpt (times the branching ratio
to μμ) is computed from the fitted numbers of occurrences of each resonance. A
graph of dσ/dpt vs pt is produced by using the pt bins: the number of
occurrences in each bin is divided by the width of the bin; this quantity is
proportional to dσ/dpt, and the real value can be obtained dividing by the
integrated luminosity times the efficiency time the acceptance.

The functions `build_cross_section_hist()` and `build_cross_section_graph()`
actually allow to specify the integrated luminosity and the
efficiency/acceptance (the latter either globally or per y-pt bin), and there
are command-line arguments for exploting this feature (see below).

Not all the fitted values are used: if the fitted mass of a resonance is
different from the known mass (from PDG) by more than a certain delta the fit
is discarded (for all the resonances, as it might alter the background
estimate). The command-line argument `--max-mass-delta` can be used to control
this behavior (see command help for default value).

Also a consistency check is performed: the fitted number of resonance events
must be non-negative for the three resonances, and the sum must be less than
the number of events. This helps excluding bad fits.

The graphs are plotted and saved to `cross_section_plots.pdf`.

Note: this part of the analysis is implemented in the `__main__.py`, so it is
not a part of the API (but can be used as example).

### Efficiency
While the integrated luminosity is a constant, the efficiency/acceptance may
depend on pt. To take this into account, the command-line argument
`--efficiency-table` is provided, which allows to specify a CSV file with the
following columns:
 - `y_min` and `y_max`: the limits of the y bin in which the efficiency is
   computed;
 - `pt_min` and `pt_max`: the limits of the pt bin in which the efficiency is
   computed;
 - `eff_y1`, `eff_y2` and `eff_y3`: the computed efficiencies for muon pairs
   with the resonances' invariant mass.

The bins in the CSV must, of course, match those of the fits. Roundoff errors
might need to be taken into account as the bins are matched by the `==`
operator.

If this infomation is provided, the dσ/dpt in each bin is divided by the
efficiency given for that bin, thus allowing to compute an unbiased estimate of
the dσ/dpt vs pt function, or the real dσ/dpt vs pt if the luminosity is also
provided via the `--luminosity` command line argument.

## Testing
The easiest way to run the tests is to use
```bash
python3 -m tests
```

For testing some of the core functions (`build_dataframe`, `book_histograms`) a
small dataset (47 KiB) is used. It is in the tests folder, and consists of the
first 1000 events of [this one](http://opendata.web.cern.ch/record/12341).

Also `pytest` is supported.
