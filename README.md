# cms-dimuon-upsilon-analysis
Analysis of the di-muon events from CMS open data aimed at resolving and
fitting the upsilon resonances' masses as a function of their transverse
momentum.

The software is distributed under the GNU GPLv3 license and requires Python 3
and PyROOT.

This analysis is based on
[this](https://twiki.cern.ch/twiki/bin/view/CMSPublic/PhysicsResultsBPH12006)
article ([arXiv](https://arxiv.org/abs/1501.07750)) and is originally intended
to be used with [this](http://opendata.web.cern.ch/record/12341) dataset.

## Event selection
The events in the dataset are selected by applying the following cuts:
 - the event must have exactly two opposite-charge muons;
 - each muon must must pass the quality cuts:
    - eta must be less than 1.6, pt more than 3 GeV;
    - if eta is less than 1.4, pt must be more than 3.5 GeV;
    - if eta is less than 1.2, pt must be more than 4.5 GeV;
 - the invariant mass of the dimuon system must be between 8.5 and 11.5 GeV;
 - the rapidity of the dimuon system must be less than 1.2;
 - the pt of the dimuon system must be between 10 and 100 GeV;

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
each resonance is computed from the gaussians' parameters.

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
