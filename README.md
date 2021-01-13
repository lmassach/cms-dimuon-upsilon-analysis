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
