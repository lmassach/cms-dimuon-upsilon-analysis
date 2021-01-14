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
"""Unit tests for ``upsilon_analysis.utils``."""
import unittest
import ROOT
import upsilon_analysis.utils as utils

__all__ = ["TestUpsilonAnalysisUtils"]


class TestUpsilonAnalysisUtils(unittest.TestCase):
    """Test case for ``upsilon_analysis.utils``."""
    def assertSequencesAlmostEqual(self, a, b):
        """
        Checks that a and b have the same length and that their items
        are almost equal.
        """
        self.assertEqual(len(a), len(b))
        for ia, ib in zip(a, b):
            self.assertAlmostEqual(ia, ib)

    def test_y_bin_edges(self):
        """Tests for function ``y_bin_edges``."""
        edges = [x for x in utils.y_bin_edges(0, 1.2, 2)]
        self.assertSequencesAlmostEqual(edges, [0, 0.6, 1.2])
        edges = [x for x in utils.y_bin_edges(0, 1.2, 3)]
        self.assertSequencesAlmostEqual(edges, [0, 0.4, 0.8, 1.2])
        edges = [x for x in utils.y_bin_edges(0, 1.2, 1)]
        self.assertSequencesAlmostEqual(edges, [0, 1.2])

    def test_pt_bin_edges(self):
        """Tests for function ``pt_bin_edges``."""
        edges = [x for x in utils.pt_bin_edges(10, 100, 2)]
        self.assertSequencesAlmostEqual(edges, [10, 12, 14, 16, 18, 20, 22, 24,
                                                26, 28, 30, 32, 34, 36, 38, 40,
                                                43, 46, 50, 55, 60, 70, 100])
        edges = [x for x in utils.pt_bin_edges(10, 100, 10)]
        self.assertSequencesAlmostEqual(edges, [10, 20, 30, 40, 55, 100])

    def test_bins(self):
        """Tests for function ``bins``."""
        self.assertEqual([x for x in utils.bins([0, 1, 2])],
                         [(0, 1), (1, 2)])
        self.assertEqual([x for x in utils.bins([0, 1])], [(0, 1)])
        self.assertEqual([x for x in utils.bins([0])], [])
        self.assertEqual([x for x in utils.bins(range(3))],
                         [(0, 1), (1, 2)])
        self.assertEqual([x for x in utils.bins(range(2))],
                         [(0, 1)])
        self.assertEqual([x for x in utils.bins(range(1))], [])

    def test_get_gaus_parameters(self):
        """Tests for the function ``get_gaus_parameters``."""
        ga = ROOT.TF1("testf", "gaus(0)", -6, 6)
        hist = ROOT.TH1F("testh", "testh", 36, -6, 6)
        bw = hist.GetBinWidth(1)
        n = 0
        for i in range(4):
            n += 5000
            hist.FillRandom("gaus", 5000)
            # Without I
            ga.SetParameters(1, 0, 1)
            hist.Fit(ga, "QBN")
            res = utils.get_gaus_parameters(ga, bw, range=(-6, 6))
            self.assertAlmostEqual(res.a / n, 1, delta=0.05)
            # With I
            ga.SetParameters(1, 0, 1)
            hist.Fit(ga, "QBNI")
            res = utils.get_gaus_parameters(ga, bw, range=(-6, 6))
            self.assertAlmostEqual(res.a / n, 1, delta=0.05)
