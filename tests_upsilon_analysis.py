#!/usr/bin/env python3
# Copyright (C) 2020 Ludovico Massaccesi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for `upsilon_analysis.py`."""
import unittest
import numbers
import os
import ROOT
import upsilon_analysis


class TestUpsilonAnalysis(unittest.TestCase):
    """Test case for `upsilon_analysis.py`."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = os.path.dirname(os.path.abspath(__file__))
        self._test_data = os.path.join(self._path,
                                       "tests_upsilon_analysis_data.root")

    def assertSequencesAlmostEqual(self, a, b):
        """
        Checks that a and b have the same length and that their items
        are almost equal.
        """
        self.assertEqual(len(a), len(b))
        for ia, ib in zip(a, b):
            self.assertAlmostEqual(ia, ib)

    def assertDictsAlmostEqual(self, a, b, places=7):
        """
        Checks that a and b have the same keys and that the respective
        values are almost equal.
        """
        self.assertEqual(len(a), len(b))
        for k in a.keys():
            self.assertAlmostEqual(a[k], b[k], places=places)

    def test_y_bin_edges(self):
        """Tests for function `y_bin_edges`."""
        edges = [x for x in upsilon_analysis.y_bin_edges(0, 1.2, 2)]
        self.assertSequencesAlmostEqual(edges, [0, 0.6, 1.2])
        edges = [x for x in upsilon_analysis.y_bin_edges(0, 1.2, 3)]
        self.assertSequencesAlmostEqual(edges, [0, 0.4, 0.8, 1.2])
        edges = [x for x in upsilon_analysis.y_bin_edges(0, 1.2, 1)]
        self.assertSequencesAlmostEqual(edges, [0, 1.2])

    def test_pt_bin_edges(self):
        """Tests for function `pt_bin_edges`."""
        edges = [x for x in upsilon_analysis.pt_bin_edges(10, 100, 2)]
        self.assertSequencesAlmostEqual(edges, [10, 12, 14, 16, 18, 20, 22, 24,
                                                26, 28, 30, 32, 34, 36, 38, 40,
                                                43, 46, 50, 55, 60, 70, 100])
        edges = [x for x in upsilon_analysis.pt_bin_edges(10, 100, 10)]
        self.assertSequencesAlmostEqual(edges, [10, 20, 30, 40, 55, 100])

    def test_bins(self):
        """Tests for function `bins`."""
        self.assertEqual([x for x in upsilon_analysis.bins([0, 1, 2])],
                         [(0, 1), (1, 2)])
        self.assertEqual([x for x in upsilon_analysis.bins([0, 1])], [(0, 1)])
        self.assertEqual([x for x in upsilon_analysis.bins([0])], [])
        self.assertEqual([x for x in upsilon_analysis.bins(range(3))],
                         [(0, 1), (1, 2)])
        self.assertEqual([x for x in upsilon_analysis.bins(range(2))],
                         [(0, 1)])
        self.assertEqual([x for x in upsilon_analysis.bins(range(1))], [])

    def test_parse_args(self):
        """Tests for function `parse_args`."""
        args = upsilon_analysis.parse_args([])
        self.assertIsInstance(args.input_file, str)
        self.assertIsInstance(args.threads, numbers.Integral)
        self.assertIsInstance(args.output_dir, str)
        self.assertIsInstance(args.pt_min, numbers.Real)
        self.assertIsInstance(args.pt_max, numbers.Real)
        self.assertIsInstance(args.pt_bin_width, numbers.Real)
        self.assertIsInstance(args.y_min, numbers.Real)
        self.assertIsInstance(args.y_max, numbers.Real)
        self.assertIsInstance(args.y_bins, numbers.Integral)
        self.assertIsInstance(args.mass_bins, numbers.Integral)
        self.assertIsInstance(args.no_quality, bool)
        self.assertIsInstance(args.v, bool)
        self.assertIsInstance(args.vv, bool)

    def test_build_dataframe(self):
        """Tests for function `build_dataframe` with example dataset."""
        print(f"Using test data {self._test_data}")
        args = upsilon_analysis.parse_args(['-i', self._test_data])
        df = upsilon_analysis.build_dataframe(args)
        report = df.Report().GetValue()
        infos = {ci.GetName(): ci.GetEff() for ci in report}
        expected = {'Two muons': 55, 'Opposite charge': 75,
                    'Quality cuts': 68, 'Mass limits': 4,
                    'Rapidity limits': 83, 'pt limits': 100}
        self.assertDictsAlmostEqual(infos, expected, places=0)

    def test_book_histograms(self):
        """Tests for function `book_histograms`."""
        args = upsilon_analysis.parse_args(['-i', self._test_data])
        df = upsilon_analysis.build_dataframe(args)
        histos = upsilon_analysis.book_histograms(df, args)
        self.assertCountEqual(list(histos.keys()),
                              [(0, 0.6), (0.6, 1.2), (0, 1.2)])
        for v in histos.values():
            self.assertEqual(list(v.keys()),
                             [(10, 12), (12, 14), (14, 16), (16, 18),
                              (18, 20), (20, 22), (22, 24), (24, 26),
                              (26, 28), (28, 30), (30, 32), (32, 34),
                              (34, 36), (36, 38), (38, 40), (40, 43),
                              (43, 46), (46, 50), (50, 55), (55, 60),
                              (60, 70), (70, 100), (10, 100)])

    def test_get_ga_parameters(self):
        """Tests for the function `get_ga_parameters`."""
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
            res = upsilon_analysis.get_ga_parameters(ga, bw, range=(-6, 6))
            self.assertAlmostEqual(res.a / n, 1, delta=0.05)
            # With I
            ga.SetParameters(1, 0, 1)
            hist.Fit(ga, "QBNI")
            res = upsilon_analysis.get_ga_parameters(ga, bw, range=(-6, 6))
            self.assertAlmostEqual(res.a / n, 1, delta=0.05)


if __name__ == "__main__":
    unittest.main()
