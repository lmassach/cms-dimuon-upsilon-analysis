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
"""Unit tests for ``upsilon_analysis.core``."""
import unittest
import numbers
import os
import upsilon_analysis.core as core

__all__ = ["TestUpsilonAnalysisCore"]


class TestUpsilonAnalysisCore(unittest.TestCase):
    """Test case for ``upsilon_analysis.core``."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = os.path.dirname(os.path.abspath(__file__))
        self._test_data = os.path.join(self._path, "test_data.root")

    def assertDictsAlmostEqual(self, a, b, places=7):
        """
        Checks that a and b have the same keys and that the respective
        values are almost equal.
        """
        self.assertEqual(len(a), len(b))
        for k in a.keys():
            self.assertAlmostEqual(a[k], b[k], places=places)

    def test_parse_args(self):
        """Tests for function ``parse_args``."""
        args = core.parse_args([])
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
        """Tests for function ``build_dataframe`` with test dataset."""
        print(f"Using test data {self._test_data}")
        df = core.build_dataframe(self._test_data)
        report = df.Report().GetValue()
        infos = {ci.GetName(): ci.GetEff() for ci in report}
        expected = {'Two muons': 55, 'Opposite charge': 75,
                    'Quality cuts': 68, 'Mass limits': 4,
                    'Rapidity limits': 83, 'pt limits': 100}
        self.assertDictsAlmostEqual(infos, expected, places=0)

    def test_book_histograms(self):
        """Tests for function ``book_histograms``. with test dataset."""
        df = core.build_dataframe(self._test_data)
        histos = core.book_histograms(df)
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
