"""Analytic tests for the frozen field gate.  Run with pytest or unittest.

CPU only: no GPU, no C++ build, no dataset.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import field_gate as fg  # noqa: E402


def uniform_tile(value=0.1):
    tile = np.zeros((256, 6), dtype=np.float64)
    tile[:, 0] = value  # Ex at every staggered point
    return tile


class S0DefinitionTest(unittest.TestCase):
    def test_uniform_field_has_s0_equal_to_the_norm(self):
        # S0_G = sqrt(256) * max_point ||G_point||; for a uniform E=(0.1,0,0)
        # this is 16 * 0.1 = 1.6 and ||A_ref|| = sqrt(256*0.1^2) = 1.6.
        s0 = fg.reference_tile_s0(uniform_tile())
        self.assertAlmostEqual(s0["E"], 1.6, places=15)
        self.assertEqual(s0["cB"], 0.0)

    def test_peak_field_uses_the_max_cell_norm(self):
        tile = np.zeros((256, 6), dtype=np.float64)
        tile[17, 1] = 0.5  # Ey only at one output point
        s0 = fg.reference_tile_s0(tile)
        self.assertAlmostEqual(s0["E"], 16.0 * 0.5, places=15)

    def test_multi_component_point_uses_vector_norm(self):
        # One point with E=(0.6, 0.8, 0): the point norm is 1.0, so
        # S0 = 16 * 1.0.  A "max over components" reading would yield
        # 16 * 0.8 = 12.8 instead.  This pins the definition.
        tile = np.zeros((256, 6), dtype=np.float64)
        tile[0, 0] = 0.6
        tile[0, 1] = 0.8
        s0 = fg.reference_tile_s0(tile)
        self.assertAlmostEqual(s0["E"], 16.0 * 1.0, places=15)
        self.assertNotAlmostEqual(s0["E"], 16.0 * 0.8, places=15)

    def test_groups_are_independent(self):
        tile = np.zeros((256, 6), dtype=np.float64)
        tile[:, 3] = 0.25  # cBx everywhere, E stays zero
        s0 = fg.reference_tile_s0(tile)
        self.assertEqual(s0["E"], 0.0)
        self.assertAlmostEqual(s0["cB"], 16.0 * 0.25, places=15)


class GateBoundaryTest(unittest.TestCase):
    """The guide uses <= 1e-3; equality must pass."""

    def _run(self, candidate, reference, config):
        return fg.evaluate(candidate, reference, config, {})

    def test_uniform_1_001_factor_passes_at_the_boundary(self):
        ref = uniform_tile(0.1)
        cand = ref * 1.001
        report = self._run(
            cand, ref, {"gate": 1e-3, "s0": {"mode": "reference_tile"}}
        )
        e = report["cases"][0]["E"]
        self.assertAlmostEqual(e["relative_l2"], 1e-3, places=15)
        self.assertTrue(e["pass"])
        # denominator is the reference norm here: S0 == ||A_ref|| for uniform data
        self.assertAlmostEqual(e["denominator"], 1.6, places=15)

    def test_just_above_the_boundary_fails(self):
        ref = uniform_tile(0.1)
        cand = ref * 1.001001
        report = self._run(
            cand, ref, {"gate": 1e-3, "s0": {"mode": "reference_tile"}}
        )
        e = report["cases"][0]["E"]
        self.assertGreater(e["relative_l2"], 1e-3)
        self.assertFalse(e["pass"])

    def test_exact_equality_passes_via_the_s0_floor(self):
        # Zero reference, S0 = 1.0, one candidate component = 1e-3 exactly,
        # so the ratio is bit-exactly the gate value and <= must accept it.
        ref = np.zeros((256, 6), dtype=np.float64)
        cand = np.zeros_like(ref)
        cand[0, 0] = 1e-3
        config = {"gate": 1e-3, "s0": {"mode": "constant", "E": 1.0, "cB": 1.0}}
        e = self._run(cand, ref, config)["cases"][0]["E"]
        self.assertEqual(e["relative_l2"], 1e-3)
        self.assertTrue(e["pass"])
        self.assertTrue(e["absolute_normalized"])

    def test_one_ulp_above_equality_fails(self):
        ref = np.zeros((256, 6), dtype=np.float64)
        cand = np.zeros_like(ref)
        cand[0, 0] = np.nextafter(np.float64(1e-3), np.float64(np.inf))
        config = {"gate": 1e-3, "s0": {"mode": "constant", "E": 1.0, "cB": 1.0}}
        e = self._run(cand, ref, config)["cases"][0]["E"]
        self.assertGreater(e["relative_l2"], 1e-3)
        self.assertFalse(e["pass"])

    def test_s0_floor_dominates_a_weak_field(self):
        # ref has norm 1.0 but S0 = 16 * 1.0 = 16.0; err = 0.016 gives 1e-3.
        ref = np.zeros((256, 6), dtype=np.float64)
        ref[0, 0] = 1.0
        cand = ref.copy()
        cand[0, 0] += 0.016 * (1.0 - 1e-12)
        e = self._run(
            cand, ref, {"gate": 1e-3, "s0": {"mode": "reference_tile"}}
        )["cases"][0]["E"]
        self.assertAlmostEqual(e["s0"], 16.0, places=15)
        self.assertAlmostEqual(e["denominator"], 16.0, places=15)
        self.assertEqual(e["denominator_source"], "s0_floor")
        self.assertAlmostEqual(e["relative_l2"], 1e-3, places=14)
        self.assertTrue(e["pass"])
        self.assertTrue(e["absolute_normalized"])

    def test_multi_point_error_uses_l2_not_l1(self):
        # Two Ex errors of magnitude 3 and 4 at different points: ||err||_2 = 5,
        # while an L1 norm would give 7.  The reference is uniform 0.1, so
        # ||A_ref|| = 1.6 and it exceeds the constant S0 = 1.0.  The asserted
        # ratio therefore changes if the implementation regresses to L1.
        ref = uniform_tile(0.1)
        cand = ref.copy()
        cand[0, 0] = 0.1 + 3.0
        cand[1, 0] = 0.1 - 4.0
        config = {"gate": 1e-3, "s0": {"mode": "constant", "E": 1.0, "cB": 1.0}}
        e = self._run(cand, ref, config)["cases"][0]["E"]
        self.assertAlmostEqual(e["error_norm"], 5.0, places=15)
        self.assertAlmostEqual(e["denominator"], 1.6, places=15)
        self.assertAlmostEqual(e["relative_l2"], 5.0 / 1.6, places=15)
        self.assertNotAlmostEqual(e["relative_l2"], 7.0 / 1.6, places=15)

    def test_E_and_cB_are_reported_separately(self):
        ref = np.zeros((256, 6), dtype=np.float64)
        ref[:, 0] = 0.1  # E only
        ref[:, 4] = 0.2  # cBy only, norm 16*0.2 = 3.2
        cand = ref.copy()
        cand[:, 0] *= 2.0  # E is wrong
        report = self._run(
            cand, ref, {"gate": 1e-3, "s0": {"mode": "reference_tile"}}
        )
        case = report["cases"][0]
        self.assertFalse(case["E"]["pass"])
        self.assertTrue(case["cB"]["pass"])
        self.assertAlmostEqual(case["cB"]["s0"], 3.2, places=15)
        self.assertAlmostEqual(case["cB"]["relative_l2"], 0.0, places=15)


class FailLoudlyTest(unittest.TestCase):
    def test_missing_s0_is_an_error(self):
        with self.assertRaises(fg.MetricError) as ctx:
            fg.evaluate(uniform_tile(), uniform_tile(), {"gate": 1e-3}, {})
        self.assertIn("s0", str(ctx.exception))

    def test_unknown_s0_mode_is_an_error(self):
        with self.assertRaises(fg.MetricError):
            fg.evaluate(
                uniform_tile(),
                uniform_tile(),
                {"s0": {"mode": "guess"}},
                {},
            )

    def test_manifest_mode_requires_the_array(self):
        with self.assertRaises(fg.MetricError):
            fg.evaluate(
                uniform_tile(),
                uniform_tile(),
                {"s0": {"mode": "manifest"}},
                {},
            )

    def test_manifest_case_count_must_match(self):
        with self.assertRaises(fg.MetricError):
            fg.evaluate(
                uniform_tile(),
                uniform_tile(),
                {"s0": {"mode": "values", "values": [[1.0, 1.0], [2.0, 2.0]]}},
                {},
            )

    def test_nan_candidate_is_rejected(self):
        cand = uniform_tile()
        cand[0, 0] = np.nan
        with self.assertRaises(fg.MetricError):
            fg.evaluate(cand, uniform_tile(), {"s0": {"mode": "reference_tile"}}, {})

    def test_exact_zero_group_passes_with_zero_denominator(self):
        # Stationary B: reference and candidate are both exactly zero.
        zero = np.zeros((256, 6), dtype=np.float64)
        report = fg.evaluate(zero, zero, {"s0": {"mode": "reference_tile"}}, {})
        b = report["cases"][0]["cB"]
        self.assertTrue(b["zero_denominator"])
        self.assertEqual(b["relative_l2"], 0.0)
        self.assertTrue(b["pass"])

    def test_nonzero_field_against_exact_zero_reference_fails(self):
        zero = np.zeros((256, 6), dtype=np.float64)
        cand = zero.copy()
        cand[0, 3] = 1e-9  # tiny but nonzero cBx
        report = fg.evaluate(cand, zero, {"s0": {"mode": "reference_tile"}}, {})
        b = report["cases"][0]["cB"]
        self.assertTrue(b["zero_denominator"])
        self.assertIsNone(b["relative_l2"])
        self.assertFalse(b["pass"])


class CommandLineTest(unittest.TestCase):
    def _write(self, directory, candidate, reference, config):
        np.savez(directory / "cand.npz", candidate=candidate)
        np.savez(directory / "ref.npz", reference=reference)
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def test_cli_success_and_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write(
                d,
                uniform_tile(),
                uniform_tile(),
                {"gate": 1e-3, "s0": {"mode": "reference_tile"}},
            )
            out = d / "report.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "field_gate.py"),
                    str(d / "cand.npz"),
                    str(d / "ref.npz"),
                    "--config",
                    str(d / "config.json"),
                    "--output",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(report["summary"]["all_pass"])
            self.assertIn("s0_definition", report["metric"])

    def test_cli_missing_s0_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write(d, uniform_tile(), uniform_tile(), {"gate": 1e-3})
            proc = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "field_gate.py"),
                    str(d / "cand.npz"),
                    str(d / "ref.npz"),
                    "--config",
                    str(d / "config.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("s0", proc.stderr.lower())


if __name__ == "__main__":
    unittest.main()
