import unittest

import numpy as np

from src.v12.oracle_expert_headroom import oracle_convex


class OracleExpertTests(unittest.TestCase):
    def test_convex_projection_and_hard_boundary(self):
        y = np.array([0.25, 2.0, -1.0, 0.4])
        e1 = np.array([0.0, 0.0, 0.0, 0.4])
        e2 = np.array([1.0, 1.0, 1.0, 0.4])
        alpha, prediction = oracle_convex(y, e1, e2)
        np.testing.assert_allclose(alpha, [0.75, 0.0, 1.0, 0.5])
        np.testing.assert_allclose(prediction, [0.25, 1.0, 0.0, 0.4])

    def test_nonfinite_or_misaligned_input_rejected(self):
        with self.assertRaises(ValueError):
            oracle_convex(np.array([0.0]), np.array([0.0, 1.0]), np.array([1.0]))
        with self.assertRaises(ValueError):
            oracle_convex(np.array([float("nan")]), np.array([0.0]), np.array([1.0]))
