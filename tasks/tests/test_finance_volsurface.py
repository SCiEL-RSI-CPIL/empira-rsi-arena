"""Arbitrage-free SVI surface: Durrleman condition, arbitrage checks, data conventions and scoring."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

try:
    from scipy.optimize import brentq, least_squares
    from scipy.stats import norm
except ImportError:  # optional: the tests that need it are skipped
    norm = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_volsurface as fv  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import have_data  # noqa: E402

VOGT = (-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)  # Gatheral-Jacquier's arbitrageable example as (a, b, rho, m, sigma)


def flat(levels):
    return np.array([[w, 0.0, 0.0, 0.0, 0.1] for w in levels])


class DurrlemanTests(unittest.TestCase):
    def test_vogt_example_has_butterfly_arbitrage_where_the_paper_shows_it(self):
        k = np.linspace(-1.5, 1.5, 30001)
        g = fv.durrleman(VOGT, k)
        negative = k[g < 0]
        self.assertAlmostEqual(negative.min(), 0.64, delta=0.01)
        self.assertAlmostEqual(negative.max(), 1.26, delta=0.01)

    @unittest.skipUnless(norm, "scipy is not installed")
    def test_g_has_the_sign_of_the_implied_density(self):
        for params in (VOGT, (0.02, 0.1, -0.6, 0.05, 0.2), (0.04, 0.4, -0.9, -0.1, 0.05)):
            strikes = np.linspace(math.exp(-1.0), math.exp(1.0), 200001)
            w = fv.svi(params, np.log(strikes))[0]
            sd = np.sqrt(w)
            d1 = -np.log(strikes) / sd + sd / 2
            calls = norm.cdf(d1) - strikes * norm.cdf(d1 - sd)
            density = np.gradient(np.gradient(calls, strikes), strikes)[2000:-2000:400]
            g = fv.durrleman(params, np.log(strikes[2000:-2000:400]))
            mask = np.abs(density) > 1e-8
            np.testing.assert_array_equal(np.sign(g[mask]), np.sign(density[mask]))

    def test_arbitrage_and_domain_violations_are_rejected(self):
        good = flat(np.linspace(0.01, 0.08, 8))
        fv.validate(good, 8)
        cases = []
        for row, values in ((0, [0.01, -0.1, 0.0, 0.0, 0.1]), (0, [0.01, 0.1, 1.0, 0.0, 0.1]), (0, [0.01, 0.1, 0.0, 0.0, 1e-6]),
                            (0, [-0.05, 0.1, 0.0, 0.0, 0.1]), (0, [0.01, 1.6, 0.5, 0.0, 0.1]), (7, list(VOGT)),
                            (3, [0.001, 0.0, 0.0, 0.0, 0.1])):
            case = good.copy()
            case[row] = values
            cases.append(case)
        sloped = good.copy()
        sloped[:, 1] = 0.2
        sloped[5, 1] = 0.1  # wing slope falls with maturity
        cases.append(sloped)
        for case in cases + [good[:7], good * np.nan]:
            with self.assertRaises(CandidateError):
                fv.validate(case, 8)


@unittest.skipUnless(have_data("dax_2002_surface.csv"), "run scripts/prepare_finance.py first")
class DaxSurfaceTests(unittest.TestCase):
    def test_surface_layout_and_seed_score(self):
        data = fv.load()
        self.assertEqual(data["implied_vol"].shape, (8, 13))
        np.testing.assert_allclose(data["forwards"], 4468.17 * np.exp(data["zero_rates"] * data["maturity_days"] / 365))
        levels = [data["total_variance"][i][np.argmin(np.abs(data["log_moneyness"][i]))] for i in range(8)]
        score, details = fv.surface_score(lambda q: flat(np.maximum.accumulate(levels)))
        self.assertAlmostEqual(score, 5.946, delta=0.01)
        self.assertEqual(details["quotes"], 104)

    @unittest.skipUnless(norm, "scipy is not installed")
    def test_independent_heston_calibration_reproduces_quantlib_reference(self):
        """QuantLib testDAXCalibration expects a squared error of 177.2 +- 1 (vol points^2) on these quotes."""
        data = fv.load()
        spot, days = data["spot"], data["maturity_days"]
        weeks = np.array([int((d + 3) / 7.0) for d in days])  # QuantLib rounds maturities to weeks
        nodes = np.r_[0.0, days]
        rates = np.interp(7 * weeks, nodes, np.r_[0.0357, data["zero_rates"]])
        T = 7 * weeks / 365.0
        disc = np.exp(-rates * T)
        forwards = spot / disc
        x, wts = np.polynomial.laguerre.laggauss(160)

        def call(F, K, t, df, v0, kappa, theta, sigma, rho):
            def phi(u):
                iu = 1j * u
                d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma ** 2 * (iu + u * u))
                g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)
                e = np.exp(-d * t)
                C = kappa * theta / sigma ** 2 * ((kappa - rho * sigma * iu - d) * t - 2 * np.log((1 - g * e) / (1 - g)))
                D = (kappa - rho * sigma * iu - d) / sigma ** 2 * (1 - e) / (1 - g * e)
                return np.exp(C + D * v0 + iu * math.log(F))
            p1 = 0.5 + np.sum(wts * np.exp(x) * np.real(np.exp(-1j * x * math.log(K)) * phi(x - 1j) / (1j * x * F))) / math.pi
            p2 = 0.5 + np.sum(wts * np.exp(x) * np.real(np.exp(-1j * x * math.log(K)) * phi(x) / (1j * x))) / math.pi
            return df * (F * p1 - K * p2)

        def black(F, K, t, vol, df, is_call):
            sd = vol * math.sqrt(t)
            d1 = math.log(F / K) / sd + sd / 2
            return df * (F * norm.cdf(d1) - K * norm.cdf(d1 - sd)) if is_call else df * (K * norm.cdf(sd - d1) - F * norm.cdf(-d1))

        def residuals(p):
            out = []
            for m in range(8):
                for s, K in enumerate(data["strikes"]):
                    is_call = K * disc[m] >= spot
                    price = call(forwards[m], K, T[m], disc[m], *p)
                    price = price if is_call else price - disc[m] * (forwards[m] - K)
                    implied = brentq(lambda v: black(forwards[m], K, T[m], v, disc[m], is_call) - price, 1e-6, 5.0, xtol=1e-14)
                    out.append(implied - data["implied_vol"][m, s])
            return np.array(out)
        fit = least_squares(residuals, [0.1, 1.0, 0.1, 0.5, -0.5], method="lm", xtol=1e-10, ftol=1e-10, gtol=1e-10)
        self.assertAlmostEqual(float(np.sum((100 * fit.fun) ** 2)), 177.2, delta=1.0)


if __name__ == "__main__":
    unittest.main()
