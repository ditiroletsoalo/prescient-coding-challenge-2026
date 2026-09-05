"""
Prescient Coding Challenge 2026 -- your submission.

THIS IS THE ONLY FILE YOU MAY CHANGE.

You implement one function. The harness calls it once per trading day and hands
you a `hist` object holding every observation STRICTLY BEFORE that day. You
return the weights you want to hold for that day.

    generate_weights(hist, prev_weights, params) -> weights

What you get
------------
hist.date                 the day you are allocating for (no data for it yet)
hist.returns              DataFrame [date x asset] of daily returns, decimals
hist.prices               DataFrame [date x asset] of total-return index levels
hist.macro                DataFrame [date x macro feature]
hist.assets               list of the six asset codes, in order
hist.benchmark            Series of benchmark weights
hist.active_weight(w)     total active weight of w -- the number rule 3 tests

prev_weights              what you held yesterday. Trading away from it costs
                          money, so look at it.
params                    the PARAMS dict below, passed straight through

Optional extras, in case you want them: hist.cov() gives an EWMA covariance
matrix and hist.te(w) an ex-ante tracking error. No rule depends on either.

What you must return
--------------------
Six weights (dict, Series or array in hist.assets order) that sum to 1, are all
non-negative, sit within 10% of their benchmark weight, have a total active
weight of no more than 40%, keep total equity at or below 75% and gold at or
below 10%. `make_legal()` below
already does all of that -- you can leave it alone.

Declare every tuneable number in PARAMS. Parameter count is part of the score.

Run `python harness.py` to test on the practice window (calendar 2025), then
`python validate.py` before you submit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Every tuneable number lives here. Fewer is better.
# --------------------------------------------------------------------------- #

# TWO parameters are absent because they are provably inert here.
#
# tau: with a single view and the He-Litterman choice of
# Omega = diag(P (tau Sigma) P'), tau cancels out of the posterior exactly --
#     middle = tau Sigma P' / (2 tau P Sigma P') = Sigma P' / (2 P Sigma P')
# -- and a sweep from 0.001 to 10 confirms it to six decimals.
#
# gamma: w_T carries a 1/gamma factor, but the result is then rescaled to a
# fixed total absolute active size, and the factor cancels in that rescaling.
# Swept 0.5 to 50 with identical weights to eight decimals.
#
# Parameter count is scored, so neither is declared.
PARAMS = {
    "mean_window":  750,    # lookback for expected returns and the macro z-score
    "view_scale":   0.01,   # view strength (return units) at a 1-sigma VIX reading
    "tilt_size":    0.20,   # scales the raw optimiser output before make_legal
    "trade_speed":  0.02,   # fraction of the gap to yesterday we close per day
}

RIDGE = 1e-6             # regularises the covariance inverse

# The rules, restated locally so this file reads on its own.
ACTIVE_BAND = 0.10       # per asset, distance from benchmark
ACTIVE_BUDGET = 0.40     # total, summed over assets
EQUITY = ["SA_EQUITY", "GLOBAL_EQUITY"]
EQUITY_CAP = 0.75        # total equity, whatever the bands allow
GOLD_CAP = 0.10


# --------------------------------------------------------------------------- #
# <<--------------------- YOUR CODE GOES BELOW THIS LINE --------------------->>
#
# This is your playground. Delete or rewrite anything here. What follows is a
# deliberately naive starting point so you can see the shape of a working
# answer. It is NOT a good answer -- on the practice window it loses to the
# benchmark. Your job is to do better.
#
# Three steps:
#   1. build a signal (here: a plain inverse-volatility tilt, which knows
#      nothing at all about expected return),
#   2. make the weights legal,
#   3. move only part of the way from yesterday, so you do not pay the full
#      trading cost every day.
#
# Steps 2 and 3 are plumbing. Keep them. Step 1 is the actual question, and
# inverse volatility is a poor answer to it: it will always prefer cash and
# bonds, whatever is happening in the world.
#
# Things worth thinking about. Which of these six assets actually diversifies
# the other five? Gold and global equity are both priced in rands -- what does
# that mean when the currency moves? The macro file has a term spread and a
# policy rate in it; what should a steepening curve do to your bond weight? And
# look at the cost table in the README before you trade property daily.
# --------------------------------------------------------------------------- #


def _zscore(series: pd.Series) -> float:
    """Most recent value of `series`, expressed in its own historical sigmas."""
    s = series.dropna()
    if len(s) < 20 or s.std() == 0:
        return 0.0
    return float((s.iloc[-1] - s.mean()) / s.std())


def _expected_returns(hist, params) -> np.ndarray:
    """Expected returns: a time-varying estimate, tilted by one view.

    The prior is the trailing mean return per asset, annualised. This matters
    more than it looks: an alternative is to reverse-optimise the prior off
    the benchmark weights, but the benchmark is a CONSTANT vector and the
    covariance moves slowly, so that prior is very nearly frozen and the
    resulting portfolio holds the same sign of active position for years at a
    time. A trailing mean moves, so the portfolio can actually change its mind.

    The view: a VIX spike (risk-off) favours GOLD over GLOBAL_EQUITY, safety
    over growth, conditioned on where the VIX sits against its own history.
    """
    assets = hist.assets
    win = int(params["mean_window"])
    sigma = hist.cov().reindex(index=assets, columns=assets).to_numpy(dtype=float)
    pi = hist.returns.tail(win).mean().reindex(assets).to_numpy(dtype=float) * 252.0

    macro = hist.macro.tail(win)
    if len(macro) < 20:
        return pi

    idx = {a: i for i, a in enumerate(assets)}
    P = np.zeros((1, len(assets)))
    P[0, idx["GOLD"]], P[0, idx["GLOBAL_EQUITY"]] = 1.0, -1.0
    V = np.array([float(params["view_scale"]) * _zscore(macro["vix"])])

    # Black-Litterman posterior in reduced form (see the note above PARAMS).
    sp = sigma @ P.T
    psp = P @ sp
    adj = sp @ np.linalg.solve(2.0 * psp + np.eye(1) * 1e-12, np.eye(1)) @ (V - P @ pi)
    return pi + adj.ravel()


def build_signal(hist, params) -> pd.Series:
    """The zero-cost active portfolio from the mutual fund separation theorem.

        w_T = (1/gamma) Sigma^-1 ( E[R] - 1 (1' Sigma^-1 E[R]) / (1' Sigma^-1 1) )

    This is an optimiser rather than a per-asset score: Sigma^-1 lets the
    correlations decide how a view on one asset should be funded out of the
    others, which a standardised score throws away. It sums to zero by
    construction, so benchmark + w_T is still fully invested.

    The result is scaled to a sensible active size and then handed to
    make_legal, which enforces the bands, the budget and the caps.
    """
    assets = hist.assets
    sigma = hist.cov().reindex(index=assets, columns=assets).to_numpy(dtype=float)
    e_r = _expected_returns(hist, params)
    n = len(assets)

    s = sigma + np.eye(n) * RIDGE
    ones = np.ones(n)
    try:
        a = np.linalg.solve(s, e_r)
        b = np.linalg.solve(s, ones)
    except np.linalg.LinAlgError:
        return pd.Series(0.0, index=assets)

    denom = float(ones @ b)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return pd.Series(0.0, index=assets)

    # (no 1/gamma factor: it cancels in the rescaling below -- see note above)
    w_t = np.nan_to_num(a - b * float(ones @ a) / denom)
    total = np.abs(w_t).sum()
    if total > 1e-12:
        w_t = w_t * (float(params["tilt_size"]) * len(assets) / total)
    return pd.Series(w_t, index=assets)


def make_legal(weights: pd.Series, hist) -> pd.Series:
    """Force `weights` to satisfy every rule. You can leave this alone.

    Everything happens in active space -- how far each asset sits from its
    benchmark weight -- because that is how the rules are written.

    The loop is there because the steps interfere: forcing the active weights
    to net to zero (so the portfolio sums to 1) can push an asset back outside
    its band. A few passes settles it. The budget scaling goes last and is safe
    there: shrinking every active weight toward zero cannot breach a band, a
    cap, or non-negativity.
    """
    bm = hist.benchmark
    active = weights.reindex(hist.assets).astype(float) - bm

    for _ in range(50):
        active = active.clip(lower=-ACTIVE_BAND, upper=ACTIVE_BAND)  # rule 2
        active = active.clip(lower=-bm)                              # keeps weights >= 0
        # rule 4: total equity cap. Trim the equity block back, sharing the
        # cut over whichever equity assets still have room to come down.
        eq_excess = (bm[EQUITY] + active[EQUITY]).sum() - EQUITY_CAP
        eq_full = eq_excess > -1e-12
        if eq_excess > 0:
            floor = np.maximum(-ACTIVE_BAND, -bm[EQUITY])
            down = (active[EQUITY] - floor).clip(lower=0)
            if down.sum() > 1e-15:
                active[EQUITY] = active[EQUITY] - eq_excess * down / down.sum()

        active["GOLD"] = min(active["GOLD"], GOLD_CAP - bm["GOLD"])  # rule 5

        excess = active.sum()          # must be zero for weights to sum to 1
        if abs(excess) < 1e-12:
            break
        # give the correction to the assets that have room to absorb it
        room = (ACTIVE_BAND - active) if excess < 0 else (active + bm).clip(lower=0)
        room = room.clip(lower=0)
        if excess < 0 and eq_full:
            room[EQUITY] = 0.0   # equity is at its cap -- top up elsewhere
        if room.sum() <= 1e-15:
            break
        active = active - excess * room / room.sum()

    total = active.abs().sum()                                       # rule 3
    if total > ACTIVE_BUDGET:
        active = active * (ACTIVE_BUDGET / total)

    return bm + active


def generate_weights(hist, prev_weights, params):
    """Return the six portfolio weights to hold on hist.date."""
    bm = hist.benchmark

    # not enough history to estimate anything: sit on the benchmark
    if len(hist.returns) < 260:
        return bm.to_dict()

    # 1. optimiser -> target weights around the benchmark
    target = make_legal(bm + build_signal(hist, params), hist)

    # 2. trade gradually toward the target rather than jumping to it
    prev = prev_weights.reindex(hist.assets)
    w = prev + float(params["trade_speed"]) * (target - prev)

    return make_legal(w, hist).to_dict()


# <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>
