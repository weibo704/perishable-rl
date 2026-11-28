"""Guard tests for the horizon-aware baselines (ss_h, newsvendor_h): no orders in the last τ days.

Orders placed on day t arrive on day t + τ, so orders in the last τ days of a T-day episode never arrive and only
cost money. The horizon-aware variants skip them; the original baselines are unchanged.
"""

import numpy as np

from perinv.baselines import newsvendor, s_S
from perinv.data.crn import load_paths
from perinv.env.config import load_env_config
from perinv.env.perishable_env import PerishableInventoryEnv
from perinv.eval.evaluate import evaluate

CFG = load_env_config("configs/env/n1_k06.yaml")
CFG5 = load_env_config("configs/env/n5_k10.yaml")


def empty_info(cfg, t):
    """A state with no stock and nothing in transit, so every baseline wants to order."""
    n = len(np.atleast_1d(cfg.demand_mean))
    return {"state": {"t": t, "inv": np.zeros((n, 1), np.int64), "pipe": np.zeros((n, 1), np.int64)}}


def profits(policy, cfg, demand):
    return evaluate(policy, PerishableInventoryEnv(cfg), demand).per_path().set_index("path_id")["profit"]


def test_no_orders_in_last_tau_days():
    for cfg in (CFG, CFG5):
        cutoff = cfg.horizon - cfg.lead_time
        pols = [s_S.SSPolicy(cfg, *s_S.per_sku(cfg, 2, 6), horizon_aware=True),
                newsvendor.NewsvendorPolicy(cfg, horizon_aware=True)]
        for pol in pols:
            for t in range(cutoff, cfg.horizon):
                assert pol.act(None, empty_info(cfg, t)).sum() == 0, (type(pol).__name__, t)
            assert pol.act(None, empty_info(cfg, cutoff - 1)).sum() > 0     # still orders right before the window


def test_default_behaviour_unchanged_before_the_window():
    """Before the window both versions give identical orders; the default is not horizon-aware."""
    cutoff = CFG.horizon - CFG.lead_time
    for t in range(cutoff):
        info = empty_info(CFG, t)
        a = s_S.SSPolicy(CFG, *s_S.per_sku(CFG, 2, 6)).act(None, info)
        b = s_S.SSPolicy(CFG, *s_S.per_sku(CFG, 2, 6), horizon_aware=True).act(None, info)
        np.testing.assert_array_equal(a, b)
    assert s_S.SSPolicy(CFG, *s_S.per_sku(CFG, 2, 6)).cutoff is None
    assert newsvendor.NewsvendorPolicy(CFG).cutoff is None


def test_skipping_the_window_never_lowers_profit():
    """Orders in the last τ days never arrive, so skipping them can only save their cost, on every path.
    If the cutoff were one day too early, an order that does arrive would be skipped and some path could lose."""
    demand = load_paths(CFG, "test")[:20]
    for make in (lambda h: s_S.SSPolicy(CFG, *s_S.per_sku(CFG, 2, 6), horizon_aware=h),
                 lambda h: newsvendor.NewsvendorPolicy(CFG, horizon_aware=h)):
        base, aware = profits(make(False), CFG, demand), profits(make(True), CFG, demand)
        assert (aware >= base - 1e-9).all()
        assert aware.mean() > base.mean()        # the originals do order in the window, so there is a real saving
