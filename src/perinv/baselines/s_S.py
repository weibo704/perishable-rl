"""(s,S) baseline (team draft §5): per SKU, if the inventory position is at or below s, order up to S.

The inventory position is on hand + in transit; stock age is ignored (that is what BSP-low-EW would add).
Orders are scaled to the remaining capacity (fit_to_capacity), so the policy is always feasible.

Parameters are grid-searched on the **tune** CRN paths only, never on the test paths (EXPERIMENTS.md rule 2).
To keep the grid two-dimensional for any N, (s, S) are set in units of the average SKU and scaled by demand:
s_i = round(s · λ_i / mean(λ)), S_i = round(S · λ_i / mean(λ)). For identical SKUs (the core configs) this is
exactly a common (s, S). Grid: 0 ≤ s < S ≤ S_max, S_max = min(C, ceil(1.5 · C / N)). Ties go to the smaller S,
then the smaller s, so the search is deterministic.

Owner: Member 2. Contract: docs/SPEC.md §8, WAREHOUSE.md §7.
"""

from __future__ import annotations

import math
import time

import numpy as np

from perinv.baselines.base import fit_to_capacity, inventory_position, remaining_capacity
from perinv.data.crn import load_paths
from perinv.env.config import EnvConfig
from perinv.eval.evaluate import evaluate


class SSPolicy:
    def __init__(self, cfg: EnvConfig, s: np.ndarray, S: np.ndarray, horizon_aware: bool = False):
        self.capacity = cfg.capacity
        self.s, self.S = np.asarray(s, np.int64), np.asarray(S, np.int64)
        # Horizon-aware variant: an order placed on day t arrives on day t + τ, so orders in the
        # last τ days of the episode never arrive. From day T − τ on, order nothing.
        self.cutoff = cfg.horizon - cfg.lead_time if horizon_aware else None

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        if self.cutoff is not None and int(info["state"]["t"]) >= self.cutoff:
            return np.zeros(len(self.s), np.int64)
        ip = inventory_position(info)
        q = np.where(ip <= self.s, self.S - ip, 0)
        return fit_to_capacity(q, remaining_capacity(info, self.capacity))


def per_sku(cfg: EnvConfig, s: int, S: int) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(cfg.demand_mean, float) / np.mean(cfg.demand_mean)
    return np.rint(s * u).astype(np.int64), np.rint(S * u).astype(np.int64)


def grid(cfg: EnvConfig) -> list[tuple[int, int]]:
    s_max = min(cfg.capacity, math.ceil(1.5 * cfg.capacity / cfg.n_skus))
    return [(s, S) for S in range(1, s_max + 1) for s in range(0, S)]


def search(cfg: EnvConfig, make_env, tune_demand: np.ndarray,
           horizon_aware: bool = False) -> tuple[tuple[int, int], float, list[dict]]:
    """Best (s, S) by mean profit on the tune paths. Returns ((s, S), profit, all candidates)."""
    env = make_env(cfg)
    rows = []
    for s, S in grid(cfg):
        pol = SSPolicy(cfg, *per_sku(cfg, s, S), horizon_aware=horizon_aware)
        profit = float(evaluate(pol, env, tune_demand).per_path()["profit"].mean())
        rows.append({"s": s, "S": S, "tune_profit": profit})
    best = max(rows, key=lambda r: (r["tune_profit"], -r["S"], -r["s"]))
    return (best["s"], best["S"]), best["tune_profit"], rows


def build(ctx, horizon_aware: bool = False) -> SSPolicy:
    """Runner builder: grid search on ctx.cfg's tune paths (the build world, never the eval world)."""
    t0 = time.perf_counter()
    tune = load_paths(ctx.cfg, "tune", ctx.crn_root)
    (s, S), profit, rows = search(ctx.cfg, ctx.make_env, tune, horizon_aware=horizon_aware)
    s_i, S_i = per_sku(ctx.cfg, s, S)
    ctx.run.update(ss={"s": s, "S": S, "s_per_sku": s_i.tolist(), "S_per_sku": S_i.tolist(),
                       "tune_profit": profit, "grid_size": len(rows), "horizon_aware": horizon_aware,
                       "search_seconds": round(time.perf_counter() - t0, 2)})
    ctx.run.log("ss_ready", s=s, S=S, tune_profit=round(profit, 3), grid_size=len(rows))
    return SSPolicy(ctx.cfg, s_i, S_i, horizon_aware=horizon_aware)


def build_h(ctx) -> SSPolicy:
    """Runner builder for the horizon-aware (s,S): no orders in the last τ days, grid search with the same rule."""
    return build(ctx, horizon_aware=True)
