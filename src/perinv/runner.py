"""Experiment runner: expand a matrix YAML into jobs, skip finished runs, run the rest (in parallel).

CLI: scripts/run_matrix.py. Contract: docs/WAREHOUSE.md §4 (keys, status), docs/EXPERIMENTS.md (matrix).

How a job runs (in its own process):
    init_process(seed) -> run_context(...) -> builder(ctx) returns a Policy (trains first if it is an RL agent)
    -> evaluate on the test CRN paths -> write_eval -> status done

Builders and the env factory are referenced as "module:function" strings, so spawned worker processes can
import them, and members plug in their code without editing this file. One runner per runs/ root at a time.

Owner: Member 3.
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cached_property
from multiprocessing import get_context
from multiprocessing.connection import wait
from pathlib import Path

import yaml

from perinv.baselines.base import Policy
from perinv.data.crn import load_paths
from perinv.env.config import EnvConfig, load_env_config
from perinv.eval.evaluate import evaluate
from perinv.eval.writer import write_eval
from perinv.utils.run_io import (
    EVAL_FILE,
    Run,
    describe_env,
    find_trained,
    mark_crashed,
    run_context,
    run_key,
    run_name,
    run_status,
    train_key,
)
from perinv.utils.seeding import init_process

log = logging.getLogger("perinv.runner")

# Default builders. Each owner implements `build(ctx) -> Policy` at this location; until then, jobs using
# that policy are reported as "no builder" and nothing runs. A matrix YAML may override any entry.
DEFAULT_BUILDERS = {
    **{algo: "perinv.train.train:build" for algo in ("ppo", "a2c", "sac", "dqn", "maskppo")},
    "dp": "perinv.baselines.dp_policy:build",
    "ss": "perinv.baselines.s_S:build",
    "newsvendor": "perinv.baselines.newsvendor:build",
    "ss_h": "perinv.baselines.s_S:build_h",                   # horizon-aware: no orders in the last τ days
    "newsvendor_h": "perinv.baselines.newsvendor:build_h",    # horizon-aware: no orders in the last τ days
}
DEFAULT_ENV_FACTORY = "perinv.env.perishable_env:PerishableInventoryEnv"


def resolve(ref: str) -> Callable:
    """'pkg.module:attr' -> the object. Raises ImportError/AttributeError with the reference in the message."""
    module, _, attr = ref.partition(":")
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as exc:
        raise type(exc)(f"cannot load {ref!r}: {exc}") from exc


# ---------------------------------------------------------------- jobs

@dataclass(frozen=True)
class Job:
    group: str
    env_path: str
    policy: str
    seed: int
    builder: str
    env_factory: str
    hparams: dict = field(hash=False)
    hparams_source: str
    train_env_steps: int | None
    runs_root: str
    crn_root: str
    eval_env_path: str | None = None   # cross-evaluation: evaluate on this config's test paths (default: env_path)

    @cached_property
    def cfg(self) -> EnvConfig:
        """The world the policy is built for (training, DP solve, heuristic tuning)."""
        return load_env_config(self.env_path)

    @cached_property
    def eval_cfg(self) -> EnvConfig:
        """The world the policy is scored in. Same as cfg unless this is a cross-evaluation."""
        return load_env_config(self.eval_env_path) if self.eval_env_path else self.cfg

    @property
    def name(self) -> str:
        cfg = self.cfg
        return run_name(self.group, cfg.n_skus, cfg.kappa if cfg.kappa is not None else 0, self.policy, self.seed)

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_root) / self.name

    def keys(self) -> tuple[str, str]:
        tk = train_key(self.cfg, policy=self.policy, hparams=self.hparams, seed=self.seed,
                       train_env_steps=self.train_env_steps, builder=self.builder, env_factory=self.env_factory,
                       crn_tune_sha256=crn_sha256(self.cfg, self.crn_root, "tune"))
        ev = self.eval_cfg
        return tk, run_key(tk, demand_hash=ev.demand_hash(), crn_test_sha256=crn_test_sha256(ev, self.crn_root),
                           eval_config_hash=ev.config_hash())


# Fields that may differ between the build config and the eval config of a cross-evaluation: only the demand
# process. Anything else (N, capacity, dynamics, costs) would change the action space or the profit yardstick.
CROSS_EVAL_MAY_DIFFER = {"demand_mean", "season", "scenario", "name"}


def check_buildable(cfg: EnvConfig) -> None:
    """Scenario configs are eval-only: the env trains on demand_mean/season and would ignore the shocks, so a
    run built on one would be labelled as a shock run while training on the base world."""
    if cfg.scenario:
        raise ValueError(f"{cfg.name} has a demand scenario and is eval-only: list it under eval_envs, "
                         f"paired with its base config under envs")


def check_cross_eval(build: EnvConfig, evaluate_on: EnvConfig) -> None:
    diff = [f for f in build.__dataclass_fields__ if f not in CROSS_EVAL_MAY_DIFFER
            and getattr(build, f) != getattr(evaluate_on, f)]
    if diff:
        raise ValueError(f"cross-evaluation {build.name} -> {evaluate_on.name}: configs may differ only in "
                         f"{sorted(CROSS_EVAL_MAY_DIFFER - {'name'})}, but differ in {diff}")
    if evaluate_on.scenario:   # shocked path k = base path k + shock, only if the base demand is the same
        moved = [f for f in ("demand_mean", "season") if getattr(build, f) != getattr(evaluate_on, f)]
        if moved:
            raise ValueError(f"shock evaluation {build.name} -> {evaluate_on.name}: a scenario config must keep "
                             f"the base demand of its build config, but {moved} differ")


def crn_sha256(cfg: EnvConfig, crn_root: str | Path, split: str) -> str:
    """sha256 of <split>.npz from the CRN manifest; fails early if the paths were never generated."""
    manifest = Path(crn_root) / cfg.demand_hash() / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"no CRN paths for {cfg.name} at {manifest.parent}; "
                                f"run: python scripts/make_crn_paths.py <env yaml>")
    return json.loads(manifest.read_text())["files"][split]["sha256"]


def crn_test_sha256(cfg: EnvConfig, crn_root: str | Path) -> str:
    return crn_sha256(cfg, crn_root, "test")


def load_hparams(policy: str, env_path: str, configs_root: Path) -> tuple[dict, str]:
    """Tuned values win over defaults: configs/tuned/<policy>-<env>.yaml, else configs/algo/<policy>.yaml."""
    tuned = configs_root / "tuned" / f"{policy}-{Path(env_path).stem}.yaml"
    for p in (tuned, configs_root / "algo" / f"{policy}.yaml"):
        if p.exists():
            return dict((yaml.safe_load(p.read_text()) or {}).get("hparams") or {}), p.as_posix()
    return {}, ""


def expand(matrix_path: str | Path, *, seeds: list[int] | None = None, groups: set[str] | None = None,
           policies: set[str] | None = None, profile: str | None = None, runs_root: str = "runs",
           crn_root: str = "data/crn", configs_root: str | Path = "configs") -> list[Job]:
    """Matrix YAML -> one Job per (group, env, policy, seed). CLI filters narrow it; budget.yaml fills defaults.

    matrix YAML:
        budget: configs/budget.yaml          # optional; default seeds and train_env_steps
        env_factory: module:Class            # optional
        builders: {name: module:function}    # optional overrides of DEFAULT_BUILDERS
        groups:
          A: {envs: [configs/env/n1_k06.yaml], policies: [ppo, dp], seeds: [0, 1]}   # seeds optional
          # a policy can also be {name: dp, seeds: [0]} for deterministic baselines
          X: {envs: [a.yaml], eval_envs: [b.yaml], policies: [ppo]}   # cross-evaluation, paired by position:
                                                                     # build on a.yaml, evaluate on b.yaml
    """
    matrix_path = Path(matrix_path)
    spec = yaml.safe_load(matrix_path.read_text())
    configs_root = Path(configs_root)
    budget = yaml.safe_load(Path(spec.get("budget", configs_root / "budget.yaml")).read_text())
    steps = budget.get("train_env_steps")
    if profile:
        steps = budget["profiles"][profile]["train_env_steps"]
    builders = DEFAULT_BUILDERS | spec.get("builders", {})
    env_factory = spec.get("env_factory", DEFAULT_ENV_FACTORY)

    jobs = []
    for group, g in spec["groups"].items():
        if groups and group not in groups:
            continue
        for env_path in g["envs"]:
            check_buildable(load_env_config(env_path))
        eval_envs = g.get("eval_envs")
        if eval_envs is not None:
            if len(eval_envs) != len(g["envs"]):
                raise ValueError(f"group {group}: eval_envs must pair 1:1 with envs "
                                 f"({len(eval_envs)} vs {len(g['envs'])})")
            for a, b in zip(g["envs"], eval_envs, strict=True):
                check_cross_eval(load_env_config(a), load_env_config(b))
        eval_of = dict(zip(g["envs"], eval_envs, strict=True)) if eval_envs is not None else {}
        for entry in g["policies"]:
            pol = entry if isinstance(entry, str) else entry["name"]
            if policies and pol not in policies:
                continue
            pol_seeds = (entry.get("seeds") if isinstance(entry, dict) else None) or g.get("seeds") or budget["seeds"]
            if seeds is not None and (isinstance(entry, str) or "seeds" not in entry):
                pol_seeds = seeds  # --seeds overrides, except for policies pinned to fixed seeds
            for env_path in g["envs"]:
                hp, src = load_hparams(pol, env_path, configs_root)
                for s in pol_seeds:
                    jobs.append(Job(group=str(group), env_path=str(env_path), policy=pol, seed=int(s),
                                    builder=builders.get(pol, ""), env_factory=env_factory, hparams=hp,
                                    hparams_source=src, train_env_steps=steps, runs_root=runs_root,
                                    crn_root=crn_root,
                                    eval_env_path=str(eval_of[env_path]) if env_path in eval_of else None))
    names = Counter(j.name for j in jobs)
    dupes = [n for n, c in names.items() if c > 1]
    if dupes:
        raise ValueError(f"two jobs map to the same run directory: {dupes[:5]}")
    return jobs


# ---------------------------------------------------------------- one job

@dataclass
class BuildContext:
    """What a policy builder gets. RL builders train (or reload when `reuse_model`), then return a Policy."""

    cfg: EnvConfig
    policy: str
    hparams: dict
    seed: int
    train_env_steps: int | None
    run: Run
    make_env: Callable[[EnvConfig], object]
    reuse_model: bool
    model_dir: Path   # where to load a reusable model from (this run, or the native run with the same train_key)
    crn_root: str = "data/crn"   # tune paths for (s,S) grid search and learning curves; never read the test split


def run_job(job: Job, official: bool = False) -> dict:
    """Execute one job in the current process. Exceptions propagate after the manifest is marked failed."""
    init_process(job.seed)
    cfg, ecfg = job.cfg, job.eval_cfg
    tk, rk = job.keys()
    meta = describe_env(cfg) | {"group": job.group, "policy": job.policy, "seed": job.seed,
                                "crn_test_sha256": crn_test_sha256(ecfg, job.crn_root),
                                "train_env_steps": job.train_env_steps,
                                "eval_env_path": job.eval_env_path or job.env_path,
                                "eval_config_hash": ecfg.config_hash(), "eval_demand_hash": ecfg.demand_hash(),
                                "eval_demand_mean": list(ecfg.demand_mean),
                                "cross_eval": job.eval_env_path is not None}
    config = {"env_path": job.env_path, "eval_env_path": job.eval_env_path, "policy": job.policy,
              "builder": job.builder, "env_factory": job.env_factory, "hparams": job.hparams,
              "hparams_source": job.hparams_source, "seed": job.seed, "train_env_steps": job.train_env_steps}
    trained_in = find_trained(Path(job.runs_root), tk, prefer=job.run_dir)
    with run_context(Path(job.runs_root), job.name, run_key=rk, train_key=tk, config=config, meta=meta,
                     official=official) as run:
        make_env = resolve(job.env_factory)
        run.stage = "build"
        run.log("stage_start")   # last log line names the stage if the process dies without unwinding
        if trained_in is not None and trained_in != run.dir:
            run.log("reuse_model", from_run=trained_in.name)
        policy: Policy = resolve(job.builder)(BuildContext(
            cfg=cfg, policy=job.policy, hparams=job.hparams, seed=job.seed, train_env_steps=job.train_env_steps,
            run=run, make_env=make_env, reuse_model=trained_in is not None, model_dir=trained_in or run.dir,
            crn_root=job.crn_root))
        run.stage = "eval"
        run.log("stage_start")
        demand = load_paths(ecfg, "test", job.crn_root)   # demand paths of the eval world
        t0 = time.perf_counter()
        result = evaluate(policy, make_env(ecfg), demand)
        write_eval(result, ecfg, demand, rk, run.dir / EVAL_FILE)
        profit = float(result.per_path()["profit"].mean())
        run.log("eval_done", paths=len(result.path_ids), mean_profit=profit,
                eval_seconds=round(time.perf_counter() - t0, 3))
    return {"run_name": job.name, "mean_profit": profit}


# ---------------------------------------------------------------- the matrix

NO_CRN = "no CRN: "
NO_BUILDER = "no builder: "


def plan(jobs: Iterable[Job], force: bool = False) -> list[tuple[Job, str, str]]:
    """(job, status, action) for each job. action: 'skip' | 'run' | 'no builder: ...' | 'no CRN: ...'."""
    out = []
    for job in jobs:
        try:
            rk = job.keys()[1]
        except FileNotFoundError as exc:   # eval-world CRN paths never generated: listed, not raised one by one
            out.append((job, "-", f"{NO_CRN}{exc}"))
            continue
        status = run_status(job.run_dir, rk)
        action = "run" if force or status != "done" else "skip"
        if action == "run":
            try:
                if job.builder:
                    resolve(job.builder)
                else:
                    action = f"{NO_BUILDER}none registered for {job.policy!r}"
            except (ImportError, AttributeError) as exc:
                action = f"{NO_BUILDER}{exc}"
        out.append((job, status, action))
    return out


def run_matrix(jobs: list[Job], n_jobs: int = 1, force: bool = False, official: bool = False) -> Counter:
    """Run every job whose action is 'run'. One failure does not stop the others. Returns outcome counts.

    Refuses to start (FileNotFoundError, nothing run) if any job lacks CRN test paths, and names all of them:
    an overnight matrix should not silently skip part of the design.
    n_jobs > 1: every job runs in its own spawned process, so a hard crash (segfault, OOM kill, os._exit)
    fails only that job; a shared pool would fail every job in flight with it.
    """
    rows = plan(jobs, force)
    no_crn = sorted({j.eval_env_path or j.env_path for j, _, a in rows if a.startswith(NO_CRN)})
    if no_crn:
        n = sum(a.startswith(NO_CRN) for _, _, a in rows)
        raise FileNotFoundError(f"{n} job(s) have no CRN test paths, so nothing was run. Generate them first "
                                f"(run: python scripts/make_crn_paths.py {' '.join(no_crn)})")
    todo = [j for j, _, action in rows if action == "run"]
    counts = Counter(skipped=len(jobs) - len(todo))
    log.info("%d jobs: %d to run, %d skipped or blocked", len(jobs), len(todo), counts["skipped"])
    if n_jobs <= 1:
        for job in todo:
            try:
                res = run_job(job, official)
            except Exception as exc:  # already recorded in the run's manifest + failure.json
                counts[_failed(job, f"{type(exc).__name__}: {exc}")] += 1
            else:
                counts[_done(job, res)] += 1
        return counts
    _run_isolated(todo, n_jobs, official, counts)
    return counts


def _child(job: Job, official: bool, conn) -> None:
    """Process target: run one job, report ('done', result) or ('failed', message). No report = crashed."""
    try:
        conn.send(("done", run_job(job, official)))
    except BaseException as exc:  # recorded in the manifest by run_context when it got that far
        conn.send(("failed", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _run_isolated(todo: list[Job], n_jobs: int, official: bool, counts: Counter) -> None:
    # spawn everywhere: same behaviour on Windows/macOS/Linux, and no forked torch thread state
    ctx = get_context("spawn")
    pending, live = list(todo), {}
    try:
        while pending or live:
            while pending and len(live) < n_jobs:
                job = pending.pop(0)
                recv, send = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_child, args=(job, official, send), name=job.name)
                proc.start()
                send.close()   # the child now holds the only write end
                live[proc.sentinel] = (proc, recv, job)
            for sentinel in wait(list(live)):
                proc, recv, job = live.pop(sentinel)
                try:
                    msg = recv.recv() if recv.poll() else None
                except (EOFError, OSError):   # Windows: poll on a pipe whose writer died is BrokenPipeError
                    msg = None
                proc.join()
                recv.close()
                if msg is None:
                    counts[_crashed(job, proc.exitcode)] += 1
                elif msg[0] == "done":
                    counts[_done(job, msg[1])] += 1
                else:
                    counts[_failed(job, msg[1])] += 1
    finally:   # Ctrl+C or an error in the parent: never leave orphan workers behind
        for proc, recv, _ in live.values():
            proc.terminate()
            proc.join()
            recv.close()


def _crashed(job: Job, exitcode: int | None) -> str:
    msg = f"worker process died without reporting (exit code {exitcode}): segfault, out of memory or os._exit"
    mark_crashed(job.run_dir, job.keys()[1], msg)
    return _failed(job, f"WorkerCrashed: {msg}")


def _failed(job: Job, message: str) -> str:
    log.error("FAILED %s: %s", job.name, message)
    return "failed"


def _done(job: Job, res: dict) -> str:
    log.info("done %s: mean profit %.2f", job.name, res["mean_profit"])
    return "done"
