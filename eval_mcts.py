#!/usr/bin/env python3
"""Run Residual MCTS on the highway environment, mirroring the notebook logic.

This script is a direct stand-alone version of the Residual MCTS section in
`Training_demo_Highway.ipynb`:
    - Same environment (`highway-ME-basic-AddRightReward-v0` by default)
    - Same evaluation loop and reward shaping
    - Same use of `agent.plan(obs)` from a config-loaded MCTS agent

With ``--safe-decide``, impermissible actions (supervisor min-violation mask) are
filtered out inside Residual MCTS prior/rollout so the returned plan already
conforms to hard constraints.

Episode metrics are written incrementally to ``--output`` (CSV). Re-running with
the same output path resumes unfinished ``episode_id``s, using ``seed + episode_id``
so unfinished work continues with the correct RNG stream.

Use ``--n-workers`` to run independent episodes in parallel across CPU cores
(default: all physical cores, at least 1). ``--n-workers 1`` preserves serial
behavior. The DQN prior is forced onto CPU (faster for many tiny MCTS queries).

Use ``--start-episode`` with ``--n-episodes`` to shard across machines: episode ids
are ``[start_episode, n_episodes)`` (e.g. ``--start-episode 50 --n-episodes 100``
runs 50–99). Each shard should write its own CSV, then concatenate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import gym
import highway_env  # noqa: F401  # registers highway environments
import numpy as np
from highway_env.vehicle.controller import ControlledVehicle
from rl_agents.agents.common.factory import load_agent, load_agent_config
from supervisor import DiscreteSupervisor

from basic_reward import compute_basic_reward, load_basic_reward_config
from highway_env.envs.merge_courtesy import courtesy_gate_gap


# Suppress noisy warnings, as in the notebook
warnings.filterwarnings("ignore")

# Match norm-supervised-highway/scripts/base_experiment.py and eval_highway.py
BASE_SEED = 42

CSV_FIELDS = [
    "episode_id",
    "seed",
    "success",
    "episode_length",
    "episode_reward",
    "basic_reward",
    "mean_speed",
    "mean_lane",
    "mean_courtesy_gap",
    "courtesy_active_steps",
    "total_norm_cost",
]

# Per-process state for parallel workers (set by _init_worker).
_WORKER: dict[str, Any] = {}


def physical_cpu_count() -> int:
    """Best-effort physical core count (falls back to logical ``os.cpu_count()``)."""
    try:
        cores: set[tuple[str, str]] = set()
        physical_id: str | None = None
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("physical id"):
                    physical_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id") and physical_id is not None:
                    cores.add((physical_id, line.split(":", 1)[1].strip()))
                    physical_id = None
        if cores:
            return len(cores)
    except OSError:
        pass
    return os.cpu_count() or 1


def default_n_workers() -> int:
    """Default parallel workers: one per physical core (min 1)."""
    return max(1, physical_cpu_count())


def is_merge_env(env_name: str) -> bool:
    return "merge" in env_name.lower()


def supervisor_profile_for_env(env_name: str) -> str:
    """Use merge courtesy norms on merge envs; right-lane otherwise."""
    return "merge_courtesy" if is_merge_env(env_name) else "right_lane"


def basic_reward_config_path_for_env(env_name: str) -> Path:
    root = Path(__file__).parent / "configs"
    if is_merge_env(env_name):
        return root / "MergeEnv" / "basic_reward.json"
    return root / "HighwayEnv" / "basic_reward.json"


def prepare_agent_config(agent_config: str | dict) -> dict:
    """Load agent JSON (if needed) and force the DQN prior onto CPU."""
    if isinstance(agent_config, dict):
        config = copy.deepcopy(agent_config)
    else:
        config = load_agent_config(str(Path(agent_config).expanduser().resolve()))
    config["device"] = "cpu"
    return config


def episode_seed(base_seed: int, episode: int) -> int:
    """Deterministic per-episode seed: base_seed + episode (independent of run length)."""
    return base_seed + episode


def make_permissibility_action_filter(supervisor: DiscreteSupervisor):
    """Build an MCTSME.action_filter matching DiscreteSupervisor._get_permissibility_mask.

    Evaluates profile constraints on the *search* environment's vehicle (the
    safe_deepcopy_env used during planning), not only the live eval env.
    Returns a boolean mask of length action_space.n; True = min-violation actions.
    """
    constraints = supervisor.profile.constraints

    def action_filter(state, observation):
        del observation  # mask depends on vehicle state, not the observation tensor
        env = state.unwrapped if hasattr(state, "unwrapped") else state
        vehicle = env.vehicle
        n_actions = state.action_space.n
        violations = np.zeros(n_actions, dtype=np.int32)
        for action in range(n_actions):
            for constraint in constraints:
                violations[action] += int(constraint.is_violating_action(vehicle, action))
        return violations == violations.min()

    return action_filter


def load_completed_episodes(output_path: Path) -> tuple[set[int], dict[str, list]]:
    """Load prior CSV rows for resume.

    Returns (completed_episode_ids, metric lists). Metric lists follow CSV row
    order; callers may sort by episode_id when summarizing.
    """
    metrics = {
        "successes": [],
        "episode_rewards": [],
        "episode_total_rewards": [],
        "episode_lengths": [],
        "episode_speeds": [],
        "episode_lanes": [],
        "episode_total_norm_costs": [],
        "episode_ids": [],
        "episode_courtesy_gaps": [],
        "episode_courtesy_active_steps": [],
    }
    completed: set[int] = set()
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return completed, metrics

    with output_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return completed, metrics
        missing = [field for field in ("episode_id",) if field not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Output CSV {output_path} is missing required columns: {missing}"
            )

        for row in reader:
            episode_id = int(row["episode_id"])
            if episode_id in completed:
                continue
            completed.add(episode_id)
            metrics["episode_ids"].append(episode_id)
            metrics["successes"].append(int(float(row["success"])))
            metrics["episode_rewards"].append(float(row["episode_reward"]))
            metrics["episode_total_rewards"].append(float(row["basic_reward"]))
            metrics["episode_lengths"].append(int(float(row["episode_length"])))
            metrics["episode_speeds"].append(float(row["mean_speed"]))
            metrics["episode_lanes"].append(float(row["mean_lane"]))
            metrics["episode_total_norm_costs"].append(float(row["total_norm_cost"]))
            gap_raw = row.get("mean_courtesy_gap", "")
            if gap_raw in ("", None):
                metrics["episode_courtesy_gaps"].append(float("nan"))
            else:
                metrics["episode_courtesy_gaps"].append(float(gap_raw))
            metrics["episode_courtesy_active_steps"].append(
                int(float(row.get("courtesy_active_steps", 0) or 0))
            )

    return completed, metrics


def append_episode_row(output_path: Path, row: dict) -> None:
    """Append one episode row, creating the CSV with a header if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.is_file() or output_path.stat().st_size == 0
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _seed_episode(env, agent, base_seed: int, episode_idx: int):
    """Seed env/agent and reset for the given episode index."""
    ep_seed = episode_seed(base_seed, episode_idx)
    env.seed(ep_seed)
    env.action_space.seed(ep_seed)
    if hasattr(agent, "seed"):
        agent.seed(ep_seed)
    return env.reset(), ep_seed


def run_single_episode(
    agent,
    env,
    supervisor: DiscreteSupervisor,
    episode_idx: int,
    seed: int,
    basic_reward_config: dict,
    env_name: str = "",
    courtesy_target_lane: int = 1,
) -> dict:
    """Run one MCTS episode and return a CSV-ready metrics row."""
    obs, ep_seed = _seed_episode(env, agent, seed, episode_idx)
    supervisor.reset_norms()

    episode_lane = 0.0
    episode_speed = 0.0
    episode_reward = 0.0
    total_reward = 0.0
    episode_norm_cost = 0.0
    courtesy_gaps: list[float] = []
    ep_len = 0
    done = False

    while not done:
        # Courtesy gap on the pre-step state (matches SCPS test_merge timing).
        gap = courtesy_gate_gap(
            env.unwrapped.vehicle if hasattr(env, "unwrapped") else env.vehicle,
            target_lane_id=courtesy_target_lane,
        )
        if gap is not None:
            courtesy_gaps.append(gap)

        actions = agent.plan(obs)
        base_action = int(actions[0])

        realized_cost = supervisor.get_norm_violation_cost([base_action])[0].item()
        episode_norm_cost += realized_cost

        obs, reward, done, _infos = env.step(base_action)

        neighbours = env.road.network.all_side_lanes(env.vehicle.lane_index)
        lane = (
            env.vehicle.target_lane_index[2]
            if isinstance(env.vehicle, ControlledVehicle)
            else env.vehicle.lane_index[2]
        )
        lane_fraction = lane / max(len(neighbours) - 1, 1)
        episode_lane += lane_fraction

        forward_speed = env.vehicle.speed * np.cos(env.vehicle.heading)
        episode_speed += forward_speed
        episode_reward += reward
        ep_len += 1

        total_reward += compute_basic_reward(
            forward_speed=forward_speed,
            crashed=bool(env.vehicle.crashed),
            on_road=bool(env.vehicle.on_road),
            config=basic_reward_config,
            lane_fraction=lane_fraction,
        )

    if is_merge_env(env_name):
        success = 0 if env.vehicle.crashed else 1
    else:
        success = 1 if ep_len == 40 else 0

    return {
        "episode_id": episode_idx,
        "seed": ep_seed,
        "success": success,
        "episode_length": ep_len,
        "episode_reward": episode_reward,
        "basic_reward": total_reward,
        "mean_speed": episode_speed / ep_len,
        "mean_lane": episode_lane / ep_len,
        "mean_courtesy_gap": (
            float(np.mean(courtesy_gaps)) if courtesy_gaps else float("nan")
        ),
        "courtesy_active_steps": len(courtesy_gaps),
        "total_norm_cost": episode_norm_cost,
    }


def _configure_agent_filter(agent, supervisor: DiscreteSupervisor, safe_decide: bool) -> None:
    if safe_decide:
        if not hasattr(agent, "planner") or not hasattr(agent.planner, "action_filter"):
            raise RuntimeError(
                "--safe-decide requires Residual MCTS (MCTSME) with action_filter support"
            )
        agent.planner.action_filter = make_permissibility_action_filter(supervisor)
    elif hasattr(agent, "planner") and hasattr(agent.planner, "action_filter"):
        agent.planner.action_filter = None


def _init_worker(
    env_name: str,
    agent_config: dict,
    safe_decide: bool,
    seed: int,
) -> None:
    """Create env/agent/supervisor once per worker process."""
    warnings.filterwarnings("ignore")
    env = gym.make(env_name)
    agent = load_agent(copy.deepcopy(agent_config), env)
    profile_name = supervisor_profile_for_env(env_name)
    supervisor = DiscreteSupervisor(
        env=env.unwrapped,
        profile_name=profile_name,
        method="nop",
        enforce_constraints=safe_decide,
        fixed_beta=None,
        kl_budget=None,
        verbose=False,
    )
    _configure_agent_filter(agent, supervisor, safe_decide)
    courtesy_norm = next(
        (n for n in supervisor.norms if str(n) == "MergeCourtesyNorm"),
        None,
    )
    _WORKER["env"] = env
    _WORKER["agent"] = agent
    _WORKER["supervisor"] = supervisor
    _WORKER["seed"] = seed
    _WORKER["env_name"] = env_name
    _WORKER["basic_reward_config"] = load_basic_reward_config(
        basic_reward_config_path_for_env(env_name)
    )
    _WORKER["courtesy_target_lane"] = (
        int(courtesy_norm.target_lane_id) if courtesy_norm is not None else 1
    )
    _WORKER["safe_decide"] = safe_decide


def _worker_run_episode(episode_idx: int) -> dict:
    """Run one episode in a worker using process-local env/agent."""
    return run_single_episode(
        agent=_WORKER["agent"],
        env=_WORKER["env"],
        supervisor=_WORKER["supervisor"],
        episode_idx=episode_idx,
        seed=_WORKER["seed"],
        basic_reward_config=_WORKER["basic_reward_config"],
        env_name=_WORKER["env_name"],
        courtesy_target_lane=_WORKER["courtesy_target_lane"],
    )


def _record_row(metrics: dict[str, list], row: dict) -> None:
    metrics["episode_ids"].append(int(row["episode_id"]))
    metrics["successes"].append(int(row["success"]))
    metrics["episode_rewards"].append(float(row["episode_reward"]))
    metrics["episode_total_rewards"].append(float(row["basic_reward"]))
    metrics["episode_lengths"].append(int(row["episode_length"]))
    metrics["episode_speeds"].append(float(row["mean_speed"]))
    metrics["episode_lanes"].append(float(row["mean_lane"]))
    metrics["episode_total_norm_costs"].append(float(row["total_norm_cost"]))
    gap = row.get("mean_courtesy_gap", float("nan"))
    metrics["episode_courtesy_gaps"].append(
        float(gap) if gap not in ("", None) else float("nan")
    )
    metrics["episode_courtesy_active_steps"].append(
        int(float(row.get("courtesy_active_steps", 0) or 0))
    )


def _print_progress(metrics: dict[str, list]) -> None:
    successes = metrics["successes"]
    episode_rewards = metrics["episode_rewards"]
    episode_total_rewards = metrics["episode_total_rewards"]
    episode_lengths = metrics["episode_lengths"]
    episode_lanes = metrics["episode_lanes"]
    episode_speeds = metrics["episode_speeds"]
    episode_total_norm_costs = metrics["episode_total_norm_costs"]
    courtesy_gaps = np.asarray(metrics["episode_courtesy_gaps"], dtype=float)

    print(f"Success rate: {100 * np.mean(successes):.2f}%")
    print(f"{len(successes)} Episodes")
    print(f"Last episode total norm cost: {episode_total_norm_costs[-1]:.2f}")
    print(
        f"Mean added reward: {np.mean(episode_rewards):.2f} "
        f"+/- {np.std(episode_rewards):.2f}"
    )
    print(
        f"Mean basic reward: {np.mean(episode_total_rewards):.2f} "
        f"+/- {np.std(episode_total_rewards):.2f}"
    )
    print(
        f"Mean episode length: {np.mean(episode_lengths):.2f} "
        f"+/- {np.std(episode_lengths):.2f}"
    )
    print(
        f"Mean lane: {np.mean(episode_lanes):.2f} "
        f"+/- {np.std(episode_lanes):.2f}"
    )
    if np.any(np.isfinite(courtesy_gaps)):
        finite = courtesy_gaps[np.isfinite(courtesy_gaps)]
        print(
            f"Mean courtesy gap: {np.mean(finite):.2f} "
            f"+/- {np.std(finite):.2f} "
            f"({len(finite)}/{len(courtesy_gaps)} episodes with active gate)"
        )
    print(
        f"Mean speed: {np.mean(episode_speeds):.2f} "
        f"+/- {np.std(episode_speeds):.2f}"
    )
    print("-----------")


def evaluation(
    env_name: str,
    agent_config: str,
    n_episodes: int,
    seed: int = BASE_SEED,
    safe_decide: bool = False,
    output_path: Path | None = None,
    resume: bool = True,
    n_workers: int = 1,
    start_episode: int = 0,
):
    """Evaluate an MCTS agent online for a fixed number of episodes.

    Env is reseeded each episode as seed + episode_index.
    When safe_decide is True, agent.planner.action_filter restricts expand/rollout
    to the supervisor's min-violation permissible set at each search state.
    When output_path is set, each finished episode is flushed to CSV immediately.
    When n_workers > 1, unfinished episodes run in a process pool.
    Episode ids run in ``[start_episode, n_episodes)``.
    """
    if start_episode < 0:
        raise ValueError(f"start_episode must be >= 0, got {start_episode}")
    if start_episode >= n_episodes:
        raise ValueError(
            f"start_episode ({start_episode}) must be < n_episodes ({n_episodes})"
        )

    completed: set[int] = set()
    if output_path is not None and resume:
        completed, metrics = load_completed_episodes(output_path)
    else:
        metrics = {
            "successes": [],
            "episode_rewards": [],
            "episode_total_rewards": [],
            "episode_lengths": [],
            "episode_speeds": [],
            "episode_lanes": [],
            "episode_total_norm_costs": [],
            "episode_ids": [],
            "episode_courtesy_gaps": [],
            "episode_courtesy_active_steps": [],
        }

    remaining = [i for i in range(start_episode, n_episodes) if i not in completed]
    if not remaining:
        print(
            f"Nothing to do: shard [{start_episode}, {n_episodes}) already saved "
            f"({len(completed)} episode(s) in CSV)."
        )
        _print_summary(metrics)
        return

    if completed:
        print(
            f"Resuming: {len(completed)} episode(s) done, "
            f"{len(remaining)} remaining -> {output_path}"
        )

    profile_name = supervisor_profile_for_env(env_name)
    print(f"Base seed = {seed}")
    print(f"Episode id range = [{start_episode}, {n_episodes})")
    print(f"Supervisor profile = {profile_name}")
    print(f"Safe decide (in-tree filter) = {safe_decide}")
    print(f"Workers = {n_workers}")
    print("DQN prior device = cpu")
    if output_path is not None:
        print(f"Writing episode results to {output_path}")

    agent_config = prepare_agent_config(agent_config)

    if n_workers <= 1:
        env = gym.make(env_name)
        agent = load_agent(copy.deepcopy(agent_config), env)
        supervisor = DiscreteSupervisor(
            env=env.unwrapped,
            profile_name=profile_name,
            method="nop",
            enforce_constraints=safe_decide,
            fixed_beta=None,
            kl_budget=None,
            verbose=False,
        )
        _configure_agent_filter(agent, supervisor, safe_decide)
        if safe_decide:
            print(
                "Safe decide: filtering MCTS prior/rollout with supervisor "
                "permissibility mask"
            )
        basic_reward_config = load_basic_reward_config(
            basic_reward_config_path_for_env(env_name)
        )
        courtesy_norm = next(
            (n for n in supervisor.norms if str(n) == "MergeCourtesyNorm"),
            None,
        )
        courtesy_target_lane = (
            int(courtesy_norm.target_lane_id) if courtesy_norm is not None else 1
        )

        for episode_idx in remaining:
            print(f"Episode {episode_idx} seed = {episode_seed(seed, episode_idx)}")
            row = run_single_episode(
                agent=agent,
                env=env,
                supervisor=supervisor,
                episode_idx=episode_idx,
                seed=seed,
                basic_reward_config=basic_reward_config,
                env_name=env_name,
                courtesy_target_lane=courtesy_target_lane,
            )
            _record_row(metrics, row)
            if output_path is not None:
                append_episode_row(output_path, row)
            _print_progress(metrics)
        env.close()
    else:
        if safe_decide:
            print(
                "Safe decide: filtering MCTS prior/rollout with supervisor "
                "permissibility mask (per worker)"
            )
        ctx = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(env_name, agent_config, safe_decide, seed),
        ) as executor:
            futures = {
                executor.submit(_worker_run_episode, episode_idx): episode_idx
                for episode_idx in remaining
            }
            for future in as_completed(futures):
                episode_idx = futures[future]
                row = future.result()
                _record_row(metrics, row)
                if output_path is not None:
                    append_episode_row(output_path, row)
                print(f"Completed episode {episode_idx} (seed={row['seed']})")
                _print_progress(metrics)

    _print_summary(metrics)


def _print_summary(metrics: dict[str, list]) -> None:
    successes = metrics.get("successes") or []
    if not successes:
        print("No completed episodes to summarize.")
        return

    episode_rewards = metrics["episode_rewards"]
    episode_lengths = metrics["episode_lengths"]
    episode_lanes = metrics["episode_lanes"]
    episode_speeds = metrics["episode_speeds"]
    episode_total_norm_costs = metrics["episode_total_norm_costs"]
    courtesy_gaps = np.asarray(
        metrics.get("episode_courtesy_gaps") or [], dtype=float
    )

    print("________________________________________________")
    print(f"Success rate: {100 * np.mean(successes):.2f}%")
    print(f"{len(successes)} Episodes")
    print(
        f"Mean reward: {np.mean(episode_rewards):.2f} "
        f"+/- {np.std(episode_rewards):.2f}"
    )
    print(
        f"Mean episode length: {np.mean(episode_lengths):.2f} "
        f"+/- {np.std(episode_lengths):.2f}"
    )
    print(
        f"Mean lane: {np.mean(episode_lanes):.2f} "
        f"+/- {np.std(episode_lanes):.2f}"
    )
    print(
        f"Mean speed: {np.mean(episode_speeds):.2f} "
        f"+/- {np.std(episode_speeds):.2f}"
    )
    if episode_total_norm_costs:
        print(
            f"Mean total norm cost: {np.mean(episode_total_norm_costs):.2f} "
            f"+/- {np.std(episode_total_norm_costs):.2f}"
        )
    if np.any(np.isfinite(courtesy_gaps)):
        finite = courtesy_gaps[np.isfinite(courtesy_gaps)]
        print(
            f"Mean courtesy gap: {np.mean(finite):.2f} "
            f"+/- {np.std(finite):.2f} "
            f"({len(finite)}/{len(courtesy_gaps)} episodes with active gate)"
        )
    print("________________________________________________")


def main():
    default_workers = default_n_workers()
    parser = argparse.ArgumentParser(
        description="Run Residual MCTS evaluation (notebook-equivalent)."
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="highway-ME-basic-AddRightReward-v0",
        help="Gym id for evaluation environment.",
    )
    parser.add_argument(
        "--agent-config",
        type=str,
        default="./configs/HighwayEnv/agents/MCTSAgent/baselineMEPreRL.json",
        help="Path to MCTS agent JSON config (as in the notebook).",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Exclusive end of the episode id range (ids are "
             "[--start-episode, --n-episodes)).",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Inclusive start of the episode id range (default 0). "
             "Use with --n-episodes to shard across machines, e.g. "
             "--start-episode 50 --n-episodes 100 runs ids 50-99.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help=f"Base random seed (default {BASE_SEED}). Each episode i uses seed + i.",
    )
    parser.add_argument(
        "--safe-decide",
        action="store_true",
        help="Filter MCTS expand/rollout actions with the supervisor permissibility "
             "mask (min-violation set), matching DiscreteSupervisor.decide hard constraints.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV path for per-episode results. Written after every episode and "
             "used for resume when the file already exists.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing --output CSV and start from episode 0 "
             "(will append duplicate episode_ids unless you delete the file).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=default_workers,
        help=(
            "Number of parallel worker processes for independent episodes "
            f"(default: {default_workers} = physical CPU cores). "
            "Use 1 for serial."
        ),
    )

    args = parser.parse_args()
    if args.n_workers < 1:
        parser.error("--n-workers must be >= 1")
    if args.start_episode < 0:
        parser.error("--start-episode must be >= 0")
    if args.start_episode >= args.n_episodes:
        parser.error("--start-episode must be < --n-episodes")

    output_path = Path(args.output).expanduser() if args.output else None

    evaluation(
        env_name=args.env_name,
        agent_config=args.agent_config,
        n_episodes=args.n_episodes,
        seed=args.seed,
        safe_decide=args.safe_decide,
        output_path=output_path,
        resume=not args.no_resume,
        n_workers=args.n_workers,
        start_episode=args.start_episode,
    )


if __name__ == "__main__":
    main()
