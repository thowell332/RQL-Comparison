#!/usr/bin/env python3
"""Evaluate a trained merge model (DQN_ME or ResidualSoftDQN) with CSV resume.

Mirrors ``eval_highway.py`` for merge environments: incremental per-episode CSV,
``seed + episode_id`` resume, optional ``--safe-decide`` with the
``merge_courtesy`` supervisor profile, and mean bumper gap while the courtesy
gate is active (same definition as SCPS ``test_merge.py``).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gym
import numpy as np
import torch as th
from stable_baselines3 import DQN_ME, ResidualSoftDQN

import highway_env  # noqa: F401
from highway_env.envs.merge_courtesy import courtesy_gate_gap
from highway_env.vehicle.controller import ControlledVehicle
from supervisor import DiscreteSupervisor

from basic_reward import compute_basic_reward, load_basic_reward_config

BASE_SEED = 42
BASIC_REWARD_CONFIG = Path(__file__).parent / "configs" / "MergeEnv" / "basic_reward.json"
ENV_CONFIG = Path(__file__).parent / "configs" / "MergeEnv" / "env.json"

CSV_FIELDS = [
    "episode_id",
    "seed",
    "success",
    "episode_length",
    "episode_reward",
    "basic_reward",
    "added_reward",
    "mean_speed",
    "mean_lane",
    "mean_courtesy_gap",
    "courtesy_active_steps",
    "total_norm_cost",
    "mean_expected_norm_cost",
]


def episode_seed(base_seed: int, episode: int) -> int:
    """Deterministic per-episode seed: base_seed + episode."""
    return base_seed + episode


def _empty_metrics() -> dict:
    return {
        "successes": [],
        "episode_rewards": [],
        "episode_basic_rewards": [],
        "episode_added_rewards": [],
        "episode_lengths": [],
        "episode_speeds": [],
        "episode_lanes": [],
        "episode_total_norm_costs": [],
        "episode_mean_expected_norm_costs": [],
        "episode_courtesy_gaps": [],
        "episode_courtesy_active_steps": [],
    }


def load_completed_episodes(output_path: Path) -> tuple[int, dict]:
    """Load prior CSV rows for resume. Returns (next_episode_idx, metric lists)."""
    metrics = _empty_metrics()
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return 0, metrics

    with output_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return 0, metrics
        if "episode_id" not in reader.fieldnames:
            raise ValueError(
                f"Output CSV {output_path} is missing required column: episode_id"
            )

        max_episode_id = -1
        for row in reader:
            episode_id = int(row["episode_id"])
            max_episode_id = max(max_episode_id, episode_id)
            metrics["successes"].append(int(float(row["success"])))
            metrics["episode_rewards"].append(float(row["episode_reward"]))
            metrics["episode_basic_rewards"].append(float(row["basic_reward"]))
            metrics["episode_added_rewards"].append(float(row.get("added_reward", 0.0)))
            metrics["episode_lengths"].append(int(float(row["episode_length"])))
            metrics["episode_speeds"].append(float(row["mean_speed"]))
            metrics["episode_lanes"].append(float(row["mean_lane"]))
            metrics["episode_total_norm_costs"].append(float(row["total_norm_cost"]))
            metrics["episode_mean_expected_norm_costs"].append(
                float(row.get("mean_expected_norm_cost", 0.0))
            )
            gap_raw = row.get("mean_courtesy_gap", "")
            if gap_raw in ("", None):
                metrics["episode_courtesy_gaps"].append(float("nan"))
            else:
                metrics["episode_courtesy_gaps"].append(float(gap_raw))
            metrics["episode_courtesy_active_steps"].append(
                int(float(row.get("courtesy_active_steps", 0) or 0))
            )

    return max_episode_id + 1, metrics


def append_episode_row(output_path: Path, row: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.is_file() or output_path.stat().st_size == 0
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _summarize_and_print(metrics: dict) -> dict:
    successes = metrics["successes"]
    if not successes:
        print("No completed episodes to summarize.")
        return {}

    courtesy_gaps = np.asarray(metrics["episode_courtesy_gaps"], dtype=float)
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Success rate: {100 * np.mean(successes):.2f}%")
    print(f"Total episodes: {len(successes)}")
    print(
        f"Mean reward: {np.mean(metrics['episode_rewards']):.2f} "
        f"+/- {np.std(metrics['episode_rewards']):.2f}"
    )
    print(
        f"Mean basic reward: {np.mean(metrics['episode_basic_rewards']):.2f} "
        f"+/- {np.std(metrics['episode_basic_rewards']):.2f}"
    )
    print(
        f"Mean added reward: {np.mean(metrics['episode_added_rewards']):.2f} "
        f"+/- {np.std(metrics['episode_added_rewards']):.2f}"
    )
    print(
        f"Mean episode length: {np.mean(metrics['episode_lengths']):.2f} "
        f"+/- {np.std(metrics['episode_lengths']):.2f}"
    )
    print(
        f"Mean speed: {np.mean(metrics['episode_speeds']):.2f} "
        f"+/- {np.std(metrics['episode_speeds']):.2f}"
    )
    print(
        f"Mean lane: {np.mean(metrics['episode_lanes']):.2f} "
        f"+/- {np.std(metrics['episode_lanes']):.2f}"
    )
    if np.any(np.isfinite(courtesy_gaps)):
        finite = courtesy_gaps[np.isfinite(courtesy_gaps)]
        print(
            f"Mean courtesy gap: {np.mean(finite):.2f} "
            f"+/- {np.std(finite):.2f} "
            f"({len(finite)}/{len(courtesy_gaps)} episodes with active gate)"
        )
    print(
        f"Mean total norm cost: {np.mean(metrics['episode_total_norm_costs']):.2f} "
        f"+/- {np.std(metrics['episode_total_norm_costs']):.2f}"
    )
    print(
        f"Mean expected norm cost: "
        f"{np.mean(metrics['episode_mean_expected_norm_costs']):.2f} "
        f"+/- {np.std(metrics['episode_mean_expected_norm_costs']):.2f}"
    )
    print("=" * 60)
    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(metrics["episode_rewards"])),
        "mean_courtesy_gap": (
            float(np.nanmean(courtesy_gaps)) if np.any(np.isfinite(courtesy_gaps)) else float("nan")
        ),
    }


def evaluation(
    model,
    env_name: str = "merge-ME-basic-v0",
    n_steps: int | None = None,
    n_episodes: int | None = None,
    seed: int = BASE_SEED,
    safe_decide: bool = False,
    output_path: str | Path | None = None,
    resume: bool = True,
):
    """Evaluate a model on a merge environment with courtesy-gap metrics."""
    metrics = _empty_metrics()
    episode_idx = 0

    if output_path is not None and resume:
        episode_idx, metrics = load_completed_episodes(Path(output_path))
        if episode_idx > 0:
            print(
                f"Resuming from episode_id={episode_idx} "
                f"({episode_idx} row(s) already in {output_path})"
            )

    env = gym.make(env_name)
    if ENV_CONFIG.exists():
        with ENV_CONFIG.open("r") as handle:
            env_config = json.load(handle)
        try:
            env.unwrapped.configure(env_config)
        except AttributeError:
            for key, value in env_config.items():
                if key in getattr(env, "config", {}):
                    env.config[key] = value

    basic_reward_config = load_basic_reward_config(BASIC_REWARD_CONFIG)
    target_episodes = n_episodes
    target_steps = n_steps

    if target_episodes is not None and episode_idx >= target_episodes:
        print(
            f"Already completed {episode_idx}/{target_episodes} episodes in {output_path}"
        )
        env.close()
        return _summarize_and_print(metrics)

    model.set_random_seed(seed)

    def reset_episode():
        ep_seed = episode_seed(seed, episode_idx)
        env.seed(ep_seed)
        env.action_space.seed(ep_seed)
        return env.reset()

    obs = reset_episode()
    supervisor = DiscreteSupervisor(
        env=env.unwrapped,
        profile_name="merge_courtesy",
        method="nop",
        enforce_constraints=safe_decide,
        fixed_beta=None,
        kl_budget=None,
        verbose=False,
    )
    supervisor.reset_norms()
    courtesy_norm = next(
        (n for n in supervisor.norms if str(n) == "MergeCourtesyNorm"),
        None,
    )
    courtesy_target_lane = (
        int(courtesy_norm.target_lane_id) if courtesy_norm is not None else 1
    )

    print(f"Env = {env_name}")
    print(f"Base seed = {seed}, episode {episode_idx} seed = {episode_seed(seed, episode_idx)}")
    print(f"Safe decide (hard constraints) = {safe_decide}")
    print(f"Supervisor profile = merge_courtesy")
    if output_path is not None:
        print(f"Writing episode results to {output_path}")

    episode_lane = 0.0
    episode_speed = 0.0
    episode_reward = 0.0
    basic_reward = 0.0
    added_reward = 0.0
    episode_norm_cost = 0.0
    episode_expected_norm_costs: list[float] = []
    courtesy_gaps: list[float] = []
    ep_len = 0
    total_steps = 0

    while True:
        with th.no_grad():
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            q_values = model.q_net(obs_tensor)
            policy = th.softmax(q_values, dim=-1)[0]
            action_probs = policy.detach().cpu().numpy()

        gap = courtesy_gate_gap(
            env.unwrapped.vehicle,
            target_lane_id=courtesy_target_lane,
        )
        if gap is not None:
            courtesy_gaps.append(gap)

        if safe_decide:
            base_action = int(supervisor.decide(policy, enforce_constraints=True))
        else:
            action, _ = model.predict(obs, deterministic=True)
            base_action = int(action)

        realized_cost = supervisor.get_norm_violation_cost([base_action])[0].item()
        episode_norm_cost += realized_cost
        cost_vector = (
            supervisor.get_norm_violation_cost(supervisor.ACTIONS_ALL)
            .detach()
            .cpu()
            .numpy()
        )
        expected_cost = float(np.dot(action_probs, cost_vector))
        episode_expected_norm_costs.append(expected_cost)

        obs, reward, done, _infos = env.step(base_action)
        total_steps += 1

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

        basic_reward += compute_basic_reward(
            forward_speed=forward_speed,
            crashed=bool(env.vehicle.crashed),
            on_road=bool(env.vehicle.on_road),
            config=basic_reward_config,
            lane_fraction=lane_fraction,
        )
        try:
            added_reward += float(env.added_reward)
        except AttributeError:
            # AddCourtesy residual env exposes added_reward; ME-basic has none.
            pass

        episode_reward += reward
        ep_len += 1

        if done:
            success = 0 if env.vehicle.crashed else 1
            mean_speed = episode_speed / ep_len
            mean_lane = episode_lane / ep_len
            mean_expected = (
                float(np.mean(episode_expected_norm_costs))
                if episode_expected_norm_costs
                else 0.0
            )
            mean_gap = (
                float(np.mean(courtesy_gaps)) if courtesy_gaps else float("nan")
            )

            metrics["successes"].append(success)
            metrics["episode_speeds"].append(mean_speed)
            metrics["episode_lanes"].append(mean_lane)
            metrics["episode_rewards"].append(episode_reward)
            metrics["episode_added_rewards"].append(added_reward)
            metrics["episode_basic_rewards"].append(basic_reward)
            metrics["episode_lengths"].append(ep_len)
            metrics["episode_total_norm_costs"].append(episode_norm_cost)
            metrics["episode_mean_expected_norm_costs"].append(mean_expected)
            metrics["episode_courtesy_gaps"].append(mean_gap)
            metrics["episode_courtesy_active_steps"].append(len(courtesy_gaps))

            if output_path is not None:
                append_episode_row(
                    Path(output_path),
                    {
                        "episode_id": episode_idx,
                        "seed": episode_seed(seed, episode_idx),
                        "success": success,
                        "episode_length": ep_len,
                        "episode_reward": episode_reward,
                        "basic_reward": basic_reward,
                        "added_reward": added_reward,
                        "mean_speed": mean_speed,
                        "mean_lane": mean_lane,
                        "mean_courtesy_gap": mean_gap,
                        "courtesy_active_steps": len(courtesy_gaps),
                        "total_norm_cost": episode_norm_cost,
                        "mean_expected_norm_cost": mean_expected,
                    },
                )

            episode_reward = 0.0
            added_reward = 0.0
            basic_reward = 0.0
            ep_len = 0
            episode_speed = 0.0
            episode_lane = 0.0
            episode_norm_cost = 0.0
            episode_expected_norm_costs = []
            courtesy_gaps = []

            if target_episodes is not None and len(metrics["successes"]) >= target_episodes:
                break

            episode_idx += 1
            obs = reset_episode()
            supervisor.reset_norms()

        if target_steps is not None and total_steps >= target_steps:
            break

    env.close()
    return _summarize_and_print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained merge model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory (containing model.zip) or model.zip file",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="merge-ME-basic-v0",
        help="Environment to evaluate on",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Number of environment steps to run evaluation",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Number of episodes to run evaluation (overrides --n-steps if set)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help=f"Base random seed (default {BASE_SEED}). Each episode i uses seed + i",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "dqn_me", "residual"],
        help="Model type: auto (detect), dqn_me, or residual",
    )
    parser.add_argument(
        "--safe-decide",
        action="store_true",
        help="Apply DiscreteSupervisor.decide with hard merge-courtesy constraints",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV path for per-episode results (resume-safe).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing --output CSV and start from episode 0.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if model_path.is_dir():
        model_zip = model_path / "model.zip"
        if not model_zip.exists():
            raise FileNotFoundError(f"Could not find model.zip in {model_path}")
        model_path = model_zip
    elif not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    print(f"Loading model from: {model_path}")
    probe_env = gym.make(args.env_name)

    if args.model_type == "residual":
        model = ResidualSoftDQN.load(str(model_path), env=probe_env)
        print("Model loaded successfully as ResidualSoftDQN!")
    elif args.model_type == "dqn_me":
        model = DQN_ME.load(str(model_path), env=probe_env)
        print("Model loaded successfully as DQN_ME!")
    else:
        try:
            model = ResidualSoftDQN.load(str(model_path), env=probe_env)
            print("Model loaded successfully as ResidualSoftDQN!")
        except Exception as exc:
            print(f"Could not load as ResidualSoftDQN ({exc}); trying DQN_ME...")
            model = DQN_ME.load(str(model_path), env=probe_env)
            print("Model loaded successfully as DQN_ME!")
    probe_env.close()

    n_steps = args.n_steps if args.n_steps is not None else (
        None if args.n_episodes is not None else 400
    )

    evaluation(
        model,
        env_name=args.env_name,
        n_steps=n_steps,
        n_episodes=args.n_episodes,
        seed=args.seed,
        safe_decide=args.safe_decide,
        output_path=args.output,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
