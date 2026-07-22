#!/usr/bin/env python3
"""Evaluate a trained model on merge-ME-basic environments."""

import argparse
import json
from pathlib import Path

import gym
import numpy as np
from stable_baselines3 import DQN_ME, ResidualSoftDQN

import highway_env  # noqa: F401
from basic_reward import compute_basic_reward, load_basic_reward_config

# Match norm-supervised-highway / eval_highway.py
BASE_SEED = 42
BASIC_REWARD_CONFIG = Path(__file__).parent / "configs" / "MergeEnv" / "basic_reward.json"
ENV_CONFIG = Path(__file__).parent / "configs" / "MergeEnv" / "env.json"


def episode_seed(base_seed: int, episode: int) -> int:
    """Deterministic per-episode seed: base_seed + episode."""
    return base_seed + episode


def evaluation(
    model,
    env_name="merge-ME-basic-v0",
    n_steps=None,
    n_episodes=None,
    seed=BASE_SEED,
):
    """Evaluate a model on a merge environment with training-aligned basic reward."""
    episode_speed = 0.0
    episode_reward = 0.0
    basic_reward = 0.0
    episode_basic_rewards = []
    episode_rewards, episode_lengths, episode_speeds = [], [], []
    ep_len = 0
    successes = []

    env = gym.make(env_name)

    if ENV_CONFIG.exists():
        with ENV_CONFIG.open("r") as f:
            env_config = json.load(f)
        try:
            env.unwrapped.configure(env_config)
        except AttributeError:
            for key, value in env_config.items():
                if key in getattr(env, "config", {}):
                    env.config[key] = value

    basic_reward_config = load_basic_reward_config(BASIC_REWARD_CONFIG)

    target_episodes = n_episodes if n_episodes is not None else None
    target_steps = n_steps if n_steps is not None else None
    episode_idx = 0

    model.set_random_seed(seed)

    def reset_episode():
        ep_seed = episode_seed(seed, episode_idx)
        env.seed(ep_seed)
        env.action_space.seed(ep_seed)
        return env.reset()

    obs = reset_episode()
    print(f"Env = {env_name}")
    print(f"simulation_frequency = {env.unwrapped.config.get('simulation_frequency')}")
    print(f"collision_reward = {env.unwrapped.config.get('collision_reward')}")
    print(f"high_speed_reward = {env.unwrapped.config.get('high_speed_reward')}")
    print(f"Base seed = {seed}, episode 0 seed = {episode_seed(seed, 0)}")

    total_steps = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        base_action = int(action)

        obs, reward, done, infos = env.step(base_action)
        total_steps += 1

        forward_speed = env.vehicle.speed * np.cos(env.vehicle.heading)
        episode_speed = episode_speed + forward_speed

        neighbours = env.road.network.all_side_lanes(env.vehicle.lane_index)
        lane = env.vehicle.target_lane_index[2] if isinstance(
            env.vehicle, highway_env.vehicle.controller.ControlledVehicle
        ) else env.vehicle.lane_index[2]
        lane_fraction = lane / max(len(neighbours) - 1, 1)

        basic_reward += compute_basic_reward(
            forward_speed=forward_speed,
            crashed=bool(env.vehicle.crashed),
            on_road=bool(env.vehicle.on_road),
            config=basic_reward_config,
            lane_fraction=lane_fraction,
        )

        episode_reward += reward
        ep_len += 1

        if done:
            # Success: reached the merge exit without crashing (x > 370 terminal).
            successes.append(0 if env.vehicle.crashed else 1)

            episode_speeds.append(episode_speed / ep_len)
            episode_rewards.append(episode_reward)
            episode_basic_rewards.append(basic_reward)
            episode_lengths.append(ep_len)

            episode_reward = 0.0
            basic_reward = 0.0
            ep_len = 0
            episode_speed = 0.0

            if target_episodes is not None and len(successes) >= target_episodes:
                break

            episode_idx += 1
            obs = reset_episode()

        if target_steps is not None and total_steps >= target_steps:
            break

    env.close()

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Success rate: {100 * np.mean(successes):.2f}%")
    print(f"Total episodes: {len(successes)}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    if episode_basic_rewards and any(r != 0 for r in episode_basic_rewards):
        print(
            f"Mean basic reward: {np.mean(episode_basic_rewards):.2f} "
            f"+/- {np.std(episode_basic_rewards):.2f}"
        )
    print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")
    print(f"Mean speed: {np.mean(episode_speeds):.2f} +/- {np.std(episode_speeds):.2f}")
    print("=" * 60)

    return {
        "success_rate": np.mean(successes),
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "mean_speed": np.mean(episode_speeds),
        "mean_basic_reward": np.mean(episode_basic_rewards) if episode_basic_rewards else 0.0,
    }


if __name__ == "__main__":
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

    env = gym.make("merge-ME-basic-v0")

    if args.model_type == "residual":
        model = ResidualSoftDQN.load(str(model_path), env=env)
        print("Model loaded successfully as ResidualSoftDQN!")
    elif args.model_type == "dqn_me":
        model = DQN_ME.load(str(model_path), env=env)
        print("Model loaded successfully as DQN_ME!")
    else:
        try:
            model = ResidualSoftDQN.load(str(model_path), env=env)
            print("Model loaded successfully as ResidualSoftDQN!")
        except Exception as e:
            print("Could not load as ResidualSoftDQN, trying DQN_ME...")
            try:
                model = DQN_ME.load(str(model_path), env=env)
                print("Model loaded successfully as DQN_ME!")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model as either ResidualSoftDQN or DQN_ME. Errors: {e}, {e2}"
                )

    n_steps = args.n_steps if args.n_steps is not None else (
        None if args.n_episodes is not None else 400
    )

    evaluation(
        model,
        env_name=args.env_name,
        n_steps=n_steps,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )
