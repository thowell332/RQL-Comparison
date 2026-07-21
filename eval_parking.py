#!/usr/bin/env python3
"""Evaluate a trained Residual SAC model on the parking environment."""

import argparse
from pathlib import Path

import gym
import numpy as np
import highway_env  # noqa: F401  # registers parking environments
from stable_baselines3 import SAC


def evaluate_residual_sac(
    model,
    env_name: str = "parking-basic-boundaryall-v0",
    n_steps: int = None,
    n_episodes: int = None,
    seed: int = 123,
):
    """Evaluate a Residual SAC model on the specified environment.

    Args:
        model: Trained ResidualSAC model.
        env_name: Gym env id to evaluate on.
        n_steps: Max number of env steps to run (if None, use n_episodes).
        n_episodes: Number of episodes to run (if None, use n_steps).
        seed: Random seed.
    """
    # Force use of boundary-all variant so env exposes basic/add-on rewards
    env = gym.make("parking-basic-boundaryall-v0")
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)

    obs = env.reset()

    total_steps = 0
    ep_reward = 0.0
    ep_len = 0

    basic_reward = 0.0
    added_reward = 0.0
    episode_basic_rewards = []
    episode_added_rewards = []

    episode_rewards = []
    episode_lengths = []
    successes = []

    target_episodes = n_episodes if n_episodes is not None else None
    target_steps = n_steps if n_steps is not None else None

    while True:
        # Residual SAC is continuous; action is a vector
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        ep_reward += float(reward)
        ep_len += 1
        total_steps += 1

        # Decomposed rewards (if provided by env, e.g. *BoundaryCostALL variants)
        try:
            basic_reward += env.basic_reward
            added_reward += env.added_reward
        except AttributeError:
            pass

        if done:
            # success flag from parking env (_info adds "is_success")
            is_success = info.get("is_success", False)
            # Handle tuple/list/array just in case
            if isinstance(is_success, (list, tuple, np.ndarray)):
                is_success = all(is_success)
            successes.append(1 if is_success else 0)

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            episode_basic_rewards.append(basic_reward)
            episode_added_rewards.append(added_reward)

            ep_reward = 0.0
            ep_len = 0
            basic_reward = 0.0
            added_reward = 0.0
            obs = env.reset()

            if target_episodes is not None and len(episode_lengths) >= target_episodes:
                break

        if target_steps is not None and total_steps >= target_steps:
            break

    env.close()

    # Aggregate results
    success_rate = np.mean(successes) if successes else 0.0
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    std_reward = np.std(episode_rewards) if episode_rewards else 0.0
    mean_ep_len = np.mean(episode_lengths) if episode_lengths else 0.0
    std_ep_len = np.std(episode_lengths) if episode_lengths else 0.0

    print("=" * 60)
    print("PARKING SAC EVALUATION")
    print("=" * 60)
    print(f"Env: parking-basic-boundaryall-v0")
    print(f"Episodes: {len(episode_lengths)}")
    print(f"Success rate: {100 * success_rate:.2f}%")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    if episode_basic_rewards:
        mean_basic = np.mean(episode_basic_rewards)
        std_basic = np.std(episode_basic_rewards)
        print(f"Mean basic reward: {mean_basic:.2f} +/- {std_basic:.2f}")
    if episode_added_rewards:
        mean_added = np.mean(episode_added_rewards)
        std_added = np.std(episode_added_rewards)
        print(f"Mean added reward: {mean_added:.2f} +/- {std_added:.2f}")
    print(f"Mean episode length: {mean_ep_len:.2f} +/- {std_ep_len:.2f}")
    print("=" * 60)

    return {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_episode_length": mean_ep_len,
        "std_episode_length": std_ep_len,
        "mean_basic_reward": np.mean(episode_basic_rewards) if episode_basic_rewards else 0.0,
        "std_basic_reward": np.std(episode_basic_rewards) if episode_basic_rewards else 0.0,
        "mean_added_reward": np.mean(episode_added_rewards) if episode_added_rewards else 0.0,
        "std_added_reward": np.std(episode_added_rewards) if episode_added_rewards else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a SAC parking model (parking-basic-boundaryall-v0)")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model .zip file or directory containing it",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Number of environment steps to run (if set, overrides n-episodes)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Number of episodes to run (if set, takes precedence over n-steps)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)

    if model_path.is_dir():
        # Try to find a single .zip file in that directory (RL Zoo style)
        zips = list(model_path.glob("*.zip"))
        if len(zips) == 0:
            raise FileNotFoundError(f"No .zip model found in directory: {model_path}")
        if len(zips) > 1:
            print(f"Multiple .zip files found in {model_path}, using: {zips[0].name}")
        model_path = zips[0]
    elif not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    print(f"Loading SAC model from: {model_path}")

    # Env is needed for SB3 load
    load_env = gym.make("parking-basic-boundaryall-v0")
    model = SAC.load(str(model_path), env=load_env)
    print("Model loaded successfully as SAC.")

    # Default behavior if neither n_steps nor n_episodes is given
    n_steps = args.n_steps
    n_episodes = args.n_episodes
    if n_steps is None and n_episodes is None:
        n_episodes = 10

    evaluate_residual_sac(
        model,
        env_name="parking-basic-boundaryall-v0",
        n_steps=n_steps,
        n_episodes=n_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
