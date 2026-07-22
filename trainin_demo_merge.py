"""Train a DQN_ME prior on merge-ME-basic-v0.

Mirrors trainin_demo_highway.py for the merge basic-reward setting.
Hyperparameters match highway-ME-basic-v0 / logs/dqn_ME/highway_basic_*.

For the Sacred train_rl pipeline (same path as the highway experiment logs), use:
    python train_rl.py --env-name merge_basic --algo dqn_ME --rollout-save-n-episodes 1000
"""

import argparse
from pathlib import Path

import gym
import numpy as np
from stable_baselines3 import DQN_ME

import highway_env  # noqa: F401

from basic_reward import compute_basic_reward, load_basic_reward_config

# Set True to skip training and load an existing checkpoint.
USE_PRETRAINED = False
PRETRAINED_PATH = "./logs/merge-ME-basic-v0_Example.zip"
SAVE_PATH = "./logs/merge-ME-basic-v0_Example"
TOTAL_TIMESTEPS = int(5e5)

BASIC_REWARD_CONFIG = Path(__file__).parent / "configs" / "MergeEnv" / "basic_reward.json"


def evaluation(model, n_steps=400, seed=42):
    """Quick smoke-eval on merge-ME-basic-v0 after training."""
    episode_speed = 0.0
    episode_reward = 0.0
    basic_reward = 0.0
    episode_basic_rewards = []
    episode_rewards, episode_lengths, episode_speeds = [], [], []
    ep_len = 0
    successes = []
    episode_idx = 0

    env = gym.make("merge-ME-basic-v0")
    basic_reward_config = load_basic_reward_config(BASIC_REWARD_CONFIG)
    model.set_random_seed(seed)

    def reset_episode():
        ep_seed = seed + episode_idx
        env.seed(ep_seed)
        env.action_space.seed(ep_seed)
        return env.reset()

    obs = reset_episode()
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(int(action))

        forward_speed = env.vehicle.speed * np.cos(env.vehicle.heading)
        episode_speed += forward_speed

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
            successes.append(0 if env.vehicle.crashed else 1)
            episode_speeds.append(episode_speed / ep_len)
            episode_rewards.append(episode_reward)
            episode_basic_rewards.append(basic_reward)
            episode_lengths.append(ep_len)

            episode_reward = 0.0
            basic_reward = 0.0
            ep_len = 0
            episode_speed = 0.0
            episode_idx += 1
            obs = reset_episode()

    env.close()
    print(f"Success rate: {100 * np.mean(successes):.2f}%")
    print(f"{len(successes)} Episodes")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(
        f"Mean basic reward: {np.mean(episode_basic_rewards):.2f} "
        f"+/- {np.std(episode_basic_rewards):.2f}"
    )
    print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")
    print(f"Mean speed: {np.mean(episode_speeds):.2f} +/- {np.std(episode_speeds):.2f}")
    print("________________________________________________")


def train_dqn_me(total_timesteps: int, save_path: str, seed: int = 1) -> DQN_ME:
    """Train DQN_ME on merge-ME-basic-v0 with highway_basic-matched hyperparameters."""
    env = gym.make("merge-ME-basic-v0")
    print("[Merge] Training new DQN_ME model from scratch on merge-ME-basic-v0...")
    model = DQN_ME(
        env=env,
        policy="MlpPolicy",
        batch_size=32,
        buffer_size=15000,
        learning_starts=200,
        learning_rate=1e-4,
        gamma=0.8,
        target_update_interval=50,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=0.7,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=seed,
        verbose=1,
    )
    model.learn(total_timesteps)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"[Merge] Saved model to {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN_ME on merge-ME-basic-v0")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help=f"Training timesteps (default {TOTAL_TIMESTEPS}, same as highway dqn_ME)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=SAVE_PATH,
        help="Where to save the trained model",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=PRETRAINED_PATH,
        help="Checkpoint to load when --use-pretrained is set",
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        default=USE_PRETRAINED,
        help="Load an existing checkpoint instead of training",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run a short evaluation after train/load",
    )
    args = parser.parse_args()

    if args.use_pretrained:
        print("[Merge] Loading pretrained DQN_ME from", args.pretrained_path)
        env = gym.make("merge-ME-basic-v0")
        model = DQN_ME.load(args.pretrained_path, env=env)
    else:
        model = train_dqn_me(
            total_timesteps=args.total_timesteps,
            save_path=args.save_path,
            seed=args.seed,
        )

    if args.eval:
        evaluation(model, seed=args.seed)
