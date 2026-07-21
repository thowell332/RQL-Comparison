#!/usr/bin/env python3
"""Evaluate a trained model from train_rl.py"""

import argparse
import json
import gym
import numpy as np
from pathlib import Path
from stable_baselines3 import DQN_ME, ResidualSoftDQN
import highway_env
from supervisor import DiscreteSupervisor
import torch as th

# Match norm-supervised-highway/scripts/base_experiment.py
BASE_SEED = 239


def episode_seed(base_seed: int, episode: int) -> int:
    """Deterministic per-episode seed: base_seed + episode (independent of run length)."""
    return base_seed + episode


def evaluation(model, env_name="highway-ME-basic-AddRightRewardALL-v0", n_steps=None, n_episodes=None, seed=BASE_SEED, safe_decide=False):
    """Evaluate a model on the specified environment.
    
    Args:
        model: The trained model to evaluate
        env_name: Environment to evaluate on
        n_steps: Number of steps to run evaluation (if None, use n_episodes)
        n_episodes: Number of episodes to run evaluation (if None, use n_steps)
        seed: Base random seed (default 239, matching norm-supervised-highway BASE_SEED).
            Each episode i is reset with seed = seed + i.
        safe_decide: If True, select actions via DiscreteSupervisor.decide with
            enforce_constraints=True (hard permissibility mask) instead of raw
            model.predict.
    """
    episode_lane = 0.0
    episode_speed = 0.0
    episode_reward = 0.0
    basic_reward = 0.0
    added_reward = 0.0
    episode_added_rewards, episode_basic_rewards = [], []
    episode_rewards, episode_lengths, episode_speeds, episode_lanes = [], [], [], []
    episode_total_norm_costs = []
    episode_norm_cost = 0.0
    expected_norm_costs = []
    ep_len = 0
    successes = []
    
    env = gym.make(env_name)

    # Override environment configuration from JSON if available
    # This allows lanes_count, vehicles_count, rewards, etc. to be controlled centrally.
    env_config_path = Path(__file__).parent / "configs" / "HighwayEnv" / "env.json"
    if env_config_path.exists():
        with env_config_path.open("r") as f:
            env_config = json.load(f)
        try:
            env.unwrapped.configure(env_config)
        except AttributeError:
            # Fallback for environments that do not expose a configure method
            for key, value in env_config.items():
                if key in getattr(env, "config", {}):
                    env.config[key] = value

    # Match test_highway.py: model RNG fixed to base seed; env reseeded per episode.
    target_episodes = n_episodes if n_episodes is not None else None
    target_steps = n_steps if n_steps is not None else None
    episode_idx = 0

    model.set_random_seed(seed)

    def reset_episode():
        """Seed and reset env for the current episode (gym API; seed before reset)."""
        ep_seed = episode_seed(seed, episode_idx)
        env.seed(ep_seed)
        env.action_space.seed(ep_seed)
        return env.reset()

    obs = reset_episode()
    supervisor = DiscreteSupervisor(
        env=env.unwrapped,
        profile_name="right_lane",
        method="nop",
        enforce_constraints=safe_decide,
        fixed_beta=None,
        kl_budget=None,
        verbose=False,
    )
    supervisor.reset_norms()
    print("Configured lanes_count =", env.unwrapped.config["lanes_count"])
    print("Configured vehicles_count =", env.unwrapped.config["vehicles_count"])
    print(f"Base seed = {seed}, episode 0 seed = {episode_seed(seed, 0)}")
    print(f"Safe decide (hard constraints) = {safe_decide}")
    
    total_steps = 0
    
    # Main evaluation loop
    while True:
        # Softmax policy from Q (also used for expected-cost metrics)
        with th.no_grad():
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            q_values = model.q_net(obs_tensor)
            policy = th.softmax(q_values, dim=-1)[0]
            action_probs = policy.detach().cpu().numpy()

        if safe_decide:
            base_action = int(supervisor.decide(policy, enforce_constraints=True))
        else:
            action, _ = model.predict(obs, deterministic=True)
            base_action = int(action)

        # Compute realized norm violation cost for the chosen action
        realized_cost = supervisor.get_norm_violation_cost([base_action])[0].item()
        episode_norm_cost += realized_cost

        # Expected norm violation cost under the model's action distribution
        cost_vector = supervisor.get_norm_violation_cost(supervisor.ACTIONS_ALL).detach().cpu().numpy()
        expected_cost = float(np.dot(action_probs, cost_vector))
        expected_norm_costs.append(expected_cost)

        obs, reward, done, infos = env.step(base_action)
        total_steps += 1
        
        neighbours = env.road.network.all_side_lanes(env.vehicle.lane_index)
        lane = env.vehicle.target_lane_index[2] if isinstance(env.vehicle, highway_env.vehicle.controller.ControlledVehicle) \
            else env.vehicle.lane_index[2]
        lane = lane / max(len(neighbours) - 1, 1)
        
        episode_lane = episode_lane + lane
        
        forward_speed = env.vehicle.speed * np.cos(env.vehicle.heading)
        episode_speed = episode_speed + forward_speed
        
        # Try to get basic/added rewards if available (for AddRightRewardALL env)
        try:
            basic_reward += env.basic_reward
            added_reward += env.added_reward
        except AttributeError:
            pass
        
        episode_reward += reward
        ep_len += 1
        
        if done:
            if ep_len == 40:
                successes.append(1)
            else:
                successes.append(0)
            
            episode_speeds.append(episode_speed/ep_len) 
            episode_lanes.append(episode_lane/ep_len) 
            episode_rewards.append(episode_reward)
            episode_added_rewards.append(added_reward)
            episode_basic_rewards.append(basic_reward)
            episode_lengths.append(ep_len)
            
            episode_reward = 0.0
            added_reward = 0.0
            basic_reward = 0.0
            ep_len = 0
            episode_speed = 0.0
            episode_lane = 0.0
            episode_total_norm_costs.append(episode_norm_cost)
            episode_norm_cost = 0.0

            # Check if we've reached target episodes before starting the next one
            if target_episodes is not None and len(successes) >= target_episodes:
                break

            episode_idx += 1
            obs = reset_episode()
            supervisor.reset_norms()
        
        # Check if we've reached target steps
        if target_steps is not None and total_steps >= target_steps:
            break
    
    env.close()
    
    # Print results
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Success rate: {100 * np.mean(successes):.2f}%")
    print(f"Total episodes: {len(successes)}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    if episode_basic_rewards and any(r != 0 for r in episode_basic_rewards):
        print(f"Mean basic reward: {np.mean(episode_basic_rewards):.2f} +/- {np.std(episode_basic_rewards):.2f}")
    if episode_added_rewards and any(r != 0 for r in episode_added_rewards):
        print(f"Mean added reward: {np.mean(episode_added_rewards):.2f} +/- {np.std(episode_added_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")
    print(f"Mean lane position: {np.mean(episode_lanes):.2f} +/- {np.std(episode_lanes):.2f}")
    print(f"Mean speed: {np.mean(episode_speeds):.2f} +/- {np.std(episode_speeds):.2f}")
    if episode_total_norm_costs:
        print(
            f"Mean total norm cost: {np.mean(episode_total_norm_costs):.2f} "
            f"+/- {np.std(episode_total_norm_costs):.2f}"
        )
    if expected_norm_costs:
        print(
            f"Mean expected norm cost: {np.mean(expected_norm_costs):.2f} "
            f"+/- {np.std(expected_norm_costs):.2f}"
        )
    print("=" * 60)
    
    return {
        'success_rate': np.mean(successes),
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_speed': np.mean(episode_speeds),
        'mean_lane': np.mean(episode_lanes),
        'mean_total_norm_cost': np.mean(episode_total_norm_costs) if episode_total_norm_costs else 0.0,
        'std_total_norm_cost': np.std(episode_total_norm_costs) if episode_total_norm_costs else 0.0,
        'mean_expected_norm_cost': np.mean(expected_norm_costs) if expected_norm_costs else 0.0,
        'std_expected_norm_cost': np.std(expected_norm_costs) if expected_norm_costs else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model directory (containing model.zip) or model.zip file')
    parser.add_argument('--env-name', type=str, default='highway-ME-basic-AddRightRewardALL-v0',
                        help='Environment to evaluate on')
    parser.add_argument('--n-steps', type=int, default=None,
                        help='Number of environment steps to run evaluation (default: 400)')
    parser.add_argument('--n-episodes', type=int, default=None,
                        help='Number of episodes to run evaluation (overrides --n-steps if set)')
    parser.add_argument('--seed', type=int, default=BASE_SEED,
                        help=f'Base random seed (default {BASE_SEED}). '
                             'Each episode i uses seed + i')
    parser.add_argument('--model-type', type=str, default='auto',
                        choices=['auto', 'dqn_me', 'residual'],
                        help='Model type: auto (detect), dqn_me, or residual')
    parser.add_argument('--safe-decide', action='store_true',
                        help='Apply DiscreteSupervisor.decide with hard constraints '
                             '(permissibility mask) before env.step')
    
    args = parser.parse_args()
    
    # Determine model path
    model_path = Path(args.model_path)
    if model_path.is_dir():
        # If directory, look for model.zip inside
        model_zip = model_path / "model.zip"
        if not model_zip.exists():
            raise FileNotFoundError(f"Could not find model.zip in {model_path}")
        model_path = model_zip
    elif not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Create environment for loading (needed by SB3)
    env = gym.make("highway-ME-basic-v0")
    
    # Load model based on specified type or auto-detect
    if args.model_type == 'residual':
        model = ResidualSoftDQN.load(str(model_path), env=env)
        print(f"Model loaded successfully as ResidualSoftDQN!")
    elif args.model_type == 'dqn_me':
        model = DQN_ME.load(str(model_path), env=env)
        print(f"Model loaded successfully as DQN_ME!")
    else:  # auto-detect
        # Try to load as ResidualSoftDQN first, then fall back to DQN_ME
        try:
            model = ResidualSoftDQN.load(str(model_path), env=env)
            print(f"Model loaded successfully as ResidualSoftDQN!")
        except Exception as e:
            print(f"Could not load as ResidualSoftDQN, trying DQN_ME...")
            try:
                model = DQN_ME.load(str(model_path), env=env)
                print(f"Model loaded successfully as DQN_ME!")
            except Exception as e2:
                raise RuntimeError(f"Failed to load model as either ResidualSoftDQN or DQN_ME. Errors: {e}, {e2}")
    
    # Set default n_steps if neither n_steps nor n_episodes is provided
    n_steps = args.n_steps if args.n_steps is not None else (None if args.n_episodes is not None else 400)
    
    # Run evaluation
    results = evaluation(
        model,
        env_name=args.env_name,
        n_steps=n_steps,
        n_episodes=args.n_episodes,
        seed=args.seed,
        safe_decide=args.safe_decide,
    )
