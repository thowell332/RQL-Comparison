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
"""

import argparse
import warnings

import gym
import highway_env  # noqa: F401  # registers highway environments
import numpy as np
from highway_env import utils
from highway_env.vehicle.controller import ControlledVehicle
from rl_agents.agents.common.factory import load_agent
from supervisor import DiscreteSupervisor


# Suppress noisy warnings, as in the notebook
warnings.filterwarnings("ignore")

# Match norm-supervised-highway/scripts/base_experiment.py and eval_highway.py
BASE_SEED = 239


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


def evaluation(agent, env, n_episodes, seed=BASE_SEED, safe_decide=False):
    """Evaluate an MCTS agent online for a fixed number of episodes.

    Env is reseeded each episode as seed + episode_index.
    When safe_decide is True, agent.planner.action_filter restricts expand/rollout
    to the supervisor's min-violation permissible set at each search state.
    """
    episode_lane = 0.0
    episode_speed = 0.0
    episode_reward = 0.0
    total_reward = 0.0
    episode_rewards, episode_lengths, episode_speeds, episode_lanes = [], [], [], []
    episode_total_rewards = []
    episode_total_norm_costs = []
    episode_norm_cost = 0.0
    ep_len = 0
    successes = []
    episode_count = 0
    timestep = 0

    def reset_episode(episode_idx: int):
        """Seed and reset env for the given episode (gym API; seed before reset)."""
        ep_seed = episode_seed(seed, episode_idx)
        env.seed(ep_seed)
        env.action_space.seed(ep_seed)
        if hasattr(agent, "seed"):
            agent.seed(ep_seed)
        return env.reset()

    obs = reset_episode(0)

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

    if safe_decide:
        if not hasattr(agent, "planner") or not hasattr(agent.planner, "action_filter"):
            raise RuntimeError(
                "--safe-decide requires Residual MCTS (MCTSME) with action_filter support"
            )
        agent.planner.action_filter = make_permissibility_action_filter(supervisor)
        print("Safe decide: filtering MCTS prior/rollout with supervisor permissibility mask")
    else:
        if hasattr(agent, "planner") and hasattr(agent.planner, "action_filter"):
            agent.planner.action_filter = None

    print(f"Base seed = {seed}, episode 0 seed = {episode_seed(seed, 0)}")
    print(f"Safe decide (in-tree filter) = {safe_decide}")

    while episode_count < n_episodes:
        actions = agent.plan(obs)
        action = actions[0]
        base_action = int(action)

        # Compute realized norm violation cost for the chosen action using the supervisor cost oracle
        realized_cost = supervisor.get_norm_violation_cost([base_action])[0].item()
        episode_norm_cost += realized_cost

        obs, reward, done, infos = env.step(base_action)
        timestep += 1

        neighbours = env.road.network.all_side_lanes(env.vehicle.lane_index)
        lane = (
            env.vehicle.target_lane_index[2]
            if isinstance(env.vehicle, ControlledVehicle)
            else env.vehicle.lane_index[2]
        )
        lane = lane / max(len(neighbours) - 1, 1)

        episode_lane = episode_lane + lane

        forward_speed = env.vehicle.speed * np.cos(env.vehicle.heading)

        episode_speed = episode_speed + forward_speed

        episode_reward += reward
        ep_len += 1

        scaled_speed = utils.lmap(
            forward_speed, env.config["reward_speed_range"], [0, 1]
        )

        reward = (
            (-0.5) * env.vehicle.crashed
            + 0.0 * lane
            + 0.4 * np.clip(scaled_speed, 0, 1)
            + 1
        )
        reward = 0 if not env.vehicle.on_road else reward

        total_reward += reward

        if done:
            if ep_len == 40:
                successes.append(1)
                episode_speeds.append(episode_speed / ep_len)
                episode_lanes.append(episode_lane / ep_len)
                episode_rewards.append(episode_reward)
                episode_total_rewards.append(total_reward)
                episode_lengths.append(ep_len)
            else:
                successes.append(0)
                episode_speeds.append(episode_speed / ep_len)
                episode_lanes.append(episode_lane / ep_len)
                episode_rewards.append(episode_reward)
                episode_total_rewards.append(total_reward)
                episode_lengths.append(ep_len)

            episode_reward = 0.0
            total_reward = 0.0
            ep_len = 0
            episode_speed = 0.0
            episode_lane = 0.0
            episode_total_norm_costs.append(episode_norm_cost)
            episode_norm_cost = 0.0

            episode_count += 1

            print(f"Success rate: {100 * np.mean(successes):.2f}%")
            print(f"{len(successes)} Episodes")
            # Per-episode norm violation cost (total over the most recent episode)
            print(
                f"Last episode total norm cost: {episode_total_norm_costs[-1]:.2f}"
            )
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
            print(
                f"Mean speed: {np.mean(episode_speeds):.2f} "
                f"+/- {np.std(episode_speeds):.2f}"
            )
            print("-----------")

            if episode_count < n_episodes:
                obs = reset_episode(episode_count)
                supervisor.reset_norms()

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
    print("________________________________________________")


def main():
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
        help="Number of episodes to run online MCTS for.",
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

    args = parser.parse_args()

    # Init env (same as notebook, but configurable)
    env = gym.make(args.env_name)

    # Load MCTS agent from config, as in the notebook
    agent = load_agent(args.agent_config, env)

    # Run evaluation (reseeds env/agent per episode)
    evaluation(
        agent,
        env,
        args.n_episodes,
        seed=args.seed,
        safe_decide=args.safe_decide,
    )


if __name__ == "__main__":
    main()
