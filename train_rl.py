import logging
import pathlib
import warnings
from typing import Any, Mapping, Optional
import zipfile

import os
import datetime
import argparse
import numpy as np

# NumPy 2.x compatibility: some libraries still expect np.bool8
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import highway_env
from sacred.observers import FileStorageObserver
from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env import VecNormalize

from imitation.data import rollout, types, wrappers
from imitation.policies import serialize
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.rewards.serialize import load_reward
from imitation.scripts.config.train_rl import train_rl_ex
from imitation.scripts.ingredients import environment
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation, rl
warnings.filterwarnings('ignore')


def is_valid_checkpoint(checkpoint_path: pathlib.Path) -> bool:
    """Check if a checkpoint file exists and is a valid zip file."""
    if not checkpoint_path.exists():
        return False
    if not checkpoint_path.is_file():
        return False
    try:
        with zipfile.ZipFile(checkpoint_path, 'r') as zip_file:
            # Try to read the zip file to verify it's valid
            zip_file.testzip()
        return True
    except (zipfile.BadZipFile, zipfile.LargeZipFile, IOError):
        return False


def find_latest_valid_checkpoint(policy_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Find the latest valid checkpoint in a policy directory.
    
    Returns the path to the latest valid checkpoint, or None if none found.
    """
    if not policy_dir.exists():
        return None
    
    # First, try numbered checkpoints (sorted by timestep, descending)
    checkpoints = sorted(
        [d for d in policy_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name),
        reverse=True
    )
    
    for checkpoint_dir in checkpoints:
        checkpoint_file = checkpoint_dir / 'model.zip'
        if is_valid_checkpoint(checkpoint_file):
            return checkpoint_file
    
    # Fall back to 'final' checkpoint
    final_checkpoint = policy_dir / 'final' / 'model.zip'
    if is_valid_checkpoint(final_checkpoint):
        return final_checkpoint
    
    return None


@train_rl_ex.main
def train_rl(
    *,
    total_timesteps: int,
    normalize_reward: bool,
    normalize_kwargs: dict,
    reward_type: Optional[str],
    reward_path: Optional[str],
    load_reward_kwargs: Optional[Mapping[str, Any]],
    rollout_save_final: bool,
    rollout_save_n_timesteps: Optional[int],
    rollout_save_n_episodes: Optional[int],
    policy_save_interval: int,
    policy_save_final: bool,
    agent_path: Optional[str],
    _rnd: np.random.Generator,
) -> Mapping[str, float]:
    """Trains an expert policy from scratch and saves the rollouts and policy.

    Checkpoints:
      At applicable training steps `step` (where step is either an integer or
      "final"):

        - Policies are saved to `{log_dir}/policies/{step}/`.
        - Rollouts are saved to `{log_dir}/rollouts/{step}.npz`.

    Args:
        total_timesteps: Number of training timesteps in `model.learn()`.
        normalize_reward: Applies normalization and clipping to the reward function by
            keeping a running average of training rewards. Note: this is may be
            redundant if using a learned reward that is already normalized.
        normalize_kwargs: kwargs for `VecNormalize`.
        reward_type: If provided, then load the serialized reward of this type,
            wrapping the environment in this reward. This is useful to test
            whether a reward model transfers. For more information, see
            `imitation.rewards.serialize.load_reward`.
        reward_path: A specifier, such as a path to a file on disk, used by
            reward_type to load the reward model. For more information, see
            `imitation.rewards.serialize.load_reward`.
        load_reward_kwargs: Additional kwargs to pass to `predict_processed`.
            Examples are 'alpha' for :class: `AddSTDRewardWrapper` and 'update_stats'
            for :class: `NormalizedRewardNet`.
        rollout_save_final: If True, then save rollouts right after training is
            finished.
        rollout_save_n_timesteps: The minimum number of timesteps saved in every
            file. Could be more than `rollout_save_n_timesteps` because
            trajectories are saved by episode rather than by transition.
            Must set exactly one of `rollout_save_n_timesteps`
            and `rollout_save_n_episodes`.
        rollout_save_n_episodes: The number of episodes saved in every
            file. Must set exactly one of `rollout_save_n_timesteps` and
            `rollout_save_n_episodes`.
        policy_save_interval: The number of training updates between in between
            intermediate rollout saves. If the argument is nonpositive, then
            don't save intermediate updates.
        policy_save_final: If True, then save the policy right after training is
            finished.
        agent_path: Path to load warm-started agent.
        _rnd: Random number generator provided by Sacred.

    Returns:
        The return value of `rollout_stats()` using the final policy.
    """
    custom_logger, log_dir = logging_ingredient.setup_logging()
    rollout_dir = log_dir / "rollouts"
    policy_dir = log_dir / "policies"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)

    post_wrappers = [lambda env, idx: wrappers.RolloutInfoWrapper(env)]
    with environment.make_venv(post_wrappers=post_wrappers) as venv:
        callback_objs = []
        if reward_type is not None:
            reward_fn = load_reward(
                reward_type,
                reward_path,
                venv,
                **load_reward_kwargs,
            )
            venv = RewardVecEnvWrapper(venv, reward_fn)
            callback_objs.append(venv.make_log_callback())
            logging.info(f"Wrapped env in reward {reward_type} from {reward_path}.")

        if normalize_reward:
            # Normalize reward. Reward scale effectively changes the learning rate,
            # so normalizing it makes training more stable. Note we do *not* normalize
            # observations here; use the `NormalizeFeaturesExtractor` instead.
            venv = VecNormalize(venv, norm_obs=False, **normalize_kwargs)
            if reward_type == "RewardNet_normalized":
                warnings.warn(
                    "Applying normalization to already normalized reward function. \
                    Consider setting normalize_reward as False",
                    RuntimeWarning,
                )

        if policy_save_interval > 0:
            save_policy_callback: callbacks.EventCallback = (
                serialize.SavePolicyCallback(policy_dir)
            )
            save_policy_callback = callbacks.EveryNTimesteps(
                policy_save_interval,
                save_policy_callback,
            )
            callback_objs.append(save_policy_callback)
        callback = callbacks.CallbackList(callback_objs)

        if agent_path is None:
            rl_algo = rl.make_rl_algo(venv)
        else:
            rl_algo = rl.load_rl_algo_from_path(agent_path=agent_path, venv=venv)
            # For off-policy algorithms, always reinitialize the replay buffer when resuming
            # This avoids corruption issues from saved checkpoints. The replay buffer will
            # fill up naturally during training (SAC handles this with learning_starts parameter)
            if hasattr(rl_algo, 'replay_buffer') and rl_algo.replay_buffer is not None:
                from stable_baselines3.common.buffers import ReplayBuffer
                # Get buffer parameters from the algorithm
                buffer_size = getattr(rl_algo, 'buffer_size', 1000000)
                optimize_memory_usage = getattr(rl_algo, 'optimize_memory_usage', False)
                replay_buffer_kwargs = getattr(rl_algo, 'replay_buffer_kwargs', {})
                
                # Always reinitialize to avoid any corruption issues
                # The buffer will fill up during training before learning starts
                rl_algo.replay_buffer = ReplayBuffer(
                    buffer_size=buffer_size,
                    observation_space=venv.observation_space,
                    action_space=venv.action_space,
                    device=rl_algo.device,
                    n_envs=venv.num_envs,
                    optimize_memory_usage=optimize_memory_usage,
                    **replay_buffer_kwargs
                )
                
                # When resuming, we need to ensure learning doesn't start until buffer has samples
                # Reset num_timesteps to 0 so learning_starts will be respected
                # This ensures the buffer fills up before training begins
                if hasattr(rl_algo, 'num_timesteps'):
                    original_timesteps = rl_algo.num_timesteps
                    rl_algo.num_timesteps = 0
                    logging.info(
                        f"Replay buffer reinitialized when resuming from checkpoint. "
                        f"Reset num_timesteps from {original_timesteps} to 0 to ensure "
                        f"buffer fills up before training starts (learning_starts={getattr(rl_algo, 'learning_starts', 'unknown')})."
                    )
                else:
                    logging.info(
                        "Replay buffer reinitialized when resuming from checkpoint. "
                        "It will fill up during training before learning starts."
                    )
        rl_algo.set_logger(custom_logger)
        rl_algo.learn(total_timesteps, callback=callback)

        # Save final artifacts after training is complete.
        if rollout_save_final:
            save_path = rollout_dir / "final.npz"
            sample_until = rollout.make_sample_until(
                rollout_save_n_timesteps,
                rollout_save_n_episodes,
            )
            types.save(
                save_path,
                rollout.rollout(rl_algo, rl_algo.get_env(), sample_until, rng=_rnd),
            )
        if policy_save_final:
            output_dir = policy_dir / "final"
            serialize.save_stable_model(output_dir, rl_algo)

        # Final evaluation of expert policy.
        return policy_evaluation.eval_policy(rl_algo, venv)


if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--env-name', type=str, default='')
    parser.add_argument('--algo', type=str, default='sac') # sac, ppo, or dqn_ME
    parser.add_argument('--rollout-save-n-episodes', type=int, default=10)
    parser.add_argument('--total-timesteps', type=int, default=int(5e5)) 
    parser.add_argument('--agent-path', type=str, default='',
                        help='Path to a saved model checkpoint to resume training from. '
                             'Can be a specific checkpoint (e.g., logs/sac/env_xxx/policies/000000100000/model.zip) '
                             'or "latest" to automatically find the latest checkpoint from the most recent run.')
    parser.add_argument('--resume-from-logdir', type=str, default='',
                        help='Resume from the latest checkpoint in the specified log directory. '
                             'Overrides --agent-path if both are provided.')
    parser.add_argument(
        '--log-dir',
        type=str,
        default='',
        help='Directory for policies/rollouts/tensorboard for this run. '
             'Defaults to logs/{algo}/{env-name}_{timestamp}/. '
             'Pass a Google Drive path on Colab so checkpoints survive disconnects.',
    )
    args = parser.parse_args()

    if args.env_name == '':
        raise ValueError('Please specify the environment.')

    # Handle resume logic
    agent_path = args.agent_path
    if args.resume_from_logdir:
        # Find the latest checkpoint in the specified log directory
        policy_dir = pathlib.Path(args.resume_from_logdir) / 'policies'
        latest_checkpoint = find_latest_valid_checkpoint(policy_dir)
        if latest_checkpoint:
            agent_path = str(latest_checkpoint)
            logging.info(f"Resuming from checkpoint: {agent_path}")
        else:
            raise ValueError(f"No valid checkpoint found in {policy_dir}")
    elif args.agent_path == 'latest':
        # Find the most recent run and use its latest checkpoint
        algo_log_dir = pathlib.Path('logs') / args.algo
        if algo_log_dir.exists():
            # Find all directories matching the env name pattern
            env_dirs = [d for d in algo_log_dir.iterdir() if d.is_dir() and args.env_name in d.name]
            if env_dirs:
                # Sort by modification time, get most recent
                latest_dir = max(env_dirs, key=lambda d: d.stat().st_mtime)
                policy_dir = latest_dir / 'policies'
                latest_checkpoint = find_latest_valid_checkpoint(policy_dir)
                if latest_checkpoint:
                    agent_path = str(latest_checkpoint)
                    logging.info(f"Resuming from latest checkpoint: {agent_path}")
                else:
                    raise ValueError(f"No valid checkpoint found in {policy_dir}")
            else:
                raise ValueError(f"No previous runs found for {args.env_name} in {algo_log_dir}")
        else:
            raise ValueError(f"Log directory not found: {algo_log_dir}")
    elif agent_path != '':
        # Validate the explicitly provided checkpoint path
        checkpoint_path = pathlib.Path(agent_path)
        if not is_valid_checkpoint(checkpoint_path):
            raise ValueError(
                f"Invalid checkpoint file: {agent_path}. "
                f"The file does not exist or is not a valid zip file. "
                f"Please check the path and ensure the checkpoint was saved correctly."
            )

    if args.log_dir:
        logdir = args.log_dir
        # If resuming into an existing Drive run dir, keep writing there.
        pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
    elif args.resume_from_logdir:
        logdir = args.resume_from_logdir
    else:
        now = datetime.datetime.now()
        timestamp = now.isoformat()
        logdir = os.path.join('logs', args.algo, args.env_name + '_{}/'.format(timestamp))

    config_updates = {
            'seed': args.seed,
            'logging.log_dir': logdir,
            'rollout_save_n_episodes': args.rollout_save_n_episodes,
        }
    if agent_path != '':
        config_updates['agent_path'] = agent_path
    if args.total_timesteps != -1:
        config_updates['total_timesteps'] = args.total_timesteps
        
    observer_path = pathlib.Path.cwd() / "logs" / "sacred" / "train_rl"
    observer = FileStorageObserver(observer_path)
    train_rl_ex.observers.append(observer)
    train_rl_ex.run(
        named_configs=[
            args.env_name, # environment named_config
            f'rl.{args.algo}', # rl algorithm named_config
            f'policy.{args.algo}' # policy named_config
        ],
        config_updates=config_updates,
    )