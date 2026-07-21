import gym
import numpy as np
from stable_baselines3 import DQN_ME, ResidualSoftDQN
import highway_env

# Configuration: use pretrained models or train from scratch
# Set this flag to True to skip training and just load existing checkpoints.
USE_PRETRAINED = True

# Paths to pretrained models (update these to match your saved checkpoints)
# For example, if you followed the original notebook, these might be:
#   ./logs/highway-ME-basic-v0_Example.zip
#   ./logs/highway-ME-basic-AddRightReward-v0_Example.zip
PRETRAINED_PRIOR_PATH = "./logs/highway-ME-basic-v0_Example.zip"
PRETRAINED_CUSTOM_PATH = "./logs/highway-ME-basic-AddRightReward-v0_Example.zip"

def evaluation(model):
    episode_lane = 0.0
    episode_speed = 0.0
    episode_reward = 0.0
    basic_reward = 0.0
    added_reward = 0.0
    episode_added_rewards,episode_basic_rewards = [],[]
    episode_rewards, episode_lengths, episode_speeds, episode_lanes = [], [], [], []
    ep_len = 0
    successes = []
    env = gym.make("highway-ME-basic-AddRightRewardALL-v0")
    obs = env.reset()
    for _ in range(400):
            action, lstm_states = model.predict(
                obs,  
                deterministic = True,
            )
            obs, reward, done, infos = env.step(int (action))
            
            neighbours = env.road.network.all_side_lanes(env.vehicle.lane_index)
            lane = env.vehicle.target_lane_index[2] if isinstance(env.vehicle, highway_env.vehicle.controller.ControlledVehicle) \
                else env.vehicle.lane_index[2]
            lane = lane / max(len(neighbours) - 1, 1)
            
            episode_lane = episode_lane + lane
            
            forward_speed = env.vehicle.speed * np.cos(env.vehicle.heading)
            
            episode_speed = episode_speed + forward_speed
            
            basic_reward +=  env.basic_reward
            added_reward +=  env.added_reward
            episode_reward += reward
            ep_len += 1
            
            if done:
                if ep_len == 40:
                    successes.append(1)
                    episode_speeds.append(episode_speed/ep_len) 
                    episode_lanes.append(episode_lane/ep_len) 
                    episode_rewards.append(episode_reward)
                    episode_added_rewards.append(added_reward)
                    episode_basic_rewards.append(basic_reward)
                    episode_lengths.append(ep_len)

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
                obs = env.reset()
                
    print(f"Success rate: {100 * np.mean(successes):.2f}%")
    print(f"{len(successes)} Episodes")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean basic reward: {np.mean(episode_basic_rewards):.2f} +/- {np.std(episode_basic_rewards):.2f}")
    print(f"Mean added reward: {np.mean(episode_added_rewards):.2f} +/- {np.std(episode_added_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")
    print(f"Mean lane: {np.mean(episode_lanes):.2f} +/- {np.std(episode_lanes):.2f}")
    print(f"Mean speed: {np.mean(episode_speeds):.2f} +/- {np.std(episode_speeds):.2f}")
    print("________________________________________________")

if __name__ == "__main__":
     # Init env
    # Init RL prior model (DQN_ME on 'highway-ME-basic-v0')
    if USE_PRETRAINED:
        print("[Prior] Loading pretrained DQN_ME from", PRETRAINED_PRIOR_PATH)
        env = gym.make("highway-ME-basic-v0")
        model = DQN_ME.load(PRETRAINED_PRIOR_PATH, env=env)
    else:
        env = gym.make("highway-ME-basic-v0")
        print("[Prior] Training new DQN_ME model from scratch...")
        # Init soft Q model
        model = DQN_ME(
            env=env,
            policy='MlpPolicy',
            batch_size=32,
            buffer_size=15000,
            learning_rate=1e-4,
            gamma=0.8,
            target_update_interval=50,
            train_freq=1,
            gradient_steps=1,
            exploration_fraction=0.7,
            policy_kwargs=dict(net_arch=[256, 256])
        )

        # Training. For better performance, we recomand train 5e5 steps 
        model.learn(int(1e5))

        # Save model
        model.save(f"./logs/highway-ME-basic-v0_Example")