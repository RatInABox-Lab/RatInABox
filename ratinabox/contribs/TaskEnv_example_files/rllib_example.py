import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from glob import glob

from ratinabox.Agent import Agent
from ratinabox.contribs.TaskEnvironment import (SpatialGoalEnvironment,
                                                SpatialGoal)

import ray
from ray import tune # for hyperparameter tuning
from ray import rllib # for RLlib: policy-learners and models
from ray.tune.registry import register_env

# Wrapper to transform a PettingZoo environment to an Rllib environment
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


def create_env():
    """
    Create a SpatialGoalEnvironment with two agents
    """
    goalcachekws = dict(agentmode="interact", goalorder="nonsequential",
                        reset_n_goals=5, verbose=False)
    rewardcachekws = dict(default_reward_level=-0.1)
    # Create a test environment
    env = SpatialGoalEnvironment(params={'dimensionality':'2D'},
                                 render_every=1, teleport_on_reset=False,
                                 goalcachekws=goalcachekws,
                                 rewardcachekws=rewardcachekws,
                                 verbose=False)
    goals = [SpatialGoal(env, pos=np.array([x, y]))
             for (x,y) in product((0.05, 0.5, 0.95), (0.05, 0.5, 0.95))]
    env.goal_cache.reset_goals = goals
    # Create rats who are part of the environment and accept action
    Ag = Agent(env);  env.add_agents(Ag) 
    # Ag2 = Agent(env); env.add_agents(Ag2)
    return ParallelPettingZooEnv(env) # wrapper transforms PettinZoo env 
                                      # to Rllib env
rllib.utils.check_env(create_env()) # Rllib's way to check if
                                    # we have a valid environment


if __name__ == "__main__":

    # Initialize ray (readies the system, spawn workers, etc.)
    if ray.is_initialized() == True: ray.shutdown()
    n_gpu = 0 # INFO: currently, only works with > 1 GPU
    ray.init(num_cpus=8, num_gpus=1)
    # ===============================================================
    # Register the environment with RLlib
    env_name = f"SpatialGoalEnvironment"
    register_env(env_name, lambda config: create_env())
    # ===============================================================
    # ===============================================================
    # Create a configuration dictionary
    # - This is where you can specify hyperparameters for the RL algorithm
    # - See: https://docs.ray.io/en/releases-1.11.0/rllib/rllib-training.html#common-parameters
    # - See: https://docs.ray.io/en/master/rllib-training.html#rllib-tune-config-dict
    # - Algos: https://docs.ray.io/en/releases-1.11.0/rllib/rllib-algorithms.html#ppo
    # ===============================================================
    config={ # Configuration dictionary
        "env": env_name, # Name of the environment
        "framework": "torch", # Use PyTorch
        "horizon": 9000, # Max timesteps per episode
        "batch_mode": "truncate_episodes",
        "rollout_fragment_length": 500,  # number of steps after which the partial trajectory will be used for an update
        "exploration_config": {
            # "type": "EpsilonGreedy",
            # "initial_epsilon": 0.8,
            # "final_epsilon": 0.25,
            # "epsilon_timesteps": 50000,  # Number of timesteps over which epsilon will decrease
            "type": "GaussianNoise",
            "stddev": 2.5,  # standard deviation of the Gaussian noise
            },
        "multiagent": {
            "policy_states_are_swappable":True,
            },
        "num_workers": 4,
        "num_envs_per_worker": 2,
        "num_cpus_per_worker": 1,
        "num_gpus": n_gpu,
        "monitor": True, # monitor gym environment
    }
    algo = "PPO" # Proximal Policy Optimization
    save_dir = "~/ray_rib_results/" + env_name
    tune.run(
        algo, # Set Proximal Policy Optimization as the RL algorithm
        name="PPO", # Name of the experiment
        stop={"timesteps_total": 5000000}, # Stop after 5 million steps
        checkpoint_freq=10, # Save a checkpoint every 10 iterations
        local_dir=save_dir, # Where to save results
        config=config,
        resume=False
    )
    ray.shutdown() # shutdown ray


# ===============================================================
# Examine the policy that was learned
# ===============================================================
from ray.rllib.agents.ppo import PPOTrainer
def load_trained_policy(checkpoint_path, config):
    trainer = PPOTrainer(config=config)
    trainer.restore(checkpoint_path)
    return trainer
def choose_action(trainer, observation):
    # Transform the observation if you used any preprocessing steps during training
    # transformed_obs = transform_observation(observation)  
    if isinstance(observation, dict):
        return {agent_id: choose_action(trainer, obs) for agent_id, obs in observation.items()}
    else:
        transformed_obs = observation
        # In this case, `transform_observation` should implement the same preprocessing steps 
        # (color reduction, resize, normalization, etc.) used in your training pipeline.
        policy = trainer.get_policy()
        action = policy.compute_single_action(transformed_obs)
        return action[0]
def get_newest_folder(folders):
    # Get all folders in the given directory
    # Find the newest folder by checking the time of last metadata change
    newest_folder = max(folders, key=os.path.getmtime)
    return newest_folder

# Find the latest checkpoint
file=f'{save_dir}/{algo}/{algo}_{env_name}_*/'
print("Sessions glob:", file)
matches = glob(os.path.expanduser(file))
matches = sort(matches)
print("Session matches:", matches)
folder = get_newest_folder(matches)
file = f'{folder}/checkpoint_*'
print("Checkpoint glob:", file)
matches = glob(os.path.expanduser(file))
matches = sort(matches)
print("Checkpoint matches:", matches)
checkpoint_path = matches[-1]
print("Using checkpoint:", checkpoint_path)
# ===============================================================
# Register the environment with RLlib
env_name = f"SpatialGoalEnvironment"
register_env(env_name, lambda config: create_env())
# ===============================================================

# Visualize the policy in action
env = create_env().par_env
trained_policy = load_trained_policy(checkpoint_path, config)
observations, info = env.reset()  
plt.ion(); env.render(); plt.show()
while not env._is_terminal_state():
    action = choose_action(trained_policy, observations)
    print("Action:", action)
    observations, reward, terminate_episode, _, info = \
            env.step(action)
    env.render()
    plt.pause(0.0001)
plt.close('all')

