import copy
import importlib
import json
import logging

import gym
import numpy as np

from rl_agents.configuration import Configurable

logger = logging.getLogger(__name__)


def agent_factory(environment, config):
    """
        Handles creation of agents.

    :param environment: the environment
    :param config: configuration of the agent, must contain a '__class__' key
    :return: a new agent
    """
    if "__class__" in config:
        path = config['__class__'].split("'")[1]
        module_name, class_name = path.rsplit(".", 1)
        agent_class = getattr(importlib.import_module(module_name), class_name)
        agent = agent_class(environment, config)
        return agent
    else:
        raise ValueError("The configuration should specify the agent __class__")


def load_agent(agent_config, env):
    """
        Load an agent from a configuration file.

    :param agent_config: dict or the path to the agent configuration file
    :param env: the environment with which the agent interacts
    :return: the agent
    """
    # Load config from file
    if not isinstance(agent_config, dict):
        agent_config = load_agent_config(agent_config)
    return agent_factory(env, agent_config)


def load_agent_config(config_path):
    """
        Load an agent configuration from file, with inheritance.
    :param config_path: path to a json config file
    :return: the configuration dict
    """
    with open(config_path) as f:
        agent_config = json.loads(f.read())
    if "base_config" in agent_config:
        base_config = load_agent_config(agent_config["base_config"])
        del agent_config["base_config"]
        agent_config = Configurable.rec_update(base_config, agent_config)
    return agent_config


def load_environment(env_config):
    """
        Load an environment from a configuration file.

    :param env_config: the configuration, or path to the environment configuration file
    :return: the environment
    """
    # Load the environment config from file
    if not isinstance(env_config, dict):
        with open(env_config) as f:
            env_config = json.loads(f.read())

    # Make the environment
    if env_config.get("import_module", None):
        __import__(env_config["import_module"])
    try:
        env = gym.make(env_config['id'])
        # Save env module in order to be able to import it again
        env.import_module = env_config.get("import_module", None)
    except KeyError:
        raise ValueError("The gym register id of the environment must be provided")
    except gym.error.UnregisteredEnv:
        # The environment is unregistered.
        print("import_module", env_config["import_module"])
        raise gym.error.UnregisteredEnv('Environment {} not registered. The environment module should be specified by '
                                        'the "import_module" key of the environment configuration'.format(
                                            env_config['id']))

    # Configure the environment, if supported
    try:
        env.unwrapped.configure(env_config)
        # Reset the environment to ensure configuration is applied
        env.reset()
    except AttributeError as e:
        logger.info("This environment does not support configuration. {}".format(e))
    return env


def preprocess_env(env, preprocessor_configs):
    """
        Apply a series of pre-processes to an environment, before it is used by an agent.
    :param env: an environment
    :param preprocessor_configs: a list of preprocessor configs
    :return: a preprocessed copy of the environment
    """
    for preprocessor_config in preprocessor_configs:
        if "method" in preprocessor_config:
            try:
                preprocessor = getattr(env.unwrapped, preprocessor_config["method"])
                if "args" in preprocessor_config:
                    env = preprocessor(preprocessor_config["args"])
                else:
                    env = preprocessor()
            except AttributeError:
                logger.warning("The environment does not have a {} method".format(preprocessor_config["method"]))
        else:
            logger.error("The method is not specified in ", preprocessor_config)
    return env


def _clone_numpy_rng(rng):
    """Clone a NumPy Generator / RandomState without broken pickle/deepcopy paths."""
    if rng is None:
        return None
    if isinstance(rng, np.random.Generator):
        bit_generator = rng.bit_generator
        new_bit_generator = bit_generator.__class__()
        new_bit_generator.state = copy.deepcopy(bit_generator.state)
        # Preserve gym's RandomNumberGenerator subclass when present.
        return type(rng)(new_bit_generator)
    if isinstance(rng, np.random.RandomState):
        cloned = np.random.RandomState()
        cloned.set_state(rng.get_state())
        return cloned
    return copy.deepcopy(rng)


def _safe_deepcopy(value, memo):
    """Deep-copy a value without calling copy.deepcopy on graphs that embed RNGs.

    A failed ``copy.deepcopy`` can leave incomplete objects in ``memo``. Later
    vehicles may keep references to an unfinished ``Road`` (no ``network``).
    We therefore recurse manually for containers and ``__dict__`` objects.
    """
    try:
        value_id = id(value)
    except Exception:
        return value

    if value_id in memo:
        return memo[value_id]

    if isinstance(value, gym.Env):
        return safe_deepcopy_env(value, memo=memo)

    if isinstance(value, (np.random.Generator, np.random.RandomState)):
        cloned = _clone_numpy_rng(value)
        memo[value_id] = cloned
        return cloned

    if isinstance(value, (type(None), bool, int, float, complex, str, bytes, type)):
        return value

    if isinstance(value, np.ndarray):
        cloned = value.copy()
        memo[value_id] = cloned
        return cloned

    if isinstance(value, dict):
        cloned = {}
        memo[value_id] = cloned
        for key, item in value.items():
            cloned[_safe_deepcopy(key, memo)] = _safe_deepcopy(item, memo)
        return cloned

    if isinstance(value, list):
        cloned = []
        memo[value_id] = cloned
        cloned.extend(_safe_deepcopy(item, memo) for item in value)
        return cloned

    if isinstance(value, tuple):
        # Placeholder keeps recursive references stable for rare self-referential tuples.
        placeholder = []
        memo[value_id] = placeholder
        placeholder.extend(_safe_deepcopy(item, memo) for item in value)
        cloned = tuple(placeholder)
        memo[value_id] = cloned
        return cloned

    if isinstance(value, set):
        cloned = set()
        memo[value_id] = cloned
        for item in value:
            cloned.add(_safe_deepcopy(item, memo))
        return cloned

    if hasattr(value, "__dict__"):
        cloned = value.__class__.__new__(value.__class__)
        memo[value_id] = cloned
        for key, item in value.__dict__.items():
            setattr(cloned, key, _safe_deepcopy(item, memo))
        return cloned

    # Arrays, numbers, and other deepcopy-safe leaves.
    try:
        return copy.deepcopy(value, memo)
    except Exception:
        # Prefer a shared reference over a hard failure for exotic leaves.
        memo[value_id] = value
        return value


def safe_deepcopy_env(obj, memo=None):
    """
        Perform a deep copy of an environment but without copying its viewer.

    NumPy>=1.24 changed Generator pickling in a way that breaks copy.deepcopy on
    gym's RandomNumberGenerator; we clone object graphs manually instead.
    """
    if memo is None:
        memo = {}
    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    cls = obj.__class__
    result = cls.__new__(cls)
    memo[obj_id] = result
    for k, v in obj.__dict__.items():
        if k not in ['viewer', '_monitor', 'grid_render', 'video_recorder', '_record_video_wrapper']:
            setattr(result, k, _safe_deepcopy(v, memo))
        else:
            setattr(result, k, None)
    return result
