"""
RL agent factory and callback builders.
"""

# Import standard libraries
import logging
import os

# Import typing
from typing import Any, Dict

# Import reinforcement learning algorithms
from stable_baselines3 import A2C, DQN, PPO

# Import base classes and callbacks
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv


# Initialize logger
logger = logging.getLogger(__name__)


# Define supported algorithms
ALGORITHMS: Dict[str, type] = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}


# Build reinforcement learning agent
def build_agent(environment: VecEnv, config):
    # Resolve algorithm
    algorithm_name = config.get("algorithm", "PPO").upper()

    algorithm_class = ALGORITHMS.get(algorithm_name)

    # Validate algorithm
    if algorithm_class is None:
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'. Available: {list(ALGORITHMS)}"
        )

    # Resolve policy and logging
    policy_name = config.get("policy", "MlpPolicy")
    log_path = config.get("log_path", "logs")

    # Base parameters
    parameters: Dict[str, Any] = {
        "learning_rate": config.get("learning_rate", 3e-4),
        "gamma": config.get("gamma", 0.99),
        "verbose": config.get("verbose", 1),
        "tensorboard_log": log_path,
    }

    # Add on-policy parameters
    if algorithm_name in ("PPO", "A2C"):
        parameters["n_steps"] = config.get("n_steps", 2048)
        parameters["ent_coef"] = config.get("ent_coef", 0.01)
        parameters["gae_lambda"] = config.get("gae_lambda", 0.95)

    # Add PPO-specific parameters
    if algorithm_name == "PPO":
        parameters["batch_size"] = config.get("batch_size", 64)
        parameters["n_epochs"] = config.get("n_epochs", 10)
        parameters["clip_range"] = config.get("clip_range", 0.2)

    # Add DQN-specific parameters
    if algorithm_name == "DQN":
        parameters["batch_size"] = config.get("batch_size", 64)
        parameters["buffer_size"] = config.get("buffer_size", 100_000)
        parameters["learning_starts"] = config.get("learning_starts", 1_000)

    # Log configuration
    logger.info("Building %s agent policy=%s", algorithm_name, policy_name)

    # Create model
    model = algorithm_class(policy_name, environment, **parameters)

    return model


# Load trained model
def load_agent(model_path, environment, algorithm="PPO"):
    # Resolve algorithm class
    algorithm_class = ALGORITHMS.get(algorithm.upper(), PPO)

    # Log loading
    logger.info("Loading %s model from %s", algorithm, model_path)

    # Load model
    model = algorithm_class.load(model_path, env=environment)

    return model


# Build training callbacks
def build_callbacks(config, evaluation_environment):
    # Resolve paths and frequency
    save_path = config.get("save_path", "models")
    log_path = config.get("log_path", "logs")
    evaluation_frequency = config.get("eval_freq", 10_000)
    evaluation_episodes = config.get("n_eval_episodes", 5)
    verbosity = config.get("verbose", 1)

    # Create directories
    os.makedirs(save_path, exist_ok=True)

    # Create evaluation callback
    evaluation_callback = EvalCallback(
        evaluation_environment,
        best_model_save_path=os.path.join(save_path, "best"),
        log_path=os.path.join(log_path, "eval"),
        eval_freq=evaluation_frequency,
        n_eval_episodes=evaluation_episodes,
        deterministic=True,
        render=False,
        verbose=verbosity,
    )

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=evaluation_frequency,
        save_path=os.path.join(save_path, "checkpoints"),
        name_prefix="trading_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Combine callbacks
    callback_list = CallbackList([evaluation_callback, checkpoint_callback])

    return callback_list