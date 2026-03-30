#!/usr/bin/env python3
"""
train.py

Main training entry point for the RL trading system.
"""

# Import standard libraries
import logging
import os
import sys

# Import reinforcement learning utilities
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Import project modules
from src.agents.rl_agent import build_agent, build_callbacks
from src.data.data_loader import load_data
from src.envs.trading_env import TechnicalStocksEnv


# Define training configuration
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
TRAIN_SPLIT = 0.8

WINDOW_SIZE = 30
FEATURES = None

ALGORITHM = "PPO"
LEARNING_RATE = 3e-4

TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
VERBOSE = 1
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5

LOG_PATH = "logs"
SAVE_PATH = "models"
MONITOR_DIR = os.path.join(LOG_PATH, "monitor")
TRAINING_LOG_PATH = "training.log"


# Configure logging
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(TRAINING_LOG_PATH, encoding="utf-8"),
        ],
    )

    return logging.getLogger(__name__)


# Build data configuration
def build_data_config():
    return {
        "ticker": TICKER,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "train_split": TRAIN_SPLIT,
    }


# Build environment configuration
def build_environment_config():
    return {
        "window_size": WINDOW_SIZE,
        "features": FEATURES,
    }


# Build agent configuration
def build_agent_config():
    return {
        "algorithm": ALGORITHM,
        "learning_rate": LEARNING_RATE,
        "log_path": LOG_PATH,
        "save_path": SAVE_PATH,
        "verbose": VERBOSE,
        "eval_freq": EVAL_FREQ,
        "n_eval_episodes": N_EVAL_EPISODES,
    }


# Create output directories
def create_output_directories():
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(MONITOR_DIR, exist_ok=True)


# Create environment factory
def make_environment_factory(dataframe, environment_config, monitor_directory=None):
    window_size = environment_config["window_size"]
    frame_bound = (window_size, len(dataframe))
    features = environment_config["features"]

    def initialize_environment():
        environment = TechnicalStocksEnv(
            df=dataframe,
            window_size=window_size,
            frame_bound=frame_bound,
            features=features,
        )

        if monitor_directory is not None:
            environment = Monitor(environment, monitor_directory)

        return environment

    return initialize_environment


# Train the model
def train_model():
    logger = configure_logging()

    # Create folders
    create_output_directories()

    # Build configurations
    data_config = build_data_config()
    environment_config = build_environment_config()
    agent_config = build_agent_config()

    # Load data
    logger.info("Loading data for %s...", data_config["ticker"])
    train_dataframe, test_dataframe = load_data(data_config)
    logger.info("train=%d rows  test=%d rows", len(train_dataframe), len(test_dataframe))

    # Resolve environment count
    algorithm = agent_config["algorithm"].upper()

    if algorithm == "DQN":
        environment_count = 1
    else:
        environment_count = N_ENVS

    # Build environments
    train_environment = DummyVecEnv(
        [make_environment_factory(train_dataframe, environment_config, MONITOR_DIR)] * environment_count
    )

    eval_environment = DummyVecEnv(
        [make_environment_factory(test_dataframe, environment_config)]
    )

    # Build agent
    logger.info("Building %s agent...", algorithm)
    model = build_agent(train_environment, agent_config)

    callbacks = build_callbacks(agent_config, eval_environment)

    # Train model
    logger.info("Training for %d timesteps...", TOTAL_TIMESTEPS)
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    # Save model
    final_model_path = os.path.join(SAVE_PATH, "final_model")
    model.save(final_model_path)
    logger.info("Final model saved to %s", final_model_path)

    # Close environments
    train_environment.close()
    eval_environment.close()

    return model, train_dataframe, test_dataframe, environment_config


# Main entry point
def main():
    train_model()


# Execute script
if __name__ == "__main__":
    main()