#!/usr/bin/env python3
"""
evaluate.py

Evaluate a trained RL trading model on test data.
"""

# Import standard libraries
import logging
import os
import sys

# Import third-party libraries
import numpy as np

# Import reinforcement learning utilities
from stable_baselines3.common.vec_env import DummyVecEnv

# Import project modules
from src.agents.rl_agent import load_agent
from src.data.data_loader import load_data
from src.envs.trading_env import TechnicalStocksEnv
from src.utils.plotting import plot_metrics, plot_trading_session


# Define evaluation configuration
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
TRAIN_SPLIT = 0.8

WINDOW_SIZE = 30
FEATURES = None

ALGORITHM = "PPO"

MODEL_PATH = "models/best/best_model"

PLOT_RESULTS = True
RESULTS_DIR = "results"


# Configure logging
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
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


# Build test environment
def build_test_environment(test_dataframe):
    return TechnicalStocksEnv(
        df=test_dataframe,
        window_size=WINDOW_SIZE,
        frame_bound=(WINDOW_SIZE, len(test_dataframe)),
        features=FEATURES,
    )


# Compute Sharpe ratio
def compute_sharpe(returns, trading_days=252):
    if returns.std() < 1e-9:
        return 0.0

    return float(np.sqrt(trading_days) * returns.mean() / returns.std())


# Compute maximum drawdown
def compute_max_drawdown(prices):
    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / (peak + 1e-9)

    return float(drawdown.max())


# Compute evaluation metrics
def compute_metrics(environment, episode_info):
    total_profit = episode_info.get("total_profit", 1.0)
    total_reward = episode_info.get("total_reward", 0.0)

    prices = environment.prices
    start_index, end_index = environment.frame_bound
    trade_prices = prices[start_index:end_index]

    log_returns = np.diff(np.log(trade_prices + 1e-9))

    sharpe_ratio = compute_sharpe(log_returns)
    max_drawdown = compute_max_drawdown(trade_prices)

    positions = [h.get("position", 0) for h in getattr(environment, "_history", [])]

    trade_count = int(
        sum(abs(positions[i] - positions[i - 1]) for i in range(1, len(positions))) // 2
    )

    return {
        "total_profit": round(total_profit, 6),
        "total_return_pct": round((total_profit - 1.0) * 100, 3),
        "total_reward": round(total_reward, 4),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 3),
        "n_trades": trade_count,
    }


# Run one evaluation episode
def run_episode(environment, model):
    observation, _ = environment.reset()

    done = False
    total_reward = 0.0
    last_info = {}

    while not done:
        action, _ = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = environment.step(action)

        done = terminated or truncated
        total_reward += float(reward)
        last_info = info

    last_info.setdefault("total_reward", total_reward)

    return compute_metrics(environment, last_info)


# Print metrics to console
def print_metrics(metrics):
    separator = "=" * 50

    print(f"\n{separator}")
    print("EVALUATION RESULTS")
    print(separator)

    for key, value in metrics.items():
        print(f"{key:<25} {value}")

    print(separator)


# Save result charts
def save_charts(environment, model, metrics):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    run_episode(environment, model)

    plot_trading_session(
        environment,
        title=f"Trading Session - {TICKER}",
        save_path=os.path.join(RESULTS_DIR, "trading_session.png"),
    )

    plot_metrics(
        metrics,
        ticker=TICKER,
        save_path=os.path.join(RESULTS_DIR, "portfolio_metrics.png"),
    )


# Evaluate model
def evaluate_model():
    logger = configure_logging()

    # Build data configuration
    data_config = build_data_config()

    # Load dataset
    logger.info("Loading test data for %s...", TICKER)
    _, test_dataframe = load_data(data_config)
    logger.info("Test split: %d rows", len(test_dataframe))

    # Build environments
    vector_env = DummyVecEnv([lambda: build_test_environment(test_dataframe)])
    environment = build_test_environment(test_dataframe)

    # Load model
    model = load_agent(MODEL_PATH, vector_env, algorithm=ALGORITHM)

    # Run evaluation episode
    logger.info("Running evaluation...")
    episode_metrics = run_episode(environment, model)

    # Print results
    print_metrics(episode_metrics)

    # Save charts
    if PLOT_RESULTS:
        save_charts(environment, model, episode_metrics)

    vector_env.close()

    return episode_metrics


# Main entry point
def main():
    evaluate_model()


# Execute script
if __name__ == "__main__":
    main()