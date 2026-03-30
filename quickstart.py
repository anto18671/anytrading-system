#!/usr/bin/env python3
"""
quickstart.py

Minimal end-to-end demo.

Downloads AAPL data, trains a PPO agent, evaluates it on the held-out test
set, and shows the trading chart.

Run:
    python quickstart.py
"""

# Import standard libraries
import os

# Import plotting
import matplotlib.pyplot as plt

# Import reinforcement learning components
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Import project data utilities
from src.data.data_loader import add_technical_indicators, fetch_data, split_data

# Import trading environment
from src.envs.trading_env import TechnicalStocksEnv


# Define configuration constants
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
WINDOW_SIZE = 20
TOTAL_TIMESTEPS = 50_000
MODEL_DIR = "models"
RESULTS_DIR = "results"
MODEL_NAME = "quickstart_model"
RESULTS_PATH = os.path.join(RESULTS_DIR, "quickstart_session.png")


# Create output directories
def create_output_directories():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


# Load and prepare the dataset
def load_and_prepare_data():
    print("⏳ Downloading data...")

    # Download raw market data
    dataframe = fetch_data(TICKER, START_DATE, END_DATE)

    # Add technical indicators
    dataframe = add_technical_indicators(dataframe)

    # Split into train and test sets
    train_dataframe, test_dataframe = split_data(dataframe, train_split=0.8)

    # Print dataset sizes
    print(f"Train: {len(train_dataframe)} rows")
    print(f"Test : {len(test_dataframe)} rows")

    return train_dataframe, test_dataframe


# Build the training environment
def build_training_environment(train_dataframe):
    # Create the technical stocks environment
    environment = TechnicalStocksEnv(
        df=train_dataframe,
        window_size=WINDOW_SIZE,
        frame_bound=(WINDOW_SIZE, len(train_dataframe)),
    )

    # Wrap the environment with a monitor
    environment = Monitor(environment)

    return environment


# Build the test environment
def build_test_environment(test_dataframe):
    # Create the technical stocks environment
    environment = TechnicalStocksEnv(
        df=test_dataframe,
        window_size=WINDOW_SIZE,
        frame_bound=(WINDOW_SIZE, len(test_dataframe)),
    )

    return environment


# Train the PPO model
def train_model(train_environment):
    print(f"\n🤖 Training PPO for {TOTAL_TIMESTEPS:,} timesteps...")

    # Create the PPO model
    model = PPO(
        "MlpPolicy",
        train_environment,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        ent_coef=0.01,
        verbose=0,
    )

    # Train the model
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    # Save the trained model
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save(model_path)

    print(f"Model saved to {model_path}.zip")

    return model


# Evaluate the model on the test environment
def evaluate_model(model, test_environment):
    print("\n📊 Evaluating on test data...")

    # Reset the environment
    observation, _ = test_environment.reset()
    done = False
    total_reward = 0.0
    last_info = {}

    # Run inference until the episode is finished
    while not done:
        # Predict the next action
        action, _ = model.predict(observation, deterministic=True)

        # Step through the environment
        observation, reward, terminated, truncated, info = test_environment.step(action)

        # Update loop state
        done = terminated or truncated
        total_reward += float(reward)
        last_info = info

    # Extract the profit multiplier
    total_profit = last_info.get("total_profit", 1.0)

    print(f"Total reward : {total_reward:.4f}")
    print(f"Profit mult. : {total_profit:.4f}  ({(total_profit - 1) * 100:.2f} %)")

    return total_reward, total_profit


# Render and save the trading session chart
def render_results(test_environment):
    print("\n📈 Rendering trading session...")

    # Render the full trading session
    test_environment.render_all()

    # Configure the chart
    plt.title(f"Quickstart - {TICKER} (PPO, {TOTAL_TIMESTEPS:,} steps)")
    plt.tight_layout()

    # Save the chart
    plt.savefig(RESULTS_PATH, dpi=150, bbox_inches="tight")

    # Show the chart
    plt.show()

    print(f"Chart saved to {RESULTS_PATH}")


# Run the full quickstart pipeline
def main():
    # Prepare output folders
    create_output_directories()

    # Load the dataset
    train_dataframe, test_dataframe = load_and_prepare_data()

    # Build environments
    train_environment = build_training_environment(train_dataframe)
    test_environment = build_test_environment(test_dataframe)

    # Train the model
    model = train_model(train_environment)

    # Evaluate the model
    evaluate_model(model, test_environment)

    # Render the results
    render_results(test_environment)


# Start the script
if __name__ == "__main__":
    main()