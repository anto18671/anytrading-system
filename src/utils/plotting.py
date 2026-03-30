"""
Visualisation helpers for the trading system.
"""

# Import standard libraries
import os

# Import typing
from typing import Dict, List, Optional

# Import third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Render trading session with buy/sell markers
def plot_trading_session(environment, title="Trading Session", save_path=None):
    # Create figure
    figure, axes = plt.subplots(1, 1, figsize=(16, 6))

    # Render environment
    environment.render_all()

    # Configure plot
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save or show
    save_or_show(save_path)


# Plot reward curve with rolling average
def plot_reward_curve(rewards, title="Episode Reward Curve", save_path=None):
    # Create figure
    plt.figure(figsize=(12, 5))

    # Plot raw rewards
    plt.plot(rewards, alpha=0.4, label="Episode reward")

    # Compute rolling window
    window_size = max(10, len(rewards) // 20)

    # Plot rolling average
    if len(rewards) >= window_size:
        rolling_average = pd.Series(rewards).rolling(window_size).mean()
        plt.plot(rolling_average, linewidth=2, label=f"Rolling avg ({window_size})")

    # Configure plot
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save or show
    save_or_show(save_path)


# Plot portfolio metrics summary
def plot_metrics(metrics, ticker="", save_path=None):
    # Create figure
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Build title
    if ticker:
        title = f"Portfolio Performance - {ticker}"
    else:
        title = "Portfolio Performance"

    figure.suptitle(title, fontsize=14, fontweight="bold")

    # Extract metrics
    total_return = metrics.get("total_return_pct", 0.0)
    sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
    max_drawdown = metrics.get("max_drawdown_pct", 0.0)
    total_profit = metrics.get("total_profit", 1.0)

    # Plot total return
    plot_metric_bar(
        axes[0, 0],
        label="Total Return (%)",
        value=total_return,
        ylabel="%",
    )

    # Plot sharpe ratio
    plot_metric_bar(
        axes[0, 1],
        label="Sharpe Ratio",
        value=sharpe_ratio,
        color=get_sharpe_color(sharpe_ratio),
        reference_line=1.0,
    )

    # Plot drawdown
    plot_metric_bar(
        axes[1, 0],
        label="Max Drawdown (%)",
        value=-abs(max_drawdown),
        color="red",
        ylabel="%",
    )

    # Plot profit multiplier
    plot_metric_bar(
        axes[1, 1],
        label="Profit Multiplier",
        value=total_profit,
        color="green" if total_profit >= 1.0 else "red",
    )

    # Final layout
    plt.tight_layout()

    # Save or show
    save_or_show(save_path)


# Plot single metric bar
def plot_metric_bar(axis, label, value, color=None, reference_line=None, ylabel=""):
    # Resolve color
    if color is None:
        if value >= 0:
            bar_color = "green"
        else:
            bar_color = "red"
    else:
        bar_color = color

    # Draw bar
    axis.bar([label], [value], color=bar_color, width=0.4)

    # Draw reference line
    if reference_line is not None:
        axis.axhline(reference_line, linestyle="--", alpha=0.7)
        axis.legend([f"Target ({reference_line})"], fontsize=9)

    # Configure axis
    axis.set_ylabel(ylabel, fontsize=10)
    axis.set_title(label, fontsize=11)
    axis.grid(True, alpha=0.3)

    # Add value text
    if value >= 0:
        vertical_alignment = "bottom"
    else:
        vertical_alignment = "top"

    axis.text(
        0,
        value,
        f"  {value:.3f}",
        va=vertical_alignment,
        fontsize=10,
        fontweight="bold",
    )


# Determine color for sharpe ratio
def get_sharpe_color(sharpe_value):
    if sharpe_value > 1:
        return "green"

    if sharpe_value > 0:
        return "orange"

    return "red"


# Save or display plot
def save_or_show(save_path):
    if save_path:
        directory = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(directory, exist_ok=True)

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved -> {save_path}")
    else:
        plt.show()

    plt.close()