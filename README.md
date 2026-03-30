# anytrading-system

A reinforcement learning trading system built on top of **[gym-anytrading](https://github.com/AminHP/gym-anytrading)** and **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)**. The agent observes a rolling window of normalised technical indicators and learns a long / flat binary trading policy directly from price history.

---

## Table of Contents

1. [How it works](#how-it-works)
2. [Project layout](#project-layout)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Configuration](#configuration)
6. [Data pipeline](#data-pipeline)
7. [Trading environment](#trading-environment)
8. [RL agent](#rl-agent)
9. [Training](#training)
10. [Evaluation](#evaluation)
11. [Visualisation](#visualisation)
12. [TensorBoard](#tensorboard)
13. [Outputs](#outputs)
14. [Extending the system](#extending-the-system)

---

## How it works

```
Yahoo Finance
     │
     ▼
 data_loader.py          ← fetch OHLCV, compute 9 technical indicators, train/test split
     │
     ▼
 trading_env.py          ← TechnicalStocksEnv wraps StocksEnv
     │  observation = z-score normalised (window_size × n_features) array
     │  action      = 0 (flat) | 1 (long)
     │  reward      = realised P&L of the current position at each bar
     ▼
 rl_agent.py             ← Stable-Baselines3 PPO / A2C / DQN
     │  train with DummyVecEnv (parallel rollouts)
     │  EvalCallback saves best model, CheckpointCallback saves snapshots
     ▼
 evaluate.py             ← run greedy policy on held-out test data
     │  compute Sharpe ratio, max drawdown, profit multiplier
     ▼
 plotting.py             ← trading session chart + 4-panel portfolio metrics
```

The training / evaluation split is **time-ordered with no look-ahead**: the last `(1 - train_split)` fraction of bars is held out as the test set.

---

## Project layout

```
anytrading-system/
│
├── train.py             ← full training pipeline (hardcoded constants at top)
├── evaluate.py          ← backtesting + metric charts (reads config.yaml)
├── quickstart.py        ← self-contained 50k-step demo, no config required
├── requirements.txt     ← pinned Python dependencies
│
└── src/
    ├── data/
    │   └── data_loader.py     fetch_data, add_technical_indicators,
    │                          split_data, load_data
    │
    ├── envs/
    │   └── trading_env.py     TechnicalStocksEnv, zscore_clip, make_env
    │
    ├── agents/
    │   └── rl_agent.py        build_agent, load_agent, build_callbacks
    │
    └── utils/
        └── plotting.py        plot_trading_session, plot_reward_curve,
                               plot_metrics, plot_metric_bar, save_or_show
```

---

## Installation

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

Core dependencies:

| Package | Role |
|---------|------|
| `gym-anytrading` | Gymnasium-compatible stock trading environment |
| `stable-baselines3[extra]` | PPO, A2C, DQN + TensorBoard + progress bar |
| `shimmy` | Gymnasium / Gym compatibility shim required by SB3 |
| `yfinance` | Free OHLCV data from Yahoo Finance |
| `ta` | Technical analysis indicators (RSI, MACD, Bollinger Bands) |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` | Charting |
| `PyYAML` | Config file parsing (used by `evaluate.py`) |

---

## Quick start

```bash
# Run the self-contained demo — AAPL, PPO, 50 000 steps, ~2 minutes
python quickstart.py
```

This single command:

1. Downloads AAPL daily bars 2020–2024 from Yahoo Finance.
2. Computes all 9 technical indicators.
3. Splits 80 / 20 into train and test sets.
4. Builds a `TechnicalStocksEnv` wrapped in `Monitor`.
5. Trains a PPO agent for 50 000 timesteps and saves `models/quickstart_model.zip`.
6. Runs the greedy policy on the test set and prints total reward and profit multiplier.
7. Saves `results/quickstart_session.png` — price chart with buy/sell markers.

For a full config-driven run:

```bash
python train.py      # trains and saves to models/
python evaluate.py   # evaluates models/best/best_model and saves charts to results/
```

---

## Configuration

`train.py` is **hardcoded** — edit the constants at the top of the file directly:

```python
# train.py — top-level constants
TICKER          = "AAPL"
START_DATE      = "2020-01-01"
END_DATE        = "2024-01-01"
TRAIN_SPLIT     = 0.8
WINDOW_SIZE     = 30
FEATURES        = None          # None = all 9 default features
ALGORITHM       = "PPO"         # "PPO" | "A2C" | "DQN"
LEARNING_RATE   = 3e-4
TOTAL_TIMESTEPS = 500_000
N_ENVS          = 4             # set to 1 when using DQN
EVAL_FREQ       = 10_000
N_EVAL_EPISODES = 5
LOG_PATH        = "logs"
SAVE_PATH       = "models"
```

`evaluate.py` reads `config.yaml`. The evaluation section controls which model is loaded and whether charts are generated:

```yaml
evaluation:
  model_path: "models/best/best_model"
  plot: true
  results_path: "results/"
```

All other `config.yaml` sections (`data`, `environment`, `agent`, `training`) are used by `evaluate.py` to reconstruct the identical data pipeline and environment that was used during training.

---

## Data pipeline

Located in `src/data/data_loader.py`.

### `fetch_data(ticker, start_date, end_date, interval="1d")`

Downloads OHLCV bars from Yahoo Finance via `yfinance`. Automatically flattens any `MultiIndex` columns produced by newer yfinance versions. Raises `ValueError` if the response is empty.

### `add_technical_indicators(dataframe)`

Appends 9 feature columns to the raw OHLCV frame:

| Column | Calculation | Library |
|--------|-------------|---------|
| `close_pct` | `Close.pct_change()` | pandas |
| `high_low_pct` | `(High - Low) / Close` | pandas |
| `rsi` | 14-period RSI | `ta.RSIIndicator` |
| `macd` | MACD line (12/26 EMA diff) | `ta.MACD` |
| `macd_signal` | 9-period EMA of MACD | `ta.MACD` |
| `macd_diff` | MACD − signal (histogram) | `ta.MACD` |
| `bb_pct` | Bollinger %B (20-period, 2σ) | `ta.BollingerBands` |
| `bb_width` | (upper − lower) / middle | `ta.BollingerBands` |
| `volume_norm` | Volume / 20-bar rolling mean | pandas |

All rows containing `NaN` (the warm-up period of the longest indicator) are dropped after computation, so the first usable bar is approximately bar 26 (MACD slow window).

### `split_data(dataframe, train_split=0.8)`

Performs a **chronological** split — no shuffling. Both halves are returned with the index reset so positional indexing works correctly inside the environment.

### `load_data(config)`

Convenience wrapper that chains `fetch_data → add_technical_indicators → split_data` using values from the config dict.

---

## Trading environment

Located in `src/envs/trading_env.py`.

### `TechnicalStocksEnv`

Subclasses `gym_anytrading.envs.StocksEnv` and overrides `_process_data` to replace the default raw-price observation with normalised technical features.

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | OHLCV + indicator DataFrame (output of `add_technical_indicators`) |
| `window_size` | `int` | Number of past bars included in each observation |
| `frame_bound` | `(int, int)` | `(start, end)` indices into `df` the episode trades over; `start >= window_size` |
| `features` | `list[str]` or `None` | Indicator columns to use; `None` defaults to all 9 |
| `render_mode` | `str` or `None` | Passed through to the base `TradingEnv` |

**Observation space:**

Shape `(window_size, n_features)` — a 2-D float32 array. Every column is independently z-score normalised across the full DataFrame and then clipped to `[-5, 5]` (see `zscore_clip`). This prevents extreme outlier values from dominating the neural network input.

**Action space:**

| Action | Meaning |
|--------|---------|
| `0` | Flat / short — close any open long position |
| `1` | Long — open or hold a long position |

**Reward:**

The reward at each timestep is the realised profit of the current position since the last action, as implemented by `gym-anytrading`. A long position earns positive reward when price rises and negative reward when it falls. A flat position earns zero reward.

**`_process_data` flow:**

1. Extract the `Close` column as the internal `prices` array (used for reward calculation — not exposed to the agent).
2. For each enabled feature column, extract the raw NumPy array and pass it through `zscore_clip`.
3. Stack all normalised arrays into a `(n_bars, n_features)` matrix returned as `signal_features`.

### `zscore_clip(array, clip_value=5.0)`

Replaces `NaN` / `inf` with 0, computes mean and standard deviation, normalises to zero mean / unit variance (skipped if `std < 1e-8` to avoid division by zero), then clips to `[-clip_value, clip_value]`.

### `make_env(df, window_size, frame_bound, features=None)`

Returns a zero-argument callable suitable for passing directly to `DummyVecEnv([make_env(...)] * n)`.

---

## RL agent

Located in `src/agents/rl_agent.py`.

### Supported algorithms

| Name | Class | Notes |
|------|-------|-------|
| `PPO` | `stable_baselines3.PPO` | Default. Works with multiple parallel envs. |
| `A2C` | `stable_baselines3.A2C` | Synchronous advantage actor-critic. Multiple envs supported. |
| `DQN` | `stable_baselines3.DQN` | Off-policy. Must use `N_ENVS = 1` (replay buffer incompatible with vectorised envs). |

### `build_agent(environment, config)`

Constructs the SB3 model from a flat config dict. Shared parameters (`learning_rate`, `gamma`, `verbose`, `tensorboard_log`) are always set. Algorithm-specific parameters are conditionally added:

- **PPO / A2C:** `n_steps`, `ent_coef`, `gae_lambda`
- **PPO only:** `batch_size`, `n_epochs`, `clip_range`
- **DQN only:** `batch_size`, `buffer_size`, `learning_starts`

### `load_agent(model_path, environment, algorithm="PPO")`

Loads a saved `.zip` checkpoint and binds it to the provided environment. The `algorithm` argument must match the class used during training.

### `build_callbacks(config, evaluation_environment)`

Returns a `CallbackList` containing:

- **`EvalCallback`** — runs `n_eval_episodes` deterministic episodes every `eval_freq` steps, saves the best model to `models/best/`, logs evaluation metrics to `logs/eval/`.
- **`CheckpointCallback`** — saves a model snapshot every `eval_freq` steps to `models/checkpoints/` with the prefix `trading_model`.

---

## Training

Located in `train.py`.

The full training pipeline runs in a single `train_model()` call:

1. **`create_output_directories()`** — creates `logs/`, `models/`, and `logs/monitor/`.
2. **`load_data(data_config)`** — downloads and prepares `train_dataframe` and `test_dataframe`.
3. **`DummyVecEnv`** — wraps `N_ENVS` copies of `TechnicalStocksEnv` (each backed by `train_dataframe`). Each env is also wrapped in `Monitor` to record episode rewards and lengths to `logs/monitor/`.
4. **Single eval env** — one unwrapped `TechnicalStocksEnv` backed by `test_dataframe` for the `EvalCallback`.
5. **`build_agent`** — constructs the SB3 model with the resolved hyperparameters.
6. **`build_callbacks`** — attaches `EvalCallback` + `CheckpointCallback`.
7. **`model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)`** — runs training. The progress bar requires `tqdm` (included in the `stable-baselines3[extra]` install).
8. **`model.save(final_model_path)`** — saves the final model regardless of best-eval performance.

To change the ticker, algorithm, or any hyperparameter, edit the constants block at the top of `train.py`.

---

## Evaluation

Located in `evaluate.py`.

The evaluation pipeline is controlled entirely by `config.yaml`.

### Metrics computed

| Metric | Description |
|--------|-------------|
| `total_profit` | Portfolio value multiplier at end of episode (1.0 = break-even) |
| `total_return_pct` | `(total_profit − 1) × 100` |
| `total_reward` | Sum of step rewards over the episode |
| `sharpe_ratio` | Annualised Sharpe from log-returns of the price series, assuming 252 trading days and risk-free rate = 0 |
| `max_drawdown_pct` | Peak-to-trough decline in the price series during the episode, expressed as a percentage |
| `n_trades` | Number of completed round-trip position flips |

### `run_episode(environment, model)`

Resets the environment and steps through the full episode using the **greedy (deterministic) policy**. Accumulates `total_reward` and captures the final `info` dict returned by the environment, which contains `total_profit` from gym-anytrading.

### Charts produced

When `evaluation.plot: true` in `config.yaml`:

- `results/trading_session.png` — full price chart with coloured position regions and buy/sell arrows rendered by `env.render_all()`.
- `results/portfolio_metrics.png` — 2×2 bar chart grid showing total return, Sharpe ratio, max drawdown, and profit multiplier.

---

## Visualisation

Located in `src/utils/plotting.py`.

### `plot_trading_session(environment, title, save_path)`

Calls `environment.render_all()` (provided by gym-anytrading) which draws the full price series with coloured background segments indicating the agent's position and arrows marking entries and exits.

### `plot_reward_curve(rewards, title, save_path)`

Plots per-episode scalar rewards with a translucent raw series and a solid rolling-average overlay. The rolling window is `max(10, len(rewards) // 20)`.

### `plot_metrics(metrics, ticker, save_path)`

4-panel summary figure:

| Panel | Colour logic |
|-------|-------------|
| Total Return (%) | Green if ≥ 0, red otherwise |
| Sharpe Ratio | Green if > 1, orange if > 0, red if ≤ 0; dashed target line at 1.0 |
| Max Drawdown (%) | Always crimson (loss metric) |
| Profit Multiplier | Green if ≥ 1.0, red otherwise |

### `save_or_show(save_path)`

Central exit point for all plot functions. If `save_path` is provided the figure is saved at 150 dpi and the path is printed to stdout; otherwise `plt.show()` is called. `plt.close()` is always called to prevent figure accumulation in long-running sessions.

---

## TensorBoard

Training metrics (`ep_rew_mean`, `ep_len_mean`, policy losses, value losses, entropy) are written to `logs/` in real time.

```bash
tensorboard --logdir logs/
```

Evaluation metrics (`eval/mean_reward`, `eval/mean_ep_length`) are written separately to `logs/eval/` by `EvalCallback`.

---

## Outputs

| Path | Created by | Content |
|------|-----------|---------|
| `models/best/best_model.zip` | `EvalCallback` | Highest eval-reward checkpoint |
| `models/checkpoints/trading_model_<N>_steps.zip` | `CheckpointCallback` | Snapshot every `EVAL_FREQ` steps |
| `models/final_model.zip` | `train.py` | Model weights at the end of training |
| `models/quickstart_model.zip` | `quickstart.py` | Quickstart demo checkpoint |
| `logs/monitor/*.csv` | `Monitor` wrapper | Per-episode reward and length during training |
| `logs/eval/` | `EvalCallback` | Evaluation episode statistics |
| `logs/PPO_*/` | SB3 TensorBoard writer | Policy & value loss curves |
| `results/trading_session.png` | `evaluate.py` | Price chart with position overlays |
| `results/portfolio_metrics.png` | `evaluate.py` | 4-panel performance summary |
| `results/quickstart_session.png` | `quickstart.py` | Quickstart trading chart |
| `training.log` | `train.py` | Full timestamped log from training run |

---

## Extending the system

**Different ticker or date range** — change `TICKER`, `START_DATE`, `END_DATE` in `train.py`, or the `data` section in `config.yaml`.

**Different features** — add indicator columns in `add_technical_indicators` in `src/data/data_loader.py`, then include the column names in the `FEATURES` list in `train.py` (or `environment.features` in `config.yaml`).

**Different algorithm** — change `ALGORITHM = "A2C"` or `"DQN"` in `train.py`. For DQN also set `N_ENVS = 1`.

**Custom reward shaping** — subclass `TechnicalStocksEnv` and override `_calculate_reward`.

**Intraday data** — set `interval` to `"1h"` or `"15m"` in the data config and increase `window_size` accordingly.