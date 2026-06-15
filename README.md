# Hi there 👋

**Zhangyao Song (宋章耀)** | PhD Student @ Southeast University (东南大学)

[![Google Scholar](https://img.shields.io/badge/Google-Scholar-4285F4?logo=google-scholar&logoColor=white)](https://scholar.google.com.hk/citations?user=KnRThI0AAAAJ&hl=zh-CN)

🎓 **Time Series Researcher** | SEU (Southeast University)

- 🔬 Research: Time Series Forecasting, Generation, and Imputation
- 🛠️ Building: TSLib - A unified toolkit for time series analysis
- 📫 Contact: [Google Scholar](https://scholar.google.com.hk/citations?user=KnRThI0AAAAJ&hl=zh-CN)

## TSLib Tool Suite

A collection of standardized Python libraries for time series analysis and forecasting.
Five composable libraries covering the full experiment lifecycle: data, models, training, evaluation, and orchestration.

```
ts_data  ------>  ts_model  ------>  ts_trainer  ------>  ts_pipeline
 (data)            (models)           (training)           (orchestration)
                        \                                      /
                         \---------->  ts_metric  <----------/
                                       (evaluation)
```

| Package | Description | Install |
|---------|-------------|---------|
| **[ts_data](https://github.com/zysongg/ts_data)** | Data loading, splitting, Dataset, DataModule | `pip install git+https://github.com/zysongg/ts_data.git` |
| **[ts_model](https://github.com/zysongg/ts-model)** | Model registry with unified `create_model` API (DLinear, NsDiff, K2VAE, CycleFlow, CSDI, ...) | `pip install git+https://github.com/zysongg/ts-model.git` |
| **[ts_trainer](https://github.com/zysongg/ts_trainer)** | PyTorch Lightning training, multi-stage pipelines, checkpoint management | `pip install git+https://github.com/zysongg/ts_trainer.git` |
| **[ts_metric](https://github.com/zysongg/ts_metric)** | Metrics (CRPS, MSE, MAE, ...), plotting, statistical tests, result export | `pip install git+https://github.com/zysongg/ts_metric.git` |
| **[ts_pipeline](https://github.com/zysongg/ts_pipeline)** | Experiment orchestration: YAML config, CLI, multi-run comparison | `pip install git+https://github.com/zysongg/ts_pipeline.git` |

### Datasets

- **[TimeSeries Dataset](https://huggingface.co/datasets/zysoong/TimeSeries)** - Preprocessed time series datasets for benchmarking
- **[ConTSG-Bench Dataset](https://huggingface.co/datasets/mldi-lab/ConTSG-Bench-Datasets)** - Time series generation benchmark dataset
- **[ConTSG-Bench Leaderboard](https://huggingface.co/spaces/mldi-lab/ConTSG-Bench-Leaderboard)** - Interactive benchmark leaderboard

### Features

- 6 Task Types: Forecast, Imputation, Generation, Conditional Generation, Classification, Anomaly Detection
- Standardized API: Consistent interfaces across all packages
- PyTorch Lightning: Scalable training with multi-stage support
- Fast Pipeline: YAML-driven experiments with CLI and auto-comparison
- Probabilistic Models: NsDiff, K2VAE, CycleFlow, CSDI, TimeGrad, DiffusionTS
- Easy Integration: Compatible with ConTSG-Bench and other frameworks

---

## Quick Start

### Install all five libraries

```bash
pip install git+https://github.com/zysongg/ts_data.git
pip install git+https://github.com/zysongg/ts-model.git
pip install git+https://github.com/zysongg/ts_trainer.git
pip install git+https://github.com/zysongg/ts_metric.git
pip install git+https://github.com/zysongg/ts_pipeline.git
```

### Method 1: Fast Pipeline (recommended)

Use `ts_pipeline` to run a full experiment from a single YAML config. This is the fastest way to get results.

**1. Create a config file** (`etth1_nsdiff.yaml`):

```yaml
run_name: etth1_NsDiff
output_root: ./output
seed: 2026

data:
  dataset_name: etth1
  data_path: ./data/ETTh1.csv
  task: forecast
  lookback: 96
  horizon: 96
  batch_size: 32

model:
  name: NsDiff
  task: forecasting
  mode: probabilistic
  params:
    num_samples: 100

train:
  epochs: 30
  lr: 0.001
  patience: 5
  accelerator: gpu

eval:
  num_samples: 100
  metrics: [CRPS, CRPS_sum, MSE_median, MAE_median]
  plot_cases: 3
```

**2. Run the full pipeline** (train + evaluate + save artifacts):

```bash
ts-pipeline run etth1_nsdiff.yaml
```

Or from Python:

```python
from ts_pipeline import run_pipeline
run_pipeline("etth1_nsdiff.yaml")
```

**3. Re-evaluate with more samples**:

```bash
ts-pipeline eval ./output/etth1_NsDiff --num-samples 200
```

**4. Compare multiple models**:

```bash
ts-pipeline compare ./output/etth1_NsDiff ./output/etth1_K2VAE ./output/etth1_CycleFlow \
  --metrics CRPS CRPS_sum MSE_median MAE_median
```

This produces CSV, JSON, and a bar chart PNG in a `comparison/` directory.

### Method 2: Compose libraries directly

For more control, use the individual libraries in Python:

```python
# 1. Data: load and split
from ts_data import DataModule
dm = DataModule("./data/ETTh1.csv", split_mode="standard", dataset_name="etth1")
train_ds = dm.create_dataset("train", "forecast", input_len=96, pred_len=96)
test_ds = dm.create_dataset("test", "forecast", input_len=96, pred_len=96)

# 2. Model: create from registry
from ts_model import create_model
model = create_model("NsDiff", task="forecasting",
    num_features=7, lookback_len=96, horizon_len=96, num_samples=100)

# 3. Train: use ts_trainer with Lightning wrapper
from ts_trainer import Trainer
module = Trainer.create_module(model, task="forecast", mode="prob", lr=1e-3)
trainer = Trainer(max_epochs=30, save_dir="./output", run_name="etth1_NsDiff",
    dataset_name="etth1", model_name="NsDiff", accelerator="gpu")
trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

# 4. Evaluate: get metrics
results = trainer.evaluate(module, dataloaders=test_loader,
    metrics=["CRPS", "CRPS_sum", "MSE_median", "MAE_median"],
    task="prediction", mode="probabilistic")

# 5. Compare: aggregate and plot
from ts_metric import MetricCalculator, plot_crps_comparison
```

### Method 3: CycleFlow two-stage pipeline

CycleFlow requires a two-stage training (CycleD pretrain then CycleFlow):

```yaml
# etth1_cycleflow.yaml
model:
  name: CycleFlow
  task: forecasting
  mode: probabilistic

cycleflow:
  cycled_epochs: 20
  cycled_lr: 0.001
  cycleflow_epochs: 20
  cycleflow_lr: 0.0001
  train_num_samples: 10
  cycled_params:
    cycle: 24
```

```bash
ts-pipeline run etth1_cycleflow.yaml
```

The pipeline automatically handles CycleD pretraining, saves flow weights, and continues with CycleFlow training.

---

## Library Details

### ts_data

Handles data loading (CSV/NPY), train/val/test splitting, and task-specific dataset creation.

```python
from ts_data import DataModule, ForecastDataset, load_data

dm = DataModule("data.csv", split_mode="standard", scale=True)
train_ds = dm.create_dataset("train", "forecast", input_len=96, pred_len=96)
```

### ts_model

Unified model registry. All models use `create_model(name, task, **params)`.

```python
from ts_model import create_model, list_models

print(list_models())  # ['DLinear', 'CycleNet', 'NsDiff', 'K2VAE', 'CycleFlow', ...]
model = create_model("K2VAE", task="forecasting", num_features=7,
    lookback_len=96, horizon_len=96, num_samples=100)
```

### ts_trainer

PyTorch Lightning-based training with multi-stage support.

```python
from ts_trainer import Trainer, TrainerConfig

# Simple single-stage training
module = Trainer.create_module(model, task="forecast", mode="prob")
trainer = Trainer(max_epochs=30, save_dir="./output", run_name="exp1",
    dataset_name="etth1", model_name="NsDiff", early_stopping_patience=5)
trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Multi-stage pipeline (e.g., CycleD -> CycleFlow)
from ts_trainer import build_cycleflow_pipeline
stages = build_cycleflow_pipeline(...)
trainer.fit_pipeline(stages)
```

### ts_metric

Metrics, plotting, and statistical tests.

```python
import ts_metric as tm

# Compute metrics
crps = tm.prediction.crps(target, samples)
mse = tm.prediction.mse(target, forecast)

# Plot
tm.plot_prob_prediction(target, samples, inputs)
tm.plot_crps_comparison(results_dict)

# Export
df = tm.to_dataframe(results)
tm.to_csv(results, "results.csv")

# Statistical test
dm_stat = tm.diebold_mariano(target, forecast_a, forecast_b)
```

### ts_pipeline

Experiment orchestration layer. Reads YAML config, wires all libraries together, manages output directories.

```bash
# Run experiment
ts-pipeline run configs/etth1_nsdiff.yaml

# Re-evaluate checkpoint with different sample count
ts-pipeline eval ./output/exp_name --num-samples 200

# Compare multiple runs
ts-pipeline compare ./output/exp1 ./output/exp2 ./output/exp3
```

---

## Output Directory Structure

Each experiment run produces a standardized directory:

```
output/{run_name}/
    config.yaml              # Experiment configuration
    manifest.json            # Run metadata
    ckpt/
        best-val.ckpt        # Best validation checkpoint
        last.ckpt            # Last epoch checkpoint
    artifacts/
        cycled_flow.pt       # (CycleFlow only) pretrained flow weights
    metrics/
        evaluate_metrics.json
    results/
        test_results.npz     # predictions, targets, samples
    plots/
        *.png                # prediction visualizations
```

---

## Supported Models

| Model | Type | Task |
|-------|------|------|
| DLinear | Point | Forecasting |
| CycleNet | Point | Forecasting |
| Autoformer | Point | Forecasting |
| NsDiff | Probabilistic | Forecasting |
| K2VAE | Probabilistic | Forecasting |
| CycleFlow | Probabilistic | Forecasting (two-stage) |
| CSDI | Probabilistic | Forecasting / Imputation |
| TimeGrad | Probabilistic | Forecasting |
| DiffusionTS | Probabilistic | Forecasting |
| TimePrism | Point | Forecasting |
| TSFlow | Probabilistic | Forecasting |

---

![GitHub Stats](https://github-readme-stats.vercel.app/api?username=zysongg&show_icons=true&theme=radical)
