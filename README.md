# Hi there 👋

[![Google Scholar](https://img.shields.io/badge/Google-Scholar-4285F4?logo=google-scholar&logoColor=white)](https://scholar.google.com.hk/citations?user=KnRThI0AAAAJ&hl=zh-CN)

🎓 **Time Series Researcher** | SEU (Southeast University)

- 🔬 Research: Time Series Forecasting, Generation, and Imputation
- 🛠️ Building: TSLib - A unified toolkit for time series analysis
- 📫 Contact: [Google Scholar](https://scholar.google.com.hk/citations?user=KnRThI0AAAAJ&hl=zh-CN)

## 🔧 TSLib Tool Suite

A collection of standardized Python libraries for time series analysis and forecasting:

| Package | Description | Install |
|---------|-------------|---------|
| **[ts_data](https://github.com/zysongg/ts_data)** | Time series data loading and processing | `pip install git+https://github.com/zysongg/ts_data.git` |
| **[ts_metric](https://github.com/zysongg/ts_metric)** | Metrics computation for prediction, imputation, and generation | `pip install git+https://github.com/zysongg/ts_metric.git` |
| **[ts_trainer](https://github.com/zysongg/ts_trainer)** | Standardized training framework based on PyTorch Lightning | `pip install git+https://github.com/zysongg/ts_trainer.git` |

### 📂 Datasets

- **[TimeSeries Dataset](https://huggingface.co/datasets/zysoong/TimeSeries)** - Preprocessed time series datasets for benchmarking
- **[ConTSG-Bench Dataset](https://huggingface.co/datasets/mldi-lab/ConTSG-Bench-Datasets)** - Time series generation benchmark dataset
- **[ConTSG-Bench Leaderboard](https://huggingface.co/spaces/mldi-lab/ConTSG-Bench-Leaderboard)** - Interactive benchmark leaderboard

### Quick Start

```python
# Data loading
from ts_data import DataModule
dm = DataModule.from_csv("data.csv", task="forecast")

# Metrics
from ts_metric import compute_metrics
results = compute_metrics(y_true, y_pred, task="forecast")

# Training
from ts_trainer import Trainer, TrainerConfig
config = TrainerConfig(max_epochs=100, logger="tensorboard")
trainer = Trainer(config)
trainer.fit(model, datamodule=dm)
```

### Features

- 🎯 **6 Task Types**: Forecast, Imputation, Generation, Conditional Generation, Classification, Anomaly Detection
- 📊 **Standardized API**: Consistent interfaces across all packages
- ⚡ **PyTorch Lightning**: Scalable training with multi-stage support
- 📦 **Easy Integration**: Compatible with ConTSG-Bench and other frameworks

---

![GitHub Stats](https://github-readme-stats.vercel.app/api?username=zysongg&show_icons=true&theme=radical)
