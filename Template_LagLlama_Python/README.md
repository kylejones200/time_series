# Lag-Llama Forecasting

Run the Lag-Llama foundation model (time-series-foundation-models/Lag-Llama) on a
univariate series.

## Installation

Lag-Llama is not published on PyPI yet. Install dependencies manually:

```bash
pip install git+https://github.com/time-series-foundation-models/lag-llama
pip install -r requirements.txt
```

Download a checkpoint from Hugging Face (for example
`time-series-foundation-models/Lag-Llama`) and update `config.yaml` with its
local path.

## Usage

```bash
python main.py
```

The template expects a CSV in `data/` (default:
`amtrak_ridership_time_series_data.csv`). Configure `context_length`,
`prediction_length`, and `checkpoint` in `config.yaml`.

Outputs (`outputs/`):

- `lag_llama_forecast.csv`
- `lag_llama_forecast.png`
- `lag_llama_metrics.yaml`

Lag-Llama runs on CPU, but using GPU is recommended for performance.
