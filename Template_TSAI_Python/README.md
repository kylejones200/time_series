# TSAI: Time Series AI

Deep learning for time series using TSAI library.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path and column names
- **Model**: Model type (InceptionTime, ResNet, XceptionTime, ROCKET), training parameters
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Model Types

- **InceptionTime**: Inception-based architecture
- **ResNet**: Residual networks
- **XceptionTime**: Xception architecture
- **ROCKET**: Random Convolutional Kernel Transform

## Outputs

Forecast plots saved to `outputs/` directory.

