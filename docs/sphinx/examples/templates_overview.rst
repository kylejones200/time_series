Templates Overview
==================

The library includes 48 production-ready forecasting templates, each implementing a specific method.

Classical Methods
-----------------

- **ARIMA** - Autoregressive Integrated Moving Average
- **ARAR** - Autoregressive Autoregressive
- **BoxJenkins** - Systematic Box-Jenkins methodology
- **MovingAverage** - Simple and exponential moving averages
- **ExponentialSmoothing** - Holt-Winters exponential smoothing
- **VAR** - Vector Autoregression
- **Differencing** - Stationarity transformations
- **Kalman** - Kalman filters

Bayesian & Statistical
------------------------

- **Bayesian** - PyMC Bayesian time series modeling
- **BayesianChangePoint** - Bayesian change point detection
- **Orbit** - Bayesian structural time series
- **PyBSTS** - Bayesian structural time series
- **ConfidenceIntervals** - Confidence interval estimation
- **SerialCorrelation** - Serial correlation analysis

Modern Forecasting Libraries
-----------------------------

- **Prophet** - Facebook Prophet
- **Darts** - Darts forecasting library
- **StatsForecast** - Nixtla StatsForecast
- **Greykite** - LinkedIn Greykite
- **Merlion** - Merlion forecasting & anomaly detection
- **Autogluon** - AutoGluon time series
- **PyCaret** - PyCaret low-code forecasting

Deep Learning
--------------

- **LSTM** - Long Short-Term Memory networks
- **NBEATS** - N-BEATS neural forecasting
- **TSAI** - TSAI deep learning
- **BERT** - BERT for time series

Foundation Models
-----------------

- **Chronos** - Amazon Chronos
- **TimesFM** - Google TimesFM
- **LagLlama** - Granite TTM (LagLlama)
- **Sundial** - Moirai forecasting

Specialized Methods
-------------------

- **TSFresh** - Automated feature extraction
- **tslearn** - Time series machine learning
- **Aeon** - Aeon time series toolkit
- **STUMPY_PyOD** - Matrix profile & anomaly detection
- **CCM** - Convergent Cross Mapping
- **Copula** - Copula methods
- **RegimeSwitching** - Markov switching models
- **TransferEntropy** - Information-theoretic causality
- **Volatility** - ARCH/GARCH models

Using Templates
---------------

Each template follows the same structure:

1. Navigate to the template directory
2. Edit ``config.yaml`` to specify your data
3. Run ``python main.py``
4. Review results in ``outputs/``

For detailed usage, see the README in each template directory.

