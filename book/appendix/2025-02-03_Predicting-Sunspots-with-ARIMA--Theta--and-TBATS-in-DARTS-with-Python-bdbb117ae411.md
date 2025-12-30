<div>

# Predicting Sunspots with ARIMA, Theta, and TBATS in DARTS with Python {#predicting-sunspots-with-arima-theta-and-tbats-in-darts-with-python .p-name}

</div>

::: {.section .p-summary field="subtitle"}
Using DARTS to forecast solar cycles
:::

::::::: {.section .e-content field="body"}
:::::: {#6472 .section .section .section--body .section--first .section--last}
::: section-divider

------------------------------------------------------------------------
:::

:::: section-content
::: {.section-inner .sectionLayout--insetColumn}
### Predicting Sunspots with ARIMA, Theta, and TBATS in DARTS with Python {#8071 .graf .graf--h3 .graf--leading .graf--title name="8071"}

#### Using DARTS to forecast solar cycles {#a378 .graf .graf--h4 .graf-after--h3 .graf--subtitle name="a378"}

Sunspot observations are one of science's longest continuous datasets
and a good case study for time series analysis. Let's explore how modern
forecasting techniques can help predict future solar activity.

The sunspot dataset, maintained by the [WDC-SILSO, Royal Observatory of
Belgium, Brussels](https://www.sidc.be/SILSO/datafiles){.markup--anchor
.markup--p-anchor data-href="https://www.sidc.be/SILSO/datafiles"
rel="noopener" target="_blank"}, contains monthly observations since
1749.

<figure id="28f8" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*aO_YmzPUDBOosmocSf6BLQ.gif"
class="graf-image" data-image-id="1*aO_YmzPUDBOosmocSf6BLQ.gif"
data-width="1500" data-height="700" data-is-featured="true" />
<figcaption>This isn an earlier version of the project. I was
experimenting with smoothing options.</figcaption>
</figure>

In our analysis, we transform this monthly data into yearly averages to
better observe the Sun's long-term patterns, particularly the roughly
11-year solar cycle first discovered by Heinrich Schwabe in 1843. This
dataset has been extensively studied by researchers worldwide, from
Daines Analytics' SARIMA modeling to Python's Gurus' work with the Darts
library. Our analysis builds upon this foundation.

Our analysis implements multiple forecasting models:

``` {#b1a1 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ARIMA, Theta, TBATS
from darts.metrics import mape, rmse
from darts.utils.utils import SeasonalityMode
import warnings

warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    def __init__(self):
        self.models = {
            'ARIMA': ARIMA(p=2, d=1, q=2, seasonal_order=(1, 1, 1, 11)),
            'Theta': Theta(season_mode=SeasonalityMode.ADDITIVE, seasonality_period=11),
            'TBATS': TBATS(use_trend=True, use_box_cox=False, seasonal_periods=[11])
        }

    def _load_csv_data(self, filepath):
        """Load and prepare sunspot data."""
        try:
            # Read the CSV file with the new format
            df = pd.read_csv(filepath)
            
            # Split the Month column into Year and Month
            df[['Year', 'Month']] = df['Month'].str.split('-', expand=True)
            
            # Convert to proper datetime
            df['Date'] = pd.to_datetime(df['Year'] + '-' + df['Month'] + '-01')
            
            # Replace 0 values with 1 to avoid log transform issues
            df['Sunspot'] = np.where(df['Sunspot'] == 0, 1, df['Sunspot'])
            
            # Convert to yearly averages
            df_yearly = df.groupby(df['Date'].dt.year)['Sunspot'].mean().reset_index()
            df_yearly['Date'] = pd.to_datetime(df_yearly['Date'].astype(str) + '-01-01')
            
            # Create TimeSeries object
            series = TimeSeries.from_dataframe(df_yearly, 'Date', 'Sunspot')
            
            return series
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None

    def load_data(self, filepath=None):
        return self._load_csv_data(filepath)

    def analyze(self, series, train_test_split=0.8):
        """Train models and evaluate predictions"""
        if series is None:
            print("No data to analyze")
            return None, None, None

        train_size = int(len(series) * train_test_split)
        train = series[:train_size]
        test = series[train_size:]
        
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(train)
                pred = model.predict(len(test))
                metrics = self._calculate_metrics(test, pred)
                results[name] = {'prediction': pred, **metrics}
                self._print_metrics(name, metrics)
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        return results, train, test

    def _calculate_metrics(self, actual, predicted):
        try:
            return {
                'MAPE': mape(actual, predicted),
                'RMSE': rmse(actual, predicted),
            }
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {'MAPE': np.nan, 'RMSE': np.nan}

    def _print_metrics(self, model_name, metrics):
        print(f"{model_name} Performance:")
        for metric, value in metrics.items():
            if not np.isnan(value):
                print(f"{metric}: {value:.2f}")

    def plot_results(self, series, results, train, test, save_path='forecast.png'):
        """Plot predictions from all models"""
        if series is None or results is None:
            print("No data to plot")
            return

        plt.figure(figsize=(15, 7))
        train.plot(label='Training', alpha=0.6)
        test.plot(label='Test', alpha=0.6)
        
        if results:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))
            for (name, result), color in zip(results.items(), colors):
                if 'prediction' in result:
                    result['prediction'].plot(
                        label=f'{name} (MAPE: {result["MAPE"]:.1f}%)',
                        color=color
                    )

        plt.title('Sunspot Number Forecasting Comparison')
        plt.xlabel('Time')
        plt.ylabel('Sunspot Number')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def main():
    analyzer = TimeSeriesAnalyzer()
    
    real_series = analyzer.load_data("SN_m_tot_V2.0.csv")  
    real_results, real_train, real_test = analyzer.analyze(real_series)
    analyzer.plot_results(real_series, real_results, 
                                real_train, real_test, 'sunspot_forecast.png')

if __name__ == "__main__":
    main()
```

<figure id="8652" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*VMkeAvN3kpJnGxzLVMorxg.png"
class="graf-image" data-image-id="1*VMkeAvN3kpJnGxzLVMorxg.png"
data-width="1489" data-height="690" />
</figure>

This real-world example demonstrates both the power and limitations of
time series forecasting. While we can capture major patterns, solar
activity's inherent complexity means predictions should be used as
guidance rather than absolute forecasts.
:::
::::
::::::
:::::::

By [Kyle Jones](https://medium.com/@kylejones_47003){.p-author .h-card}
on [February 3, 2025](https://medium.com/p/bdbb117ae411).

[Canonical
link](https://medium.com/@kylejones_47003/predicting-sunspots-with-arima-theta-and-tbats-in-darts-with-python-bdbb117ae411){.p-canonical}

Exported from [Medium](https://medium.com) on February 9, 2025.
