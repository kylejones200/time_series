<div>

# Aeon for Time Series Clustering with Python {#aeon-for-time-series-clustering-with-python .p-name}

</div>

::: {.section .p-summary field="subtitle"}
Aeon is an open-source time series library for TS classification,
regression, clustering, forecasting, and transformation. It seems to
be...
:::

::::::: {.section .e-content field="body"}
:::::: {#ecfd .section .section .section--body .section--first .section--last}
::: section-divider

------------------------------------------------------------------------
:::

:::: section-content
::: {.section-inner .sectionLayout--insetColumn}
### Aeon for Time Series Clustering with Python {#e979 .graf .graf--h3 .graf--leading .graf--title name="e979"}

**Aeon** is an open-source time series library for TS classification,
regression, clustering, forecasting, and transformation. It seems to be
well maintained and there are articles about it from 2024.

> Warning --- I could only get Aeon to work with Pandas==1.4.0 (an older
> version). Newer versions of Pandas use a different index API which
> breaks a lot of time series libraries.

My favorite part are the visualizations which are pretty. But I miss the
flexibility that I have with Matplotlib.

#### Let's take a look at visualizing Time Series with Aeon {#b530 .graf .graf--h4 .graf-after--p name="b530"}

Aeon includes plotting utilities for exploratory data analysis. Let's
look at a basic Line Plot.

<figure id="ae28" class="graf graf--figure graf--iframe graf-after--p">

</figure>

<figure id="3b17" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*NnNMdIQqhSFAyidZJ8gDgw.png"
class="graf-image" data-image-id="1*NnNMdIQqhSFAyidZJ8gDgw.png"
data-width="1315" data-height="391" data-is-featured="true" />
</figure>

Not bad. Nothing spectactular but simple and nice.

### Time Series clustering and classification {#4620 .graf .graf--h3 .graf-after--p name="4620"}

Some time series follow predictable patterns but it can be hard to
distinguish to which group those series belong. Given a random signal,
how do we classify it with other similar signals?

This is where clustering algorithms help. The code generates 50 random
samples that follow one of three set patterns (sine, cosine, or
sine(2x)). Then we use aeon to classify which pattern each sample
belongs to. It uses the k-nearest neighbor algo do sort things out. As a
result, we can easily separate the different series into like buckets.
From here, we could do more analysis on a specific bucket. Looking all
all these series together just appears like noise. So the clustering
really helps us in this case.

<figure id="70ed" class="graf graf--figure graf--iframe graf-after--p">

</figure>

<figure id="ff27" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*d8rZaVV-7nU9IJ9KfvIa3w.png"
class="graf-image" data-image-id="1*d8rZaVV-7nU9IJ9KfvIa3w.png"
data-width="1489" data-height="989" />
</figure>

These graphs are much prettier than the line plot of the Passangers
dataset.

We can take these labels and have aeon classify the data using a
classifier. Because this is simulated data, i'm not surprised that the
algo perfectly separates things.

<figure id="c691" class="graf graf--figure graf--iframe graf-after--p">

</figure>

``` {#0290 .graf .graf--pre .graf-after--figure .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="yaml"}
Accuracy: 1.00

Data shape: (50, 30)
Number of classes: 3
Class distribution: [13 21 16]
```

Aeon is supposed to have an 'ETSForecaster' but I couldn't get this to
work.

### So what? {#55f4 .graf .graf--h3 .graf-after--p name="55f4"}

Aeon classification, regression, clustering, and visualization work
well. The API is simple and well documented. Compared to other tools,
though, Aeon is not my favorite and I don't intent to use this one
often.

#### Code {#4153 .graf .graf--h4 .graf-after--p name="4153"}

The code for this project is available on
[GitHub](https://github.com/kylejones200/time_series/blob/main/medium/Aeon%20for%20Time%20Series%20Forecasting%20with%C2%A0Python.ipynb){.markup--anchor
.markup--p-anchor
data-href="https://github.com/kylejones200/time_series/blob/main/medium/Aeon%20for%20Time%20Series%20Forecasting%20with%C2%A0Python.ipynb"
rel="noopener" target="_blank"}.
:::
::::
::::::
:::::::

By [Kyle Jones](https://medium.com/@kylejones_47003){.p-author .h-card}
on [January 10, 2025](https://medium.com/p/82229ac63282).

[Canonical
link](https://medium.com/@kylejones_47003/aeon-for-time-series-clustering-with-python-82229ac63282){.p-canonical}

Exported from [Medium](https://medium.com) on February 9, 2025.
