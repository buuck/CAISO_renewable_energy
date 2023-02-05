# Analysis of 2021-2022 CAISO Power Source Data

This data comes from [Kaggle](https://www.kaggle.com/datasets/karatechop/caiso-renewable-energy-data-20212022), and presumably ultimately from CAISO. The units are not labeled, so I am guessing a bit here on what is what. I'm going to do some exploratory analysis, and see if we can do any meaningful forecasting.


```python
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy
from time import time
```


```python
sns.set_theme(context='notebook', style='whitegrid')
```

### Read and preprocess data

Make sure that the date/time columns in the DataFrame are coded as such so that `seaborn`/`matplotlib` don't treat them as strings, which causes major problems when plotting.

It's not clear what the units are for the different energy sources, but based on what is on the [CAISO website](https://www.caiso.com/TodaysOutlook/Pages/default.aspx), I'm guessing they're MW.


```python
power_sources = ['Solar', 'Wind', 'Geothermal', 'Biomass',
       'Biogas', 'Small hydro', 'Coal', 'Nuclear', 'Natural Gas',
       'Large Hydro', 'Batteries', 'Imports']
other = ['Date', 'Time', 'DateTime', 'Month', 'Year']
```


```python
df = pd.read_csv('caiso_2021-22.csv', date_parser=['Date', 'Time'])
df.drop(columns=['Other'], inplace=True) # There is hardly any of this, and it just makes the plots look weird
df.Date = pd.to_datetime(df.Date)
df.Time = pd.to_datetime(df.Time)
```


```python
# Compute the total 
df['Total power'] = df['Solar']
for ps in power_sources[1:]:
    df['Total power'] += df[ps]
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Date</th>
      <th>Time</th>
      <th>Solar</th>
      <th>Wind</th>
      <th>Geothermal</th>
      <th>Biomass</th>
      <th>Biogas</th>
      <th>Small hydro</th>
      <th>Coal</th>
      <th>Nuclear</th>
      <th>Natural Gas</th>
      <th>Large Hydro</th>
      <th>Batteries</th>
      <th>Imports</th>
      <th>DateTime</th>
      <th>Month</th>
      <th>Year</th>
      <th>Total power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2021-09-01</td>
      <td>2023-02-05 00:00:00</td>
      <td>-34.0</td>
      <td>4547.0</td>
      <td>928.0</td>
      <td>281.0</td>
      <td>195.0</td>
      <td>168.0</td>
      <td>18.0</td>
      <td>2263.0</td>
      <td>8875.0</td>
      <td>1261.0</td>
      <td>-186.0</td>
      <td>8145.0</td>
      <td>2021-09-01 00:00:00</td>
      <td>9</td>
      <td>2021</td>
      <td>26461.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2021-09-01</td>
      <td>2023-02-05 00:05:00</td>
      <td>-34.0</td>
      <td>4528.0</td>
      <td>929.0</td>
      <td>283.0</td>
      <td>201.0</td>
      <td>169.0</td>
      <td>18.0</td>
      <td>2262.0</td>
      <td>9086.0</td>
      <td>1109.0</td>
      <td>-13.0</td>
      <td>7717.0</td>
      <td>2021-09-01 00:05:00</td>
      <td>9</td>
      <td>2021</td>
      <td>26255.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2021-09-01</td>
      <td>2023-02-05 00:10:00</td>
      <td>-34.0</td>
      <td>4511.0</td>
      <td>929.0</td>
      <td>281.0</td>
      <td>208.0</td>
      <td>146.0</td>
      <td>18.0</td>
      <td>2263.0</td>
      <td>9168.0</td>
      <td>985.0</td>
      <td>37.0</td>
      <td>7553.0</td>
      <td>2021-09-01 00:10:00</td>
      <td>9</td>
      <td>2021</td>
      <td>26065.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2021-09-01</td>
      <td>2023-02-05 00:15:00</td>
      <td>-34.0</td>
      <td>4514.0</td>
      <td>929.0</td>
      <td>280.0</td>
      <td>214.0</td>
      <td>140.0</td>
      <td>19.0</td>
      <td>2262.0</td>
      <td>9167.0</td>
      <td>962.0</td>
      <td>34.0</td>
      <td>7458.0</td>
      <td>2021-09-01 00:15:00</td>
      <td>9</td>
      <td>2021</td>
      <td>25945.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2021-09-01</td>
      <td>2023-02-05 00:20:00</td>
      <td>-34.0</td>
      <td>4515.0</td>
      <td>929.0</td>
      <td>281.0</td>
      <td>215.0</td>
      <td>140.0</td>
      <td>18.0</td>
      <td>2262.0</td>
      <td>9176.0</td>
      <td>949.0</td>
      <td>35.0</td>
      <td>7342.0</td>
      <td>2021-09-01 00:20:00</td>
      <td>9</td>
      <td>2021</td>
      <td>25828.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>105099</th>
      <td>283</td>
      <td>2022-08-31</td>
      <td>2023-02-05 23:35:00</td>
      <td>-1.0</td>
      <td>2576.0</td>
      <td>872.0</td>
      <td>339.0</td>
      <td>203.0</td>
      <td>206.0</td>
      <td>4.0</td>
      <td>2263.0</td>
      <td>16228.0</td>
      <td>2679.0</td>
      <td>-434.0</td>
      <td>8044.0</td>
      <td>2022-08-31 23:35:00</td>
      <td>8</td>
      <td>2022</td>
      <td>32979.0</td>
    </tr>
    <tr>
      <th>105100</th>
      <td>284</td>
      <td>2022-08-31</td>
      <td>2023-02-05 23:40:00</td>
      <td>0.0</td>
      <td>2589.0</td>
      <td>871.0</td>
      <td>340.0</td>
      <td>204.0</td>
      <td>206.0</td>
      <td>5.0</td>
      <td>2264.0</td>
      <td>16098.0</td>
      <td>2687.0</td>
      <td>-487.0</td>
      <td>8078.0</td>
      <td>2022-08-31 23:40:00</td>
      <td>8</td>
      <td>2022</td>
      <td>32855.0</td>
    </tr>
    <tr>
      <th>105101</th>
      <td>285</td>
      <td>2022-08-31</td>
      <td>2023-02-05 23:45:00</td>
      <td>0.0</td>
      <td>2558.0</td>
      <td>872.0</td>
      <td>339.0</td>
      <td>204.0</td>
      <td>206.0</td>
      <td>5.0</td>
      <td>2263.0</td>
      <td>16034.0</td>
      <td>2636.0</td>
      <td>-514.0</td>
      <td>8120.0</td>
      <td>2022-08-31 23:45:00</td>
      <td>8</td>
      <td>2022</td>
      <td>32723.0</td>
    </tr>
    <tr>
      <th>105102</th>
      <td>286</td>
      <td>2022-08-31</td>
      <td>2023-02-05 23:50:00</td>
      <td>0.0</td>
      <td>2521.0</td>
      <td>871.0</td>
      <td>339.0</td>
      <td>204.0</td>
      <td>206.0</td>
      <td>5.0</td>
      <td>2264.0</td>
      <td>15976.0</td>
      <td>2642.0</td>
      <td>-521.0</td>
      <td>8163.0</td>
      <td>2022-08-31 23:50:00</td>
      <td>8</td>
      <td>2022</td>
      <td>32670.0</td>
    </tr>
    <tr>
      <th>105103</th>
      <td>287</td>
      <td>2022-08-31</td>
      <td>2023-02-05 23:55:00</td>
      <td>0.0</td>
      <td>2513.0</td>
      <td>871.0</td>
      <td>341.0</td>
      <td>205.0</td>
      <td>206.0</td>
      <td>5.0</td>
      <td>2264.0</td>
      <td>15581.0</td>
      <td>2650.0</td>
      <td>-389.0</td>
      <td>7993.0</td>
      <td>2022-08-31 23:55:00</td>
      <td>8</td>
      <td>2022</td>
      <td>32240.0</td>
    </tr>
  </tbody>
</table>
<p>105104 rows × 19 columns</p>
</div>



From the description below, we can see that Natural Gas is clearly the largest power source, and also the second most variable after Solar (2nd highest standard deviation).


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Solar</th>
      <th>Wind</th>
      <th>Geothermal</th>
      <th>Biomass</th>
      <th>Biogas</th>
      <th>Small hydro</th>
      <th>Coal</th>
      <th>Nuclear</th>
      <th>Natural Gas</th>
      <th>Large Hydro</th>
      <th>Batteries</th>
      <th>Imports</th>
      <th>Month</th>
      <th>Year</th>
      <th>Total power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>105104.000000</td>
      <td>105092.000000</td>
      <td>105092.000000</td>
      <td>105092.000000</td>
      <td>105092.000000</td>
      <td>105092.000000</td>
      <td>105092.000000</td>
      <td>105092.000000</td>
      <td>105092.000000</td>
      <td>105092.000000</td>
      <td>105092.000000</td>
      <td>105092.000000</td>
      <td>105092.000000</td>
      <td>105104.000000</td>
      <td>105104.000000</td>
      <td>105092.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>143.478231</td>
      <td>4193.116555</td>
      <td>2454.372731</td>
      <td>869.873949</td>
      <td>287.631047</td>
      <td>200.969284</td>
      <td>191.277966</td>
      <td>12.498525</td>
      <td>2076.552973</td>
      <td>8614.644778</td>
      <td>1457.924219</td>
      <td>77.379268</td>
      <td>5612.521467</td>
      <td>6.526374</td>
      <td>2021.665703</td>
      <td>26048.762760</td>
    </tr>
    <tr>
      <th>std</th>
      <td>83.125935</td>
      <td>5046.487510</td>
      <td>1456.446929</td>
      <td>76.966457</td>
      <td>45.488711</td>
      <td>14.678127</td>
      <td>94.729647</td>
      <td>4.994506</td>
      <td>406.780691</td>
      <td>3905.161266</td>
      <td>855.182969</td>
      <td>572.085363</td>
      <td>2963.652848</td>
      <td>3.447995</td>
      <td>0.471747</td>
      <td>4762.813769</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-180.000000</td>
      <td>28.000000</td>
      <td>474.000000</td>
      <td>-278.000000</td>
      <td>132.000000</td>
      <td>46.000000</td>
      <td>-6.000000</td>
      <td>446.000000</td>
      <td>1494.000000</td>
      <td>-494.000000</td>
      <td>-1848.000000</td>
      <td>-4459.000000</td>
      <td>1.000000</td>
      <td>2021.000000</td>
      <td>15916.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>71.000000</td>
      <td>-33.000000</td>
      <td>1200.000000</td>
      <td>823.000000</td>
      <td>255.000000</td>
      <td>195.000000</td>
      <td>153.000000</td>
      <td>9.000000</td>
      <td>2250.000000</td>
      <td>5692.000000</td>
      <td>878.000000</td>
      <td>-234.000000</td>
      <td>3395.000000</td>
      <td>4.000000</td>
      <td>2021.000000</td>
      <td>22661.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>143.000000</td>
      <td>170.000000</td>
      <td>2211.000000</td>
      <td>878.000000</td>
      <td>288.000000</td>
      <td>204.000000</td>
      <td>190.000000</td>
      <td>14.000000</td>
      <td>2264.000000</td>
      <td>8116.000000</td>
      <td>1279.000000</td>
      <td>3.000000</td>
      <td>6208.000000</td>
      <td>7.000000</td>
      <td>2022.000000</td>
      <td>25121.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>215.000000</td>
      <td>9391.000000</td>
      <td>3649.000000</td>
      <td>903.000000</td>
      <td>320.000000</td>
      <td>211.000000</td>
      <td>226.000000</td>
      <td>17.000000</td>
      <td>2268.000000</td>
      <td>10828.000000</td>
      <td>1908.000000</td>
      <td>319.000000</td>
      <td>7999.000000</td>
      <td>10.000000</td>
      <td>2022.000000</td>
      <td>27983.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>287.000000</td>
      <td>14288.000000</td>
      <td>6429.000000</td>
      <td>1134.000000</td>
      <td>412.000000</td>
      <td>242.000000</td>
      <td>3316.000000</td>
      <td>91.000000</td>
      <td>2287.000000</td>
      <td>25441.000000</td>
      <td>4556.000000</td>
      <td>3053.000000</td>
      <td>11587.000000</td>
      <td>12.000000</td>
      <td>2022.000000</td>
      <td>46679.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Exploratory Data Analysis

This dataset comes in 5 minute intervals spanning a full year. That is >100k samples, which is difficult to visualize meaningfully all at once. Furthermore, there are at least 2 meaningful periods of variation in this dataset: daily and annually. For these reasons, we'll make 2 different plots, one showing the daily average power generation over the course of the full year, and one showing hourly power generation over one day, averaging over every day of the dataset.

If we do this for only the Solar power data, we see the following figures


```python
sns.lineplot(data=df, x='Date', y='Solar')
```




    <AxesSubplot: xlabel='Date', ylabel='Solar'>




    
![png](CAISO_analysis_files/CAISO_analysis_11_1.png)
    



```python
ax = sns.lineplot(data=df, x='Time', y='Solar')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.show()
```


    
![png](CAISO_analysis_files/CAISO_analysis_12_0.png)
    


The curves are almost exactly what we should expect. The annual curve peaks in late June at the summer solstice, and has a trough in late December during the winter solstice. The light blue shaded region shows the 95% inner quantile range.

In order to make more plots easily with Seaborn, we need to convert the DataFrame from wide to long format.


```python
df_long = df.drop(columns=['Unnamed: 0'])\
    .melt(id_vars=['Date', 'Time', 'DateTime', 'Month', 'Year'],
          var_name='Source',
          value_name='Power (MW)')
df_long
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>DateTime</th>
      <th>Month</th>
      <th>Year</th>
      <th>Source</th>
      <th>Power (MW)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-09-01</td>
      <td>2023-02-05 00:00:00</td>
      <td>2021-09-01 00:00:00</td>
      <td>9</td>
      <td>2021</td>
      <td>Solar</td>
      <td>-34.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-09-01</td>
      <td>2023-02-05 00:05:00</td>
      <td>2021-09-01 00:05:00</td>
      <td>9</td>
      <td>2021</td>
      <td>Solar</td>
      <td>-34.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-09-01</td>
      <td>2023-02-05 00:10:00</td>
      <td>2021-09-01 00:10:00</td>
      <td>9</td>
      <td>2021</td>
      <td>Solar</td>
      <td>-34.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-09-01</td>
      <td>2023-02-05 00:15:00</td>
      <td>2021-09-01 00:15:00</td>
      <td>9</td>
      <td>2021</td>
      <td>Solar</td>
      <td>-34.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-09-01</td>
      <td>2023-02-05 00:20:00</td>
      <td>2021-09-01 00:20:00</td>
      <td>9</td>
      <td>2021</td>
      <td>Solar</td>
      <td>-34.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1366347</th>
      <td>2022-08-31</td>
      <td>2023-02-05 23:35:00</td>
      <td>2022-08-31 23:35:00</td>
      <td>8</td>
      <td>2022</td>
      <td>Total power</td>
      <td>32979.0</td>
    </tr>
    <tr>
      <th>1366348</th>
      <td>2022-08-31</td>
      <td>2023-02-05 23:40:00</td>
      <td>2022-08-31 23:40:00</td>
      <td>8</td>
      <td>2022</td>
      <td>Total power</td>
      <td>32855.0</td>
    </tr>
    <tr>
      <th>1366349</th>
      <td>2022-08-31</td>
      <td>2023-02-05 23:45:00</td>
      <td>2022-08-31 23:45:00</td>
      <td>8</td>
      <td>2022</td>
      <td>Total power</td>
      <td>32723.0</td>
    </tr>
    <tr>
      <th>1366350</th>
      <td>2022-08-31</td>
      <td>2023-02-05 23:50:00</td>
      <td>2022-08-31 23:50:00</td>
      <td>8</td>
      <td>2022</td>
      <td>Total power</td>
      <td>32670.0</td>
    </tr>
    <tr>
      <th>1366351</th>
      <td>2022-08-31</td>
      <td>2023-02-05 23:55:00</td>
      <td>2022-08-31 23:55:00</td>
      <td>8</td>
      <td>2022</td>
      <td>Total power</td>
      <td>32240.0</td>
    </tr>
  </tbody>
</table>
<p>1366352 rows × 7 columns</p>
</div>



Seaborn will order the line colors based on the order in which they show up in the DataFrame. This is fine, but I want them to be listed in the legend in the same order as they will appear visually on the plot.


```python
argsort = df_long.groupby('Source').mean(numeric_only=True)['Power (MW)']\
            .argsort().values[::-1]
hue_order = np.array(
    df_long.groupby('Source').mean(numeric_only=True)\
        ['Power (MW)'][argsort].index
    )
# pal = np.concatenate(([[0,0,0]], np.array(sns.color_palette('Paired', len(hue_order)-1))))
pal = np.array(sns.color_palette('husl', len(hue_order)))[argsort]
pal[0] = np.array([0,0,0])
```

Next we'll make the same plots as above, but this time for all power sources. We can do this easily with the long-form DataFrame and Seaborn.


```python
fig, ax = plt.subplots(figsize=(15,15/scipy.constants.golden))
ax = sns.lineplot(
    ax=ax,
    data=df_long,
    x='Date',
    y='Power (MW)',
    hue='Source',
    palette=pal,
    hue_order=hue_order
    )
ax.set(yscale='log')
ax.set_ylim([1,5e4])
plt.show()
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/seaborn/_oldcore.py:200: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if palette in QUAL_PALETTES:



    
![png](CAISO_analysis_files/CAISO_analysis_18_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15,15/scipy.constants.golden))
ax = sns.lineplot(
    ax=ax,
    data=df_long,
    x='Date',
    y='Power (MW)',
    hue='Source',
    palette=pal,
    hue_order=hue_order
    )
plt.show()
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/seaborn/_oldcore.py:200: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if palette in QUAL_PALETTES:



    
![png](CAISO_analysis_files/CAISO_analysis_19_1.png)
    


There are a number of interesting features happening in the two figures above.
- A weeklong variable period is apparent in the Total Power curve. January 1st 2022 was a Saturday and also lines up with one of the troughs, which indicates to me that weekends generally put substantially less load on the power system.
- In the linear plot, we can see that the Total Power drawn is generally higher in the summer and lower in the winter, although there is a notable increase in power draw around Christmastime. Christmas lights? Or perhaps it was just cold.
- Unsurprisingly, nuclear power output is generally extremely steady, although we can see that it did drop off a few times.
- We can again see the seasonal variability of solar power, probably more clearly in the linear plot than the log.
- On a day-to-day basis, wind power is extremely variable, more so than solar.
- California imports more power in the winter than summer. I would guess this may be because electricity gets more expensive in the summer, and therefore harder to import.
- Biomass, biogas, small hydro, and batteries play a pretty small role.
- There is virtually no coal power in the system. That's because there is only one 63 MW coal plant operating in the state of California, in Trona.

Now let's take a look at the hourly data.


```python
fig, ax = plt.subplots(figsize=(15,15/scipy.constants.golden))
ax = sns.lineplot(
    ax=ax,
    data=df_long,
    x='Time',
    y='Power (MW)',
    hue='Source',
    palette=pal,
    hue_order=hue_order
    )
ax.set(yscale='log')
ax.set_ylim([5,4e4])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.show()
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/seaborn/_oldcore.py:200: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if palette in QUAL_PALETTES:



    
![png](CAISO_analysis_files/CAISO_analysis_21_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15,15/scipy.constants.golden))
ax = sns.lineplot(
    ax=ax,
    data=df_long,
    x='Time',
    y='Power (MW)',
    hue='Source',
    palette=pal,
    hue_order=hue_order
    )
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.show()
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/seaborn/_oldcore.py:200: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if palette in QUAL_PALETTES:



    
![png](CAISO_analysis_files/CAISO_analysis_22_1.png)
    


Again we can make a few interesting observations.
- Although they make up only a small part of CAISO's power capacity, batteries are playing a nontrivial role at certain times of day, particularly in the evening. We can also see the batteries charging during the day in the linear plot (where the curve goes negative).
- The solar power curve has two "tails". I don't know what is causing them, but my hypothesis of what is has two components:
  - California is a long state N/S, and summer is the part of the year where the sun is up latest into the evening. During the summer, the evening terminator is mostly perpendicular to the length of the state, meaning that there should be a gradual dropoff in solar power as it gets dark from south to north. That explains the fact that there is at least one bump.
  - There are two bumps instead of one because of daylight savings time.
  - This does not occur in the morning because on summer mornings, the terminator faces the opposite direction and so all parts of the state start receiving solar power at about the same time.
- Wind power does indeed see variation complementary to solar power as advertised, but it is a much smaller source than solar and therefore is not able to offset much of the solar variability.

For the last plots in this section, I'm going to separate sources into dispatchable and variable. Dispatchable resources are those that can be adjusted during the day to compensate for uncontrollable variability in other sources and demand. In practice, I am separating sources into these two categories based on whether they appear to have controllable daily variation in the plot above. The dispatchable resources that seem to have this:
- Natural Gas
- Imports
- Large Hydro
- Small Hydro
- Batteries
- (Theoretically coal would go here as well, but it's too small for me to see evidence of daily variation.)

In principle, some of the other resources can be ramped up or down, but I would guess that they aren't because they are smaller and more distributed, and may not be technically set up to be ramped on demand.


```python
df_long = df_long[df_long['Source']!='Total power']
dispatchable = ['Natural Gas', 'Imports', 'Large Hydro', 'Small hydro', 'Batteries', 'Coal']
variable = ['Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas', 'Nuclear']
df_long['Dispatchable'] = df_long['Source'].apply(lambda x: x in dispatchable)
```

    /tmp/ipykernel_1387/2591602897.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_long['Dispatchable'] = df_long['Source'].apply(lambda x: x in dispatchable)



```python
df_dispatchable = df_long.groupby(by=['DateTime', 'Time', 'Date', 'Dispatchable']).sum()
df_dispatchable.reset_index(level=3, inplace=True)
```

    /tmp/ipykernel_1387/4263355843.py:1: FutureWarning: The default value of numeric_only in DataFrameGroupBy.sum is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.
      df_dispatchable = df_long.groupby(by=['DateTime', 'Time', 'Date', 'Dispatchable']).sum()



```python
fig, ax = plt.subplots(figsize=(15,15/scipy.constants.golden))
ax = sns.lineplot(data=df_dispatchable, x='Time', y='Power (MW)', hue='Dispatchable')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
```


    
![png](CAISO_analysis_files/CAISO_analysis_26_0.png)
    



```python
fig, ax = plt.subplots(figsize=(15,15/scipy.constants.golden))
ax = sns.lineplot(data=df_dispatchable, x='Date', y='Power (MW)', hue='Dispatchable')
ax.set_ylim([0, ax.get_ylim()[1]])
```




    (0.0, 26955.196124131944)




    
![png](CAISO_analysis_files/CAISO_analysis_27_1.png)
    


There's not quite as much to see here as in the previous plots, except that the duck curve is alive and well in the hourly plot. The daily plot is dominated by random variation and the seasonal variation of solar power.

## Time-series Modeling

Let's see if we can extract any utility (haha) from time-series analysis of this data. First we'll try an ARIMA analysis.


```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
```

We have a full year of 5-minute samples available, which translates to 105,104 samples. This (empirially) is too many to fit on my laptop, so I'm going to downsample the data to hourly. In the real world, 5-minute predictions may be useful, but we're going to make do with 1-hr predictions for now.


```python
df['Hour'] = df.DateTime.apply(datetime.datetime.fromisoformat).dt.hour
df_hourly = df.groupby(by=['Date', 'Hour']).mean()
df_hourly.dropna(inplace=True) # Daylight savings causes a NaN
```

    /tmp/ipykernel_1618/3178447348.py:2: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.
      df_hourly = df.groupby(by=['Date', 'Hour']).mean()


We're going to try to predict the total amount of power from dispatchable resources required at any point in time. This is similar to what a real power company or ISO has to do in real time: tune the dispatchable power sources to cover the difference between variable supply and demand at all times.

Since the amount of variable resource supply affects how much dispatchable power will be needed, it is reasonable to look for relationships between previous values of variable power output and the current amount of dispatchable power. Below I create a series of plots showing the relationship between a series of lagged variable power samples and dispatchable power.


```python
df_hourly['Dispatchable'] = np.sum([df_hourly[s] for s in dispatchable], axis=0)
df_hourly['Variable'] = np.sum([df_hourly[s] for s in variable], axis=0)
for i in range(1, 10):
    df_hourly['Variable.L{}'.format(i)] = np.roll(df_hourly['Variable'], i)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[14], line 1
    ----> 1 df_hourly['Dispatchable'] = np.sum([df_hourly[s] for s in dispatchable], axis=0)
          2 df_hourly['Variable'] = np.sum([df_hourly[s] for s in variable], axis=0)
          3 for i in range(1, 10):


    NameError: name 'dispatchable' is not defined


First let's take a look at the autocorrelation and partial autocorrelation functions. These will help us decide what kind of model is most appropriate.


```python
# I don't know if it's a bug in statsmodels or what, but this makes two copies of one or both plots for some reason
plot_acf(df_hourly['Dispatchable'], lags=100)
plot_pacf(df_hourly['Dispatchable'], lags=100)
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/graphics/tsaplots.py:348: FutureWarning: The default method 'yw' can produce PACF values outside of the [-1,1] interval. After 0.13, the default will change tounadjusted Yule-Walker ('ywm'). You can use this method now by setting method='ywm'.
      warnings.warn(





    
![png](CAISO_analysis_files/CAISO_analysis_36_1.png)
    




    
![png](CAISO_analysis_files/CAISO_analysis_36_2.png)
    



    
![png](CAISO_analysis_files/CAISO_analysis_36_3.png)
    


We see pretty clear evidence of a daily pattern in the autocorrelation function. The partial autocorrelation function suggests that there is meaningful information to be extracted up to lags of about 24, which makes sense since this data is taken over a 24 hour daily period. Unsurprisingly, we see evidence of "seasonality", where in this case a season corresponds to a 24 hr day. This suggests that an ARIMA model of ARIMA(24, 0, 0)x(1, 0, 0, 24) could be appropriate. However, this is way too many lags to fit in a reasonable amount of time, so we'll just have to see how many we can do before I run out of computation power or they stop adding reasonable predictive power.

We should also look at the real-time relationship between variable and dispatchable resources. Clearly there is a strong relationship, but if we are trying to predict future values of dispatchable power, we will only have access to past values of variable power. Therefore, we need to take at least one lag to make this realistic.


```python
sns.kdeplot(data=df_hourly, x='Dispatchable', y='Variable')
```




    <AxesSubplot: xlabel='Dispatchable', ylabel='Variable'>




    
![png](CAISO_analysis_files/CAISO_analysis_39_1.png)
    



```python
sns.kdeplot(data=df_hourly[1:], x='Dispatchable', y='Variable.L1')
```




    <AxesSubplot: xlabel='Dispatchable', ylabel='Variable.L1'>




    
![png](CAISO_analysis_files/CAISO_analysis_40_1.png)
    



```python
sns.kdeplot(data=df_hourly[1:], x='Dispatchable', y='Variable.L2')
```




    <AxesSubplot: xlabel='Dispatchable', ylabel='Variable.L2'>




    
![png](CAISO_analysis_files/CAISO_analysis_41_1.png)
    



```python
sns.kdeplot(data=df_hourly[1:], x='Dispatchable', y='Variable.L3')
```




    <AxesSubplot: xlabel='Dispatchable', ylabel='Variable.L3'>




    
![png](CAISO_analysis_files/CAISO_analysis_42_1.png)
    



```python
sns.kdeplot(data=df_hourly[9:], x='Dispatchable', y='Variable.L9')
```




    <AxesSubplot: xlabel='Dispatchable', ylabel='Variable.L9'>




    
![png](CAISO_analysis_files/CAISO_analysis_43_1.png)
    


The first two variable lags have a reasonably strong relationship with the dispatchable power output, but by the time we get to the third lag, there isn't much left. The ninth lag also shows essentially no relationship between the two, so we'll limit the model to the first two lags of variable power output.

Given the results of the autocorrelation plots and the Variable vs. Dispatchable plots above, let's build a series of models with 1 and 2 Variable source lags, building up the number of autocorrelated lags as far as we reasonably can. We'll also include one 24-hr seasonal term.


```python
# Baseline, no seasonal autocorrelation, no exogenous variables
model_0_1_0 = ARIMA(endog=df_hourly['Dispatchable'], order=(1,0,0))
res_0_1_0 = model_0_1_0.fit()
print(res_0_1_0.summary())
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)


                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:           Dispatchable   No. Observations:                 8759
    Model:                 ARIMA(1, 0, 0)   Log Likelihood              -79321.828
    Date:                Sun, 05 Feb 2023   AIC                         158649.656
    Time:                        10:16:15   BIC                         158670.890
    Sample:                             0   HQIC                        158656.891
                                   - 8759                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       1.597e+04    420.877     37.936      0.000    1.51e+04    1.68e+04
    ar.L1          0.9439      0.004    239.155      0.000       0.936       0.952
    sigma2        4.3e+06   5.13e+04     83.832      0.000     4.2e+06     4.4e+06
    ===================================================================================
    Ljung-Box (L1) (Q):                4591.85   Jarque-Bera (JB):              1675.00
    Prob(Q):                              0.00   Prob(JB):                         0.00
    Heteroskedasticity (H):               1.03   Skew:                             0.66
    Prob(H) (two-sided):                  0.50   Kurtosis:                         4.69
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).



```python
# Baseline, no autoregression, 24-hr "seasonal" autoregression
model_0_0_1 = ARIMA(endog=df_hourly['Dispatchable'], order=(0,0,0), seasonal_order=(1,0,0,24))
res_0_0_1 = model_0_0_1.fit()
print(res_0_0_1.summary())
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)


                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:           Dispatchable   No. Observations:                 8759
    Model:             ARIMA(1, 0, 0, 24)   Log Likelihood              -80386.596
    Date:                Sun, 05 Feb 2023   AIC                         160779.192
    Time:                        10:16:20   BIC                         160800.425
    Sample:                             0   HQIC                        160786.427
                                   - 8759                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       1.597e+04    358.556     44.530      0.000    1.53e+04    1.67e+04
    ar.S.L24       0.9298      0.004    236.051      0.000       0.922       0.938
    sigma2      5.462e+06   6.02e+04     90.769      0.000    5.34e+06    5.58e+06
    ===================================================================================
    Ljung-Box (L1) (Q):                8069.84   Jarque-Bera (JB):              1334.72
    Prob(Q):                              0.00   Prob(JB):                         0.00
    Heteroskedasticity (H):               1.12   Skew:                             0.08
    Prob(H) (two-sided):                  0.00   Kurtosis:                         4.91
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).



```python
# Baseline with both autocorrelation and seasonal autocorrelation
model_0_1_1 = ARIMA(endog=df_hourly['Dispatchable'], order=(1,0,0), seasonal_order=(1, 0, 0, 24))
res_0_1_1 = model_0_1_1.fit()
print(res_0_1_1.summary())
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)


                                        SARIMAX Results                                     
    ========================================================================================
    Dep. Variable:                     Dispatchable   No. Observations:                 8759
    Model:             ARIMA(1, 0, 0)x(1, 0, 0, 24)   Log Likelihood              -69337.252
    Date:                          Sun, 05 Feb 2023   AIC                         138682.504
    Time:                                  10:05:54   BIC                         138710.815
    Sample:                                       0   HQIC                        138692.150
                                             - 8759                                         
    Covariance Type:                            opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       1.597e+04   1.48e-11   1.08e+15      0.000     1.6e+04     1.6e+04
    ar.L1          1.0000    1.6e-05   6.23e+04      0.000       1.000       1.000
    ar.S.L24       0.9491      0.002    415.030      0.000       0.945       0.954
    sigma2      4.363e+05   1.99e-10    2.2e+15      0.000    4.36e+05    4.36e+05
    ===================================================================================
    Ljung-Box (L1) (Q):                1712.24   Jarque-Bera (JB):             12230.48
    Prob(Q):                              0.00   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.70   Skew:                             0.04
    Prob(H) (two-sided):                  0.00   Kurtosis:                         8.79
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    [2] Covariance matrix is singular or near-singular, with condition number 3.27e+30. Standard errors may be unstable.



```python
# AR(1)xSAR(1)xVar.L1
model_1_1 = ARIMA(endog=df_hourly['Dispatchable'], exog=df_hourly[['Variable.L1']], order=(1,0,0), seasonal_order=(1, 0, 0, 24))
res_1_1 = model_1_1.fit()
print(res_1_1.summary())
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)


                                        SARIMAX Results                                     
    ========================================================================================
    Dep. Variable:                     Dispatchable   No. Observations:                 8759
    Model:             ARIMA(1, 0, 0)x(1, 0, 0, 24)   Log Likelihood              -69250.928
    Date:                          Sun, 05 Feb 2023   AIC                         138511.856
    Time:                                  10:17:19   BIC                         138547.245
    Sample:                                       0   HQIC                        138523.915
                                             - 8759                                         
    Covariance Type:                            opg                                         
    ===============================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const        2.301e+04   8.65e-09   2.66e+12      0.000     2.3e+04     2.3e+04
    Variable.L1    -0.4229      0.000   -898.260      0.000      -0.424      -0.422
    ar.L1           1.0000   1.77e-05   5.64e+04      0.000       1.000       1.000
    ar.S.L24        0.9987      0.001   1724.759      0.000       0.998       1.000
    sigma2       4.925e+05   7.78e-10   6.33e+14      0.000    4.92e+05    4.92e+05
    ===================================================================================
    Ljung-Box (L1) (Q):                 401.93   Jarque-Bera (JB):             11334.48
    Prob(Q):                              0.00   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.75   Skew:                            -0.11
    Prob(H) (two-sided):                  0.00   Kurtosis:                         8.57
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    [2] Covariance matrix is singular or near-singular, with condition number 1.1e+28. Standard errors may be unstable.


`ar.L1 == 1` is a random walk, where the most likely next value is the current value. On top of that, we have a strong relationship between the next value and the sample 24 hours ago, which is also close to a random walk with a 24 hr period. Finally, we also see that there is an inverse relationship between the forecasted dispatchable power output and the previous variable power output, as expected. We'll use the information criteria in the upper right to choose our model. 


```python
# AR(2)xSAR(1)xVar.L1
model_1_2 = ARIMA(endog=df_hourly['Dispatchable'], exog=df_hourly[['Variable.L1']], order=(2, 0, 0), seasonal_order=(1, 0, 0, 24))
res_1_2 = model_1_2.fit()
print(res_1_2.summary())
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)


                                        SARIMAX Results                                     
    ========================================================================================
    Dep. Variable:                     Dispatchable   No. Observations:                 8759
    Model:             ARIMA(2, 0, 0)x(1, 0, 0, 24)   Log Likelihood              -68125.930
    Date:                          Sun, 05 Feb 2023   AIC                         136263.861
    Time:                                  07:26:56   BIC                         136306.328
    Sample:                                       0   HQIC                        136278.331
                                             - 8759                                         
    Covariance Type:                            opg                                         
    ===============================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const        2.301e+04   1527.241     15.066      0.000       2e+04     2.6e+04
    Variable.L1     0.0347      0.013      2.749      0.006       0.010       0.059
    ar.L1           1.4371      0.010    150.823      0.000       1.418       1.456
    ar.L2          -0.4938      0.010    -50.632      0.000      -0.513      -0.475
    ar.S.L24        0.9263      0.003    328.655      0.000       0.921       0.932
    sigma2       3.317e+05   2769.802    119.763      0.000    3.26e+05    3.37e+05
    ===================================================================================
    Ljung-Box (L1) (Q):                  23.19   Jarque-Bera (JB):             11503.60
    Prob(Q):                              0.00   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.75   Skew:                            -0.10
    Prob(H) (two-sided):                  0.00   Kurtosis:                         8.61
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).


The parameters have changed a bit now that we've added the extra lag. Besides the fit taking quite a bit longer, we now have a negative coefficient for the `L2` term and a positive coefficient for the `L1` term. This corresponds roughly to making a linear extrapolation from the previous two sample points. The coefficient of the `Variable.L1` term has also nearly disappeared, but it is still quite statistically significant.


```python
# AR(3)xSAR(1)xVar.L1
model_1_3 = ARIMA(endog=df_hourly['Dispatchable'], exog=df_hourly[['Variable.L1']], order=(3, 0, 0), seasonal_order=(1, 0, 0, 24))
res_1_3 = model_1_3.fit()
print(res_1_3.summary())
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)


                                        SARIMAX Results                                     
    ========================================================================================
    Dep. Variable:                     Dispatchable   No. Observations:                 8759
    Model:             ARIMA(3, 0, 0)x(1, 0, 0, 24)   Log Likelihood              -68048.912
    Date:                          Sun, 05 Feb 2023   AIC                         136111.824
    Time:                                  10:20:22   BIC                         136161.369
    Sample:                                       0   HQIC                        136128.706
                                             - 8759                                         
    Covariance Type:                            opg                                         
    ===============================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const        2.301e+04   1802.777     12.763      0.000    1.95e+04    2.65e+04
    Variable.L1     0.1533      0.016      9.590      0.000       0.122       0.185
    ar.L1           1.5700      0.014    108.283      0.000       1.542       1.598
    ar.L2          -0.7789      0.025    -30.808      0.000      -0.828      -0.729
    ar.L3           0.1616      0.013     12.646      0.000       0.137       0.187
    ar.S.L24        0.9258      0.003    328.230      0.000       0.920       0.931
    sigma2        3.26e+05   2735.865    119.140      0.000    3.21e+05    3.31e+05
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.20   Jarque-Bera (JB):             11390.09
    Prob(Q):                              0.65   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.77   Skew:                            -0.08
    Prob(H) (two-sided):                  0.00   Kurtosis:                         8.58
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).



```python
# AR(4)xSAR(1)xVar.L1
model_1_4 = ARIMA(endog=df_hourly['Dispatchable'], exog=df_hourly[['Variable.L1']], order=(4, 0, 0), seasonal_order=(1, 0, 0, 24))
res_1_4 = model_1_4.fit()
print(res_1_4.summary())
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/base/model.py:604: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "


                                        SARIMAX Results                                     
    ========================================================================================
    Dep. Variable:                     Dispatchable   No. Observations:                 8759
    Model:             ARIMA(4, 0, 0)x(1, 0, 0, 24)   Log Likelihood              -68061.618
    Date:                          Sun, 05 Feb 2023   AIC                         136139.236
    Time:                                  07:51:19   BIC                         136195.859
    Sample:                                       0   HQIC                        136158.529
                                             - 8759                                         
    Covariance Type:                            opg                                         
    ===============================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const        2.301e+04   2668.673      8.622      0.000    1.78e+04    2.82e+04
    Variable.L1     0.2007      0.017     12.149      0.000       0.168       0.233
    ar.L1           1.6240      0.015    106.777      0.000       1.594       1.654
    ar.L2          -0.8668      0.029    -29.703      0.000      -0.924      -0.810
    ar.L3           0.2462      0.023     10.583      0.000       0.201       0.292
    ar.L4          -0.0364      0.010     -3.502      0.000      -0.057      -0.016
    ar.S.L24        0.9268      0.003    329.352      0.000       0.921       0.932
    sigma2       3.321e+05   2853.443    116.401      0.000    3.27e+05    3.38e+05
    ===================================================================================
    Ljung-Box (L1) (Q):                   1.26   Jarque-Bera (JB):             12334.06
    Prob(Q):                              0.26   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.77   Skew:                            -0.06
    Prob(H) (two-sided):                  0.00   Kurtosis:                         8.81
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).


Non-convergence means that we are done adding terms, and `model_1_3` is the best we are going to do with only `Variable.L1`.


```python
# AR(1)xSAR(1)xVar.L1xVar.L2
model_2_1 = ARIMA(endog=df_hourly['Dispatchable'], exog=df_hourly[['Variable.L1','Variable.L2']], order=(1, 0, 0), seasonal_order=(1, 0, 0, 24))
res_2_1 = model_2_1.fit()
print(res_2_1.summary())
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)


                                        SARIMAX Results                                     
    ========================================================================================
    Dep. Variable:                     Dispatchable   No. Observations:                 8759
    Model:             ARIMA(1, 0, 0)x(1, 0, 0, 24)   Log Likelihood              -68898.499
    Date:                          Sun, 05 Feb 2023   AIC                         137808.998
    Time:                                  10:28:14   BIC                         137851.465
    Sample:                                       0   HQIC                        137823.468
                                             - 8759                                         
    Covariance Type:                            opg                                         
    ===============================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const         2.21e+04   2.71e-08   8.16e+11      0.000    2.21e+04    2.21e+04
    Variable.L1    -0.3860      0.009    -42.275      0.000      -0.404      -0.368
    Variable.L2     0.1625      0.008     20.866      0.000       0.147       0.178
    ar.L1           1.0000   5.91e-07   1.69e+06      0.000       1.000       1.000
    ar.S.L24        0.9338      0.003    353.493      0.000       0.929       0.939
    sigma2       3.995e+05    8.4e-09   4.76e+13      0.000       4e+05       4e+05
    ===================================================================================
    Ljung-Box (L1) (Q):                 507.22   Jarque-Bera (JB):              9530.79
    Prob(Q):                              0.00   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.76   Skew:                             0.10
    Prob(H) (two-sided):                  0.00   Kurtosis:                         8.11
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    [2] Covariance matrix is singular or near-singular, with condition number 2.78e+28. Standard errors may be unstable.



```python
# AR(2)xSAR(1)xVar.L1xVar.L2
model_2_2 = ARIMA(endog=df_hourly['Dispatchable'], exog=df_hourly[['Variable.L1','Variable.L2']], order=(2, 0, 0), seasonal_order=(1, 0, 0, 24))
res_2_2 = model_2_2.fit()
print(res_2_2.summary())
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)


                                        SARIMAX Results                                     
    ========================================================================================
    Dep. Variable:                     Dispatchable   No. Observations:                 8759
    Model:             ARIMA(2, 0, 0)x(1, 0, 0, 24)   Log Likelihood              -68275.361
    Date:                          Sun, 05 Feb 2023   AIC                         136564.722
    Time:                                  08:09:44   BIC                         136614.267
    Sample:                                       0   HQIC                        136581.604
                                             - 8759                                         
    Covariance Type:                            opg                                         
    ===============================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const         2.21e+04   3.35e-08   6.59e+11      0.000    2.21e+04    2.21e+04
    Variable.L1     0.0411      0.010      4.307      0.000       0.022       0.060
    Variable.L2     0.1649      0.009     18.478      0.000       0.147       0.182
    ar.L1           1.4987      0.009    169.438      0.000       1.481       1.516
    ar.L2          -0.4987      0.009    -56.384      0.000      -0.516      -0.481
    ar.S.L24        0.9265      0.003    353.314      0.000       0.921       0.932
    sigma2       3.426e+05   7.36e-09   4.66e+13      0.000    3.43e+05    3.43e+05
    ===================================================================================
    Ljung-Box (L1) (Q):                  12.19   Jarque-Bera (JB):             12506.26
    Prob(Q):                              0.00   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.76   Skew:                            -0.02
    Prob(H) (two-sided):                  0.00   Kurtosis:                         8.85
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    [2] Covariance matrix is singular or near-singular, with condition number 1.51e+29. Standard errors may be unstable.



```python
# AR(4)xSAR(1)
model_0_4 = ARIMA(endog=df_hourly['Dispatchable'], order=(4, 0, 0), seasonal_order=(1, 0, 0, 24))
res_0_4 = model_0_4.fit()
print(res_0_4.summary())
```

    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /home/mbuuck/miniconda3/envs/caiso_analysis/lib/python3.11/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)


                                        SARIMAX Results                                     
    ========================================================================================
    Dep. Variable:                     Dispatchable   No. Observations:                 8759
    Model:             ARIMA(4, 0, 0)x(1, 0, 0, 24)   Log Likelihood              -68073.396
    Date:                          Sun, 05 Feb 2023   AIC                         136160.792
    Time:                                  08:14:48   BIC                         136210.337
    Sample:                                       0   HQIC                        136177.673
                                             - 8759                                         
    Covariance Type:                            opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       1.597e+04   1329.281     12.011      0.000    1.34e+04    1.86e+04
    ar.L1          1.4725      0.008    185.271      0.000       1.457       1.488
    ar.L2         -0.6368      0.015    -42.424      0.000      -0.666      -0.607
    ar.L3          0.1400      0.016      8.704      0.000       0.108       0.171
    ar.L4         -0.0319      0.010     -3.300      0.001      -0.051      -0.013
    ar.S.L24       0.9174      0.003    316.633      0.000       0.912       0.923
    sigma2      3.279e+05   2754.964    119.008      0.000    3.22e+05    3.33e+05
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):             10359.17
    Prob(Q):                              0.96   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.78   Skew:                            -0.06
    Prob(H) (two-sided):                  0.00   Kurtosis:                         8.33
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).


We can plot a barchart of the AIC values for each model, which will help us decide which is optimal.


```python
AICs = pd.DataFrame(
    {
        'AIC':[158649.656, 160779.192, 138682.504,
               138511.856, 136263.861, 136111.824,
               137808.998, 136564.722, 136160.792],
        'Model': ['AR(1)', 'SAR(1)', 'AR(1)xSAR(1)',
                  'AR(1)xSAR(1)xVar.L1', 'AR(2)xSAR(1)xVar.L1', 'AR(3)xSAR(1)xVar.L1',
                  'AR(1)xSAR(1)xVar.L1xVar.L2', 'AR(2)xSAR(1)xVar.L1xVar.L2', 'AR(4)xSAR(1)']
    }
)
AICs['Relative AIC'] = AICs['AIC'] - np.min(AICs['AIC'])
```


```python
sns.barplot(data=AICs, x='Relative AIC', y='Model')
```




    <AxesSubplot: xlabel='Relative AIC', ylabel='Model'>




    
![png](CAISO_analysis_files/CAISO_analysis_60_1.png)
    


The Relative AIC is just the difference between the given model's AIC and the lowest AIC amongst all models. It is a bit hard to see, but the `AR(3)xSAR(1)xVar.L1` model has the lowest AIC, followed closely by the `AR(4)xSAR(1)` model. The fact that these are so close, coupled with the fact that including `Var.L1` does seem to improve things (see `AR(1)xSAR(1)` vs. `AR(1)xSAR(1)xVar.L1`), suggests that a `AR(4)xSAR(1)xVar.L1` model would likely perform even better if it had converged. Interestingly, adding the second `Variable` resource lag (`Var.L2`) seems to make things worse at `AR(2)`.

Next we'll look directly at some predictions and residuals. First, we'll make a plot showing the predicted power vs the actual dispatchable power over the course of the year.


```python
preds = pd.DataFrame(res_1_3.predict()).rename(columns={'predicted_mean':'Predicted Power (MW)'})
preds['Power (MW)'] = df_hourly['Dispatchable']
preds['Residual (MW)'] = preds['Predicted Power (MW)'] - preds['Power (MW)']
```


```python
ax = sns.lineplot(data=preds, x='Hour', y='Predicted Power (MW)')
ax = sns.lineplot(ax=ax, data=preds, x='Hour', y='Power (MW)')
```


    
![png](CAISO_analysis_files/CAISO_analysis_63_0.png)
    



```python
ax = sns.lineplot(data=preds, x='Date', y='Residual (MW)')
```


    
![png](CAISO_analysis_files/CAISO_analysis_64_0.png)
    


The residuals are small enough that they don't noticeably show up on the first plot. By eye it looks like the residuals may be higher in the summer than in the winter, but that is hard to say. That wouldn't surprise me very much, since solar power is much stronger in the summer and makes up the largest share of the variable resources. Weather effects, which should be a large contributor to solar variablity, would have a bigger effect on the overall market during periods with greater sunshine.

Next we'll look at some plots of 3 arbitrary days of data with different models, comparing the actual and predicted dispatchable power output. First we'll look at the final chosen model.


```python
ax = sns.lineplot(x=range(72), y=preds[24:96]['Power (MW)'], label='Power (MW)')
ax = sns.lineplot(ax=ax, x=range(72), y=res_1_3.predict()[24:96], label='Predicted Power (MW)')
```


    
![png](CAISO_analysis_files/CAISO_analysis_66_0.png)
    


We see excellent agreement in general, as expected.

Next we'll look at the `AR(1)` model, a.k.a. one autoregressive lag, no seasonal component, and no dependence on prior variable resource output.


```python
ax = sns.lineplot(x=range(72), y=preds[24:96]['Power (MW)'], label='Power (MW)')
ax = sns.lineplot(ax=ax, x=range(72), y=res_0_1_0.predict()[24:96], label='Predicted Power (MW)')
```


    
![png](CAISO_analysis_files/CAISO_analysis_68_0.png)
    


This model tends to predict slight mean reversion based on the previous sample, which makes sense on average, but is pretty bad in general.

The next model includes only the "seasonal" (a.k.a. daily) term.


```python
ax = sns.lineplot(x=range(72), y=preds[24:96]['Power (MW)'], label='Power (MW)')
ax = sns.lineplot(ax=ax, x=range(72), y=res_0_0_1.predict()[24:96], label='Predicted Power (MW)')
```


    
![png](CAISO_analysis_files/CAISO_analysis_70_0.png)
    


The prediction of this model is only informed by the value at the same time the day before, so while it doesn't show the lagged behavior that the previous plot does, it is not able to react to any information more recent than 24 hours ago, and consequently is not very good.

Finally, let's plot the residuals of these models as well as the `AR(1)xSAR(1)xVar.L1` model together. We can see that both the green and red curve perform significanty better than the blue and orange curves, showing that having at least 4 terms (not including the constant term) in the model is important. Red and green are quite similar, but we know from the box-plot above that the more complex model (red) has a lower AIC than green. Since it also has more parameters, that must mean its likelihood is higher, and therefore it should have smaller residuals on average.


```python
days = 3
ax = sns.lineplot(x=range(days*24), y=res_0_1_0.predict()[:days*24]-preds[:days*24]['Power (MW)'], label='Model AR(1) Residual (MW)')
ax = sns.lineplot(ax=ax, x=range(days*24), y=res_0_0_1.predict()[:days*24]-preds[:days*24]['Power (MW)'], label='Model SAR(1) Residual (MW)')
ax = sns.lineplot(ax=ax, x=range(days*24), y=res_1_1.predict()[:days*24]-preds[:days*24]['Power (MW)'], label='Model Var.L1xAR(1)xSAR(1) Residual (MW)')
ax = sns.lineplot(ax=ax, x=range(days*24), y=res_1_3.predict()[:days*24]-preds[:days*24]['Power (MW)'], label='Model Var.L1xAR(3)xSAR(1) Residual (MW)')
```


    
![png](CAISO_analysis_files/CAISO_analysis_72_0.png)
    


It would be interesting to fit an RNN to this data.
