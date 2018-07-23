
### This is a PROMISE Software Engineering Repository data set made publicly available in order to encourage repeatable, verifiable, refutable, and/or improvable predictive models of software engineering.

If you publish material based on PROMISE data sets then, please
follow the acknowledgment guidelines posted on the PROMISE repository
web page http://promise.site.uottawa.ca/SERepository .


```python
import math
from scipy.io import arff
from scipy.stats.stats import pearsonr
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, classification_report, r2_score
from sklearn.model_selection import LeaveOneOut 
from sklearn.cross_validation import train_test_split

# Formatação mais bonita para os notebooks
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15,5)
```


```python
df_desharnais = pd.read_csv('../Datasets/02.desharnais.csv',  header=0)
df_desharnais.head()
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
      <th>id</th>
      <th>Project</th>
      <th>TeamExp</th>
      <th>ManagerExp</th>
      <th>YearEnd</th>
      <th>Length</th>
      <th>Effort</th>
      <th>Transactions</th>
      <th>Entities</th>
      <th>PointsNonAdjust</th>
      <th>Adjustment</th>
      <th>PointsAjust</th>
      <th>Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>85</td>
      <td>12</td>
      <td>5152</td>
      <td>253</td>
      <td>52</td>
      <td>305</td>
      <td>34</td>
      <td>302</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>86</td>
      <td>4</td>
      <td>5635</td>
      <td>197</td>
      <td>124</td>
      <td>321</td>
      <td>33</td>
      <td>315</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>85</td>
      <td>1</td>
      <td>805</td>
      <td>40</td>
      <td>60</td>
      <td>100</td>
      <td>18</td>
      <td>83</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>86</td>
      <td>5</td>
      <td>3829</td>
      <td>200</td>
      <td>119</td>
      <td>319</td>
      <td>30</td>
      <td>303</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>86</td>
      <td>4</td>
      <td>2149</td>
      <td>140</td>
      <td>94</td>
      <td>234</td>
      <td>24</td>
      <td>208</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Applying Pearson’s correlation

In this section, the correlations between attributes of Desharnais dataset and software effort are analyzed and applicability of the regression analysis is examined. The correlation between two variables is a measure of how well the variables are related. The most common measure of correlation in statistics is the Pearson Correlation (or the Pearson Product Moment Correlation - PPMC) which shows the linear relationship between two variables. 

Pearson correlation coefficient analysis produces a result between `-1` and `1`. A result of `-1` means that there is a perfect negative correlation between the two values at all, while a result of `1` means that there is a perfect positive correlation between the two variables. 

Results between `0.5` and `1.0` indicate high correlation.Correlation coefficients are used in statistics to measure how strong a relationship is between two variables. There are several types of correlation coefficient. `Pearson’s correlation` (also called Pearson’s R) is a correlation coefficient commonly used in linear regression.


```python
df_desharnais.corr()
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
      <th>id</th>
      <th>Project</th>
      <th>TeamExp</th>
      <th>ManagerExp</th>
      <th>YearEnd</th>
      <th>Length</th>
      <th>Effort</th>
      <th>Transactions</th>
      <th>Entities</th>
      <th>PointsNonAdjust</th>
      <th>Adjustment</th>
      <th>PointsAjust</th>
      <th>Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-0.006007</td>
      <td>0.214294</td>
      <td>0.096486</td>
      <td>0.255187</td>
      <td>0.126153</td>
      <td>0.265891</td>
      <td>0.028787</td>
      <td>0.226076</td>
      <td>-0.207774</td>
      <td>0.202608</td>
      <td>0.391475</td>
    </tr>
    <tr>
      <th>Project</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-0.006007</td>
      <td>0.214294</td>
      <td>0.096486</td>
      <td>0.255187</td>
      <td>0.126153</td>
      <td>0.265891</td>
      <td>0.028787</td>
      <td>0.226076</td>
      <td>-0.207774</td>
      <td>0.202608</td>
      <td>0.391475</td>
    </tr>
    <tr>
      <th>TeamExp</th>
      <td>-0.006007</td>
      <td>-0.006007</td>
      <td>1.000000</td>
      <td>0.424687</td>
      <td>-0.210335</td>
      <td>0.143948</td>
      <td>0.119529</td>
      <td>0.103768</td>
      <td>0.256608</td>
      <td>0.203805</td>
      <td>0.235629</td>
      <td>0.222884</td>
      <td>-0.079112</td>
    </tr>
    <tr>
      <th>ManagerExp</th>
      <td>0.214294</td>
      <td>0.214294</td>
      <td>0.424687</td>
      <td>1.000000</td>
      <td>-0.011519</td>
      <td>0.211324</td>
      <td>0.158303</td>
      <td>0.138146</td>
      <td>0.206644</td>
      <td>0.207748</td>
      <td>-0.066821</td>
      <td>0.187399</td>
      <td>0.205521</td>
    </tr>
    <tr>
      <th>YearEnd</th>
      <td>0.096486</td>
      <td>0.096486</td>
      <td>-0.210335</td>
      <td>-0.011519</td>
      <td>1.000000</td>
      <td>-0.095027</td>
      <td>-0.048367</td>
      <td>0.034331</td>
      <td>0.001686</td>
      <td>0.028234</td>
      <td>-0.056743</td>
      <td>0.012106</td>
      <td>0.342233</td>
    </tr>
    <tr>
      <th>Length</th>
      <td>0.255187</td>
      <td>0.255187</td>
      <td>0.143948</td>
      <td>0.211324</td>
      <td>-0.095027</td>
      <td>1.000000</td>
      <td>0.693280</td>
      <td>0.620711</td>
      <td>0.483504</td>
      <td>0.723849</td>
      <td>0.266086</td>
      <td>0.714092</td>
      <td>-0.023810</td>
    </tr>
    <tr>
      <th>Effort</th>
      <td>0.126153</td>
      <td>0.126153</td>
      <td>0.119529</td>
      <td>0.158303</td>
      <td>-0.048367</td>
      <td>0.693280</td>
      <td>1.000000</td>
      <td>0.581881</td>
      <td>0.510328</td>
      <td>0.705449</td>
      <td>0.463865</td>
      <td>0.738271</td>
      <td>-0.261942</td>
    </tr>
    <tr>
      <th>Transactions</th>
      <td>0.265891</td>
      <td>0.265891</td>
      <td>0.103768</td>
      <td>0.138146</td>
      <td>0.034331</td>
      <td>0.620711</td>
      <td>0.581881</td>
      <td>1.000000</td>
      <td>0.185041</td>
      <td>0.886419</td>
      <td>0.341906</td>
      <td>0.880923</td>
      <td>0.136778</td>
    </tr>
    <tr>
      <th>Entities</th>
      <td>0.028787</td>
      <td>0.028787</td>
      <td>0.256608</td>
      <td>0.206644</td>
      <td>0.001686</td>
      <td>0.483504</td>
      <td>0.510328</td>
      <td>0.185041</td>
      <td>1.000000</td>
      <td>0.618913</td>
      <td>0.234747</td>
      <td>0.598401</td>
      <td>-0.056439</td>
    </tr>
    <tr>
      <th>PointsNonAdjust</th>
      <td>0.226076</td>
      <td>0.226076</td>
      <td>0.203805</td>
      <td>0.207748</td>
      <td>0.028234</td>
      <td>0.723849</td>
      <td>0.705449</td>
      <td>0.886419</td>
      <td>0.618913</td>
      <td>1.000000</td>
      <td>0.383842</td>
      <td>0.985945</td>
      <td>0.082737</td>
    </tr>
    <tr>
      <th>Adjustment</th>
      <td>-0.207774</td>
      <td>-0.207774</td>
      <td>0.235629</td>
      <td>-0.066821</td>
      <td>-0.056743</td>
      <td>0.266086</td>
      <td>0.463865</td>
      <td>0.341906</td>
      <td>0.234747</td>
      <td>0.383842</td>
      <td>1.000000</td>
      <td>0.513197</td>
      <td>-0.199167</td>
    </tr>
    <tr>
      <th>PointsAjust</th>
      <td>0.202608</td>
      <td>0.202608</td>
      <td>0.222884</td>
      <td>0.187399</td>
      <td>0.012106</td>
      <td>0.714092</td>
      <td>0.738271</td>
      <td>0.880923</td>
      <td>0.598401</td>
      <td>0.985945</td>
      <td>0.513197</td>
      <td>1.000000</td>
      <td>0.046672</td>
    </tr>
    <tr>
      <th>Language</th>
      <td>0.391475</td>
      <td>0.391475</td>
      <td>-0.079112</td>
      <td>0.205521</td>
      <td>0.342233</td>
      <td>-0.023810</td>
      <td>-0.261942</td>
      <td>0.136778</td>
      <td>-0.056439</td>
      <td>0.082737</td>
      <td>-0.199167</td>
      <td>0.046672</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.set(font_scale=1.05)
sns.heatmap(df_desharnais.drop(['id'], axis=1).astype(float).corr(),linewidths=0.1,vmax=1.0, square=True,cmap=colormap, linecolor='white', annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x116305ac8>




![png](output_6_1.png)



```python
# split into train and test
neigh = KNeighborsRegressor(n_neighbors=3, weights='uniform')

features = [ 'TeamExp', 'ManagerExp', 'YearEnd', 'Length', 'Transactions', 'Entities',
        'PointsNonAdjust', 'Adjustment', 'PointsAjust']

max_corr_features = ['Length', 'Transactions', 'Entities','PointsNonAdjust','PointsAjust']

X = df_desharnais[max_corr_features]
y = df_desharnais['Effort']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=30)
```


```python
neigh.fit(X_train, y_train) 
print(neigh.score(X_test, y_test))
```

    0.7379861869550943



```python
model = LinearRegression()

features = ['TeamExp', 'ManagerExp', 'YearEnd', 'Length', 'Transactions', 'Entities',
        'PointsNonAdjust', 'Adjustment', 'PointsAjust']

max_corr_features = ['Length', 'Transactions', 'Entities','PointsNonAdjust','PointsAjust']

X = df_desharnais[max_corr_features]
y = df_desharnais['Effort']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)
```


```python
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

    0.7680074954440712



```python
plt.figure(figsize=(18,6))

for i, feature in enumerate(max_corr_features):
   
    xs, ys = zip(*sorted(zip(X_test[feature], neigh.fit(X_train, y_train).predict(X_test))))
    
    model_xs, model_ys = zip(*sorted(zip(X_test[feature], model.fit(X_train, y_train).predict(X_test))))

    
    plt.scatter(X_test[feature], y_test, label='Real data', lw=1,alpha= 0.7, c='k' )
    plt.plot(model_xs, model_ys , lw=1, label='Linear Regression Model')
    plt.plot(xs, ys , lw=1,label='K Neighbors Regressor Model', c='r')
    plt.xlabel(feature)
    plt.ylabel('Effort')
    plt.legend()
    plt.show()
```


![png](output_11_0.png)



![png](output_11_1.png)



![png](output_11_2.png)



![png](output_11_3.png)



![png](output_11_4.png)

