 
# Q3问题

使用线性回归或者其它方法预测PM2.5的数值。用前九个小时的测量值，预测第十个小时的PM2.5的值。

## 数据

data文件夹下train.csv和test_X.csv.

训练集: 
列分别是 日期/测站/测项/24h测试数据
训练数据总共有     12个月*20天*18项 = 4320行

测试数据:
排除train.csv中剩餘的資料，取連續9小時的資料當feature，預測第10小時的PM2.5值。
總共取240筆不重複的test data.
240*18 = 4320行。

## 数据预处理

pandas 按格式读取:

```
colnames = ['Date', "Site", "Item"] + [str(x) for x in range(24)]
df = pd.read_csv(path, names=colnames, skiprows=1, encoding="GB2312")
```

remove the column "site"

```
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html#pandas.DataFrame.loc
df = df.loc[:, ['Date', 'Item'] + [str(x) for x in range(24)]]
```

数据按时间排序,像这样:。
```text
              Date        Item Hour Value
0         2014/1/1    AMB_TEMP    0    14
1         2014/1/1         CH4    0   1.8
2         2014/1/1          CO    0  0.51
...            ...         ...  ...   ...
103650  2014/12/19         NOx   23   5.1
103651  2014/12/19          O3   23    32
```

melt:

```
df = pd.melt(df, id_vars=['Date', 'Item'], value_vars=[str(x) for x in range(24)],
                 var_name='Hour', value_name='Value')
```

将Data和Hour合并成Datatime,置NR为0：

```python
# generate "Datatime"
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html#pandas.to_datetime
df["Datetime"] = pd.to_datetime(df.Date + " " + df.Hour + ":00:00")
df = df.loc[:, ['Datetime', 'Item', 'Value']]

# replace NR to 0
df.loc[df.Value == "NR", "Value"] = 0
```

将Item项转换为列，像这样：
```text
Item                AMB_TEMP  CH4    CO  ...  WIND_DIREC WIND_SPEED WS_HR
Datetime                                 ...                             
2014-01-01 00:00:00       14  1.8  0.51  ...          35        1.4   0.5
2014-01-01 01:00:00       14  1.8  0.41  ...          79        1.8   0.9
2014-01-01 02:00:00       14  1.8  0.39  ...         2.4          1   0.6
2014-01-01 03:00:00       13  1.8  0.37  ...          55        0.6   0.3
```

```python
# pivot "Item" to columns
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html#pandas.pivot_table
df = df.pivot_table(values='Value', index='Datetime', columns='Item', aggfunc='sum')

```

最后取12月为验证集:
```python
### obtain training set and validation set ###
df_12m = df.loc[df.index.month == 12, :]
df_not_12m = df.loc[df.index.month != 12, :]
```

生成训练集和验证集.

用每隔9小时的item和第十小时PM2.5划分数据集，则共有 5181 = 24 * 20 * 11 - 9 * 11
笔数据，feature列数 162 = 18 * 9.所以训练集为5181*162.验证集同理.

## Linear Models


### Linear Regression

```bash
Linear Regression Mean squared error: 29.62
Linear Regression Variance score: 0.85
```


### Ridge Regression 

```bash
Ridge Regression Mean squared error: 29.62
Ridge Regression Variance score: 0.85
```
