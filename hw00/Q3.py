import sys, os, math
import numpy as np
import pandas as pd
from collections import OrderedDict
from datetime import timedelta
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

path = "./data/train.csv"
# preproc data
# format
colnames = ['Date', "Site", "Item"] + [str(x) for x in range(24)]
df = pd.read_csv(path, names=colnames, skiprows=1, encoding="GB2312")

# remove the column "site"
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html#pandas.DataFrame.loc
df = df.loc[:, ['Date', 'Item'] + [str(x) for x in range(24)]]

# melt "Hour" to column
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.melt.html
df = pd.melt(df, id_vars=['Date', 'Item'], value_vars=[str(x) for x in range(24)],
                 var_name='Hour', value_name='Value')

# generate "Datatime"
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html#pandas.to_datetime
df["Datetime"] = pd.to_datetime(df.Date + " " + df.Hour + ":00:00")
df = df.loc[:, ['Datetime', 'Item', 'Value']]


# replace NR to 0
df.loc[df.Value == "NR", "Value"] = 0

# pivot "Item" to columns
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html#pandas.pivot_table
df = df.pivot_table(values='Value', index='Datetime', columns='Item', aggfunc='sum')

### obtain training set and validation set ###
df_12m = df.loc[df.index.month == 12, :]
df_not_12m = df.loc[df.index.month != 12, :]


def gen_regression_form(df):
    data = OrderedDict()
    item_list = df.columns.tolist()
    datetime_list = df.index
    for i in range(9):
        for item in item_list:
            data['{:02d}h__{}'.format(i + 1, item)] = []
    data['10h__PM2.5'] = []

    d1h = timedelta(hours=1)
    # unique http://pandas.pydata.org/pandas-docs/stable/generated/pandas.unique.html#pandas.unique
    for m in pd.unique(datetime_list.month):
        for timestamp in (df.loc[df.index.month == m, :]).index:
            start = timestamp
            end = timestamp + 9 * d1h
            sub_df = df.loc[(start <= df.index) & (df.index <= end), :]
            if sub_df.shape[0] == 10:
                for i in range(9):
                    for item in item_list:
                        data['{:02d}h__{}'.format(i + 1, item)].append(
                            sub_df.loc[timestamp + i * d1h, item])
                data['10h__PM2.5'].append(sub_df.loc[timestamp + 9 * d1h, 'PM2.5'])

    return pd.DataFrame(data)

# valid_data
path_valid_data = './valid_data.csv'
if os.path.isfile(path_valid_data):
    valid_data = pd.read_csv(path_valid_data)
else:
    valid_data = gen_regression_form(df_12m)
    valid_data.to_csv(path_valid_data, index=None)

# train_data
path_train_data = './train_data.csv'
if os.path.isfile(path_train_data):
    train_data = pd.read_csv(path_train_data)
else:
    train_data = gen_regression_form(df_not_12m)
    train_data.to_csv(path_train_data, index=None)

# 162 = 18 * 9
# 5181 = 24 * 20 * 11 - 9 * 11
# 471 = 24* 20 * 1 - 9

valid_X = np.array(valid_data.loc[:, valid_data.columns != '10h__PM2.5'])
valid_y = np.array(valid_data.loc[:, '10h__PM2.5'])

train_X = np.array(train_data.loc[:, train_data.columns != '10h__PM2.5'])
train_y = np.array(train_data.loc[:, '10h__PM2.5'])
print(valid_X.shape)
print(train_X.shape)


###########################################
# Linear Regression                       #
###########################################
reg = linear_model.LinearRegression()
reg.fit(train_X,train_y)

# Make predictions using the testing set
valid_y_pred = reg.predict(valid_X)

# The coefficients
#print('Coefficients: \n', reg.coef_)
# The mean squared error
print("Linear Regression Mean squared error: %.2f"
      % mean_squared_error(valid_y, valid_y_pred))
# Explained variance score: 1 is perfect prediction
print('Linear Regression Variance score: %.2f' % r2_score(valid_y, valid_y_pred))

###########################################
# Ridge Regression                        #
###########################################
regr = linear_model.Ridge(alpha=0.4)
regr.fit(train_X, train_y)

valid_y_pred_r = regr.predict(valid_X)

# The coefficients
# print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Ridge Regression Mean squared error: %.2f"
          % mean_squared_error(valid_y, valid_y_pred))
# Explained variance score: 1 is perfect prediction
print('Ridge Regression Variance score: %.2f' % r2_score(valid_y, valid_y_pred))

# record the order of columns
colname_X = (train_data.loc[:,train_data.columns!='10h__PM2.5']).columns

### testing
col_names = ['ID','Item']+ ['{:02d}h'.format(x) for x in range(1,10)]
test = pd.read_csv('./data/test_X.csv', names = col_names, header=None)

# replace NR to 0
for col in ['{:02d}h'.format(x) for x in range(1,10)]:
    test.loc[(test.Item=='RAINFALL')&(test[col]=='NR'),col] = 0

# ['ID','Item','Hour','Value'] form
test =  test.pivot_table( index=['ID','Item'], aggfunc='sum')
test = test.stack()
test = test.reset_index()
test.columns = ['ID','Item','Hour','Value']


# combine 'Hour' and 'Item' to 'Col'
test['Col'] = test.Hour + "__" + test.Item
test = test[['ID','Col','Value']]

# pivot 'Col' to columns
test = test.pivot_table(values='Value',index='ID',columns='Col', aggfunc='sum').reset_index()
test.name = ''

# re-order
test['ID_Num'] = test.ID.str.replace('id_','').astype('int')
test = test.sort_values(by='ID_Num')
test = test.reset_index(drop=True)

# predict
X_test = np.array(test[colname_X],dtype='float64')
test['Predict'] = regr.predict(X_test).round(2)

# output
test[['ID','Predict']].to_csv('test__out.csv',header=None,index=None)