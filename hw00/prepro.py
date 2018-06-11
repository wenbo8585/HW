import sys, os, math
import numpy as np
import pandas as pd
from GradientDescent import GradientDescent
from collections import OrderedDict
from datetime import timedelta

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

valid_X = np.array(valid_data.loc[:, valid_data.columns != '10h__PM2.5'])
valid_y = np.array(valid_data.loc[:, '10h__PM2.5'])

train_X = np.array(train_data.loc[:, train_data.columns != '10h__PM2.5'])
train_y = np.array(train_data.loc[:, '10h__PM2.5'])

# record the order of columns
colname_X = (train_data.loc[:, train_data.columns != '10h__PM2.5'])
#print(colname_x)

# gradient descent
gd = GradientDescent()

gd.train_by_pseudo_inverse(train_X, train_y, alpha=0.5, validate_data=(valid_X, valid_y))

# train-->
#
#init_wt = gd.wt
#init_b = gd.b
#
#gd.train(train_X, train_y, epoch=10, rate=0.00001, batch=100, alpha=0.0000001,
#        init_wt=np.array(init_wt), init_b=init_b, validate_data = (valid_X, valid_y))
#
# train <--

### testing
col_names = ['ID','Item']+[lambda x:'{:02d}h'.format(x) for x in range(1,10)]
test = pd.read_csv('./data/test_X.csv', names = col_names, header=None)

# record ordfer of test.ID
id_test = test.ID

#map(lambda x:'{:02d}h'.format(x),range(1,10)):
# replace NR to 0
for col in [lambda x:'{:02d}h'.format(x) for x in range(1,10)]:
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
test['Predict'] = gd.predict(X_test)

# output
test[['ID','Predict']].to_csv('linear_regression.csv',header=None,index=None)