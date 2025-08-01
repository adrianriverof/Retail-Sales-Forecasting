#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

from FunctionsRetail import *



#Load the data

data_path = '../../02_Data/02_Validation/DataForProduction.csv'
df = pd.read_csv(data_path,sep=',',parse_dates=['date'],index_col='date')


final_variables = ['store_id',
                     'item_id',
                     'event_name_1',                     
                     'month',
                     'sell_price',                      
                     'wday',
                     'weekday',
                     'sales']
df = df[final_variables]


#Launch prediction
forecast = recursive_forecast(df).sort_values(by = ['store_id','item_id'])



