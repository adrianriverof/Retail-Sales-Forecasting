#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

from FunctionsRetail import *


#Load data
data_path = '../../02_Data/03_Work/work.csv'
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


step1_df = data_quality(df)
step2_df = create_variables(step1_df)


launch_training(step2_df)

