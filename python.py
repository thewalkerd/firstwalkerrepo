# -*- coding: utf-8 -*-
import json 
import pandas as pd
import pickle 
import numpy as np
import xgboost as xgb
from numpy import nan


# load the model from disk
loaded_model = pickle.load(open(r'C:\Users\ipek.kaygusuz\Desktop\PS\ipek py\results\FINAL Models\Model no trx vr no prelimination\MODEL TEST\Model with Ilkin output-json conversion-Final.sav', 'rb'))


# "customerId":598170
x='{"has_transaction":1,"trxdt_count":51,"salary_calculated":1725.91,"avg_amt":-11.801650485436891,"maxBalanceOfDayEnd0":586.44,"min_amt":-450,"age":44,"numberOfNegativeTransactions2":25,"standardDDeviationOfDayEnds3":392.86368326048813,"interval2loanValuesnegativeSum":0,"interval0loanValuesnegativeSum":0,"minBalanceOfDayEnd1":364.38,"interval3loanValuesbalance":0,"interval3loanValuesnegativeSum":0,"numberOfNegativeTransactions0":27,"maxBalanceOfDayEnd3":1660.6,"avg_bal":599.1051456310674,"trx_dens_daily":2.019607843137255,"standardDDeviationOfDayEnds2":259.21007351417154,"numberOfDaysBelowAverage3":30,"interval3loanValuespositiveCount":0,"maxBalanceOfDayEnd2":1660.6,"numberOfPositiveTransactions3":11,"avgOfDayEnds0":283.04242424242426,"sumOfNegativeTransactionsAmount3":-7161.49,"numberOfPositiveTransactions2":4,"minBalanceOfDayEnd0":-21.23,"max_bal":2007.32,"interval0loanValuesstandardDeviation":0,"min_bal":-79.71,"sumOfNegativeTransactionsAmount2":-2209.18,"interval0loanValuesbalance":0,"sumOfNegativeTransactionsAmount0":-2334.52,"interval3loanValuesstandardDeviation":0,"interval0loanValuesnegativeCount":0,"interval1loanValuespositiveCount":0,"maxBalanceOfDayEnd1":1243.46,"interval1loanValuesbalance":0,"avgOfDayEnds1":849.2513333333334,"sumOfNegativeTransactionsAmount1":-2617.79,"barcelona_flag":0,"avgOfDayEnds2":1161.9123333333334,"sumOfPositiveTransactionsAmount3":5945.92,"other_town_flag":null,"trx_count":103,"numberOfDaysBelowAverage1":9,"sumOfPositiveTransactionsAmount1":2003.29,"interval1loanValuespositiveSum":0,"max_amt":1753.29,"numberOfNegativeTransactions1":32,"minBalanceOfDayEnd2":266.88,"numberOfDaysBelowAverage2":21,"sevilla_flag":0,"numberOfDaysAboveAverage0":23,"numberOfDaysBelowAverage0":10,"madrid_flag":0,"minBalanceOfDayEnd3":-21.23,"interval2loanValuespositiveSum":0,"sumOfPositiveTransactionsAmount2":1993.72,"paybackrisk_new":null,"numberOfPositiveTransactions1":3,"numberOfNegativeTransactions3":84,"avgOfDayEnds3":927.4817204301075,"interval2loanValuesstandardDeviation":0,"interval3loanValuespositiveSum_corrected":0,"numberOfOverdrawnDays3":1,"numberOfDaysAboveAverage1":21,"standardDDeviationOfDayEnds1":211.7528801808582,"outlook_flag":0,"numberOfOverdrawnDays0":2,"interval2loanValuesbalance":0,"numberOfPositiveTransactions0":4,"interval3loanValuesnegativeCount":0,"numberOfOverdrawnDays2":0,"local_flag":1,"trx_dens":1.7647058823529411,"numberOfDaysAboveAverage2":9,"sumOfPositiveTransactionsAmount0":1948.91,"other_domain_flag":0,"standardDDeviationOfDayEnds0":154.74442701142019,"trxdt_range":90,"interval1loanValuesstandardDeviation":0}'

# parse x:
y = json.loads(x)
dfs = []
dfs.append(pd.DataFrame([y]))
df = pd.concat(dfs, ignore_index=True, sort=False)


# input içerisinden other_town_flag çıkarttığında aşağıdaki koddan silebiliriz
df_2 = df[df.columns[~df.columns.isin(['has_transaction','other_town_flag'])]]
df_2 = df_2.fillna(value=np.nan)

score=loaded_model.predict_proba(df_2)[:,1].tolist()

# has_transaction=0 için sonuç None doner
if  df['has_transaction'].to_numpy()==1:
    final_score=round(score[0],2)
else:
    final_score=None
    
    
#sonuc 0.49 olmalı
print(final_score)