import pandas as pd
import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

trx = load_dataset('./data/trx')
pdtrx = trx['train'].to_pandas()
pdtrx.rename(columns={
    "'timestamp'":'timestamp', " 'price'":'price', " 'datetime'":'datetime', " 'cost'":'cost', " 'id'":'id', " 'fee'":'fee', " 'fee1'":'fee1', " 'order'":'order', " 'symbol'":'symbol', " 'amount'":'amount', " 'type'":'type', " 'side'":'side'
},inplace=True)

table = pdtrx[['datetime','amount']]
table['datetime'] = pd.to_datetime(table['datetime'])
volume = table.groupby(['datetime']).sum()
hours = volume.index.to_series().dt.hour
hours.rename('hour',inplace=True)
mins = volume.index.to_series().dt.minute
mins.rename('min',inplace=True)
time=pd.concat([hours,mins],axis=1)
quant=pd.concat([time,volume],axis=1)

Q1=quant.groupby(['hour','min']).quantile(0.25)
Q1.rename(columns={
    'amount':'Q1'},inplace=True)
Q2=quant.groupby(['hour','min']).quantile(0.5)
Q2.rename(columns={
    'amount':'Q2'},inplace=True)
Q3=quant.groupby(['hour','min']).quantile(0.75)
Q3.rename(columns={
    'amount':'Q3'},inplace=True)
Q4=quant.groupby(['hour','min']).quantile(1)
Q4.rename(columns={
    'amount':'Q4'},inplace=True)

Qs=pd.concat([Q1,Q2,Q3],axis=1)

avg = quant.groupby(['hour','min']).mean()
avg['y'] = Q4['Q4']/avg['amount']
avg['logy'] = np.log(avg['log'])

avg.hist(column='logy',bins=1440)

if __name__ == "__main__":
    Qs.plot()
    avg.hist(column='logy', bins=1440)