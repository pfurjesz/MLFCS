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
#Different from Aleksandr grouping. My time interval is (t,t+1]. His is [t,t+1).
table['datetime'] = table['datetime'].dt.ceil('min')

volume = table.groupby('datetime',as_index=False).sum()
volume.set_index('datetime',verify_integrity=True,inplace=True)
new_index = pd.date_range(volume.index[0],volume.index[-1],freq = 'min')
volume = volume.reindex(new_index,fill_value=0)
volume['time'] = volume.index.time
quantiles = volume[['time','amount']].groupby('time').quantile([0.25,0.5,0.75])
quantiles = quantiles.reset_index().rename(columns={'level_1':'quantiles'})
quantiles = quantiles.pivot(columns='quantiles',index='time',values='amount')
volume['mean'] = volume[['time','amount']].groupby('time').transform('mean')
volume['y'] = volume['amount']/volume['mean']
volume['logy'] = np.log(volume['amount']/volume['mean'])
volume.replace(-np.inf, np.nan,inplace=True)
volume.dropna(inplace=True)

if __name__ == "__main__":
    quantiles.plot()
    volume.hist(column='logy',bins=800)