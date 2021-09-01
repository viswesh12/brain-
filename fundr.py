import numpy as np
import pandas as pd
from binance import Client
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from scipy import stats

#####Binance key###
api_key = "6dc316731b204420a9df36b3316df02e0365f40488c945cf0bdf947996216270"
api_secret = "89e9c0cf9e751a2617f3d0e2add79254ad8d9e78ef00e0429bce4901e8b06c0b"
client=Client(api_key,api_secret,testnet=True)

#### Data input ###
symbol="IOTAUSDT"
start_time="1610582400000"
end_time="1623628800000"
start_price="1610553600000"

### Price fetching ###
price=client.futures_historical_klines(symbol,Client.KLINE_INTERVAL_8HOUR,start_price,end_time)
price=pd.DataFrame(price,columns=['open time','open','high','low','close','volume','close time','quote','no. of trade','taker','base','ignore' ])
price['time']=pd.to_datetime(price['open time'],unit='ms')
price['close']=pd.to_numeric(price['close'])
price['open']=pd.to_numeric(price['open'])
price['pct chg']=price.open.pct_change(fill_method='ffill')
train_P=price.iloc[:300]
test_P=price.iloc[300,:]
print(price)

#### Funding Rate fetching ###
fd= client.futures_funding_rate(symbol= symbol, startTime=start_time,endTime=end_time,limit='1000')
fd=pd.DataFrame(fd)
fdd=pd.to_numeric(fd.fundingRate)
fd['time']=pd.to_datetime(fd['fundingTime'],unit='ms')
fd.set_index=fd['time']
fd['open']=price['open']
fd['pct chng']=price['pct chg']
train_fd=fd.iloc[:300]
train_fdd=fdd.iloc[:300]

#### ADF Test ###
t_stat, p_value, _, _, critical_values, _  = adfuller(train_fdd.values, autolag='AIC')
print(f'ADF Statistic: {t_stat:.2f}')
print(f'p-value: {p_value:.2f}')
for key, value in critical_values.items():
    print('Critial Values:')
    print(f'   {key}, {value:.2f}')

### Z score plot ##
pd.set_option('display.max_rows',None)
pd.set_option('display.width',None)
pd.set_option('display.max_columns',None)
zscore=stats.zscore(fdd)
fd['zscore']=pd.DataFrame(zscore)
print(fd)
plt.plot(fd.time,zscore)
plt.axhline(0, color='black')
plt.axhline(2.0, color='red', linestyle='--')
plt.axhline(-2.0, color='green', linestyle='--')
plt.legend(['Z-Score', 'Mean', '+2', '-2'])
plt.show()
test_fd=fd.iloc[300:].copy()

#### Strategy and Backtest####
exitZscore =0
entryZscore = 2
# set up num units long
df1=pd.DataFrame()
df1['long entry'] = ((fd['zscore'] < - entryZscore) & (fd['zscore'].shift(1) > - entryZscore))
df1['long exit'] = ((fd['zscore'] > - exitZscore) &(fd['zscore'].shift(1) < - exitZscore))
df1['num units long'] = np.nan
df1.loc[df1['long entry'], 'num units long'] = 1
df1.loc[df1['long exit'], 'num units long'] = 0
df1['num units long'][0] = 0
df1['num units long'] = df1['num units long'].fillna(method='pad')

# set up num units short
df1['short entry'] = ((fd.zscore > entryZscore) & (fd.zscore.shift(1) < entryZscore))
df1['short exit'] = ((fd.zscore < exitZscore) & (fd.zscore.shift(1) > exitZscore))
df1.loc[df1['short entry'], 'num units short'] = -1
df1.loc[df1['short exit'], 'num units short'] = 0
df1['num units short'][0] = 0
df1['num units short'] = df1['num units short'].fillna(method='pad')
df1['numUnits'] = df1['num units long'] + df1['num units short']

df1['port rets'] = price['pct chg'] * df1['numUnits'].shift(1)
#df1['port rets +1']=df1['port rets']+1
#df1['cum rets']=df1['port rets +1'].cumprod()
df1['cum rets1'] = df1['port rets'].cumsum()
df1['cum rets1'] = df1['cum rets1'] + 1
plt.plot(df1['cum rets1'])
plt.axvline(300, color='red', linestyle='--')
pd.set_option('display.max_rows',None)
pd.set_option('display.width',None)
pd.set_option('display.max_columns',None)


print(df1)
plt.show()