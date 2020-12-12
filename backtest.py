import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style
import time
import datetime
import os
from pandas.core.common import flatten
from functools import partial,reduce
import talib

account_size=10000
slippage=2
size=1
ATR_SL=2
plot_trades = True
plt.style.use('ggplot')

path='/Users/davidliao/Documents/code/Python/MyStudy/Backtest/data'
pairs_list=[]
for filename in os.listdir(path):
    if filename.endswith('.csv'):
        pairs_list.append(filename.split('.')[0])

df={}
for pair in range(len(pairs_list)):
    df[pair]=pd.read_csv(path+'/'+pairs_list[pair]+'.csv',parse_dates=['DateTime'])
    df[pair]=pd.DataFrame(df[pair])
    df[pair]['DateTime'] = pd.to_datetime(df[pair]['DateTime'],unit='s') 
    df[pair]=df[pair].set_index('DateTime')

plt.figure(figsize=(24,8))
plt.plot(df[0]['Close'],color='black')
plt.xlabel('Data',fontsize=18)
plt.ylabel('Price',fontsize=18)



def SMA(df,fast,slow):
    df['sma_fast']=df['Close'].rolling(fast).mean()
    df['sma_slow']=df['Close'].rolling(slow).mean()
    return df

def ATR(df,n):
    df=df.copy()
    df['High-Low']=abs(df['High']-df['Low'])
    df['High-prevClose']=abs(df['High']-df['Close'].shift(1))
    df['Low-prevClose']=abs(df['Low']-df['Close'].shift(1))
    df['TR']=df[['High-Low','High-prevClose','Low-prevClose']].max(axis=1,skipna=False)
    df['ATR']=df['TR'].rolling(n).mean()
    df=df.drop(['High-Low','High-prevClose','Low-prevClose'],axis=1)
    return df

# def RSI(df,n):
#     # signal=False
#     # df=df.copy()
   
#     # closes = self.df['Close']
#     df['RSI']=talib.RSI(df['Close'], timeperiod=n)
#     # df['RSI']=talib.RSI(np.array(closes), timeperiod=3)
    
#     # ul=80
#     # dl=20
#     # condition1=a[-1]<dl and a[-2]>=dl
#     # condition2=a[-1]>ul and a[-2]<=ul

#     # if condition1:
#     #     signal='BUY' if self.d==1 else 'SELL'

#     # if condition2:
#     #     signal='SELL' if self.d==1 else 'BUY'
#     print('df:',df)
#     return df

def RSI(df, period=3):
    # 整理資料
    Chg = df['Close']-df['Close'].shift(1)
    # Chg = Close - Close.shift(1)
    Chg_pos = pd.Series(index=Chg.index, data=Chg[Chg>0])
    Chg_pos = Chg_pos.fillna(0)
    Chg_neg = pd.Series(index=Chg.index, data=-Chg[Chg<0])
    Chg_neg = Chg_neg.fillna(0)
    
    # 計算period日平均漲跌幅度
    up_mean = []
    down_mean = []
    for i in range(period+1, len(Chg_pos)+1):
        up_mean.append(np.mean(Chg_pos.values[i-period:i]))
        down_mean.append(np.mean(Chg_neg.values[i-period:i]))
    
    # 計算 RSI
    rsi = []
    for i in range(len(up_mean)):
        rsi.append(100 * up_mean[i] / (up_mean[i]+down_mean[i]))
    rsi_series = pd.Series(index = df['Close'].index[period:], data = rsi)
    df['RSI']=rsi_series
    return df



for pair in range(len(pairs_list)):
    df[pair]['ATR']=ATR(df[pair],20)['ATR']
    df[pair]['sma_fast']=SMA(df[pair],50,200)
    df[pair]['RSI']=RSI(df[pair],3)['RSI']
    if 'JPY' not in pairs_list[pair]:
        df[pair]['spread']=float(slippage)/float(10000)
        df[pair]['size']=float(size)*float(10000)
        print('Pair: ',pairs_list[pair],'a')
    else:
        df[pair]['spread']=float(slippage)/float(100)
        df[pair]['size']=float(size)*float(100)
        print('Pair: ',pairs_list[pair],'b')
    




# print(df)
open_trade={}
trade={}
lep={}
sep={}
ltp={}
stp={}
lsl={}
ssl={}

for pair in range(len(pairs_list)):
    open_trade[pair]=[]
    trade[pair]={}
    lep[pair]=[]
    sep[pair]=[]
    ltp[pair]=[]
    stp[pair]=[]
    lsl[pair]=[]
    ssl[pair]=[]
    for i in range(20,len(df[pair])):
        # Buy
        if df[pair]['RSI'][i-1]>20 and df[pair]['RSI'][i]<=20 and len(open_trade[pair])==0:
        # if df[pair]['sma_fast'][i-1]<df[pair]['sma_slow'][i-1] and df[pair]['sma_fast'][i]>=df[pair]['sma_slow'][i] and len(open_trade[pair])==0:
            print(i,'New Long trade at price:',round(df[pair]['Close'][i],4),'On day:',df[pair].index[i],'Pair:',pairs_list[pair])
            trade[pair][i]={'ID':i,
                    'date_of_trade':df[pair].index[i],
                    'entry_price':round(df[pair]['Close'][i],4),
                    'signal':'Buy',
                    'result':0,
                    'TP':round(df[pair]['Close'][i]+df[pair]['ATR'][i]*ATR_SL,4),
                    'SL':round(df[pair]['Close'][i]-df[pair]['ATR'][i]*ATR_SL,4)
                    }
            open_trade[pair].append(i)
            lep[pair].append(trade[pair][i]['entry_price'])
            ltp[pair].append(trade[pair][i]['TP'])
            lsl[pair].append(trade[pair][i]['SL'])

        # Sell
        if df[pair]['RSI'][i-1]<80 and df[pair]['RSI'][i]>=80 and len(open_trade[pair])==0:
        # if df[pair]['sma_fast'][i-1]>df[pair]['sma_slow'][i-1] and df[pair]['sma_fast'][i]<=df[pair]['sma_slow'][i] and len(open_trade[pair])==0:
            print(i,'New Short trade at price:',round(df[pair]['Close'][i],4),'On day:',df[pair].index[i],'Pair:',pairs_list[pair])
            trade[pair][i]={'ID':i,
                    'date_of_trade':df[pair].index[i],
                    'entry_price':round(df[pair]['Close'][i],4),
                    'signal':'Sell',
                    'result':0,
                    'TP':round(df[pair]['Close'][i]-df[pair]['ATR'][i]*ATR_SL,4),
                    'SL':round(df[pair]['Close'][i]+df[pair]['ATR'][i]*ATR_SL,4)
                    }
            open_trade[pair].append(i)
            sep[pair].append(trade[pair][i]['entry_price'])
            stp[pair].append(trade[pair][i]['TP'])
            ssl[pair].append(trade[pair][i]['SL'])
        
        # Exit trades ----------------------------------------------------------------------------------------------
        # Buy profit
        if any(y<=df[pair]['Close'][i] for y in ltp[pair]):
            for j in open_trade[pair]:
                if trade[pair][j].get('result',{})==0 and trade[pair][j].get('signal',{})=='Buy':
                    if df[pair]['Close'][i]>=trade[pair][j]['TP']:
                        trade[pair][j].update({'result':(trade[pair][j]['TP']-trade[pair][j]['entry_price']-df[pair]['spread'][i])*df[pair]['size'][i]})
                        print(j,
                            'Long profit at price:',round(df[pair]['Close'][i],4),
                            'On day:',df[pair].index[i],
                            'With profit:',round(trade[pair][j]['result'],4),'\n')
                        open_trade[pair].remove(j)
                        ltp[pair].remove(trade[pair][j]['TP'])
                        lsl[pair].remove(trade[pair][j]['SL'])

        # Buy loss
        if any(y>=df[pair]['Close'][i] for y in lsl[pair]):
            for j in open_trade[pair]:
                if trade[pair][j].get('result',{})==0 and trade[pair][j].get('signal',{})=='Buy':
                    if df[pair]['Close'][i]<=trade[pair][j]['SL']:
                        trade[pair][j].update({'result':(trade[pair][j]['SL']-trade[pair][j]['entry_price']-df[pair]['spread'][i])*df[pair]['size'][i]})
                        print(j,
                            'Long loss at price:',round(df[pair]['Close'][i],4),
                            'On day:',df[pair].index[i],
                            'With loss:',round(trade[pair][j]['result'],4),'\n')
                        open_trade[pair].remove(j)
                        ltp[pair].remove(trade[pair][j]['TP'])
                        lsl[pair].remove(trade[pair][j]['SL'])

        # Sell profit
        if any(y>=df[pair]['Close'][i] for y in stp[pair]):
            for j in open_trade[pair]:
                if trade[pair][j].get('result',{})==0 and trade[pair][j].get('signal',{})=='Sell':
                    if df[pair]['Close'][i]<=trade[pair][j]['TP']:
                        trade[pair][j].update({'result':(trade[pair][j]['entry_price']-trade[pair][j]['TP']-df[pair]['spread'][i])*df[pair]['size'][i]})
                        print(j,
                            'Short profit at price:',round(df[pair]['Close'][i],4),
                            'On day:',df[pair].index[i],
                            'With profit:',round(trade[pair][j]['result'],4),'\n')
                        open_trade[pair].remove(j)
                        stp[pair].remove(trade[pair][j]['TP'])
                        ssl[pair].remove(trade[pair][j]['SL'])

        # Sell loss
        if any(y<=df[pair]['Close'][i] for y in ssl[pair]):
            for j in open_trade[pair]:
                if trade[pair][j].get('result',{})==0 and trade[pair][j].get('signal',{})=='Sell':
                    if df[pair]['Close'][i]>=trade[pair][j]['SL']:
                        trade[pair][j].update({'result':(trade[pair][j]['entry_price']-trade[pair][j]['SL']-df[pair]['spread'][i])*df[pair]['size'][i]})
                        print(j,
                            'Short loss at price:',round(df[pair]['Close'][i],4),
                            'On day:',df[pair].index[i],
                            'With loss:',round(trade[pair][j]['result'],4),'\n')
                        open_trade[pair].remove(j)
                        stp[pair].remove(trade[pair][j]['TP'])
                        ssl[pair].remove(trade[pair][j]['SL'])

        # Exit after time
        if len(open_trade[pair]) != 0:
            for j in open_trade[pair]:
                if (i-trade[pair][j]['ID']) >= 12 and trade[pair][j].get('result',{})==0 and trade[pair][j].get('signal',{})=='Buy': 
                    trade[pair][j].update({'result':(df[pair]['Close'][i]-trade[pair][j]['entry_price']-df[pair]['spread'][i])*df[pair]['size'][i]})
                    print(j,
                        'Long exit after 12 bars:',round(df[pair]['Close'][i],4),
                        'On day:',df[pair].index[i],
                        'With profit:',round(trade[pair][j]['result'],4),'\n')
                    open_trade[pair].remove(j)
                    ltp[pair].remove(trade[pair][j]['TP'])
                    lsl[pair].remove(trade[pair][j]['SL'])
                    # plot trade
                    # if plot_trades == True:
                    #     trade_plot(df[pair][i-1000:i+30],trade[pair][j],df[pair]['Close'][i],df[pair].index[i])
                elif (i-trade[pair][j]['ID']) >= 12 and trade[pair][j].get('result',{})==0 and trade[pair][j].get('signal',{})=='Sell': 
                    trade[pair][j].update({'result':(trade[pair][j]['entry_price']-df[pair]['Close'][i]-df[pair]['spread'][i])*df[pair]['size'][i]})
                    print(j,
                        'Short exit after 12 bars:',round(df[pair]['Close'][i],4),
                        'On day:',df[pair].index[i],
                        'With profit:',round(trade[pair][j]['result'],4),'\n')
                    open_trade[pair].remove(j)
                    stp[pair].remove(trade[pair][j]['TP'])
                    ssl[pair].remove(trade[pair][j]['SL'])
                    # plot trade
                    # if plot_trades == True:
                    #     trade_plot(df[pair][i-1000:i+30],trade[pair][j],df[pair]['Close'][i],df[pair].index[i])

pairs_results={}
profits={}
losses={}
be={}

for pair in range(len(pairs_list)):
    profits[pair]=[]
    losses[pair]=[]
    be[pair]=[]

    pairs_results[pair]=pd.DataFrame.from_dict({(i,j):trade[pair][j] for j in trade[pair].keys()},orient='index')
    pairs_results[pair]=pairs_results[pair].drop(['signal','ID','TP','SL',],axis=1)
    pairs_results[pair].set_index('date_of_trade',inplace=True)
    pairs_results[pair]['cum_res']=pairs_results[pair]['result'].cumsum()+account_size
  

for t in trade[pair]:
    profits[pair].append(trade[pair][t]['result']) if trade[pair][t]['result']>0.1 else ''
    losses[pair].append(trade[pair][t]['result']) if trade[pair][t]['result']<-0.1 else ''
    be[pair].append(trade[pair][t]['result']) if -0.1<= trade[pair][t]['result']<=0.1 else ''

my_reduce=partial(pd.merge,on='date_of_trade',how='outer')
strategy_results=reduce(my_reduce,pairs_results.values())
strategy_results=strategy_results.sort_index()
strategy_results['final_res']=strategy_results.filter(like='result',axis=1).sum(axis=1)
strategy_results['cum_res']=strategy_results['final_res'].cumsum()+account_size

print(strategy_results)

profits_keys=list(profits.keys())
profits_values=[profits[x] for x in profits_keys]
str_profits=list(flatten(profits_values))

losses_keys=list(losses.keys())
losses_values=[losses[x] for x in losses_keys]
str_losses=list(flatten(losses_values))

be_keys=list(be.keys())
be_values=[be[x] for x in be_keys]
str_be=list(flatten(be_values))

plt.figure(figsize=(26,10))
for pair in range(len(pairs_list)):
    plt.plot(pairs_results[pair]['cum_res'],label=pairs_list[pair])
plt.legend()
plt.title('Return of each pair',fontsize=18)

print('Strategy returns:',round(strategy_results['cum_res'][-1])-account_size)